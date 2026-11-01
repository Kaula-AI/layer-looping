"""Downstream benchmarks with per-token layer routing (Ouro and the other supported models).

Multiple-choice tasks are scored like lm-eval-harness: the log-likelihood of each answer continuation
given the prompt; `acc` picks the highest total, `acc_norm` the highest per byte. The prompt always runs
the base path (all rounds); every ANSWER token is then decoded one at a time on its policy's path with a
first-pass K/V cache (decode.step_path), so later answer tokens see the K/V the routed tokens wrote.

Policies (each reports accuracy and mean layer applications per answer token):
  base             the unmodified model (all rounds)
  rounds:R         every answer token runs R rounds (Ouro; R = total+1 is one extra round)
  gate:q           Ouro's own learned exit gate, decided in-pass: exit after the first round whose
                   cumulative exit probability reaches q (OuroForCausalLM's exit_threshold)
  static:NAME      every answer token runs one program from the router file, e.g. static:repeat@40x2
  router:lam       the trained ridge router (probe.py's router_lag_interp.npz) on token t-1's base-pass
                   features: argmin predicted dNLL + lam * cost, edits only where that is < 0
  shuffled:lam     control: the router's choices for this item, randomly permuted across its answer
                   tokens (same edits, same compute, no token-level signal)

  python bench.py --model ByteDance/Ouro-1.4B --router ouro_quick/router_lag_interp.npz \\
      --tasks arc_challenge hellaswag --n 500 --policies base gate:0.5 router:0 shuffled:0
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from decode import Cache, _attend, _token_ctx, step_path
from oracle import GO, make_g
from ouro_model import load_causal_lm


# ---------------------------------------------------------------- tasks: (context, [continuations], gold)
def _hs_pre(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def load_task(name, n, seed=0):
    if name in ("arc_challenge", "arc_easy"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge" if name == "arc_challenge" else "ARC-Easy", split="test")
        items = [(f"Question: {r['question']}\nAnswer:", [" " + c for c in r["choices"]["text"]],
                  r["choices"]["label"].index(r["answerKey"])) for r in ds]
    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        items = [(_hs_pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()),
                  [" " + _hs_pre(e) for e in r["endings"]], int(r["label"])) for r in ds]
    elif name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        items = []
        for r in ds:
            q = r["question"].strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip("ABCD", r["choices"])) + "Answer:"
            items.append((q, [" A", " B", " C", " D"], int(r["answer"])))
    elif name == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        items = []
        for r in ds:  # partial scoring: the option fills the blank in the context, score the shared remainder
            i = r["sentence"].index("_")
            rest = " " + r["sentence"][i + 1:].strip()
            items.append(([r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]], [rest, rest],
                          int(r["answer"]) - 1))
    else:
        raise ValueError(name)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))[:n] if n and n < len(items) else np.arange(len(items))
    return [items[i] for i in sorted(idx)]


# ---------------------------------------------------------------- programs
def program_seqs(R):
    """Layer sequence for every program in a router file."""
    L = int(R["n_layers"])
    base = list(range(L))
    out = []
    for o, l, e, k in zip(R["cfg_op"].astype(str), R["cfg_layer"], R["cfg_end"], R["cfg_k"]):
        l, e, k = int(l), int(e), int(k)
        if o == "repeat":
            out.append(base[: l + 1] + [l] * (k - 1) + base[l + 1:])
        elif o == "skip":
            out.append(base[:l] + base[l + 1:])
        elif o == "swap":
            out.append(base[:l] + [l + 1, l] + base[l + 2:])
        elif o == "block":
            out.append(base[: e + 1] + list(range(l, e + 1)) * (k - 1) + base[e + 1:])
        elif o == "exit":
            out.append(base[: l + 1])
        else:
            raise ValueError(o)
    return out


def prog_name(R, c):
    o, l, k = str(R["cfg_op"][c]), int(R["cfg_layer"][c]), int(R["cfg_k"][c])
    return f"{o}@{l}" + (f"x{k}" if o in ("repeat", "block") else "")


# ---------------------------------------------------------------- Ouro gate, decided in-pass
@torch.no_grad()
def step_gate(g, cache, token, pos, q):
    """Base path, but stop after the first round whose cumulative exit probability >= q. Layers not run
    write K/V projected from the final state (step_path's rule). Returns final state and layers run."""
    L, n = len(g.layers), g.n_phys
    x, cs = _token_ctx(g, token, pos)
    pend, rem, cum, ran = [None] * L, torch.ones(x.shape[0], device=x.device), 0.0, L
    for li in range(L):
        qq, k, v = g.qkv(li, x, cs)
        pend[li] = (k, v)
        x, _ = _attend(g, li, x, qq, k, v, cache.k[li], cache.v[li], cs)
        if li % n == n - 1 and li < L - 1:
            lam = torch.sigmoid(g.m.early_exit_gate(g.norm(x)).float()[:, 0, 0])
            cum = cum + lam * rem
            rem = rem * (1 - lam)
            if bool((cum >= q).all()):
                ran = li + 1
                break
    for li in range(ran, L):
        pend[li] = g.kv_of(li, x, cs)
    for li in range(L):
        cache.append(li, *pend[li])
    return x, ran


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score(g, tok, ctx, cont, policy, R, seqs, dev, rng):
    """Log-likelihood of cont given ctx under a policy. Returns (sum logprob, n answer tokens, layers run)."""
    c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    a_ids = tok(cont, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([[tok.bos_token_id] + c_ids + a_ids], device=dev)
    T, na = ids.shape[1], len(a_ids)
    L = len(g.layers)
    hs, kvs, attn_ent, cs = g.base_pass(ids)
    if policy == "base":
        lp = g.logits(hs[-1][0, T - na - 1: T - 1]).log_softmax(-1)
        return float(lp.gather(-1, ids[0, T - na:, None]).sum()), na, float(L)
    p0 = T - na - 1  # first position whose prediction is scored (the last prompt token)
    cache = Cache(L)
    for li in range(L):
        cache.k[li], cache.v[li] = kvs[li][0][:, :, :p0], kvs[li][1][:, :, :p0]
    kind, _, arg = policy.partition(":")
    paths = None
    if kind == "rounds":
        paths = [list(range(g.n_phys * int(arg)))] * na if int(arg) <= g.rounds else \
            [list(range(L)) + list(range(L - g.n_phys, L)) * (int(arg) - g.rounds)] * na
    elif kind == "static":
        c = [prog_name(R, i) for i in range(len(seqs))].index(arg)
        paths = [seqs[c]] * na
    elif kind in ("router", "shuffled"):
        ent, agree, upd = g.lens_features(hs)
        feat = torch.cat([ent, agree, upd, attn_ent.float()], -1)[0].float().cpu().numpy()  # T, 4L
        X = (feat[p0 - 1: T - 2] - R["mean"]) / R["scale"]  # token t-1's features for t = p0 .. T-2
        P = X @ R["coef"].T + R["intercept"]
        obj = P + float(arg) * R["cfg_cost"][None]
        ch = obj.argmin(1)
        ch = np.where(obj[np.arange(len(ch)), ch] < 0, ch, -1)
        if kind == "shuffled":
            ch = rng.permutation(ch)
        paths = [list(range(L)) if c < 0 else seqs[c] for c in ch]
    lp_sum, layers = 0.0, 0
    for t in range(p0, T - 1):
        if kind == "gate":
            h, ran = step_gate(g, cache, ids[:, t], t, float(arg))
        else:
            path = paths[t - p0]
            h, ran = step_path(g, cache, ids[:, t], t, path), len(path)
        lp_sum += float(g.logits(h[:, 0]).log_softmax(-1)[0, ids[0, t + 1]])
        layers += ran
    return lp_sum, na, layers / na


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--router", default=None)
    p.add_argument("--tasks", nargs="+", default=["arc_challenge", "arc_easy", "hellaswag", "mmlu", "winogrande"])
    p.add_argument("--n", type=int, default=500, help="items per task (fixed random subset)")
    p.add_argument("--policies", nargs="+", default=["base", "gate:0.5", "router:0", "shuffled:0"])
    p.add_argument("--out", default="bench_out")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16, attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    R = dict(np.load(a.router)) if a.router else None
    seqs = program_seqs(R) if R is not None else None
    for pol in a.policies:
        assert pol in ("base",) or pol.split(":")[0] in ("rounds", "gate", "static", "router", "shuffled"), pol
        assert not pol.startswith(("rounds", "gate")) or isinstance(g, GO), f"{pol} needs a looped model"
        assert not pol.startswith(("static", "router", "shuffled")) or R is not None, f"{pol} needs --router"
    summary = {}
    for task in a.tasks:
        items = load_task(task, a.n)
        for pol in a.policies:
            rng = np.random.default_rng(0)
            t0, rows = time.time(), []
            for ctx, conts, gold in items:
                ctxs = ctx if isinstance(ctx, list) else [ctx] * len(conts)
                sc = [score(g, tok, c_, k_, pol, R, seqs, dev, rng) for c_, k_ in zip(ctxs, conts)]
                ll = np.array([s[0] for s in sc])
                nb = np.array([len(k_.encode()) for k_ in conts])
                rows.append(dict(gold=gold, ll=ll.tolist(), acc=int(ll.argmax() == gold),
                                 acc_norm=int((ll / nb).argmax() == gold), layers=float(np.mean([s[2] for s in sc]))))
            acc = np.array([r["acc"] for r in rows])
            accn = np.array([r["acc_norm"] for r in rows])
            boot = np.random.default_rng(1).integers(0, len(rows), (1000, len(rows)))
            res = dict(n=len(rows), acc=float(acc.mean()), acc_norm=float(accn.mean()),
                       acc_ci=[float(np.percentile(acc[boot].mean(1), q)) for q in (2.5, 97.5)],
                       layers_per_answer_token=float(np.mean([r["layers"] for r in rows])),
                       minutes=(time.time() - t0) / 60)
            summary[f"{task}|{pol}"] = res
            print(f"{task:14s} {pol:14s} acc {res['acc']:.3f} [{res['acc_ci'][0]:.3f},{res['acc_ci'][1]:.3f}] "
                  f"acc_norm {res['acc_norm']:.3f}  layers/tok {res['layers_per_answer_token']:6.1f}  "
                  f"({res['minutes']:.1f} min)", flush=True)
            with open(os.path.join(a.out, f"{task}_{pol.replace(':', '_')}.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
