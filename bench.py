"""Downstream benchmarks with per-token layer routing (Ouro and the other supported models).

Multiple-choice tasks are scored like lm-eval-harness: the log-likelihood of each answer continuation
given the prompt; `acc` picks the highest total, `acc_norm` the highest per byte.

Semantics: every ANSWER token follows its policy's layer path; at each step it attends to the other
tokens' base-pass K/V of the layer being run plus its own current K/V (the diagonal swap, oracle.py).
That is exact for each token's own prediction, is the setting the router was trained in, and matches
OuroForCausalLM's exit_threshold evaluation (all rounds are computed for every token; each token's
logits come from its exit round). Batched: one base pass per group of sequences, then run_paths.

Policies (each reports accuracy with a 95% bootstrap CI and mean layer applications per answer token):
  base             the unmodified model (all rounds)
  rounds:R         every answer token runs R rounds (Ouro; R = total+1 adds one extra round)
  gate:q           Ouro's own learned exit gate: exit after the first round whose cumulative exit
                   probability reaches q (OuroForCausalLM's exit_threshold)
  static:NAME      every answer token runs one program from the router file, e.g. static:repeat@40x2
  router:lam       the trained ridge router (probe.py's router_lag_interp.npz) on token t-1's base-pass
                   features: argmin predicted dNLL + lam * cost, edits only where that is < 0
  routed           a routed_ouro.py checkpoint (--weights) with its trained router, hard decisions
  shuffled:lam     control: the router's choices for the whole task, randomly permuted across answer
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


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def answer_logprobs(g, batch, policies, R, seqs, dev, rmodel=None):
    """batch: list of (ids list, n answer tokens). Returns {policy: list of (per-token logprobs, layers per
    token)} for every sequence, plus the raw router choices (program index or -1) per answer token."""
    L = len(g.layers)
    B, T = len(batch), max(len(x) for x, _ in batch)
    ids = torch.zeros(B, T, dtype=torch.long, device=dev)
    for b, (x, _) in enumerate(batch):
        ids[b, : len(x)] = torch.tensor(x, device=dev)  # right padding: causal, so it never affects real tokens
    hs, kvs, attn_ent, cs = g.base_pass(ids)
    rows = [(b, t) for b, (x, na) in enumerate(batch) for t in range(len(x) - na - 1, len(x) - 1)]
    bi = torch.tensor([r[0] for r in rows], device=dev)
    ti = torch.tensor([r[1] for r in rows], device=dev)
    tgt = ids[bi, ti + 1]

    def lp_of(h_rows):  # h_rows: (n, d) final states of the scored rows
        return g.logits(h_rows).log_softmax(-1).gather(-1, tgt[:, None])[:, 0].float().cpu().numpy()

    def run_rows(paths):  # paths: one layer list per scored row; each distinct path runs once over the batch
        out = hs[-1][bi, ti].clone()
        groups = {}
        for k, p_ in enumerate(paths):
            if p_ != list(range(L)):
                groups.setdefault(tuple(p_), []).append(k)
        for p_, ks in groups.items():
            ks_t = torch.tensor(ks, device=dev)
            out[ks_t] = g.run_program(hs, kvs, cs, list(p_))[bi[ks_t], ti[ks_t]]
        return out

    choice = None
    if any(p_.split(":")[0] in ("router", "shuffled") for p_ in policies):
        ent, agree, upd = g.lens_features(hs)
        feat = torch.cat([ent, agree, upd, attn_ent.float()], -1).float()
        X = ((feat[bi, ti - 1].cpu().numpy() - R["mean"]) / R["scale"])  # token t-1's features
        choice = X @ R["coef"].T + R["intercept"]  # predicted dNLL per program
    out = {}
    for pol in policies:
        kind, _, arg = pol.partition(":")
        if kind == "base":
            lp, lay = lp_of(hs[-1][bi, ti]), np.full(len(rows), float(L))
        elif kind == "routed":  # stage-2 model: its trained router picks each token's path in-pass
            lg, lr_, _, _ = rmodel(ids, hard_eval=True)
            lp = lg[bi, ti].float().log_softmax(-1).gather(-1, tgt[:, None])[:, 0].cpu().numpy()
            lay = lr_[bi, ti].float().cpu().numpy()
        elif kind == "rounds" and int(arg) <= g.rounds:
            n = g.n_phys * int(arg)
            lp, lay = lp_of(hs[n][bi, ti]), np.full(len(rows), float(n))
        elif kind == "rounds":
            p_ = list(range(L)) + list(range(L - g.n_phys, L)) * (int(arg) - g.rounds)
            lp, lay = lp_of(run_rows([p_] * len(rows))), np.full(len(rows), float(len(p_)))
        elif kind == "gate":
            cum = g.exit_gate_probs(hs)[bi, ti].cumsum(-1)
            hit = cum >= float(arg)
            s_ = torch.where(hit.any(-1), hit.float().argmax(-1) + 1, torch.full_like(hit[:, 0], g.rounds, dtype=torch.long))
            H = torch.stack([hs[g.n_phys * (r + 1)][bi, ti] for r in range(g.rounds)], 1)
            lp = lp_of(H[torch.arange(len(rows), device=dev), s_ - 1])
            lay = (s_ * g.n_phys).float().cpu().numpy()
        elif kind == "static":
            c = [prog_name(R, i) for i in range(len(seqs))].index(arg)
            lp, lay = lp_of(run_rows([seqs[c]] * len(rows))), np.full(len(rows), float(len(seqs[c])))
        elif kind in ("router", "shuffled"):
            obj = choice + float(arg) * R["cfg_cost"][None]
            ch = obj.argmin(1)
            ch = np.where(obj[np.arange(len(ch)), ch] < 0, ch, -1)
            out[f"_choice:{arg}"] = ch
            continue  # filled in by the caller (shuffled needs choices pooled over the task)
        else:
            raise ValueError(pol)
        out[pol] = (lp, lay)
    return out, rows, run_rows, lp_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--router", default=None)
    p.add_argument("--tasks", nargs="+", default=["arc_challenge", "arc_easy", "hellaswag", "mmlu", "winogrande"])
    p.add_argument("--n", type=int, default=500, help="items per task (fixed random subset)")
    p.add_argument("--policies", nargs="+", default=["base", "gate:0.5", "router:0", "shuffled:0"])
    p.add_argument("--batch", type=int, default=32, help="sequences per batched pass")
    p.add_argument("--weights", default=None, help="routed_ouro.py model.pt: fine-tuned weights (+ router)")
    p.add_argument("--out", default="bench_out")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16, attn_implementation="eager").to(dev).eval()
    rmodel = None
    if a.weights:
        from routed_ouro import RoutedOuro

        ck = torch.load(a.weights, map_location=dev)
        model.load_state_dict(ck["ouro"])
        if ck["use_router"]:
            rmodel = RoutedOuro(model).to(dev).eval()
            rmodel.heads.load_state_dict(ck["heads"])
    g = make_g(model)
    R = dict(np.load(a.router)) if a.router else None
    seqs = program_seqs(R) if R is not None else None
    for pol in a.policies:
        kind = pol.split(":")[0]
        assert kind in ("base", "rounds", "gate", "static", "router", "shuffled", "routed"), pol
        assert kind != "routed" or rmodel is not None, "routed needs --weights from a router checkpoint"
        assert kind not in ("rounds", "gate") or isinstance(g, GO), f"{pol} needs a looped model"
        assert kind not in ("static", "router", "shuffled") or R is not None, f"{pol} needs --router"
    L = len(g.layers)
    summary = json.load(open(os.path.join(a.out, "summary.json"))) if os.path.exists(os.path.join(a.out, "summary.json")) else {}
    for task in a.tasks:
        t0 = time.time()
        items = load_task(task, a.n)
        seq_list, owner = [], []  # every (item, choice) sequence
        for ii, (ctx, conts, gold) in enumerate(items):
            ctxs = ctx if isinstance(ctx, list) else [ctx] * len(conts)
            for ci, (c_, k_) in enumerate(zip(ctxs, conts)):
                ci_ids = tok(c_, add_special_tokens=False)["input_ids"]
                a_ids = tok(k_, add_special_tokens=False)["input_ids"]
                seq_list.append(([tok.bos_token_id] + ci_ids + a_ids, len(a_ids)))
                owner.append((ii, ci))
        order = np.argsort([len(x) for x, _ in seq_list])  # length-sorted batches: little padding
        tok_lp = {pol: [None] * len(seq_list) for pol in a.policies}
        tok_lay = {pol: [None] * len(seq_list) for pol in a.policies}
        pending = {}
        for s0 in range(0, len(order), a.batch):
            if s0 // a.batch % 20 == 0:
                print(f"  {task}: {s0}/{len(order)} sequences, {(time.time() - t0) / 60:.1f} min", flush=True)
            idx = order[s0: s0 + a.batch]
            batch = [seq_list[i] for i in idx]
            out, rows, run_rows, lp_of = answer_logprobs(g, batch, a.policies, R, seqs, dev, rmodel)
            seg = np.cumsum([0] + [na for _, na in batch])
            for pol in a.policies:
                kind, _, arg = pol.partition(":")
                if kind in ("router", "shuffled"):
                    ch = out[f"_choice:{arg}"]
                    if kind == "shuffled":
                        pending.setdefault(arg, []).append((idx, seg, ch))
                        continue
                    paths = [list(range(L)) if c < 0 else seqs[c] for c in ch]
                    lp, lay = lp_of(run_rows(paths)), np.array([len(q) for q in paths], float)
                else:
                    lp, lay = out[pol]
                for k, i in enumerate(idx):
                    tok_lp[pol][i], tok_lay[pol][i] = lp[seg[k]: seg[k + 1]], lay[seg[k]: seg[k + 1]]
        # shuffled control: pool the router's choices over the task, permute, re-run with the same batches
        for arg, parts in pending.items():
            pol = f"shuffled:{arg}"
            allc = np.random.default_rng(0).permutation(np.concatenate([ch for _, _, ch in parts]))
            pos = 0
            for idx, seg, ch in parts:
                batch = [seq_list[i] for i in idx]
                _, rows, run_rows, lp_of = answer_logprobs(g, batch, ["base"], R, seqs, dev)
                chs = allc[pos: pos + len(ch)]
                pos += len(ch)
                paths = [list(range(L)) if c < 0 else seqs[c] for c in chs]
                lp, lay = lp_of(run_rows(paths)), np.array([len(q) for q in paths], float)
                for k, i in enumerate(idx):
                    tok_lp[pol][i], tok_lay[pol][i] = lp[seg[k]: seg[k + 1]], lay[seg[k]: seg[k + 1]]
        for pol in a.policies:
            ll = {}
            for (ii, ci), lp in zip(owner, tok_lp[pol]):
                ll.setdefault(ii, {})[ci] = float(lp.sum())
            rows_out = []
            for ii, (ctx, conts, gold) in enumerate(items):
                v = np.array([ll[ii][ci] for ci in range(len(conts))])
                nb = np.array([len(k_.encode()) for k_ in conts])
                rows_out.append(dict(gold=gold, ll=v.tolist(), acc=int(v.argmax() == gold),
                                     acc_norm=int((v / nb).argmax() == gold)))
            acc = np.array([r["acc"] for r in rows_out])
            accn = np.array([r["acc_norm"] for r in rows_out])
            boot = np.random.default_rng(1).integers(0, len(rows_out), (1000, len(rows_out)))
            lay = float(np.concatenate(tok_lay[pol]).mean())
            res = dict(n=len(rows_out), acc=float(acc.mean()), acc_norm=float(accn.mean()),
                       acc_ci=[float(np.percentile(acc[boot].mean(1), q)) for q in (2.5, 97.5)],
                       acc_norm_ci=[float(np.percentile(accn[boot].mean(1), q)) for q in (2.5, 97.5)],
                       layers_per_answer_token=lay, minutes=(time.time() - t0) / 60)
            summary[f"{task}|{pol}"] = res
            print(f"{task:14s} {pol:14s} acc {res['acc']:.3f} [{res['acc_ci'][0]:.3f},{res['acc_ci'][1]:.3f}]  "
                  f"acc_norm {res['acc_norm']:.3f} [{res['acc_norm_ci'][0]:.3f},{res['acc_norm_ci'][1]:.3f}]  "
                  f"layers/tok {lay:6.1f}", flush=True)
            with open(os.path.join(a.out, f"{task}_{pol.replace(':', '_')}.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows_out)
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        print(f"  ({task}: {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
