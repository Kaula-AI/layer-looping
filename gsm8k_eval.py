"""
E8: GSM8K accuracy with per-token layer programs during greedy generation, at matched compute.

Policies:
  base                          unmodified model
  static --program repeat@13x2  every token gets that edit (also skip@L)
  router --lam X                ridge router (router_lag_interp.npz) on token t-1's features

Prompt tokens are prefilled with routed_forward (routes from base-pass features of t-1).
Generated tokens are decoded one at a time with decode.step and a first-pass KV cache; the router
reads features of the previous token's ACTUAL (routed) pass. That is a mild distribution shift from
training (base-pass features); report it. Swap programs are not used for routing.

Prints accuracy, mean signed extra layer applications per generated token, and writes per-problem
results to <out>/gsm8k_<policy>_<tag>.jsonl.
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
from ouro_model import load_causal_lm

from decode import Cache, step, token_features
from oracle import make_g
from routed_eval import lens_features, parse_program, routed_forward


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--policy", choices=["base", "static", "router"], default="base")
    p.add_argument("--program", default="repeat@13x2")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--router", default="oracle_out_full/router_lag_interp.npz")
    p.add_argument("--n_problems", type=int, default=250)
    p.add_argument("--n_shots", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out", default="oracle_out_full")
    return p.parse_args()


def extract(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    else:
        s = m.group(1)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


class Policy:
    def __init__(self, a, L, dev):
        self.kind, self.lam, self.L = a.policy, a.lam, L
        if a.policy == "static":
            self.fixed = parse_program(a.program)
        if a.policy == "router":
            r = np.load(a.router)
            self.mean, self.scale = r["mean"], r["scale"]
            self.coef, self.icpt = r["coef"], r["intercept"]
            self.lay, self.kk, self.cost = r["cfg_layer"], r["cfg_k"], r["cfg_cost"].astype(np.float32)
            self.ok = np.isin(r["cfg_op"].astype(str), ["repeat", "skip"])

    def route(self, prev_feat):
        """prev_feat: numpy (4L,) features of token t-1, or None. Returns (layer, k)."""
        if self.kind == "base" or (self.kind == "router" and prev_feat is None):
            return (-1, 1)
        if self.kind == "static":
            return self.fixed
        P = ((prev_feat - self.mean) / self.scale) @ self.coef.T + self.icpt
        obj = np.where(self.ok, P + self.lam * self.cost, np.inf)
        c = int(obj.argmin())
        return (int(self.lay[c]), int(self.kk[c])) if obj[c] < 0 else (-1, 1)


def cost_of(route):
    l, k = route
    return 0 if l < 0 else (-1 if k == 0 else k - 1)


@torch.no_grad()
def solve(g, tok, pol, prompt, max_new, dev):
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    T = ids.shape[1]
    sw = getattr(g.cfg, "sliding_window", None)
    assert sw is None or T + max_new <= sw, "prompt + generation must fit in the sliding window"
    rl = torch.full((1, T), -1, dtype=torch.long, device=dev)
    rk = torch.ones((1, T), dtype=torch.long, device=dev)
    feat = lens_features(g, ids)[0].float().cpu().numpy() if pol.kind == "router" else None
    for t in range(1, T):
        r = pol.route(feat[t - 1] if feat is not None else None)
        rl[0, t], rk[0, t] = r
    h, kvs = routed_forward(g, ids, rl, rk, return_kv=True)
    cache = Cache(len(g.layers))
    cache.k, cache.v = [k for k, _ in kvs], [v for _, v in kvs]
    prev_feat = feat[T - 1] if feat is not None else None
    nxt = g.logits(h[:, -1]).argmax(-1)
    out_ids, extra = [], 0
    for i in range(max_new):
        out_ids.append(int(nxt))
        text = tok.decode(out_ids)
        if "\n\nQuestion" in text or re.search(r"####\s*-?[\d,]*\.?\d+\s*\n", text) or int(nxt) == tok.eos_token_id:
            break
        r = pol.route(prev_feat)
        extra += cost_of(r)
        x, hs_t, ae = step(g, cache, nxt, T + i, r, want_hs=pol.kind == "router") if pol.kind == "router" \
            else (step(g, cache, nxt, T + i, r), None, None)
        if pol.kind == "router":
            prev_feat = token_features(g, hs_t, ae)[0].float().cpu().numpy()
        nxt = g.logits(x[:, 0]).argmax(-1)
    return tok.decode(out_ids).split("\n\nQuestion")[0], extra, len(out_ids)


def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    pol = Policy(a, len(g.layers), dev)
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n_problems))
    shots = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(a.n_shots))
    tag = {"base": "base", "static": a.program, "router": f"lam{a.lam}"}[a.policy]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"gsm8k_{a.policy}_{tag}.jsonl")
    correct, tot_extra, tot_tok, t0 = 0, 0, 0, time.time()
    with open(path, "w") as f:
        for n, ex in enumerate(test, 1):
            gen, extra, ntok = solve(g, tok, pol, shots + f"Question: {ex['question']}\nAnswer:", a.max_new_tokens, dev)
            gold = extract(ex["answer"])
            pred = extract(gen)
            ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
            correct += ok
            tot_extra += extra
            tot_tok += ntok
            f.write(json.dumps({"i": n - 1, "pred": pred, "gold": gold, "correct": ok, "extra_layers": extra,
                                "tokens": ntok, "generation": gen}) + "\n")
            if n % 10 == 0:
                print(f"{n}/{len(test)}  acc {correct / n:.3f}  extra/token {tot_extra / max(tot_tok, 1):+.3f}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"policy={a.policy} {tag}: accuracy {correct / len(test):.4f} ({correct}/{len(test)}), "
          f"extra layer applications per generated token {tot_extra / max(tot_tok, 1):+.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
