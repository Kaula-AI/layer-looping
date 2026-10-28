"""
E12: does the per-token path gain survive when EVERY token follows its chosen path at once?

Token-by-token teacher-forced decoding (decode.decode_nll_paths, first-pass KV cache) on held-out
chunks. Paths are chosen per token from the library by:
  oracle      argmin of the library's exact per-token dNLL + lambda*cost
  controller  the trained MLP (controller.pt, 'mlp_lag') on token t-1's base-pass features
Reports routed dNLL against the sum of the independent per-token deltas (the interaction gap)
and the extra layer applications per token.

  python program_eval.py --library programs_quick/library.npz --controller programs_quick/controller.pt \\
         --n_chunks 8 --lam 0.0
"""
import argparse
import json
import os

import numpy as np
import torch

from controller import Controller
from decode import decode_nll_paths
from oracle import G2, lens_features_np, make_chunks

PAD = -1


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--library", default="programs_out/library.npz")
    p.add_argument("--controller", default="programs_out/controller.pt")
    p.add_argument("--policy", choices=["oracle", "controller"], nargs="+", default=["oracle", "controller"])
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--n_chunks", type=int, default=8, help="held-out chunks to decode (each 511 steps)")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--n_tokens", type=int, default=100_000, help="must match library.py")
    p.add_argument("--out", default="programs_out")
    return p.parse_args()


def route_and_decode(g, ids, P, cost, paths, base_idx, lam):
    """P: (T-1, V) predicted or exact dNLL per position and library path. Returns choices and routed NLL."""
    obj = P + lam * cost[None]
    ch = obj.argmin(1)
    ch = np.where((obj[np.arange(len(P)), ch] < 0) & (ch != base_idx), ch, -1)
    L = len(g.layers)
    route = [paths[q] if q >= 0 else list(range(L)) for q in ch] + [list(range(L))]
    return ch, decode_nll_paths(g, ids, route)[0].float().cpu().numpy()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    lib = np.load(a.library)
    T1 = a.seq_len - 1
    delta = lib["delta"].astype(np.float32).reshape(-1, T1, lib["delta"].shape[1])
    cost = lib["cfg_cost"].astype(np.float32)
    paths = [[int(v) for v in row if v != PAD] for row in lib["cfg_path"]]
    base_idx = int(np.where(lib["cfg_op"].astype(str) == "base")[0][0])
    nll_all = lib["base_nll"].reshape(-1, T1)
    ck = torch.load(a.controller, map_location="cpu", weights_only=False) if "controller" in a.policy else None
    if ck is not None:
        m = ck["models"]["mlp_lag"]
        ctrl = Controller(m["d_in"], len(paths), ck["hidden"]).to(dev).eval()
        ctrl.load_state_dict(m["state"])
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = G2(model)
    chunks, _ = make_chunks(tok, a.n_tokens, a.seq_len)
    test_ids = [c for c in range(len(chunks)) if c % 5 == 0][: a.n_chunks]
    res = {}
    for pol in a.policy:
        tot = dict(routed=0.0, indep=0.0, extra=0.0, n=0)
        for c in test_ids:
            ids = torch.tensor([chunks[c]], device=dev)
            if pol == "oracle":
                P = delta[c]
            else:
                feat = lens_features_np(g, ids)  # (T-1, 4L) base-pass features
                X = np.c_[feat[: T1 - 1], nll_all[c][: T1 - 1]]  # token t-1's features predict position t
                X = (X - m["mean"]) / m["scale"]
                with torch.no_grad():
                    Pp = ctrl(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
                P = np.vstack([np.full((1, len(paths)), 1e9, np.float32), Pp])  # position 0: no edit
            ch, routed = route_and_decode(g, ids, P, cost, paths, base_idx, a.lam)
            tot["routed"] += float((routed - nll_all[c]).sum())
            tot["indep"] += float(np.where(ch >= 0, delta[c][np.arange(T1), np.maximum(ch, 0)], 0).sum())
            tot["extra"] += float(np.where(ch >= 0, cost[np.maximum(ch, 0)], 0).sum())
            tot["n"] += T1
            print(f"  [{pol}] chunk {c}: routed dNLL {(routed - nll_all[c]).mean():+.4f}", flush=True)
        n = tot["n"]
        res[pol] = dict(routed_dnll=tot["routed"] / n, independent_sum_dnll=tot["indep"] / n,
                        extra_layers=tot["extra"] / n, tokens=n, lam=a.lam)
        r = res[pol]
        print(f"{pol}: routed dNLL {r['routed_dnll']:+.4f} vs independent {r['independent_sum_dnll']:+.4f} "
              f"(gap {r['routed_dnll'] - r['independent_sum_dnll']:+.4f}); extra layers/token {r['extra_layers']:+.3f}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"program_eval_lam{a.lam}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
