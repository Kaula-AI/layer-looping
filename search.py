"""
E9: per-token beam search over layer paths (repeat / skip / swap / reverse / loop), frozen model.

Round 1 scores every single edit of the base path for all positions at once (shared path, cheap).
Rounds 2..R extend each token's own beam: edits of each beam path anchored on the token's
`focus` most promising layers from round 1, scored with the mixed-path executor (G2.run_paths),
which gives each position's exact next-token loss under its own path.

For every position we keep the Pareto front over (cost, dNLL) of all paths evaluated for it.
Writes <out>/search.npz.

  python search.py --n_tokens 20000 --out programs_quick
"""
import argparse
import os
import time

import numpy as np
import torch

from programs import candidate_edits, path_cost

PAD = -1


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=100_000)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--focus", type=int, default=6, help="layers kept per token for rounds >= 2")
    p.add_argument("--max_cands", type=int, default=24, help="candidates per beam path in rounds >= 2")
    p.add_argument("--lam_search", type=float, default=0.01, help="beam objective dNLL + lam*cost")
    p.add_argument("--protect", type=int, default=2)
    p.add_argument("--max_extra", type=int, default=6)
    p.add_argument("--max_skip", type=int, default=4)
    p.add_argument("--keep", type=int, default=8, help="Pareto points stored per token")
    p.add_argument("--raw_layer", type=int, default=None)
    p.add_argument("--out", default="programs_out")
    return p.parse_args(argv)


def pareto(pool, keep):
    """pool: dict path_tuple -> (dnll, cost). Non-dominated points (lower dnll, lower cost), best first."""
    pts = sorted(pool.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    front, best = [], float("inf")
    for path, (dn, c) in pts:
        if dn < best - 1e-9:
            front.append((path, dn, c))
            best = dn
    front.sort(key=lambda z: z[1])
    return front[:keep]


@torch.no_grad()
def search_batch(g, ids, hs, kvs, cs, base_nll, a):
    """Returns, for each (b, t) with t < T-1, the Pareto front list [(path, dnll, cost)]."""
    L = len(g.layers)
    B, T = ids.shape
    base = list(range(L))
    kw = dict(protect=a.protect, max_extra=a.max_extra, max_skip=a.max_skip)
    pools = [[{} for _ in range(T - 1)] for _ in range(B)]

    # round 1: every single edit of the base path, shared by all positions
    r1 = candidate_edits(base, L, **kw)
    d1 = torch.empty(B, T - 1, len(r1), device=ids.device)
    for c, (_, path) in enumerate(r1):
        d1[..., c] = g.nll(g.run_program(hs, kvs, cs, path), ids) - base_nll
    d1 = d1.float().cpu().numpy()
    costs1 = np.array([path_cost(p, L) for _, p in r1], dtype=np.float32)
    anchor = np.array([base[e[1]] for e, _ in r1])
    beams, focus = {}, {}
    for b in range(B):
        for t in range(T - 1):
            row = d1[b, t]
            pool = pools[b][t]
            for c, (_, path) in enumerate(r1):
                pool[tuple(path)] = (float(row[c]), float(costs1[c]))
            obj = row + a.lam_search * costs1
            top = np.argsort(obj)[: a.beam]
            beams[b, t] = [r1[c][1] for c in top]
            focus[b, t] = set(list(dict.fromkeys(anchor[np.argsort(row)].tolist()))[: a.focus])

    # rounds 2..R: per-token candidates, scored with the mixed-path executor
    for _ in range(a.rounds - 1):
        cand = {}
        for (b, t), bl in beams.items():
            lst = []
            for path in bl:
                ce = candidate_edits(path, L, focus=focus[b, t], **kw)
                if len(ce) > a.max_cands:  # deterministic thinning, keep spread across the path
                    step = len(ce) / a.max_cands
                    ce = [ce[int(q * step)] for q in range(a.max_cands)]
                lst.extend(p for _, p in ce)
            cand[b, t] = [p for p in dict.fromkeys(map(tuple, lst)) if p not in pools[b][t]]
        n_slots = max((len(v) for v in cand.values()), default=0)
        if n_slots == 0:
            break
        S = max(len(p) for v in cand.values() for p in v)
        results = {}
        for m in range(n_slots):
            P = torch.full((B, T, S), PAD, dtype=torch.long)
            P[:, :, :L] = torch.arange(L)
            live = []
            for (b, t), v in cand.items():
                if m < len(v):
                    P[b, t] = PAD
                    P[b, t, : len(v[m])] = torch.tensor(v[m])
                    live.append((b, t))
            if not live:
                continue
            dn = (g.nll(g.run_paths(hs, kvs, cs, P.to(ids.device)), ids) - base_nll).float().cpu().numpy()
            for b, t in live:
                results[b, t, m] = float(dn[b, t])
        for (b, t), v in cand.items():
            pool = pools[b][t]
            for m, path in enumerate(v):
                if (b, t, m) in results:
                    pool[path] = (results[b, t, m], float(len(path) - L))
            ranked = sorted(pool.items(), key=lambda kv: kv[1][0] + a.lam_search * kv[1][1])
            beams[b, t] = [list(p) for p, _ in ranked[: a.beam]]
    return [[pareto(pools[b][t], a.keep) for t in range(T - 1)] for b in range(B)]


def main():
    from transformers import AutoTokenizer
    from ouro_model import load_causal_lm

    from oracle import base_records, feat_names, make_chunks, make_g

    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    L = len(g.layers)
    raw_layer = a.raw_layer if a.raw_layer is not None else L // 2
    chunks, names = make_chunks(tok, a.n_tokens, a.seq_len)
    Smax = L + a.max_extra
    rec = {k: [] for k in ["front_path", "front_dnll", "front_cost", "feat", "base_nll", "raw", "tok", "chunk", "pos"]}
    t0 = time.time()
    for bi in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[bi : bi + a.batch], device=dev)
        B, T = ids.shape
        hs, kvs, cs, base_nll, feat, raw = base_records(g, ids, raw_layer)
        fronts = search_batch(g, ids, hs, kvs, cs, base_nll, a)
        fp = np.full((B, T - 1, a.keep, Smax), PAD, dtype=np.int8)
        fd = np.full((B, T - 1, a.keep), np.nan, dtype=np.float32)
        fc = np.full((B, T - 1, a.keep), np.nan, dtype=np.float32)
        for b in range(B):
            for t in range(T - 1):
                for q, (path, dn, c) in enumerate(fronts[b][t]):
                    fp[b, t, q, : len(path)] = path
                    fd[b, t, q], fc[b, t, q] = dn, c
        rec["front_path"].append(fp.reshape(-1, a.keep, Smax))
        rec["front_dnll"].append(fd.reshape(-1, a.keep))
        rec["front_cost"].append(fc.reshape(-1, a.keep))
        rec["feat"].append(feat.half().cpu().numpy().reshape(-1, feat.shape[-1]))
        rec["base_nll"].append(base_nll.float().cpu().numpy().reshape(-1))
        rec["raw"].append(raw.half().cpu().numpy().reshape(-1, raw.shape[-1]))
        rec["tok"].append(ids[:, :-1].cpu().numpy().reshape(-1))
        rec["chunk"].append(np.repeat(np.arange(bi, bi + B), T - 1))
        rec["pos"].append(np.tile(np.arange(T - 1), B))
        done, el = bi + B, time.time() - t0
        print(f"{done}/{len(chunks)} chunks | {el / 60:.1f} min | ETA {el / done * (len(chunks) - done) / 60:.1f} min",
              flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out.update(source=np.array(names), n_layers=np.array(L), feat_names=feat_names(L))
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "search.npz"), **out)
    best = np.nanmin(out["front_dnll"], 1)
    print(f"saved {a.out}/search.npz  ({len(best):,} positions); mean best dNLL {np.minimum(best, 0).mean():+.4f}")


if __name__ == "__main__":
    main()
