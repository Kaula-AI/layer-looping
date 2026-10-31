"""
E10: build a shared program library and score it exactly for every position.

Library = base-path neighbours (every single edit of the base path, so a controller can always
fall back to what the single-edit router could do) + the `n_composite` multi-edit paths that
appear most often on TRAIN-chunk Pareto fronts from search.npz. Each library path is then scored
for every position with one shared-path pass (G2.run_program, exact for each position's own loss).

Writes <out>/library.npz in the same format as oracle.npz (delta, feat, raw, base_nll, cfg_*), so
probe.py and analyze_tokens.py run on it unchanged, plus cfg_path / cfg_name with the paths.

  python library.py --search programs_quick/search.npz --n_tokens 20000 --out programs_quick
"""
import argparse
import collections
import os
import time

import numpy as np
import torch

from programs import path_cost, path_key, single_edit_library

PAD = -1


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--search", default="programs_out/search.npz")
    p.add_argument("--n_composite", type=int, default=128)
    p.add_argument("--front_weight", default="best", choices=["best", "front"],
                   help="count only each token's best path, or every path on its front")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=100_000, help="must match search.py")
    p.add_argument("--protect", type=int, default=2)
    p.add_argument("--max_extra", type=int, default=6)
    p.add_argument("--max_skip", type=int, default=4)
    p.add_argument("--raw_layer", type=int, default=None)
    p.add_argument("--out", default="programs_out")
    return p.parse_args(argv)


def build_library(S, L, a):
    """S: loaded search.npz. Returns list of dicts: op, layer, k, cost, path, name."""
    lib = [dict(op="base", layer=-1, k=1, path=list(range(L)))]
    for kind, anchor, w, path in single_edit_library(L, protect=a.protect, max_extra=a.max_extra,
                                                     max_skip=a.max_skip):
        lib.append(dict(op=kind, layer=anchor, k=w, path=path))
    have = {path_key(e["path"]) for e in lib}
    train = S["chunk"] % 5 != 0
    fp, fd = S["front_path"][train], S["front_dnll"][train]
    counts = collections.Counter()
    for r in range(len(fp)):
        q_idx = [int(np.nanargmin(fd[r]))] if a.front_weight == "best" and not np.all(np.isnan(fd[r])) else \
            [q for q in range(fd.shape[1]) if not np.isnan(fd[r, q])]
        for q in q_idx:
            if fd[r, q] < 0:
                path = [int(v) for v in fp[r, q] if v != PAD]
                k = path_key(path)
                if k not in have:
                    counts[k] += 1
    for k, n in counts.most_common(a.n_composite):
        lib.append(dict(op="composite", layer=-1, k=n, path=[int(v) for v in k.split(",")]))
    for e in lib:
        e["cost"] = path_cost(e["path"], L)
        e["name"] = e["op"] + ":" + path_key(e["path"])
    return lib


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from oracle import base_records, feat_names, make_chunks, make_g

    a = get_args()
    S = np.load(a.search)
    L = int(S["n_layers"])
    lib = build_library(S, L, a)
    print(f"library: {len(lib)} paths ({sum(e['op'] == 'composite' for e in lib)} composite)", flush=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    raw_layer = a.raw_layer if a.raw_layer is not None else L // 2
    chunks, names = make_chunks(tok, a.n_tokens, a.seq_len)
    rec = {k: [] for k in ["delta", "feat", "base_nll", "raw", "tok", "chunk", "pos"]}
    t0 = time.time()
    for bi in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[bi : bi + a.batch], device=dev)
        B, T = ids.shape
        hs, kvs, cs, base_nll, feat, raw = base_records(g, ids, raw_layer)
        delta = torch.zeros(B, T - 1, len(lib), device=dev)
        for c, e in enumerate(lib):
            if e["op"] != "base":
                delta[..., c] = g.nll(g.run_program(hs, kvs, cs, e["path"]), ids) - base_nll
        rec["delta"].append(delta.half().cpu().numpy().reshape(-1, len(lib)))
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
    assert len(out["delta"]) == len(S["chunk"]), "use the same --n_tokens/--seq_len as search.py"
    Smax = max(len(e["path"]) for e in lib)
    cp = np.full((len(lib), Smax), PAD, dtype=np.int16)
    for c, e in enumerate(lib):
        cp[c, : len(e["path"])] = e["path"]
    out.update(cfg_op=np.array([e["op"] for e in lib]), cfg_layer=np.array([e["layer"] for e in lib]),
               cfg_end=np.array([e["layer"] for e in lib]), cfg_k=np.array([e["k"] for e in lib]),
               cfg_cost=np.array([e["cost"] for e in lib], dtype=np.float32),
               cfg_name=np.array([e["name"] for e in lib]), cfg_path=cp,
               source=np.array(names), n_layers=np.array(L), feat_names=feat_names(L))
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "library.npz"), **out)
    print(f"saved {a.out}/library.npz  delta {out['delta'].shape}")


if __name__ == "__main__":
    main()
