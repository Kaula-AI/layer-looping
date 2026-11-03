"""
Noise null for the per-token oracle: is the hindsight gain selection or signal?

The single-edit oracle (oracle.py) takes, for every token, the best of ~140 layer edits after seeing
the answer. If most of that gain is "best of many perturbations happens to favor the right token",
then the same best-of-N over RANDOM perturbations of the same size should recover most of it.

For every token and every (layer l, draw j), this adds a random-direction vector to the token's input
to layer l, with norm s * ||h_{l+1} - h_l|| (s times that token's own update at layer l: s = 1 is the
size of what skipping layer l removes), and runs the base order from there with the diagonal swap, so
each delta is exact for the token's own next-token loss, the same accounting as oracle.py.

  python noise_null.py --n_tokens 20000 --draws 4 --scales 1 --out noise_quick
  python noise_null.py --model ByteDance/Ouro-1.4B --n_tokens 20000 --draws 2 --out noise_ouro

Saves noise.npz: delta (N, P) in oracle.npz's position order (same chunks for the same --n_tokens and
--seq_len), plus base_nll, chunk, pos, cfg_layer, cfg_draw, cfg_scale. Compare with
`python selection_curve.py --oracle <dir>/oracle.npz --noise <dir>/noise.npz`.
"""
import argparse
import os
import time

import numpy as np
import torch

from oracle import make_chunks, make_g
from ouro_model import load_causal_lm
from transformers import AutoTokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=20_000)
    p.add_argument("--draws", type=int, default=4, help="random directions per (layer, scale)")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0], help="noise norm / own layer-update norm")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="layers to perturb (default: all)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="noise_out")
    return p.parse_args()


def unit_noise(shape, gen, dev, dtype):
    """Random directions, one unit vector per position: (B, T, d)."""
    z = torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32)
    return (z / z.norm(dim=-1, keepdim=True)).to(dev, dtype)


@torch.no_grad()
def noise_deltas(g, hs, kvs, cs, ids, base_nll, cfgs, gen):
    """(B, T-1, len(cfgs)) exact per-token dNLL for each (layer, draw, scale) perturbation."""
    L = len(g.layers)
    base = list(range(L))
    B, T = ids.shape
    out = torch.empty(B, T - 1, len(cfgs), device=ids.device)
    for ci, (l, _, s) in enumerate(cfgs):
        upd = (hs[l + 1] - hs[l]).float().norm(dim=-1, keepdim=True)  # B, T, 1: this token's own update size
        x0 = hs[l] + (s * upd).to(hs[l].dtype) * unit_noise(hs[l].shape, gen, hs[l].device, hs[l].dtype)
        out[..., ci] = g.nll(g.run_program(hs, kvs, cs, base, x0=x0, start=l), ids) - base_nll
    return out


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_causal_lm(a.model, getattr(torch, a.dtype), attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    L = len(g.layers)
    layers = a.layers if a.layers else list(range(L))
    cfgs = [(l, j, s) for s in a.scales for l in layers for j in range(a.draws)]
    chunks, _ = make_chunks(tok, a.n_tokens, a.seq_len)
    print(f"{len(chunks)} chunks x {a.seq_len} tokens, {len(cfgs)} noise programs "
          f"({len(layers)} layers x {a.draws} draws x scales {a.scales}), device {dev}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rec = {k: [] for k in ["delta", "base_nll", "chunk", "pos"]}
    t0 = time.time()
    for bi in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[bi : bi + a.batch], device=dev)
        B, T = ids.shape
        hs, kvs, _, cs = g.base_pass(ids)
        base_nll = g.nll(hs[-1], ids)
        delta = noise_deltas(g, hs, kvs, cs, ids, base_nll, cfgs, gen)
        rec["delta"].append(delta.half().cpu().numpy().reshape(-1, len(cfgs)))
        rec["base_nll"].append(base_nll.float().cpu().numpy().reshape(-1))
        rec["chunk"].append(np.repeat(np.arange(bi, bi + B), T - 1))
        rec["pos"].append(np.tile(np.arange(T - 1), B))
        done, el = bi + B, time.time() - t0
        print(f"{done}/{len(chunks)} chunks | {el / 60:.1f} min | ETA {el / done * (len(chunks) - done) / 60:.1f} min",
              flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["cfg_layer"] = np.array([c[0] for c in cfgs])
    out["cfg_draw"] = np.array([c[1] for c in cfgs])
    out["cfg_scale"] = np.array([c[2] for c in cfgs], dtype=np.float32)
    out["n_layers"] = np.array(L)
    path = os.path.join(a.out, "noise.npz")
    np.savez(path, **out)
    print(f"saved {path}  ({len(out['delta']):,} positions)")


if __name__ == "__main__":
    main()
