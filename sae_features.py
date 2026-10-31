"""
E4: Gemma Scope SAE features for every oracle position (re-runs the base pass on the SAME chunks).

Gemma Scope residual SAEs for layer l read the residual stream after block l, i.e. hs[l + 1] in
oracle.py's indexing. For each position we keep the top-k active latents per layer.

Usage:
  pip install sae-lens
  python sae_features.py --n_tokens 200000 --out oracle_out_full      # writes sae.npz next to oracle.npz
  python probe.py --path oracle_out_full/oracle.npz --out_dir oracle_out_full --sae oracle_out_full/sae.npz
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from oracle import make_chunks, make_g


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--width", default="16k")
    p.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=200_000)
    p.add_argument("--out", default="oracle_out_full")
    return p.parse_args()


def main():
    from sae_lens import SAE

    a = get_args()
    if AutoConfig.from_pretrained(a.model).model_type != "gemma2":
        raise SystemExit(f"{a.model}: Gemma Scope residual SAEs exist only for Gemma-2 (and Gemma-3 via Gemma Scope 2), "
                         "not Gemma-4, so E4's SAE step is deferred for this model. Run with --model google/gemma-2-2b "
                         "only on an oracle collected with that model.")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    saes = [SAE.from_pretrained(a.release, f"layer_{l}/width_{a.width}/canonical", device=dev) for l in a.layers]
    width = int(saes[0].cfg.d_sae)
    chunks, _ = make_chunks(tok, a.n_tokens, a.seq_len)

    idx_all, val_all = [], []
    t0 = time.time()
    for bi in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[bi : bi + a.batch], device=dev)
        with torch.no_grad():
            hs, _, _, _ = g.base_pass(ids)
            per_layer_i, per_layer_v = [], []
            for l, sae in zip(a.layers, saes):
                x = hs[l + 1][:, :-1].float().reshape(-1, hs[0].shape[-1])  # aligned with oracle positions
                z = sae.encode(x)
                v, i = z.topk(a.topk, dim=-1)
                per_layer_i.append(i.to(torch.int16).cpu())
                per_layer_v.append(v.half().cpu())
        idx_all.append(torch.stack(per_layer_i, 1).numpy())
        val_all.append(torch.stack(per_layer_v, 1).numpy())
        done, el = bi + ids.shape[0], time.time() - t0
        print(f"{done}/{len(chunks)} chunks | {el / 60:.1f} min | ETA {el / done * (len(chunks) - done) / 60:.1f} min",
              flush=True)

    idx, val = np.concatenate(idx_all), np.concatenate(val_all)
    path = os.path.join(a.out, "sae.npz")
    os.makedirs(a.out, exist_ok=True)
    np.savez(path, idx=idx, val=val, layers=np.array(a.layers), width=np.array(width),
             neuronpedia=np.array([f"gemma-2-2b/{l}-gemmascope-res-{a.width}" for l in a.layers]))
    print(f"saved {path}  idx {idx.shape}")


if __name__ == "__main__":
    main()
