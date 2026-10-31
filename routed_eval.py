"""
Routed teacher-forced evaluation (E5 interaction gap): apply a per-token route to a whole
sequence at once and let the effects propagate to later tokens, then compare against the
oracle's independent per-token deltas.

Supported per-token edits: repeat layer l k times, skip layer l. (Swaps are oracle-only for now;
routes are chosen among repeat/skip programs.)

Semantics ("first-pass KV caching"):
  - At every layer, every position's K/V are projected from that position's input to the layer,
    so upstream edits DO change later layers and later tokens.
  - A skipped position keeps its residual unchanged at layer l, but its K/V at l are still
    projected from its input (later tokens can attend to it). State this in the paper.
  - A repeating position's extra iterations of layer l attend to the other positions'
    first-pass K/V at l plus its own current K/V (diagonal swap).
  This equals autoregressive decoding where each token's KV entry at layer l is written on its
  first pass through l. TODO: verify against a token-by-token decode (tests/test_routed_eval.py).

Policies:
  none                          base model
  static --program repeat@13x2  every token gets that edit (also skip@20)
  oracle --lam X                per-token argmin of oracle dNLL + lam*cost (needs oracle.npz of the SAME chunks)
  router --lam X                router_lag_interp.npz on token t-1's base-pass features
"""
import argparse
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle import make_chunks, make_g


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--policy", choices=["none", "static", "oracle", "router"], default="none")
    p.add_argument("--program", default="repeat@13x2", help="for --policy static: repeat@LxK or skip@L")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--oracle", default="oracle_out/oracle.npz")
    p.add_argument("--router", default="oracle_out/router_lag_interp.npz")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=20_000)
    p.add_argument("--test_only", action="store_true", help="evaluate only chunks with index %% 5 == 0")
    return p.parse_args()


@torch.no_grad()
def routed_forward(g, ids, route_l, route_k, return_kv=False):
    """route_l, route_k: (B, T) long. route_l = -1: no edit. route_k >= 2: repeat layer route_l
    route_k times. route_k == 0: skip layer route_l. With return_kv, also returns the first-pass
    K/V per layer (a prefill cache for decode.step)."""
    h = g.embed(ids)
    cs = g.context(ids, h)
    kvs, latest = [], {}  # latest: each token's own K/V of K/V-source layers (Gemma-4), from its latest visit
    for li in range(len(g.layers)):
        sl = g.kv_src(li)
        if sl == li:
            out, kv, _ = g.layer(li, h, cs)
            own = None
        else:  # K/V-shared layer: earlier tokens' first-pass source K/V, own latest K/V on the diagonal
            kv, own = kvs[sl], latest[sl]
            out, _, _ = g.layer(li, h, cs, base_kv=kv, own_kv=own)
        kvs.append(kv)
        if li in g.kv_sources:
            latest[li] = kv
        here = route_l == li
        rep = here & (route_k >= 2)
        if rep.any():
            x = out
            for it in range(2, int(route_k[rep].max().item()) + 1):
                nxt, kv2, _ = g.layer(li, x, cs, base_kv=kv, own_kv=own)
                m = rep & (route_k >= it)
                x = torch.where(m.unsqueeze(-1), nxt, x)
                if li in g.kv_sources:
                    mm = m[:, None, :, None]
                    latest[li] = tuple(torch.where(mm, new_, old_) for new_, old_ in zip(kv2, latest[li]))
            out = x
        skp = here & (route_k == 0)
        if skp.any():
            out = torch.where(skp.unsqueeze(-1), h, out)
        h = out
    return (h, kvs) if return_kv else h


def lens_features(g, ids):
    hs, kvs, attn_ent, _ = g.base_pass(ids)
    ent, agree, upd = g.lens_features(hs)
    return torch.cat([ent, agree, upd, attn_ent.float()], -1)


def parse_program(s):
    m = re.fullmatch(r"repeat@(\d+)x(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"skip@(\d+)", s)
    if m:
        return int(m.group(1)), 0
    raise ValueError(f"unsupported program {s!r}; use repeat@LxK or skip@L")


def main():
    a = get_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    g = make_g(model)
    chunks, _ = make_chunks(tok, a.n_tokens, a.seq_len)
    idx = [c for c in range(len(chunks)) if (c % 5 == 0 or not a.test_only)]

    src = np.load(a.oracle) if a.policy == "oracle" else (np.load(a.router) if a.policy == "router" else None)
    if src is not None:
        op, lay, kk, cost = src["cfg_op"].astype(str), src["cfg_layer"], src["cfg_k"], src["cfg_cost"]
        allowed = torch.tensor(np.isin(op, ["repeat", "skip"]), device=dev)
        if (~np.isin(op, ["repeat", "skip"])).any():
            print("note: swap and block programs excluded from routing (oracle-only)")
        lay_t, kk_t = torch.tensor(lay, device=dev), torch.tensor(kk, device=dev)
        cost_t = torch.tensor(cost, dtype=torch.float32, device=dev)
    if a.policy == "oracle":
        D = src["delta"].astype(np.float32).reshape(-1, a.seq_len - 1, len(op))  # (chunk, pos, cfg)
    if a.policy == "router":
        rt = src

    tot_base, tot_routed, tot_extra, tot_pred, n = 0.0, 0.0, 0.0, 0.0, 0
    for bi in range(0, len(idx), a.batch):
        cb = idx[bi : bi + a.batch]
        ids = torch.tensor([chunks[c] for c in cb], device=dev)
        B, T = ids.shape
        rl = torch.full((B, T), -1, dtype=torch.long, device=dev)
        rk = torch.ones((B, T), dtype=torch.long, device=dev)
        pred = torch.zeros(B, T - 1, device=dev)
        if a.policy == "static":
            l_, k_ = parse_program(a.program)
            rl[:], rk[:] = l_, k_
        elif a.policy in ("oracle", "router"):
            if a.policy == "oracle":
                P = torch.tensor(D[cb], device=dev)
            else:
                feat = lens_features(g, ids)
                X = (feat[:, :-2].float().cpu().numpy() - rt["mean"]) / rt["scale"]
                P = torch.full((B, T - 1, len(op)), 1e9, device=dev)
                P[:, 1:] = torch.tensor(X @ rt["coef"].T + rt["intercept"], dtype=torch.float32, device=dev)
            obj = torch.where(allowed, P + a.lam * cost_t, torch.full_like(P, 1e9))
            best, c = obj.min(-1)
            go = best < 0
            rl[:, :-1] = torch.where(go, lay_t[c], -1)
            rk[:, :-1] = torch.where(go, kk_t[c], 1)
            if a.policy == "oracle":
                pred = torch.where(go, P.gather(-1, c.unsqueeze(-1)).squeeze(-1), 0.0)
        hs, _, _, _ = g.base_pass(ids)
        base = g.nll(hs[-1], ids)
        routed = g.nll(routed_forward(g, ids, rl, rk), ids)
        edit = rl[:, :-1] >= 0
        extra = torch.where(edit & (rk[:, :-1] == 0), -1, torch.where(edit, rk[:, :-1] - 1, 0))
        tot_base += base.sum().item()
        tot_routed += routed.sum().item()
        tot_extra += extra.sum().item()
        tot_pred += pred.sum().item()
        n += base.numel()

    print(f"policy={a.policy} lam={a.lam}  tokens={n:,}")
    print(f"base NLL    {tot_base / n:.4f}")
    print(f"routed NLL  {tot_routed / n:.4f}   (dNLL {(tot_routed - tot_base) / n:+.4f})")
    print(f"extra layer applications per token  {tot_extra / n:+.3f}")
    if a.policy == "oracle":
        print(f"sum of independent oracle deltas    {tot_pred / n:+.4f}   (gap = interaction effect)")


if __name__ == "__main__":
    main()
