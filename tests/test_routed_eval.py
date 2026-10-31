"""
Token-by-token decoding with a first-pass KV cache (decode.py) must match the whole-sequence
routed forward (routed_eval.routed_forward) for mixed repeat/skip routes. Also checks that
decode features on an unedited route equal oracle.py's base-pass features.
Run: python tests/test_routed_eval.py
"""
import os
import sys

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decode import Cache, decode_nll, step, token_features  # noqa: E402
from oracle import G2  # noqa: E402
from routed_eval import routed_forward  # noqa: E402

torch.manual_seed(1)
CFG = Gemma2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                   sliding_window=4096, max_position_embeddings=256)
CFG._attn_implementation = "eager"
MODEL = Gemma2ForCausalLM(CFG).float().eval()
G = G2(MODEL)
L = CFG.num_hidden_layers


def random_route(T, gen):
    rl = torch.full((1, T), -1, dtype=torch.long)
    rk = torch.ones((1, T), dtype=torch.long)
    for t in range(T):
        u = torch.rand(1, generator=gen).item()
        if u < 0.3:
            rl[0, t], rk[0, t] = torch.randint(0, L, (1,), generator=gen).item(), torch.randint(2, 4, (1,), generator=gen).item()
        elif u < 0.55:
            rl[0, t], rk[0, t] = torch.randint(0, L, (1,), generator=gen).item(), 0
    return rl, rk


@torch.no_grad()
def test_decode_matches_routed_forward():
    gen = torch.Generator().manual_seed(0)
    for T in (6, 11, 16):
        ids = torch.randint(0, 97, (1, T), generator=gen)
        rl, rk = random_route(T, gen)
        whole = G.nll(routed_forward(G, ids, rl, rk), ids)
        inc = decode_nll(G, ids, rl, rk)
        assert (whole - inc).abs().max() < 1e-4, (T, (whole - inc).abs().max())


@torch.no_grad()
def test_no_route_matches_base_and_features():
    ids = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(4))  # own RNG: test-order independent
    hs, _, attn_ent, _ = G.base_pass(ids)
    ent, agree, upd = G.lens_features(hs)
    feat_base = torch.cat([ent, agree, upd, attn_ent.float()], -1)[0]
    cache = Cache(L)
    for t in range(9):
        h, hs_t, ae = step(G, cache, ids[:, t], t, want_hs=True)
        f = token_features(G, hs_t, ae)[0]
        assert (h[0, 0] - hs[-1][0, t]).abs().max() < 1e-5
        assert (f - feat_base[t]).abs().max() < 1e-4


@torch.no_grad()
def test_prefill_then_decode():
    """routed_forward prefill (return_kv) of the first m tokens, then step() the rest == whole routed_forward."""
    gen = torch.Generator().manual_seed(5)
    T, m = 14, 6
    ids = torch.randint(0, 97, (1, T), generator=gen)
    rl, rk = random_route(T, gen)
    whole = G.nll(routed_forward(G, ids, rl, rk), ids)
    h, kvs = routed_forward(G, ids[:, :m], rl[:, :m], rk[:, :m], return_kv=True)
    cache = Cache(L)
    cache.k, cache.v = [k for k, _ in kvs], [v for _, v in kvs]
    nll = [torch.nn.functional.cross_entropy(G.logits(h[:, -1]), ids[:, m])]
    for t in range(m, T - 1):
        x = step(G, cache, ids[:, t], t, (int(rl[0, t]), int(rk[0, t])))
        nll.append(torch.nn.functional.cross_entropy(G.logits(x[:, 0]), ids[:, t + 1]))
    assert (whole[0, m - 1 :] - torch.stack(nll)).abs().max() < 1e-4


if __name__ == "__main__":
    for f in [test_decode_matches_routed_forward, test_no_route_matches_base_and_features, test_prefill_then_decode]:
        f()
        print("ok", f.__name__)
