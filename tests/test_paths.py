"""
Mixed-path executor: every position follows its own arbitrary layer path (repeats, skips, swaps,
reversed and looped blocks) in one pass. Must equal an independent per-position recompute.
Run: python tests/test_paths.py
"""
import os
import random
import sys

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decode import decode_nll, decode_nll_paths  # noqa: E402
from oracle import G2  # noqa: E402
from routed_eval import routed_forward  # noqa: E402

torch.manual_seed(0)
CFG = Gemma2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                   sliding_window=4096, max_position_embeddings=256)
CFG._attn_implementation = "eager"
MODEL = Gemma2ForCausalLM(CFG).float().eval()
G = G2(MODEL)
L = CFG.num_hidden_layers
IDS = torch.randint(0, 97, (3, 10))


def random_path(rng):
    p = list(range(L))
    for _ in range(rng.randint(0, 3)):
        i = rng.randint(1, len(p) - 2)
        kind = rng.choice(["repeat", "skip", "swap", "reverse", "loop"])
        if kind == "repeat":
            p = p[: i + 1] + [p[i]] + p[i + 1 :]
        elif kind == "skip" and len(p) > 3:
            p = p[:i] + p[i + 1 :]
        elif kind == "swap" and i + 1 < len(p):
            p[i], p[i + 1] = p[i + 1], p[i]
        elif kind == "reverse":
            j = min(len(p), i + 3)
            p = p[:i] + p[i:j][::-1] + p[j:]
        elif kind == "loop":
            j = min(len(p), i + 2)
            p = p[:j] + p[i:j] + p[j:]
    return p


@torch.no_grad()
def test_run_paths_matches_independent():
    rng = random.Random(0)
    B, T = IDS.shape
    hs, kvs, _, cs = G.base_pass(IDS)
    paths = [[random_path(rng) if rng.random() < 0.8 else list(range(L)) for _ in range(T)] for _ in range(B)]
    S = max(len(p) for row in paths for p in row)
    P = torch.full((B, T, S), -1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            P[b, t, : len(paths[b][t])] = torch.tensor(paths[b][t])
    out = G.run_paths(hs, kvs, cs, P, chunk_rows=7)
    nll = G.nll(out, IDS)
    for b in range(B):
        for t in range(T - 1):
            ref = G.independent_program_nll(hs, IDS, paths[b][t], t, b)
            assert abs(nll[b, t] - ref) < 1e-4, (b, t, paths[b][t], nll[b, t].item(), ref.item())


@torch.no_grad()
def test_run_paths_shared_equals_run_program():
    hs, kvs, _, cs = G.base_pass(IDS)
    seq = [0, 1, 3, 2, 2, 4, 5]
    B, T = IDS.shape
    P = torch.tensor(seq).expand(B, T, len(seq)).clone()
    assert (G.run_paths(hs, kvs, cs, P) - G.run_program(hs, kvs, cs, seq)).abs().max() < 1e-5
    Pb = torch.arange(L).expand(B, T, L).clone()
    assert (G.run_paths(hs, kvs, cs, Pb) - hs[-1]).abs().max() < 1e-5


@torch.no_grad()
def test_decode_paths_matches_routed_forward_for_repeat_skip():
    """Arbitrary-path decoding reduces to the repeat/skip semantics already checked against routed_forward."""
    gen = torch.Generator().manual_seed(3)
    T = 12
    ids = torch.randint(0, 97, (1, T), generator=gen)
    rl = torch.full((1, T), -1, dtype=torch.long)
    rk = torch.ones((1, T), dtype=torch.long)
    paths = []
    for t in range(T):
        u = torch.rand(1, generator=gen).item()
        l = int(torch.randint(0, L, (1,), generator=gen))
        if u < 0.35:
            k = int(torch.randint(2, 4, (1,), generator=gen))
            rl[0, t], rk[0, t] = l, k
            paths.append(list(range(l + 1)) + [l] * (k - 1) + list(range(l + 1, L)))
        elif u < 0.6:
            rl[0, t], rk[0, t] = l, 0
            paths.append([q for q in range(L) if q != l])
        else:
            paths.append(list(range(L)))
    whole = G.nll(routed_forward(G, ids, rl, rk), ids)
    assert (decode_nll_paths(G, ids, paths) - whole).abs().max() < 1e-4
    assert (decode_nll(G, ids, rl, rk) - whole).abs().max() < 1e-4


@torch.no_grad()
def test_decode_paths_single_token_matches_oracle():
    """Only token t edited (any path): its loss equals the exact oracle; earlier tokens unchanged."""
    ids = IDS[:1]
    T = ids.shape[1]
    hs, kvs, _, cs = G.base_pass(ids)
    base = G.nll(hs[-1], ids)
    rng = random.Random(7)
    for t in (2, 5, 8):
        path = random_path(rng)
        paths = [list(range(L))] * T
        paths = paths[:t] + [path] + paths[t + 1 :]
        nll = decode_nll_paths(G, ids, paths)
        ref = G.independent_program_nll(hs, ids, path, t, 0)
        assert abs(nll[0, t] - ref) < 1e-4, (t, path)
        assert (nll[0, :t] - base[0, :t]).abs().max() < 1e-5


if __name__ == "__main__":
    for f in [test_run_paths_matches_independent, test_run_paths_shared_equals_run_program,
              test_decode_paths_matches_routed_forward_for_repeat_skip, test_decode_paths_single_token_matches_oracle]:
        f()
        print("ok", f.__name__)
