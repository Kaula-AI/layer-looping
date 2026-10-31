"""
CPU tests on a tiny random Gemma-4 (no download, no GPU), the same exactness checks as the Gemma-2
tests, on an architecture with sliding/full layers of different head dims, per-layer embeddings,
per-layer output scalars and K/V sharing (layers 4-7 attend to the K/V of layer 3 or 2):
  1. manual G4 forward == HF forward
  2. diagonal swap with nothing edited == base pass
  3. oracle delta == independent recompute for repeat, skip, swap and block programs, including
     edits of the K/V-source layers and of K/V-shared layers
  4. routed_forward with one edited token == oracle at that token; earlier tokens untouched
  5. token-by-token decode == routed_forward; prefill + decode; decode features == base features
  6. mixed-path executor == independent recompute for random paths (reversed / looped blocks
     across the sharing boundary); arbitrary-path decoding == oracle and == routed_forward
Run: python -m pytest -q tests/test_gemma4.py
"""
import os
import random
import sys

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decode import Cache, decode_nll, decode_nll_paths, step, token_features  # noqa: E402
from oracle import G4, build_programs, make_g  # noqa: E402
from routed_eval import routed_forward  # noqa: E402

torch.manual_seed(0)
S_, F_ = "sliding_attention", "full_attention"
CFG = Gemma4TextConfig(
    vocab_size=97, vocab_size_per_layer_input=97, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=1, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=8,
    num_kv_shared_layers=4, use_double_wide_mlp=True, initializer_range=0.2, sliding_window=64,
    layer_types=[S_, S_, F_, S_, S_, F_, S_, F_], max_position_embeddings=256, final_logit_softcapping=30.0,
    rope_parameters={F_: {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
                     S_: {"rope_theta": 1e4, "rope_type": "default"}})
CFG._attn_implementation = "eager"
MODEL = Gemma4ForCausalLM(CFG).float().eval()
with torch.no_grad():  # non-trivial norm weights and layer scalars, so every term is exercised
    for n_, p_ in MODEL.named_parameters():
        if "norm" in n_ and p_.dim() == 1:
            p_.uniform_(0.5, 1.5)
    for i, l_ in enumerate(MODEL.model.layers):
        l_.layer_scalar.fill_(0.8 + 0.05 * i)
G = make_g(MODEL)
L = CFG.num_hidden_layers
IDS = torch.randint(0, 97, (2, 12))
# Activations here reach ~20 (vs ~2 in the Gemma-2 tests) and Gemma-4's RMSNorm always computes in
# float32, so exact-in-float64 identities show ~1e-6 relative error. Real bugs give errors of order 1.
TOL = 5e-4  # NLL
HTOL = 5e-4  # hidden states


def test_sharing_layout():
    assert isinstance(G, G4)
    assert G.src == [0, 1, 2, 3, 3, 2, 3, 2] and G.kv_sources == [2, 3]


@torch.no_grad()
def test_manual_matches_hf():
    hf = MODEL(IDS).logits.float()
    hs, _, _, _ = G.base_pass(IDS)
    assert (hf - G.logits(hs[-1])).abs().max() < TOL


@torch.no_grad()
def test_swap_identity():
    hs, kvs, _, cs = G.base_pass(IDS)
    assert (G.run_program(hs, kvs, cs, list(range(L))) - hs[-1]).abs().max() < HTOL
    x = hs[1]
    for li in range(1, L):
        own = None if G.has_kv(li) else kvs[G.kv_src(li)]
        x, _, _ = G.layer(li, x, cs, base_kv=kvs[li], own_kv=own)
    assert (x - hs[-1]).abs().max() < HTOL


@torch.no_grad()
def test_oracle_exact_all_ops():
    hs, kvs, _, cs = G.base_pass(IDS)
    base = G.nll(hs[-1], IDS)
    progs = build_programs(L, ("repeat", "skip", "swap"), (2, 3), blocks=[(1, 3, 2), (2, 5, 2), (3, 6, 3)])
    assert {p["op"] for p in progs} == {"repeat", "skip", "swap", "block"}
    for pr in progs:
        d_all = G.nll(G.run_program(hs, kvs, cs, pr["seq"]), IDS) - base
        for t, b in [(0, 0), (5, 0), (9, 1), (10, 1)]:
            d_ind = G.independent_program_nll(hs, IDS, pr["seq"], t, b) - base[b, t]
            assert abs(d_all[b, t] - d_ind) < TOL, (pr["op"], pr["layer"], t, d_all[b, t].item(), d_ind.item())


@torch.no_grad()
def test_source_edit_reaches_shared_layers():
    """Repeating a K/V-source layer must change what the shared layers see for that token: the
    result differs from re-running the shared layers on first-pass source K/V."""
    hs, kvs, _, cs = G.base_pass(IDS)
    seq = [0, 1, 2, 3, 3, 4, 5, 6, 7]
    x = hs[3]
    for li in (3, 3):
        x, _, _ = G.layer(li, x, cs, base_kv=kvs[li])
    for li in range(4, L):  # stale: own K/V of the source from the base pass
        x, _, _ = G.layer(li, x, cs, base_kv=kvs[li], own_kv=kvs[G.kv_src(li)])
    assert (G.run_program(hs, kvs, cs, seq) - x).abs().max() > 1e-3


@torch.no_grad()
def test_routed_single_token_matches_oracle():
    hs, kvs, _, cs = G.base_pass(IDS)
    base = G.nll(hs[-1], IDS)
    t = 7
    for l, k in [(1, 2), (2, 2), (3, 3), (5, 2), (6, 3), (2, 0), (3, 0), (4, 0), (7, 0)]:
        seq = [q for q in range(L) if q != l] if k == 0 else list(range(l + 1)) + [l] * (k - 1) + list(range(l + 1, L))
        d_oracle = G.nll(G.run_program(hs, kvs, cs, seq), IDS)[:, t] - base[:, t]
        rl = torch.full(IDS.shape, -1)
        rk = torch.ones(IDS.shape, dtype=torch.long)
        rl[:, t], rk[:, t] = l, k
        routed = G.nll(routed_forward(G, IDS, rl, rk), IDS)
        assert (routed[:, t] - base[:, t] - d_oracle).abs().max() < TOL, (l, k)
        assert (routed[:, :t] - base[:, :t]).abs().max() < HTOL  # causal: earlier tokens untouched


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
        assert (whole - decode_nll(G, ids, rl, rk)).abs().max() < TOL, T


@torch.no_grad()
def test_no_route_matches_base_and_features():
    ids = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(4))  # own RNG: test-order independent
    hs, _, attn_ent, _ = G.base_pass(ids)
    ent, agree, upd = G.lens_features(hs)
    feat_base = torch.cat([ent, agree, upd, attn_ent.float()], -1)[0]
    cache = Cache(L)
    for t in range(9):
        h, hs_t, ae = step(G, cache, ids[:, t], t, want_hs=True)
        assert (h[0, 0] - hs[-1][0, t]).abs().max() < HTOL
        assert (token_features(G, hs_t, ae)[0] - feat_base[t]).abs().max() < TOL


@torch.no_grad()
def test_prefill_then_decode():
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
    assert (whole[0, m - 1 :] - torch.stack(nll)).abs().max() < TOL


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
    ids = torch.randint(0, 97, (3, 10), generator=torch.Generator().manual_seed(2))
    B, T = ids.shape
    hs, kvs, _, cs = G.base_pass(ids)
    paths = [[random_path(rng) if rng.random() < 0.8 else list(range(L)) for _ in range(T)] for _ in range(B)]
    paths[0][4] = [0, 1, 4, 3, 2, 5, 6, 7]  # shared layer 4 runs before its source 3 is revisited
    paths[1][6] = [0, 1, 3, 4, 5, 2, 6, 7]  # source 2 skipped, then visited after a layer that shares it
    S = max(len(p) for row in paths for p in row)
    P = torch.full((B, T, S), -1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            P[b, t, : len(paths[b][t])] = torch.tensor(paths[b][t])
    nll = G.nll(G.run_paths(hs, kvs, cs, P, chunk_rows=7), ids)
    for b in range(B):
        for t in range(T - 1):
            ref = G.independent_program_nll(hs, ids, paths[b][t], t, b)
            assert abs(nll[b, t] - ref) < TOL, (b, t, paths[b][t], nll[b, t].item(), ref.item())


@torch.no_grad()
def test_run_paths_shared_equals_run_program():
    hs, kvs, _, cs = G.base_pass(IDS)
    B, T = IDS.shape
    for seq in ([0, 1, 3, 2, 2, 4, 5, 6, 7], [0, 1, 2, 4, 3, 5, 5, 6, 7]):
        P = torch.tensor(seq).expand(B, T, len(seq)).clone()
        assert (G.run_paths(hs, kvs, cs, P) - G.run_program(hs, kvs, cs, seq)).abs().max() < HTOL
    Pb = torch.arange(L).expand(B, T, L).clone()
    assert (G.run_paths(hs, kvs, cs, Pb) - hs[-1]).abs().max() < HTOL


@torch.no_grad()
def test_decode_paths():
    """Arbitrary-path decoding: == routed_forward for repeat/skip routes, and == the oracle when
    only token t is edited (any path), with earlier tokens unchanged."""
    gen = torch.Generator().manual_seed(3)
    T = 12
    ids = torch.randint(0, 97, (1, T), generator=gen)
    rl, rk = random_route(T, gen)
    paths = []
    for t in range(T):
        l, k = int(rl[0, t]), int(rk[0, t])
        paths.append(list(range(L)) if l < 0 else [q for q in range(L) if q != l] if k == 0
                     else list(range(l + 1)) + [l] * (k - 1) + list(range(l + 1, L)))
    whole = G.nll(routed_forward(G, ids, rl, rk), ids)
    assert (decode_nll_paths(G, ids, paths) - whole).abs().max() < TOL

    hs, _, _, _ = G.base_pass(ids)
    base = G.nll(hs[-1], ids)
    rng = random.Random(7)
    for t, path in [(2, random_path(rng)), (5, [0, 1, 4, 3, 2, 5, 6, 7]), (8, [0, 1, 3, 4, 5, 2, 6, 7]),
                    (9, random_path(rng))]:
        ps = [list(range(L))] * T
        nll = decode_nll_paths(G, ids, ps[:t] + [path] + ps[t + 1 :])
        assert abs(nll[0, t] - G.independent_program_nll(hs, ids, path, t, 0)) < TOL, (t, path)
        assert (nll[0, :t] - base[0, :t]).abs().max() < HTOL


if __name__ == "__main__":
    for f in [test_sharing_layout, test_manual_matches_hf, test_swap_identity, test_oracle_exact_all_ops,
              test_source_edit_reaches_shared_layers, test_routed_single_token_matches_oracle,
              test_decode_matches_routed_forward, test_no_route_matches_base_and_features, test_prefill_then_decode,
              test_run_paths_matches_independent, test_run_paths_shared_equals_run_program, test_decode_paths]:
        f()
        print("ok", f.__name__)
