"""
CPU tests on a tiny random Gemma-2 (no download, no GPU):
  1. manual G2 forward == HF forward
  2. diagonal swap with nothing edited == base pass
  3. oracle delta == independent recompute, for repeat, skip, swap and block-loop programs
     (independent: truncate the sequence at t, run each layer of the program normally on
      [other positions' base inputs to that layer] + [token t's current state])
  4. routed_forward with one edited token (repeat or skip) == oracle at that token, and
     earlier tokens are untouched
Run: python -m pytest -q tests/test_tiny.py   (or: python tests/test_tiny.py)
"""
import os
import sys

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import G2, build_programs  # noqa: E402
from routed_eval import routed_forward  # noqa: E402

torch.manual_seed(0)
CFG = Gemma2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                   sliding_window=4096, max_position_embeddings=256)
CFG._attn_implementation = "eager"
MODEL = Gemma2ForCausalLM(CFG).float().eval()
G = G2(MODEL)
L = CFG.num_hidden_layers
IDS = torch.randint(0, 97, (2, 12))
TOL = 1e-4


@torch.no_grad()
def independent_nll(seq, t, b, hs):
    """Token t of sequence b follows program `seq`; everything else is the base model."""
    ids = IDS[b : b + 1, : t + 2]
    pos = torch.arange(t + 1).unsqueeze(0)
    cos, sin = G.rotary(hs[0][b : b + 1, : t + 1], pos)
    x = hs[0][b : b + 1, t : t + 1]
    for li in seq:
        inp = torch.cat([hs[li][b : b + 1, :t], x], 1)  # others: base input to layer li
        out, _, _ = G.layer(li, inp, cos, sin)
        x = out[:, -1:]
    logits = G.logits(x)[0, 0]
    return torch.nn.functional.cross_entropy(logits[None], ids[0, t + 1][None])


@torch.no_grad()
def test_manual_matches_hf():
    hf = MODEL(IDS).logits.float()
    hs, _, _, _ = G.base_pass(IDS)
    assert (hf - G.logits(hs[-1])).abs().max() < TOL


@torch.no_grad()
def test_swap_identity():
    hs, kvs, _, cs = G.base_pass(IDS)
    assert (G.run_program(hs, kvs, cs, list(range(L))) - hs[-1]).abs().max() < 1e-5
    x = hs[1]
    for li in range(1, L):
        x, _, _ = G.layer(li, x, *cs, base_kv=kvs[li])
    assert (x - hs[-1]).abs().max() < 1e-5


@torch.no_grad()
def test_oracle_exact_all_ops():
    hs, kvs, _, cs = G.base_pass(IDS)
    base = G.nll(hs[-1], IDS)
    progs = build_programs(L, ("repeat", "skip", "swap"), (2, 3), blocks=[(1, 3, 2), (2, 3, 3)])
    assert {p["op"] for p in progs} == {"repeat", "skip", "swap", "block"}
    for pr in progs:
        d_all = G.nll(G.run_program(hs, kvs, cs, pr["seq"]), IDS) - base
        for t, b in [(0, 0), (5, 0), (9, 1), (10, 1)]:
            d_ind = independent_nll(pr["seq"], t, b, hs) - base[b, t]
            assert abs(d_all[b, t] - d_ind) < TOL, (pr["op"], pr["layer"], t, d_all[b, t].item(), d_ind.item())


@torch.no_grad()
def test_routed_single_token_matches_oracle():
    hs, kvs, _, cs = G.base_pass(IDS)
    base = G.nll(hs[-1], IDS)
    t = 7
    for l, k, seq in [(2, 2, [0, 1, 2, 2, 3, 4]), (3, 3, [0, 1, 2, 3, 3, 3, 4]), (2, 0, [0, 1, 3, 4])]:
        d_oracle = G.nll(G.run_program(hs, kvs, cs, seq), IDS)[:, t] - base[:, t]
        rl = torch.full(IDS.shape, -1)
        rk = torch.ones(IDS.shape, dtype=torch.long)
        rl[:, t], rk[:, t] = l, k
        routed = G.nll(routed_forward(G, IDS, rl, rk), IDS)
        assert (routed[:, t] - base[:, t] - d_oracle).abs().max() < TOL, (l, k)
        assert (routed[:, :t] - base[:, :t]).abs().max() < 1e-5  # causal: earlier tokens untouched


if __name__ == "__main__":
    for f in [test_manual_matches_hf, test_swap_identity, test_oracle_exact_all_ops,
              test_routed_single_token_matches_oracle]:
        f()
        print("ok", f.__name__)
