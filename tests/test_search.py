"""
Search sanity on a tiny random Gemma-2: every stored front point's dNLL equals an independent
recompute of that token under that path, fronts are non-dominated, and the base path is never stored.
Run: python tests/test_search.py
"""
import os
import sys

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import G2  # noqa: E402
from programs import apply_edit, candidate_edits  # noqa: E402
from search import get_args, search_batch  # noqa: E402

torch.manual_seed(0)
CFG = Gemma2Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, query_pre_attn_scalar=16,
                   sliding_window=4096, max_position_embeddings=256)
CFG._attn_implementation = "eager"
G = G2(Gemma2ForCausalLM(CFG).float().eval())
L = CFG.num_hidden_layers


def test_grammar():
    base = list(range(L))
    assert apply_edit(base, ("loop", 2, 3))[:8] == [0, 1, 2, 3, 4, 2, 3, 4]
    assert apply_edit(base, ("reverse", 2, 3))[:6] == [0, 1, 4, 3, 2, 5]
    ce = candidate_edits(base, L, protect=2)
    kinds = {e[0] for e, _ in ce}
    assert kinds == {"repeat", "skip", "swap", "reverse", "loop"}
    for _, p in ce:  # protected ends untouched
        assert p[:2] == [0, 1] and p[-2:] == [L - 2, L - 1]


@torch.no_grad()
def test_search_fronts_exact():
    ids = torch.randint(0, 97, (2, 9))
    hs, kvs, _, cs = G.base_pass(ids)
    base_nll = G.nll(hs[-1], ids)
    a = get_args(["--rounds", "3", "--beam", "2", "--focus", "3", "--max_cands", "6", "--keep", "5",
                  "--max_extra", "4", "--max_skip", "2"])
    fronts = search_batch(G, ids, hs, kvs, cs, base_nll, a)
    for b in range(2):
        for t in range(8):
            fr = fronts[b][t]
            assert fr and tuple(range(L)) not in [p for p, _, _ in fr]
            for path, dn, c in fr:
                ref = (G.independent_program_nll(hs, ids, list(path), t, b) - base_nll[b, t]).item()
                assert abs(dn - ref) < 1e-4, (b, t, path, dn, ref)
                assert c == len(path) - L
            cs_ = sorted(fr, key=lambda z: z[2])
            for i in range(1, len(cs_)):  # higher cost must buy lower loss
                assert cs_[i][1] < cs_[i - 1][1] + 1e-9


if __name__ == "__main__":
    for f in [test_grammar, test_search_fronts_exact]:
        f()
        print("ok", f.__name__)
