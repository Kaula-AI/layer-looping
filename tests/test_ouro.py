"""
CPU tests on a tiny random Ouro (looped LM: 3 physical layers x 3 rounds = 9 virtual layers):
  1. GO's unrolled forward == the reference Ouro forward at every exit step
  2. exit programs reproduce Ouro's early exit; exit-gate distribution == reference
  3. diagonal swap with nothing edited == base pass
  4. oracle delta == independent recompute for repeat, skip, swap, extra-round block and exit programs
  5. mixed-path executor == independent recompute for random paths that loop and revisit layers
Run: python -m pytest -q tests/test_ouro.py
"""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle import GO, build_programs, make_g  # noqa: E402
from ouro_model import Ouro, config_from_dict  # noqa: E402

torch.manual_seed(0)
CFG = config_from_dict(dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=4, head_dim=16, rms_norm_eps=1e-6,
                            rope_theta=1e4, total_ut_steps=3))
MODEL = Ouro(CFG).double().eval()
with torch.no_grad():
    for n_, p_ in MODEL.named_parameters():
        if "norm" in n_:
            p_.uniform_(0.5, 1.5)
        elif p_.dim() == 2:
            p_.normal_(0, 0.2)
G = make_g(MODEL)
N, R = 3, 3
L = N * R
IDS = torch.randint(0, 97, (2, 12))
TOL = 1e-8  # hidden states (float64)
NTOL = 1e-5  # NLL: logits are cast to float32


def test_unrolled_layout():
    assert isinstance(G, GO) and len(G.layers) == L
    assert all(G.layers[j] is MODEL.model.layers[j % N] for j in range(L))


@torch.no_grad()
def test_matches_reference_every_exit():
    hs, _, _, _ = G.base_pass(IDS)
    for s in range(R):
        ref = MODEL(IDS, exit_at_step=s).logits.float()
        assert (G.logits(hs[N * (s + 1)]) - ref).abs().max() < 1e-6, s


@torch.no_grad()
def test_exit_programs_and_gate():
    hs, kvs, _, cs = G.base_pass(IDS)
    for s in range(1, R):
        out = G.run_program(hs, kvs, cs, list(range(N * s)))
        assert (G.logits(out) - MODEL(IDS, exit_at_step=s - 1).logits.float()).abs().max() < 1e-6
    pdf = G.exit_gate_probs(hs)
    _, gates = MODEL.rounds(IDS)
    lam = [torch.sigmoid(g_.float()[..., 0]) for g_ in gates]
    ref = torch.stack([lam[0], (1 - lam[0]) * lam[1], (1 - lam[0]) * (1 - lam[1])], -1)
    assert (pdf - ref).abs().max() < 1e-6 and (pdf.sum(-1) - 1).abs().max() < 1e-6


@torch.no_grad()
def test_swap_identity():
    hs, kvs, _, cs = G.base_pass(IDS)
    assert (G.run_program(hs, kvs, cs, list(range(L))) - hs[-1]).abs().max() < TOL
    x = hs[1]
    for li in range(1, L):
        x, _, _ = G.layer(li, x, cs, base_kv=kvs[li])
    assert (x - hs[-1]).abs().max() < TOL


@torch.no_grad()
def test_oracle_exact_all_ops():
    hs, kvs, _, cs = G.base_pass(IDS)
    base = G.nll(hs[-1], IDS)
    progs = build_programs(L, ("repeat", "skip", "swap"), (2, 3), blocks=[(N * (R - 1), L - 1, 2), (2, 4, 2)])
    progs += [dict(op="exit", layer=N * s - 1, seq=list(range(N * s))) for s in range(1, R)]
    for pr in progs:
        d_all = G.nll(G.run_program(hs, kvs, cs, pr["seq"]), IDS) - base
        for t, b in [(0, 0), (5, 0), (9, 1), (10, 1)]:
            d_ind = G.independent_program_nll(hs, IDS, pr["seq"], t, b) - base[b, t]
            assert abs(d_all[b, t] - d_ind) < NTOL, (pr["op"], pr["layer"], t, d_all[b, t].item(), d_ind.item())


def random_path(rng):
    """Loops, skips and revisits across rounds, e.g. 0 1 1 1 3 3 2 4 4 ... (cf. a user's example path)."""
    p = list(range(L))
    for _ in range(rng.randint(1, 5)):
        i = rng.randrange(1, len(p) - 1)
        k = rng.choice(["rep", "skip", "back"])
        if k == "rep":
            p = p[: i + 1] + [p[i]] * rng.randint(1, 2) + p[i + 1:]
        elif k == "skip" and len(p) > L - 3:
            p = p[:i] + p[i + 1:]
        elif k == "back":
            p = p[: i + 1] + [p[rng.randrange(0, i + 1)]] + p[i + 1:]
    return p


@torch.no_grad()
def test_run_paths_matches_independent():
    rng = random.Random(0)
    ids = torch.randint(0, 97, (2, 10), generator=torch.Generator().manual_seed(2))
    B, T = ids.shape
    hs, kvs, _, cs = G.base_pass(ids)
    paths = [[random_path(rng) for _ in range(T)] for _ in range(B)]
    paths[0][3] = [0, 0, 0, 2, 2, 1, 3, 3, 4, 5, 6, 7, 8]  # 3x layer 0, skip 1, 2x layer 2, back to 1, 2x 3 ...
    S = max(len(p) for row in paths for p in row)
    P = torch.full((B, T, S), -1, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            P[b, t, : len(paths[b][t])] = torch.tensor(paths[b][t])
    nll = G.nll(G.run_paths(hs, kvs, cs, P, chunk_rows=5), ids)
    for b in range(B):
        for t in range(T - 1):
            ref = G.independent_program_nll(hs, ids, paths[b][t], t, b)
            assert abs(nll[b, t] - ref) < NTOL, (b, t, paths[b][t], nll[b, t].item(), ref.item())


@torch.no_grad()
def test_bench_batched_scoring():
    """bench.py: right-padded batches give the unpadded log-likelihoods; gate never firing == base;
    an extra round for answer tokens == the independent recompute of that path."""
    from bench import answer_logprobs

    a_ = IDS[0, :12].tolist()
    b_ = IDS[1, :8].tolist()
    batch = [(a_, 3), (b_, 2)]
    out, rows, run_rows, lp_of = answer_logprobs(G, batch, ["base", "rounds:3", "gate:2.0", "rounds:4"],
                                                 None, None, "cpu")
    ref = []
    for x, na in batch:
        ids = torch.tensor([x])
        hs, _, _, _ = G.base_pass(ids)
        T = len(x)
        ref.append(G.logits(hs[-1][0, T - na - 1: T - 1]).log_softmax(-1).gather(-1, ids[0, T - na:, None])[:, 0])
    ref = torch.cat(ref).numpy()
    for pol in ("base", "rounds:3", "gate:2.0"):
        assert abs(out[pol][0] - ref).max() < NTOL, pol
    extra = list(range(L)) + list(range(L - N, L))
    for k, (b, t) in enumerate(rows):
        x = torch.tensor([batch[b][0]])
        hs, _, _, _ = G.base_pass(x)
        ind = -G.independent_program_nll(hs, x, extra, t, 0)
        assert abs(out["rounds:4"][0][k] - ind) < NTOL, (b, t)
    assert (out["rounds:4"][1] == len(extra)).all() and (out["gate:2.0"][1] == L).all()


@torch.no_grad()
def test_routed_ouro_paths():
    """routed_ouro: at init (every head says 'once') the routed model is the reference model; forcing
    'twice' / 'skip' at one virtual layer equals running that layer twice / not at all for every token."""
    from routed_ouro import RoutedOuro

    rm = RoutedOuro(MODEL, protect=1).double()
    lg, lay, _, _ = rm(IDS, hard_eval=True)
    assert (lg - MODEL(IDS).logits).abs().max() < 1e-4 and (lay == L).all()
    m = MODEL.model
    pos = torch.arange(IDS.shape[1])[None].expand(IDS.shape[0], -1)
    for j, act in ((4, 2), (5, 0)):
        with torch.no_grad():
            rm.heads.bias.view(L, 3)[j] = torch.tensor([3.0, -3.0, -3.0] if act == 0 else [-3.0, -3.0, 3.0])
        lg, lay, _, _ = rm(IDS, hard_eval=True)
        rm.heads.bias.view(L, 3)[j] = torch.tensor([-3.0, 3.0, -3.0])
        x = m.embed_tokens(IDS)
        cos, sin = m.rotary_emb(x, pos)
        for i in range(L):
            if i > 0 and i % N == 0:
                x = m.norm(x)
            reps = act if i == j else 1
            for _ in range(reps):
                x = MODEL._layer(m.layers[i % N], x, cos, sin)
        ref = MODEL.lm_head(m.norm(x))
        assert (lg - ref).abs().max() < 1e-4, (j, act)  # SDPA vs manual attention rounding
        assert (lay == L - 1 + act).all()
