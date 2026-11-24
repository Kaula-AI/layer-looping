"""
CPU tests for exit_signals.py (E15) on a tiny random Ouro (3 layers x 3 rounds, float64):
  1. round_features' normed states and per-round CE == the reference Ouro forward at every exit step
  2. attention probabilities are causal distributions, and the probability path reproduces Ouro's layer
  3. no look-ahead: round-r features are identical whether or not later rounds are run
  4. no label or future leakage: features at position t do not change when tokens t+1.. change
  5. own / attn / layers / src / src_perm / pos features == brute-force recomputation
  6. Gate initialized from Ouro's gate == Ouro's exit distribution; per-round gates; objective, rules, oracle
Run: python -m pytest -q tests/test_exit_signals.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exit_signals import (ARMS, Gate, arm_inputs, decide, expected_objective, kl_rule,  # noqa: E402
                          layer_with_probs, oracle_exit, round_features)
from ouro_model import Ouro, config_from_dict  # noqa: E402

torch.manual_seed(0)
CFG = config_from_dict(dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                            rope_theta=1e4, total_ut_steps=3))
MODEL = Ouro(CFG).double().eval()
with torch.no_grad():
    for n_, p_ in MODEL.named_parameters():
        if "norm" in n_:
            p_.uniform_(0.5, 1.5)
        elif p_.dim() == 2:
            p_.normal_(0, 0.2)
N, R = 3, 3
EOS = 5
IDS = torch.randint(6, 97, (3, 12))
IDS[0, 4] = EOS
IDS[1, 8] = EOS
SRC = (0, 2)
KEYS = ("hn", "own", "attn", "layers", "src", "src_perm", "pos")
TOL = 1e-8


def feats(ids, **kw):
    return round_features(MODEL, ids, src_layers=SRC, eos_id=EOS, **kw)


def test_states_and_ce_match_reference():
    rd = feats(IDS)
    outs, _ = MODEL.rounds(IDS)
    for r in range(R):
        assert (rd[r]["hn"] - outs[r]).abs().max() < TOL
        lg = MODEL(IDS, exit_at_step=r).logits.double()
        ce = torch.nn.functional.cross_entropy(lg[:, :-1].transpose(1, 2), IDS[:, 1:], reduction="none")
        assert (rd[r]["ce"] - ce).abs().max() < 1e-6


def test_layer_probs():
    m = MODEL.model
    x = m.embed_tokens(IDS)
    pos = torch.arange(IDS.shape[1])[None].expand(3, -1)
    cos, sin = m.rotary_emb(x, pos)
    y, p = layer_with_probs(MODEL, m.layers[0], x, cos, sin)
    assert (y - MODEL._layer(m.layers[0], x, cos, sin)).abs().max() < TOL
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)))
    assert (p.triu(1) == 0).all()
    y2, _, ao, mo = layer_with_probs(MODEL, m.layers[0], x, cos, sin, return_parts=True)
    assert (y2 - (x + ao + mo)).abs().max() < TOL


def test_no_lookahead_rounds():
    full = feats(IDS)
    for r in range(1, R):
        part = feats(IDS, n_rounds=r)
        for k in KEYS + ("ce",):
            assert (part[r - 1][k] - full[r - 1][k]).abs().max() < TOL, (r, k)


def test_no_label_or_future_leakage():
    full = feats(IDS)
    for t in (0, 3, 7, 10):
        ids2 = IDS.clone()
        ids2[:, t + 1:] = torch.randint(6, 97, ids2[:, t + 1:].shape)  # change the label t+1 and everything after
        alt = feats(ids2)
        for r in range(R):
            for k in KEYS:
                assert (alt[r][k][:, : t + 1] - full[r][k][:, : t + 1]).abs().max() < TOL, (t, r, k)


def test_features_brute_force():
    rd = feats(IDS)
    m = MODEL.model
    B, T = IDS.shape
    x = m.embed_tokens(IDS)
    pos = torch.arange(T)[None].expand(B, -1)
    cos, sin = m.rotary_emb(x, pos)
    prev_p, prev_h = None, None
    for r in range(R):
        probs, parts, ins = [], [], []
        for li in range(N):
            ins.append(x.clone())
            x, p, ao, mo = layer_with_probs(MODEL, m.layers[li], x, cos, sin, return_parts=True)
            probs.append(p)
            parts.append((ao, mo))
        h = x.clone()
        x = m.norm(x)
        pr = torch.softmax(MODEL.lm_head(x).double(), -1)
        kl_all = torch.zeros(B, T) if prev_p is None else (pr * (pr.log() - prev_p.log())).sum(-1)
        ent_all = -(pr * pr.log()).sum(-1)
        for b in range(B):
            last_eos = -1
            for t in range(T):
                if IDS[b, t] == EOS:
                    last_eos = t
                own = rd[r]["own"][b, t]
                assert abs(own[2] - ent_all[b, t]) < 1e-6 and abs(own[3] - pr[b, t].max()) < 1e-6
                if r == 0:
                    assert own[0] == 0 and own[1] == 0 and own[4] == 1
                else:
                    dh = (h[b, t] - prev_h[b, t]).norm() / h[b, t].norm()
                    assert abs(own[0] - kl_all[b, t]) < 1e-6 and abs(own[1] - dh) < 1e-6 and own[4] == 0
                since = t - last_eos if last_eos >= 0 else t
                assert (rd[r]["pos"][b, t] - torch.tensor([math.log1p(t), math.log1p(since)])).abs().max() < 1e-6
                for li in range(N):
                    pt = probs[li][b, :, t, : t + 1]  # (H, t+1)
                    e = -(pt * pt.clamp_min(1e-30).log()).sum(-1)
                    e = e / math.log(t + 1) if t > 0 else torch.zeros_like(e)
                    want = torch.stack([e.mean(), pt[:, 0].mean(), pt[:, t].mean()])
                    assert (rd[r]["attn"][b, t, 3 * li:3 * li + 3] - want).abs().max() < 1e-6
                    ao, mo = parts[li]
                    xn = ins[li][b, t].norm()
                    want = torch.stack([ao[b, t].norm() / xn, mo[b, t].norm() / xn, (ao[b, t] + mo[b, t]).norm() / xn])
                    got = rd[r]["layers"][b, t, 3 * li:3 * li + 3].double()  # float32 features: relative tolerance
                    assert ((got - want).abs() / want.abs()).max() < 1e-6
                for si, li in enumerate(SRC):
                    w = probs[li][b, :, t, :].mean(0).clone()
                    w[0] = 0
                    w[t:] = 0  # self and later positions excluded
                    w = w / w.sum() if w.sum() > 0 else w
                    donor = (b - 1) % B
                    for j, (vals_self, vals_donor) in enumerate([(kl_all[b], kl_all[donor]), (ent_all[b], ent_all[donor])]):
                        assert abs(rd[r]["src"][b, t, 2 * si + j] - (w * vals_self).sum()) < 1e-6
                        assert abs(rd[r]["src_perm"][b, t, 2 * si + j] - (w * vals_donor).sum()) < 1e-6
        prev_p, prev_h = pr, h


def test_gate_objective_and_rules():
    rd = feats(IDS)
    outs, gates = MODEL.rounds(IDS)
    g = Gate(CFG.hidden_size, R, ouro_gate=MODEL.model.early_exit_gate).double()
    pdf = g([x["hn"] for x in rd])
    lam = [torch.sigmoid(gl[..., 0]) for gl in gates]
    ref, rem = [], torch.ones_like(lam[0])
    for r in range(R):
        ref.append(lam[r] * rem if r < R - 1 else rem)
        rem = rem * (1 - lam[r])
    # Gate keeps Ouro's gate weights in float32 (exact for the real bf16 gate; ~1e-8 off for this float64 toy)
    assert (pdf - torch.stack(ref, -1)).abs().max() < 1e-6
    assert torch.allclose(pdf.sum(-1), torch.ones_like(pdf.sum(-1)))
    # per-round weights: each round uses its own row
    f = arm_inputs(rd, "own")
    gp = Gate(5, R, per_round=True)  # float32, like the features
    with torch.no_grad():
        gp.w.copy_(torch.randn(R, 5))
    lam_p = [torch.sigmoid(f[r] @ gp.w[r]) for r in range(R)]
    assert (gp(f)[..., 0] - lam_p[0]).abs().max() < 1e-12
    # every arm builds inputs of the right width
    widths = {"hidden": 64, "own": 5, "hidden_own": 69, "attn_own": 3 * N + 5, "layers_own": 3 * N + 5,
              "src_own": 2 * len(SRC) + 5, "srcperm_own": 2 * len(SRC) + 5, "all": 64 + 5 + 3 * N + 2 * len(SRC),
              "pos": 2}
    for arm, w in widths.items():
        assert arm_inputs(rd, arm)[0].shape == (3, 11, w), arm
        assert all(torch.isfinite(v).all() for v in arm_inputs(rd, arm)), arm
    assert set(widths) | {"bias"} == set(ARMS)
    # objective == brute force
    ce = torch.stack([x["ce"] for x in rd], -1)
    p2 = pdf[:, :-1]
    obj, lay = expected_objective(p2, ce, 0.01, N)
    bf = sum(p2[..., r] * (ce[..., r] + 0.01 * N * (r + 1)) for r in range(R)).mean()
    assert abs(obj - bf) < 1e-10 and abs(lay - sum(p2[..., r] * N * (r + 1) for r in range(R)).mean()) < 1e-10
    # bias-only gate broadcasts
    pb = Gate(0, R).double()(arm_inputs(rd, "bias"))
    assert pb.shape == (R,) and abs(float(pb.sum()) - 1) < 1e-12
    expected_objective(pb, ce, 0.01, N)
    # decision rule: first round with cumulative prob >= q
    toy = torch.tensor([[[0.1, 0.3, 0.6], [0.6, 0.3, 0.1]]])
    assert decide(toy, 0.35).tolist() == [[1, 0]] and decide(toy, 0.95).tolist() == [[2, 2]]
    # zero-parameter rule: round 1 by entropy, later rounds by KL
    own = [torch.zeros(1, 4, 5) for _ in range(R)]
    own[0][0, :, 2] = torch.tensor([0.05, 2.0, 2.0, 2.0])
    own[1][0, :, 0] = torch.tensor([0.5, 0.01, 0.5, 0.5])
    own[2][0, :, 0] = torch.tensor([0.5, 0.5, 0.01, 0.5])
    assert kl_rule(own, 0.1, 0.1).tolist() == [[0, 1, 2]]
    # oracle: argmin of CE + price * layers
    toyce = torch.tensor([[[1.0, 0.5, 0.49]]])
    assert oracle_exit(toyce, 0.0, N).item() == 2 and oracle_exit(toyce, 0.01, N).item() == 1
