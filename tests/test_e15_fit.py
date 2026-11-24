"""
CPU tests for e15_fit.py (E15 scoring), on synthetic data:
  1. exact budget matching hits the budget for gates, the KL rule and charged costs
  2. the static mixed policy hits every budget
  3. charged costs add readout and attention overheads as specified
  4. the chunk bootstrap covers the mean
Run: python -m pytest -q tests/test_e15_fit.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from e15_fit import (BUDGETS, N_LAYERS, R, bias_rounds, boot, cost_fn, gate_rounds, kl_rounds,  # noqa: E402
                     match_budget)

G = torch.Generator().manual_seed(0)
N = 20000
PDF = torch.softmax(torch.randn(N, R, generator=G) * 2, -1)
U = torch.rand(N, generator=G)


def test_gate_budget_matching_is_exact():
    for B in BUDGETS:
        rr = match_budget(gate_rounds(PDF), 0.0, 1.0 + 1e-6, B, U)
        assert abs(N_LAYERS * (rr + 1).float().mean().item() - B) < 0.5
    assert match_budget(gate_rounds(PDF), 0.0, 1.0 + 1e-6, 120, U) is None  # beyond 96 layers: unreachable


def test_bias_policy_hits_budgets():
    for B in BUDGETS + [24, 96]:
        rr = bias_rounds(B, U)
        assert rr.min() >= 0 and rr.max() <= R - 1
        assert abs(N_LAYERS * (rr + 1).float().mean().item() - B) < 0.5


def test_kl_rule_matching_and_charges():
    own = [torch.rand(N, 5, generator=G) for _ in range(R)]
    import math
    for B in (60, 80):
        rr = match_budget(lambda s: kl_rounds(own, 0.1, math.exp(-s)), -5.0, 25.0, B, U)
        assert rr is not None and abs(N_LAYERS * (rr + 1).float().mean().item() - B) < 0.5
    rr = torch.tensor([0, 1, 3])
    assert torch.allclose(cost_fn("hidden", 2.0, 0.05)(rr), torch.tensor([24.0, 48.0, 96.0]))
    assert torch.allclose(cost_fn("own", 2.0, 0.05)(rr), torch.tensor([24.0, 50.0, 102.0]))
    assert torch.allclose(cost_fn("src_own", 2.0, 0.05)(rr), torch.tensor([24 + 1.2, 50 + 2.4, 102 + 4.8]))
    # charged matching hits the charged budget
    rr = match_budget(gate_rounds(PDF), 0.0, 1.0 + 1e-6, 70, U, layers_of=cost_fn("all", 2.0, 0.05))
    assert abs(cost_fn("all", 2.0, 0.05)(rr).mean().item() - 70) < 0.6


def test_bootstrap():
    x = np.random.default_rng(0).normal(0.3, 1.0, 200)
    lo, hi = boot(x)
    assert lo < x.mean() < hi and hi - lo < 0.4
    lo90, hi90 = boot(x, level=0.90)
    assert lo < lo90 < hi90 < hi


def test_gate_bank_rows_equal_gates_and_adam():
    from e15_fit import PRICES, GateBank
    from exit_signals import Gate, expected_objective
    torch.manual_seed(0)
    n, dim = 500, 7
    feats = [torch.randn(n, dim) for _ in range(R)]
    ce = torch.rand(n, R) * 3
    mean, std = torch.randn(dim) * 0.1, torch.rand(dim) + 0.5
    for per_round in (False, True):
        keys = [("x", lw, lb, p) for lw in (1e-3, 1e-2) for lb in (1e-2,) for p in PRICES]
        bank = GateBank(dim, keys, per_round, mean=mean, std=std)
        gates, opts = [], []
        for k in keys:
            g = Gate(dim, R, mean=mean, std=std, per_round=per_round)
            gates.append(g)
            opts.append(torch.optim.Adam([{"params": [g.w], "lr": k[1]}, {"params": [g.b], "lr": k[2]}]))
        for _ in range(3):  # three optimizer steps: the bank's manual Adam == torch Adam, gate by gate
            bank.step(feats, ce)
            for k, g, o in zip(keys, gates, opts):
                obj, _ = expected_objective(g(feats), ce, k[3], N_LAYERS)
                o.zero_grad()
                obj.backward()
                o.step()
        pdf = bank.pdf(feats)
        for i, g in enumerate(gates):
            assert (pdf[:, i] - g(feats)).abs().max() < 1e-5
            assert (bank.W[i] - g.w.detach()).abs().max() < 1e-5
        # gate_state reproduces the best row as a Gate
        bank.check(feats, ce)
        g2 = Gate(dim, R, per_round=per_round)
        g2.load_state_dict(bank.gate_state(1))
        assert (g2(feats) - bank.pdf(feats, bank.best_W, bank.best_b)[:, 1]).abs().max() < 1e-6


def test_gate_bank_early_stopping_keeps_best():
    from e15_fit import GateBank
    feats = [torch.randn(100, 3) for _ in range(R)]
    ce = torch.rand(100, R)
    bank = GateBank(3, [("x", 1e-2, 1e-2, 0.001)], False)
    bank.check(feats, ce)
    best = bank.best.clone()
    W0 = bank.best_W.clone()
    bank.W += 100.0  # a much worse state must not replace the best one
    bank.check(feats, ce)
    assert torch.equal(bank.best, best) and torch.equal(bank.best_W, W0) and bank.bad[0] == 1
    bank.check(feats, ce)
    assert not bank.active[0]  # patience 2 reached
