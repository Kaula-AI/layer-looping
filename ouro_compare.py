"""Ouro: per-token layer paths vs Ouro's own learned exit gate, at matched compute.

Ouro's gate (trained with the model) picks how many of its 4 rounds each token runs. For a threshold q it
exits at the first round whose cumulative exit probability reaches q (OuroForCausalLM's exit_threshold).
Each threshold gives one (extra layers per token, dNLL vs 4 full rounds) point. For every point, this
prints the best dNLL each probe.py router / the oracle reaches with no more extra layers, so a router
row below the gate row means per-token paths beat the gate at that compute.

  python ouro_compare.py --path oracle_out/oracle.npz --results oracle_out/results.json
"""
import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--path", required=True)
p.add_argument("--results", required=True)
p.add_argument("--qs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99])
p.add_argument("--out", default=None)
a = p.parse_args()

d = np.load(a.path)
delta = d["delta"].astype(np.float32)
op = d["cfg_op"].astype(str)
cost = d["cfg_cost"].astype(np.float32)
pdf = d["exit_pdf"].astype(np.float32)
R = pdf.shape[1]
test = (d["chunk"] % 5 == 0) & (d["pos"] > 0)  # probe.py's held-out chunks; pos > 0 so lagged routers exist
exit_col = {int(k): c for c, k in zip(np.where(op == "exit")[0], d["cfg_k"][op == "exit"])}  # round s -> column
assert sorted(exit_col) == list(range(1, R)), exit_col


def at_budget(fr, b):
    ok = [q["dnll"] for q in fr if q["extra_layers"] <= b + 1e-9]
    return float(min(ok)) if ok else None


def gate_point(q):
    cum = np.cumsum(pdf, 1)
    s = np.where((cum >= q).any(1), (cum >= q).argmax(1) + 1, R)  # rounds run, 1..R
    dn = np.zeros(len(s), np.float32)
    ex = np.zeros(len(s), np.float32)
    for r in range(1, R):
        m = s == r
        dn[m] = delta[m, exit_col[r]]
        ex[m] = cost[exit_col[r]]
    return float(dn[test].mean()), float(ex[test].mean()), float((s[test] < R).mean())


# oracle over exits only (hindsight): the best a perfect exit router could do, at each lambda
ex_cols = [exit_col[r] for r in range(1, R)]
D = np.c_[np.zeros(len(delta)), delta[:, ex_cols]][test]
C = np.r_[0.0, cost[ex_cols]]
lams = np.concatenate([[0.0], np.geomspace(1e-4, 0.2, 40)])
ex_front = []
for l_ in lams:
    ch = (D + l_ * C).argmin(1)
    ex_front.append(dict(dnll=float(D[np.arange(len(ch)), ch].mean()), extra_layers=float(C[ch].mean())))

res = json.load(open(a.results))
fronts = dict(res["frontiers"])
fronts["oracle_exits_only"] = ex_front
names = ["oracle", "oracle_exits_only", "same_interp", "lag_interp", "same_raw", "lag_raw", "static"]
names = [n for n in names if n in fronts]

rows = []
print(f"{'gate q':>7} {'exit%':>6} {'+layers':>8} {'gate dNLL':>10}  " + "  ".join(f"{n:>17}" for n in names))
for q in a.qs:
    dn, ex, frac = gate_point(q)
    best = {n: at_budget(fronts[n], ex) for n in names}
    rows.append(dict(q=q, exit_frac=frac, extra_layers=ex, gate_dnll=dn, best=best))
    cells = "  ".join(f"{'—' if v is None else f'{v:+.4f}':>17}" for v in best.values())
    print(f"{q:7.2f} {frac:6.1%} {ex:+8.2f} {dn:+10.4f}  {cells}")
print("\nA router beats Ouro's gate at a budget if its value is below the gate dNLL in that row.")
json.dump(dict(rows=rows), open(a.out or a.results.replace("results.json", "ouro_compare.json"), "w"), indent=1)
