"""
Analysis of oracle.npz (single-edit layer programs: repeat / skip / swap).

  E1  Headroom by op: oracle gain, fraction helped, fraction skippable, winning op/layer.
  E2  Compute-accuracy frontier: route each token to argmin_c [dNLL_hat(c) + lambda*cost(c)]
      (or the base program when nothing is below 0). Costs are signed: skip = -1, repeat k = k-1,
      swap = 0, so lambda > 0 lets skips pay for loops. Budget-matched NLL at fixed extra-layer budgets.
  E3  Lag: same-token vs t-1 routers, interp vs raw features; share of the same-token gain recovered.
  E4  Ablation: which interpretable feature group carries the lagged signal.

Negative dNLL = the edit helps. Train/test split by chunk (chunk % 5 == 0 is test).
95% CIs by bootstrap over test chunks.

Outputs (in --out_dir): results.json, frontier.png, router_lag_interp.npz
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

p = argparse.ArgumentParser()
p.add_argument("--path", default="oracle_out/oracle.npz")
p.add_argument("--out_dir", default="oracle_out")
p.add_argument("--thr", type=float, default=0.05, help="nats of gain that count as 'helped'")
p.add_argument("--skip_eps", type=float, default=0.01, help="a skip is 'safe' if dNLL < this")
p.add_argument("--max_train", type=int, default=200_000)
p.add_argument("--alpha", type=float, default=10.0)
p.add_argument("--n_lambdas", type=int, default=40)
p.add_argument("--n_boot", type=int, default=200)
p.add_argument("--budgets", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.25, 0.5])
p.add_argument("--sae", default=None, help="sae.npz from sae_features.py (adds lag_sae and lag_interp_sae routers)")
p.add_argument("--sae_select", type=int, default=1024, help="SAE latents kept, selected on train rows")
a = p.parse_args()
rng = np.random.default_rng(0)
os.makedirs(a.out_dir, exist_ok=True)

d = np.load(a.path)
delta = d["delta"].astype(np.float32)
L = int(d["n_layers"])
if "cfg_op" in d:
    op, lay, kk, cost = d["cfg_op"].astype(str), d["cfg_layer"], d["cfg_k"], d["cfg_cost"].astype(np.float32)
else:  # older repeat-only files
    op = np.array(["repeat"] * len(d["cfgs"]))
    lay, kk = d["cfgs"][:, 0], d["cfgs"][:, 1]
    cost = (kk - 1).astype(np.float32)
OPS = [o for o in dict.fromkeys(op.tolist()) if o != "base"]  # oracle.npz or library.npz
end = d["cfg_end"] if "cfg_end" in d else lay


cfg_name = d["cfg_name"].astype(str) if "cfg_name" in d else None


def name(c):
    if cfg_name is not None:
        return cfg_name[c]
    if op[c] == "block":
        return f"block@{lay[c]}-{end[c]}x{kk[c]}"
    return f"{op[c]}@{lay[c]}" + (f"x{kk[c]}" if op[c] == "repeat" else "")
feat, nll, raw = d["feat"].astype(np.float32), d["base_nll"].astype(np.float32), d["raw"]
chunk, pos = d["chunk"], d["pos"]
res = {"programs": [name(c) for c in range(len(op))]}

# ---------------------------------------------------------------- E1 headroom
best, arg = delta.min(1), delta.argmin(1)
gain = np.clip(-best, 0, None)
helped = gain > a.thr
static = delta.mean(0)
s = static.argmin()
h = {
    "positions": int(len(delta)),
    "base_nll": float(nll.mean()),
    "oracle_nll_reduction": float(gain.mean()),
    "oracle_pct_of_base": float(100 * gain.mean() / nll.mean()),
    "frac_helped": float(helped.mean()),
    "best_static": {"program": name(s), "mean_dnll": float(static[s])},
    "by_op": {},
}
for o in OPS:
    m = op == o
    g_o = np.clip(-delta[:, m].min(1), 0, None)
    h["by_op"][o] = {
        "oracle_gain_restricted": float(g_o.mean()),
        "frac_tokens_best_op": float(((op[arg] == o) & helped).mean()),
        "winning_layer_hist": np.bincount(np.clip(lay[arg[helped & (op[arg] == o)]], 0, None), minlength=L).tolist()
        if (lay[op == o] >= 0).all() else None,  # composite paths have no single anchor layer
        "best_static": name(np.where(m)[0][static[m].argmin()]),
    }
if "skip" in OPS:
    sk = op == "skip"
    h["frac_tokens_with_safe_skip"] = float((delta[:, sk].min(1) < a.skip_eps).mean())
    h["safe_skip_frac_by_layer"] = (delta[:, sk] < a.skip_eps).mean(0).tolist()
res["headroom"] = h
print("== E1 Oracle headroom ==")
for k_, v in h.items():
    if k_ not in ("by_op", "safe_skip_frac_by_layer"):
        print(f"  {k_}: {v}")
for o, v in h["by_op"].items():
    print(f"  [{o}] gain if restricted: {v['oracle_gain_restricted']:.4f}  best-op share: "
          f"{v['frac_tokens_best_op']:.1%}  best static: {v['best_static']}")

# ---------------------------------------------------------------- routers
i = np.where(pos > 0)[0]
j = i - 1
assert (chunk[j] == chunk[i]).all() and (pos[j] == pos[i] - 1).all()
test = chunk[i] % 5 == 0
train = ~test
Y = delta[i]
Yte = Y[test]
ar = np.arange(len(Yte))
te_chunks = chunk[i][test]
uniq = np.unique(te_chunks)
chunk_idx = [np.where(te_chunks == c)[0] for c in uniq]


def fit(X):
    Xtr, Ytr = X[train].astype(np.float32), Y[train]
    if len(Xtr) > a.max_train:
        k_ = rng.choice(len(Xtr), a.max_train, replace=False)
        Xtr, Ytr = Xtr[k_], Ytr[k_]
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=a.alpha).fit(sc.transform(Xtr), Ytr)
    return sc, m, m.predict(sc.transform(X[test].astype(np.float32)))


def route(P, lam):
    obj = P + lam * cost[None]
    c = obj.argmin(1)
    return np.where(obj[ar, c] < 0, c, -1)  # -1 = base program


def per_token(choice):
    hit = choice >= 0
    cc = np.maximum(choice, 0)
    return np.where(hit, Yte[ar, cc], 0.0), np.where(hit, cost[cc], 0.0)


def realize(choice):
    dn, ex = per_token(choice)
    return float(dn.mean()), float(ex.mean())


def boot_ci(v):
    means = [np.concatenate([v[chunk_idx[q]] for q in rng.integers(0, len(uniq), len(uniq))]).mean()
             for _ in range(a.n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


lams = np.concatenate([[0.0], np.geomspace(1e-3, 2.0, a.n_lambdas - 1)])


def frontier(P):
    base_pt = [dict(lam=None, dnll=0.0, extra_layers=0.0)]  # no edits at all
    return base_pt + [dict(lam=float(l_), dnll=dn, extra_layers=ex) for l_ in lams for dn, ex in [realize(route(P, l_))]]


def at_budget(fr, b):
    """Best NLL change among frontier points using <= b extra layers per token (None if unreachable)."""
    ok = [q["dnll"] for q in fr if q["extra_layers"] <= b + 1e-9]
    return float(min(ok)) if ok else None


sets = {
    "same_interp": feat[i],
    "lag_interp": feat[j],
    "lag_interp_surprisal": np.c_[feat[j], nll[j]],
    "same_raw": raw[i],
    "lag_raw": raw[j],
}

# ---------------------------------------------------------------- E4 SAE features (optional)
if a.sae:
    import scipy.sparse as sp

    S = np.load(a.sae)
    sidx, sval, s_layers, W = S["idx"].astype(np.int64), S["val"].astype(np.float32), S["layers"], int(S["width"])
    assert len(sidx) == len(delta), f"sae.npz has {len(sidx)} positions, oracle has {len(delta)}: use the same --n_tokens"
    N, nL, K = sidx.shape
    cols = (sidx + (np.arange(nL) * W)[None, :, None]).reshape(N, -1)
    Xs = sp.csr_matrix((sval.reshape(N, -1).ravel(), (np.repeat(np.arange(N), nL * K), cols.ravel())),
                       shape=(N, nL * W))
    Xs.sum_duplicates()
    # select latents on TRAIN rows only, by correlation of token t-1's latent with token t's best gain per op
    Xl = Xs[j[train]]
    targets = {"any": np.clip(-Y[train].min(1), 0, None)}
    for o in OPS:
        targets[o] = np.clip(-Y[train][:, op == o].min(1), 0, None)
    n = Xl.shape[0]
    mx = np.asarray(Xl.mean(0)).ravel()
    sx = np.sqrt(np.maximum(np.asarray(Xl.multiply(Xl).mean(0)).ravel() - mx ** 2, 1e-12))
    corr = {}
    for tn, y in targets.items():
        cxy = np.asarray(Xl.T @ (y - y.mean())).ravel() / n
        corr[tn] = cxy / (sx * (y.std() + 1e-12))
    score = np.max(np.abs(np.stack(list(corr.values()))), 0)
    score[np.asarray((Xl != 0).sum(0)).ravel() < 20] = 0  # ignore latents that almost never fire
    sel = np.argsort(-score)[: a.sae_select]
    Xsel = Xs[:, sel].toarray().astype(np.float32)
    sets["lag_sae"] = Xsel[j]
    sets["lag_interp_sae"] = np.c_[feat[j], Xsel[j]]
    top = []
    for c in sel[:25]:
        li, lat = divmod(int(c), W)
        npid = str(S["neuronpedia"][li])
        top.append({"layer": int(s_layers[li]), "latent": lat, "fire_rate": float((Xl[:, c] != 0).mean()),
                    **{f"corr_{tn}": float(corr[tn][c]) for tn in corr},
                    "neuronpedia": f"https://www.neuronpedia.org/{npid}/{lat}"})
    res["sae_top_features"] = top
    print("\n== E4 SAE latents on token t-1 most correlated with token t's gain (train rows) ==")
    for q in top[:10]:
        print(f"  L{q['layer']:>2} #{q['latent']:<6} fire {q['fire_rate']:.1%}  "
              + " ".join(f"{tn}:{q['corr_' + tn]:+.3f}" for tn in corr) + f"  {q['neuronpedia']}")

res["routers"], res["frontiers"] = {}, {"oracle": frontier(Yte)}
res["frontiers"]["static"] = [dict(program=name(c), dnll=float(Yte[:, c].mean()), extra_layers=float(cost[c]))
                              for c in range(len(op))]
s_tr = Y[train].mean(0).argmin()
print("\n== E2/E3 Routers on held-out chunks (lambda = 0) ==")
print(f"  oracle:      {np.minimum(Yte.min(1), 0).mean():+.4f}")
print(f"  best static: {Yte[:, s_tr].mean():+.4f}  ({name(s_tr)}, chosen on train)")
print(f"  {'features':24s} {'dNLL':>8s} {'95% CI':>19s} {'AUC':>6s} {'edit%':>6s} {'+layers':>8s}  ops used")
for nm, X in sets.items():
    sc, m, P = fit(X)
    ch = route(P, 0.0)
    dn_tok, _ = per_token(ch)
    dn, ex = realize(ch)
    lab = Yte.min(1) < -a.thr
    auc = roc_auc_score(lab, -P.min(1)) if 0 < lab.sum() < len(lab) else float("nan")
    used = {o: float((op[ch[ch >= 0]] == o).mean()) if (ch >= 0).any() else 0.0 for o in OPS}
    fr = frontier(P)
    res["routers"][nm] = {"dnll": dn, "ci95": boot_ci(dn_tok), "auc": float(auc), "edit_frac": float((ch >= 0).mean()),
                          "extra_layers": ex, "ops_used": used,
                          "budget_matched": {str(b): at_budget(fr, b) for b in a.budgets}}
    res["frontiers"][nm] = fr
    ci = res["routers"][nm]["ci95"]
    print(f"  {nm:24s} {dn:+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {auc:6.3f} {100 * (ch >= 0).mean():5.1f}% {ex:+8.3f}  "
          + " ".join(f"{o}:{v:.0%}" for o, v in used.items()))
    if nm == "lag_interp":
        np.savez(os.path.join(a.out_dir, "router_lag_interp.npz"), mean=sc.mean_, scale=sc.scale_, coef=m.coef_,
                 intercept=m.intercept_, cfg_op=op, cfg_layer=lay, cfg_k=kk, cfg_cost=cost)

# ---------------------------------------------------------------- prior-work heuristic baselines
# Training-free per-token rules in the spirit of EGLR (arXiv 2606.16620: re-apply top layers when the
# next-token entropy is high) and ALRR (arXiv 2606.06188: repeat the layer where the hidden-state norm
# peaks). Thresholds are quantiles of the TRAIN scores; each threshold gives one frontier point.
ent_final = feat[:, L - 1]           # logit-lens entropy at the last layer = the model's next-token entropy
upd = feat[:, 2 * L : 3 * L]          # relative residual-update norm per layer
rep2 = {int(lay[c]): c for c in range(len(op)) if op[c] == "repeat" and kk[c] == 2}
top_rep = [c for c in range(len(op)) if op[c] == "repeat" and lay[c] >= L - 4]
c_top = top_rep[int(np.argmin(Y[train][:, top_rep].mean(0)))] if top_rep else None


def rule_frontier(score_rows, prog_rows):
    sc_tr, sc_te, pr_te = score_rows[train], score_rows[test], prog_rows[test]
    pts = [dict(lam=None, dnll=0.0, extra_layers=0.0)]
    for q in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
        ch = np.where(sc_te > np.quantile(sc_tr, q), pr_te, -1)
        dn, ex = realize(ch)
        pts.append(dict(lam=q, dnll=dn, extra_layers=ex))
    return pts


rules = {}
if c_top is not None:
    rules["rule_entropy_same"] = (ent_final[i], np.full(len(i), c_top))  # EGLR-like, decided after t's pass
    rules["rule_entropy_lag"] = (ent_final[j], np.full(len(i), c_top))   # same gate, one token early
if rep2:
    peak = upd[i].argmax(1)
    rules["rule_normpeak_same"] = (upd[i].max(1), np.array([rep2.get(int(l_), -1) for l_ in peak]))  # ALRR-like
print("\n== Prior-work heuristic baselines (training-free) ==")
for nm, (sc, pr) in rules.items():
    fr = rule_frontier(sc, pr)
    res["frontiers"][nm] = fr
    best_pt = min(fr, key=lambda q: q["dnll"])
    print(f"  {nm:24s} best dNLL {best_pt['dnll']:+.4f} at {best_pt['extra_layers']:+.3f} layers "
          f"(threshold quantile {best_pt['lam']})")
if c_top is not None:
    print(f"  (entropy rules repeat {name(c_top)}, the best top-4-layer repeat on train)")

print("\n== E2 Budget-matched dNLL (best frontier point with <= budget extra layers/token) ==")
print("  " + f"{'':24s}" + "".join(f"{b:>+9.2f}" for b in a.budgets))
res["budget_matched"] = {}
for nm in ["oracle"] + list(sets) + list(rules):
    row = [at_budget(res["frontiers"][nm], b) for b in a.budgets]
    res["budget_matched"][nm] = dict(zip(map(str, a.budgets), row))
    print(f"  {nm:24s}" + "".join(f"{v:+9.4f}" if v is not None else f"{'—':>9s}" for v in row))

same, lag = res["routers"]["same_interp"]["dnll"], res["routers"]["lag_interp"]["dnll"]
res["lag_recovery"] = float(lag / same) if same < 0 else None
print(f"\n  E3 lag recovery (t-1 interp / same-token interp at lambda=0): {res['lag_recovery']}")

# ---------------------------------------------------------------- E4 ablation
print("\n== E4 Lagged interp feature groups (lambda = 0) ==")
res["ablation"] = {}
for gi, gname in enumerate(["lens_ent", "lens_agree", "upd_norm", "attn_ent"]):
    _, _, P = fit(feat[j][:, gi * L : (gi + 1) * L])
    dn, ex = realize(route(P, 0.0))
    res["ablation"][gname] = {"dnll": dn, "extra_layers": ex}
    print(f"  {gname:24s} {dn:+8.4f}  +layers {ex:+.3f}")

with open(os.path.join(a.out_dir, "results.json"), "w") as f:
    json.dump(res, f, indent=1)

# ---------------------------------------------------------------- frontier plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pareto(points):
        """Non-dominated points: no other point saves more NLL with no more compute."""
        pts = sorted(points, key=lambda q: (q["extra_layers"], q["dnll"]))
        out, best = [], float("inf")
        for q in pts:
            if q["dnll"] < best - 1e-12:
                out.append(q)
                best = q["dnll"]
        return out

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    style = {"oracle": ("#d4a017", "-"), "same_interp": ("#2b8cbe", "-"), "same_raw": ("#2b8cbe", ":"),
             "lag_interp": ("#41ab5d", "-"), "lag_interp_surprisal": ("#41ab5d", "--"), "lag_raw": ("#41ab5d", ":"),
             "lag_sae": ("#756bb1", ":"), "lag_interp_sae": ("#756bb1", "-"),
             "rule_entropy_same": ("#636363", "--"), "rule_entropy_lag": ("#636363", ":"),
             "rule_normpeak_same": ("#969696", "-.")}
    style = {k_: v for k_, v in style.items() if k_ in res["frontiers"]}
    for nm, (c, ls) in style.items():
        fr = pareto(res["frontiers"][nm])
        ax.plot([q["extra_layers"] for q in fr], [-q["dnll"] for q in fr], ls, color=c, label=nm, lw=2, marker=".", ms=3)
    st = res["frontiers"]["static"]
    mk = {"repeat": "o", "skip": "v", "swap": "s", "block": "D", "exit": "x"}
    for o in OPS:
        pts = [q for q, oo in zip(st, op) if oo == o]
        ax.scatter([q["extra_layers"] for q in pts], [-q["dnll"] for q in pts], s=12, marker=mk.get(o, "."),
                   color="#de2d26", alpha=0.6, label=f"static {o}")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("extra layer applications per token (negative = fewer)")
    ax.set_ylabel("NLL saved (nats / token)")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Per-token layer programs: accuracy vs compute (Pareto fronts)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "frontier.png"), dpi=200)
    print(f"\nsaved {a.out_dir}/frontier.png and results.json")
except ImportError:
    print("\nmatplotlib not installed; skipped frontier.png (results.json saved)")
