"""
Selection curve: hindsight oracle gain vs the number of options, for real layer edits and for the
noise null (noise_null.py), on the same held-out positions.

If best-of-N random perturbations recover most of the real oracle's gain at the same N, the hindsight
gain is selection, not routable signal. The "selection-corrected oracle" is real minus noise at matched N.

  python selection_curve.py --oracle out/oracle.npz --noise out/noise.npz --out out/selection
  python selection_curve.py ... --max_cost 0     # real edits restricted to no extra compute

Writes selection.json and selection.png. Gains are positive nats/token: mean over positions of
max(0, -min over the option subset of dNLL). CIs bootstrap over held-out chunks (chunk % 5 == 0).
"""
import argparse
import json
import os

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--out", default="selection")
    p.add_argument("--max_cost", type=float, default=None, help="keep real edits with cost <= this")
    p.add_argument("--subsets", type=int, default=20, help="random option subsets per N")
    p.add_argument("--boot", type=int, default=1000)
    return p.parse_args()


def best_gain(delta):
    """Per-position hindsight gain over the given options (N,) >= 0."""
    return np.maximum(0.0, -delta.min(1))


def curve(delta, ns, subsets, rng):
    """Mean best-of-n gain, averaged over random option subsets of size n."""
    P = delta.shape[1]
    out = []
    for n in ns:
        reps = 1 if n == P else subsets
        out.append(float(np.mean([best_gain(delta[:, rng.choice(P, n, replace=False)]).mean() for _ in range(reps)])))
    return out


def boot_ci(x, chunk, n_boot, rng):
    """95% CI of mean(x) resampling whole chunks."""
    ids = np.unique(chunk)
    sums = np.array([x[chunk == c].sum() for c in ids])
    cnts = np.array([(chunk == c).sum() for c in ids])
    b = rng.integers(0, len(ids), (n_boot, len(ids)))
    m = sums[b].sum(1) / cnts[b].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    a = get_args()
    rng = np.random.default_rng(0)
    o, z = np.load(a.oracle), np.load(a.noise)
    key_o = o["chunk"].astype(np.int64) * 100_000 + o["pos"]
    key_z = z["chunk"].astype(np.int64) * 100_000 + z["pos"]
    assert np.array_equal(key_o, key_z), "oracle and noise must cover the same positions (same --n_tokens, --seq_len)"
    gap = np.abs(o["base_nll"] - z["base_nll"]).max()
    assert gap < 0.05, f"base NLL differs by {gap:.3f}: different model, data or dtype"

    test = o["chunk"] % 5 == 0
    real = o["delta"][test].astype(np.float32)
    if a.max_cost is not None:
        real = real[:, o["cfg_cost"] <= a.max_cost]
    noise_all = z["delta"][test].astype(np.float32)
    chunk, base_nll = o["chunk"][test], o["base_nll"][test]

    res = {"positions": int(test.sum()), "base_nll": float(base_nll.mean()), "max_cost": a.max_cost,
           "n_real": int(real.shape[1]), "by_scale": {}}
    ns_all = sorted({int(round(v)) for v in np.geomspace(1, max(real.shape[1], noise_all.shape[1]), 12)})
    curves = {"real": (ns := [n for n in ns_all if n <= real.shape[1]], curve(real, ns, a.subsets, rng))}
    for s in np.unique(z["cfg_scale"]):
        noise = noise_all[:, z["cfg_scale"] == s]
        N = min(real.shape[1], noise.shape[1])  # matched option count
        # the same real subset for every scale with the same N, so scales are compared on equal footing
        r_sub = real[:, np.random.default_rng(N).choice(real.shape[1], N, replace=False)] if N < real.shape[1] else real
        z_sub = noise[:, rng.choice(noise.shape[1], N, replace=False)] if N < noise.shape[1] else noise
        g_r, g_z = best_gain(r_sub), best_gain(z_sub)
        q = np.quantile(base_nll, [0.2, 0.4, 0.6, 0.8])
        quint = np.digitize(base_nll, q)
        res["by_scale"][f"{s:g}"] = {
            "n_matched": int(N),
            "real_gain": float(g_r.mean()), "real_ci": boot_ci(g_r, chunk, a.boot, rng),
            "noise_gain": float(g_z.mean()), "noise_ci": boot_ci(g_z, chunk, a.boot, rng),
            "noise_share_of_real": float(g_z.mean() / max(g_r.mean(), 1e-9)),
            "corrected_gain": float((g_r - g_z).mean()), "corrected_ci": boot_ci(g_r - g_z, chunk, a.boot, rng),
            "frac_helped_real": float((g_r > 0.05).mean()), "frac_helped_noise": float((g_z > 0.05).mean()),
            "by_base_nll_quintile": [{"real": float(g_r[quint == k].mean()), "noise": float(g_z[quint == k].mean())}
                                     for k in range(5)],
        }
        ns = [n for n in ns_all if n <= noise.shape[1]]
        curves[f"noise s={s:g}"] = (ns, curve(noise, ns, a.subsets, rng))
    res["curves"] = {k: {"n": v[0], "gain": v[1]} for k, v in curves.items()}

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    print(f"held-out positions {res['positions']:,}, base NLL {res['base_nll']:.3f}, real options {res['n_real']}")
    for s, r in res["by_scale"].items():
        print(f"  noise scale {s}: N={r['n_matched']}  real {r['real_gain']:.3f}  noise {r['noise_gain']:.3f} "
              f"({100 * r['noise_share_of_real']:.0f}% of real)  corrected {r['corrected_gain']:+.3f} "
              f"[{r['corrected_ci'][0]:+.3f}, {r['corrected_ci'][1]:+.3f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (ns, g) in curves.items():
        ax.plot(ns, g, "o-", lw=2, ms=4, label="real layer edits" if k == "real" else f"random noise ({k[6:]})")
    ax.set_xscale("log")
    ax.set_xlabel("Options per token (best chosen in hindsight)")
    ax.set_ylabel("Hindsight loss reduction (nats/token)")
    ax.set_title("Best-of-N noise vs best-of-N layer edits", loc="left", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "selection.png"), dpi=200)
    print(f"saved {os.path.join(a.out, 'selection.json')} and selection.png")


if __name__ == "__main__":
    main()
