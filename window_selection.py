"""
Window-level selection test: does coarser routing carry signal that per-token routing lacks?

E13 (selection_curve.py) showed that, per token, best-of-N random perturbations match best-of-N
layer edits: the per-token hindsight gain is selection. Library of Layers (arXiv 2611.03526) routes
whole windows of tokens instead of single tokens. Here every window of W consecutive positions must
take ONE option (layer edit, or noise draw (layer, j)) for all its tokens, chosen in hindsight:

  gain(window) = max(0, -min_c sum_{t in window} dNLL_t(c))      (base program if nothing helps)

A real edit is the same program at every token of the window; a noise option is the same (layer,
draw index) with an independent random direction per token. If layer edits have effects that are
consistent across neighbouring tokens and noise does not, real gain should pull ahead of noise as W
grows. W = 1 reproduces E13; W = 511 is one choice per 512-token chunk.

APPROXIMATION: a window's loss change is the sum of single-token oracle deltas (each measured with
only that token edited). It ignores how edited tokens change what later tokens in the window attend
to (E5's interaction gap). Treat this as a screen for whether an exact window-level run is worth it.

Also reports a static baseline: the one option with the best mean dNLL on train chunks, applied to
every test token.

  python window_selection.py --oracle results/gemma-4-e2b-e13/oracle.npz --noise results/gemma-4-e2b-e13/noise.npz \
      --out results/window-selection/gemma-4-e2b
"""
import argparse
import json
import os

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[1, 8, 32, 128, 511])
    p.add_argument("--boot", type=int, default=1000)
    return p.parse_args()


def window_gain(delta, chunk, pos, W):
    """Per-window hindsight gain (total nats) and each window's chunk and token count."""
    key = chunk.astype(np.int64) * 100_000 + pos // W
    _, g = np.unique(key, return_inverse=True)
    sums = np.zeros((g.max() + 1, delta.shape[1]), dtype=np.float64)
    np.add.at(sums, g, delta)
    gain = np.maximum(0.0, -sums.min(1))
    w_chunk = np.zeros(len(gain), dtype=np.int64)
    w_chunk[g] = chunk
    return gain, w_chunk, np.bincount(g)


def per_chunk(gain, w_chunk, ids):
    return np.array([gain[w_chunk == c].sum() for c in ids])


def boot(num, den, n_boot, rng):
    """Mean and 95% CI of sum(num)/sum(den), resampling chunks."""
    b = rng.integers(0, len(num), (n_boot, len(num)))
    m = num[b].sum(1) / den[b].sum(1)
    return float(num.sum() / den.sum()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    a = get_args()
    rng = np.random.default_rng(0)
    o, z = np.load(a.oracle), np.load(a.noise)
    key_o = o["chunk"].astype(np.int64) * 100_000 + o["pos"]
    key_z = z["chunk"].astype(np.int64) * 100_000 + z["pos"]
    assert np.array_equal(key_o, key_z), "oracle and noise must cover the same positions"

    test = o["chunk"] % 5 == 0
    chunk, pos = o["chunk"][test], o["pos"][test]
    ids = np.unique(chunk)
    tok = np.array([(chunk == c).sum() for c in ids], dtype=np.float64)
    res = {"positions": int(test.sum()), "windows": a.windows, "by_options": {}}

    for name, max_cost in [("all", None), ("cost<=0", 0.0)]:
        keep = np.ones(o["delta"].shape[1], bool) if max_cost is None else o["cfg_cost"] <= max_cost
        real_all = o["delta"][:, keep].astype(np.float32)
        # static baseline: best mean option on train chunks, applied to every test token
        st = int(real_all[~test].mean(0).argmin())
        static = {"option": st, "gain": float(-real_all[test][:, st].mean())}
        out = {"n_real": int(keep.sum()), "static_real": static, "by_scale": {}}
        for s in np.unique(z["cfg_scale"]):
            noise_all = z["delta"][:, z["cfg_scale"] == s].astype(np.float32)
            N = min(real_all.shape[1], noise_all.shape[1])  # matched option count, subsets as in selection_curve.py
            r_idx = np.random.default_rng(N).choice(real_all.shape[1], N, replace=False) if N < real_all.shape[1] \
                else np.arange(real_all.shape[1])
            z_idx = rng.choice(noise_all.shape[1], N, replace=False) if N < noise_all.shape[1] \
                else np.arange(noise_all.shape[1])
            real, noise = real_all[test][:, r_idx], noise_all[test][:, z_idx]
            zs = int(noise_all[~test].mean(0).argmin())
            row = {"n_matched": int(N), "static_noise_gain": float(-noise_all[test][:, zs].mean()), "by_window": {}}
            for W in a.windows:
                gr, wc, _ = window_gain(real, chunk, pos, W)
                gz, _, _ = window_gain(noise, chunk, pos, W)
                cr, cz = per_chunk(gr, wc, ids), per_chunk(gz, wc, ids)
                br = np.random.default_rng(1)  # same chunk resamples for real, noise and their difference
                m_r, ci_r = boot(cr, tok, a.boot, br)
                br = np.random.default_rng(1)
                m_z, ci_z = boot(cz, tok, a.boot, br)
                br = np.random.default_rng(1)
                m_d, ci_d = boot(cr - cz, tok, a.boot, br)
                row["by_window"][str(W)] = {"real": m_r, "real_ci": ci_r, "noise": m_z, "noise_ci": ci_z,
                                            "noise_share": m_z / max(m_r, 1e-9), "corrected": m_d, "corrected_ci": ci_d,
                                            "frac_windows_helped_real": float((gr > 0).mean()),
                                            "frac_windows_helped_noise": float((gz > 0).mean())}
            out["by_scale"][f"{s:g}"] = row
        res["by_options"][name] = out

    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "window_selection.json"), "w"), indent=1)
    print(f"held-out positions {res['positions']:,}; gains in nats/token; corrected = real - noise [95% chunk CI]")
    for name, out in res["by_options"].items():
        print(f"\n== options: {name}  (real {out['n_real']}; static best-on-train real edit: "
              f"{out['static_real']['gain']:+.3f})")
        for s, row in out["by_scale"].items():
            print(f"  noise scale {s}, N={row['n_matched']} (static best noise option: {row['static_noise_gain']:+.3f})")
            for W, r in row["by_window"].items():
                print(f"    W={W:>4}: real {r['real']:.3f}  noise {r['noise']:.3f} ({100 * r['noise_share']:.0f}%)  "
                      f"corrected {r['corrected']:+.3f} [{r['corrected_ci'][0]:+.3f}, {r['corrected_ci'][1]:+.3f}]  "
                      f"windows helped {r['frac_windows_helped_real']:.2f}/{r['frac_windows_helped_noise']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, out) in zip(axs, res["by_options"].items()):
        Ws = a.windows
        first = True
        for s, row in out["by_scale"].items():
            if first:
                ax.plot(Ws, [row["by_window"][str(W)]["real"] for W in Ws], "o-", lw=2, color="k", label="real layer edits")
                first = False
            ax.plot(Ws, [row["by_window"][str(W)]["noise"] for W in Ws], "o--", lw=1.5, label=f"noise s={s}")
        ax.axhline(out["static_real"]["gain"], color="grey", lw=1, ls=":", label="static best edit")
        ax.set_xscale("log")
        ax.set_xlabel("Window size W (tokens sharing one choice)")
        ax.set_title(f"options: {name}", loc="left")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Hindsight loss reduction (nats/token)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "window_selection.png"), dpi=200)
    print(f"\nsaved {os.path.join(a.out, 'window_selection.json')} and window_selection.png")


if __name__ == "__main__":
    main()
