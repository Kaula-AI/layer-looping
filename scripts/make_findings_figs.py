"""Figures for docs/findings.md, built only from committed result files.

  python scripts/make_findings_figs.py   # writes docs/figures/*.png
"""
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (light)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


def fig_hindsight():
    """Potential (hindsight) gain vs what the best router captures at zero extra compute."""
    runs = [("GPT-2 small\n(pilot)", "pilot/gpt2/results.json"),
            ("Qwen3-1.7B\n(pilot)", "pilot/qwen3-1.7b-base/results.json"),
            ("Gemma-4 E2B", "results/gemma-4-e2b-quick/results.json"),
            ("Ouro-1.4B", "results/ouro-1.4b/ouro_quick/results.json")]
    names, oracle, best = [], [], []
    for n, p in runs:
        d = load(p)
        names.append(n)
        oracle.append(d["headroom"]["oracle_nll_reduction"])
        bm = [v["budget_matched"].get("0.0") for v in d["routers"].values()]
        best.append(max(0.0, -min(b for b in bm if b is not None)))  # NLL reduction actually captured
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    h = 0.34
    ax.barh(y + h / 2 + 0.02, oracle, height=h, color=BLUE)
    ax.barh(y - h / 2 - 0.02, best, height=h, color=ORANGE)
    for yi, o, b in zip(y, oracle, best):
        ax.text(o + 0.015, yi + h / 2 + 0.02, f"{o:.2f}", va="center", color=INK, fontsize=10)
        ax.text(b + 0.015, yi - h / 2 - 0.02, f"{b:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Loss reduction per token (nats), at no extra compute")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Hindsight finds big gains; no router captures them")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Best edit per token, chosen knowing the answer"),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Best trained router (chooses before the answer)")],
              loc="lower right", frameon=False, fontsize=9)
    ax.set_ylim(y[-1] - 1.1, y[0] + 0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_hindsight_vs_router.png"), dpi=200)
    plt.close(fig)


def paired(dirs, a, b):
    """Run bench_compare.py and parse rows: task -> (diff, lo, hi, layers_a, layers_b)."""
    cmd = [sys.executable, os.path.join(ROOT, "bench_compare.py"), "--dir", *dirs, "--a", a, "--b", b,
           "--metric", "acc_norm"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
    rows = {}
    for line in out[2:]:
        f = line.split()
        lo, hi = f[5].strip("[]").split(",")
        rows[f[0]] = (float(f[4]), float(lo), float(hi), float(f[7]), float(f[8]))
    return rows


def fig_gate_vs_router():
    """Paired accuracy differences: Ouro's own gate vs fixed rounds, and our router vs Ouro's gate."""
    o = os.path.join(ROOT, "results", "ouro-1.4b")
    g = paired([os.path.join(o, "bench_fast"), os.path.join(o, "stage2", "bench_orig")], "gate:0.2", "rounds:3")
    r = paired([os.path.join(o, "stage2", "bench_routed_frozen"), os.path.join(o, "bench_fast"),
                os.path.join(o, "stage2", "bench_orig")], "routed", "gate:0.2")
    tasks = ["arc_easy", "arc_challenge", "mmlu", "hellaswag", "winogrande"]
    label = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "mmlu": "MMLU", "hellaswag": "HellaSwag",
             "winogrande": "Winogrande"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(tasks))[::-1]
    for rows, col, off, name in ((g, BLUE, 0.17, "Ouro's trained exit gate minus fixed 3 rounds"),
                                 (r, ORANGE, -0.17, "Our trained per-layer router minus Ouro's gate")):
        for yi, t in zip(y, tasks):
            d, lo, hi, la, lb = rows[t]
            ax.plot([100 * lo, 100 * hi], [yi + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(100 * d, yi + off, "o", ms=8, color=col, mec=SURFACE, mew=2)
            ax.text(max(100 * hi, 0) + 0.3, yi + off,
                    f"{100 * d:+.1f}  ({la:.0f} vs {lb:.0f} layers)".replace("-", "\u2212"), va="center",
                    fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [label[t] for t in tasks])
    ax.set_xlabel("Accuracy difference, points (acc_norm, 1,000 items, paired 95% CI)")
    ax.set_xlim(-8.5, 9.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Ouro's own gate beats fixed compute; our router loses to it")
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", lw=2, label="Ouro's trained exit gate minus fixed 3 rounds"),
                       plt.Line2D([], [], color=ORANGE, marker="o", lw=2, label="Our trained per-layer router minus Ouro's gate")],
              loc="lower left", frameon=False, fontsize=9)
    ax.set_ylim(y[-1] - 1.3, y[0] + 0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_gate_vs_router.png"), dpi=200)
    plt.close(fig)


DIAG = "results/ouro-1.4b/stage2/router72_frozen_v2/diag_frozen/diagnose.json"


def fig_compute_curve():
    """Held-out NLL vs layers per token: Ouro's gate, fixed rounds, and our router with its controls."""
    d = load(DIAG)
    pts = {c["policy"]: c for c in d["curve"]}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    gate = sorted([c for c in d["curve"] if c["policy"].startswith("gate")], key=lambda c: c["layers"])
    ax.plot([c["layers"] for c in gate], [c["nll"] for c in gate], "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5,
            label="Ouro's trained exit gate (thresholds 0.05–0.9)")
    rnd = [pts[f"rounds:{r}"] for r in (2, 3, 4)]
    ax.plot([c["layers"] for c in rnd], [c["nll"] for c in rnd], "s", color=AQUA, ms=8, mec=SURFACE, mew=1.5,
            label="Fixed rounds for every token")
    for r, c in zip((2, 3, 4), rnd):
        ax.annotate(f"{r} rounds", (c["layers"], c["nll"]), xytext=(6, 8), textcoords="offset points", fontsize=9, color=INK2)
    router = [("router_argmax", "our router"), ("router_static_schedule", "same router, one schedule for all tokens"),
              ("router_shuffled", "router decisions shuffled across tokens"), ("router_sampled", "router, sampled as in training")]
    for k, lab in router:
        c = pts[k]
        ax.plot(c["layers"], c["nll"], "D", color=ORANGE, ms=8, mec=SURFACE, mew=1.5)
        ax.annotate(lab, (c["layers"], c["nll"]), xytext=(9, -3), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("Layers per token (of 96)")
    ax.set_ylabel("Held-out loss (nats per token)")
    ax.set_xlim(44, 128)
    ax.set_ylim(2.32, 2.68)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.plot([], [], "D", color=ORANGE, ms=8, label="Our trained per-layer router (Ouro frozen)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("At any compute, Ouro's gate has lower loss than our router")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_compute_curve.png"), dpi=200)
    plt.close(fig)


def fig_skip_map():
    """Where the router skips: rate per virtual layer, by round."""
    d = load(DIAG)
    sk = np.array(d["skip_rate_by_layer"])
    n = len(sk) // 4
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(sk))
    ax.bar(x, sk, width=0.8, color=ORANGE)
    for r in range(1, 4):
        ax.axvline(r * n - 0.5, color=INK2, lw=1, ls=(0, (2, 3)))
    for r in range(4):
        ax.text(r * n + n / 2, 1.08, f"Round {r + 1}: {100 * sk[r * n:(r + 1) * n].mean():.0f}% skipped", ha="center",
                fontsize=9.5, color=INK2)
    ax.set_xlim(-1, len(sk))
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Ouro's 96 unrolled layers (24 layers × 4 rounds)")
    ax.set_ylabel("Tokens skipping")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Our router cuts the middle of round 3 and never touches round 4", pad=22)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_skip_map.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_hindsight()
    fig_gate_vs_router()
    fig_compute_curve()
    fig_skip_map()
    print("wrote", sorted(os.listdir(OUT)))
