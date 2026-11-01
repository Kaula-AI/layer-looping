"""Paired comparison of two bench.py policies on the same items: accuracy difference with a 95% bootstrap
CI over items, the McNemar exact p-value, and both policies' layers per answer token.

  python bench_compare.py --dir bench_fast --a gate:0.2 --b rounds:3 [--metric acc_norm]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest

p = argparse.ArgumentParser()
p.add_argument("--dir", nargs="+", required=True, help="bench.py output dirs (summary.json + per-task jsonl)")
p.add_argument("--a", required=True)
p.add_argument("--b", required=True)
p.add_argument("--metric", default="acc", choices=["acc", "acc_norm"])
a = p.parse_args()

summ = {}
for d in a.dir:
    summ.update(json.load(open(os.path.join(d, "summary.json"))))


def rows(task, pol):
    for d in a.dir:
        f = os.path.join(d, f"{task}_{pol.replace(':', '_')}.jsonl")
        if os.path.exists(f):
            return [json.loads(l)[a.metric] for l in open(f)]
    return None


tasks = sorted({os.path.basename(f).split("_" + a.a.split(":")[0])[0] for d in a.dir
                for f in glob.glob(os.path.join(d, f"*_{a.a.replace(':', '_')}.jsonl"))})
print(f"{a.metric}: {a.a} minus {a.b}")
print(f"{'task':14s} {'n':>5} {'A':>6} {'B':>6} {'A-B':>7} {'95% CI':>17} {'McNemar p':>10} {'layers A':>9} {'layers B':>9}")
for t in tasks:
    ra, rb = rows(t, a.a), rows(t, a.b)
    if ra is None or rb is None:
        continue
    ra, rb = np.array(ra), np.array(rb)
    diff = ra - rb
    boot = np.random.default_rng(0).integers(0, len(diff), (10000, len(diff)))
    lo, hi = np.percentile(diff[boot].mean(1), [2.5, 97.5])
    n01, n10 = int(((ra == 1) & (rb == 0)).sum()), int(((ra == 0) & (rb == 1)).sum())
    pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    la = summ.get(f"{t}|{a.a}", {}).get("layers_per_answer_token", float("nan"))
    lb = summ.get(f"{t}|{a.b}", {}).get("layers_per_answer_token", float("nan"))
    print(f"{t:14s} {len(diff):5d} {ra.mean():6.3f} {rb.mean():6.3f} {diff.mean():+7.3f} [{lo:+.3f},{hi:+.3f}] "
          f"{pv:10.4f} {la:9.1f} {lb:9.1f}")
