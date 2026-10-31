"""Label-free self-selection: run base + a few candidate programs, keep the most confident output.

Needs an oracle.npz written with `oracle.py --save_conf`. No router is trained: candidates are picked on train
chunks (best mean dNLL, or most often helpful), and the choice per token uses only the model's own output
(max log-prob or entropy). A margin m makes it conservative: switch only if the candidate beats the base
pass's confidence by > m. Held-out chunks (chunk % 5 == 0), 95% bootstrap CI over chunks.

  python selfselect.py --path oracle_out_fp32/oracle.npz [--compare oracle_out_quick/oracle.npz]
"""
import argparse
import json
import os

import numpy as np


def boot_ci(x, chunk, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(chunk)
    s = np.array([x[chunk == c].sum() for c in u])
    m = np.array([(chunk == c).sum() for c in u])
    idx = rng.integers(0, len(u), (n, len(u)))
    b = s[idx].sum(1) / m[idx].sum(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--compare", default=None, help="another oracle.npz on the same tokens (e.g. bf16) for noise")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = np.load(a.path)
    delta = d["delta"].astype(np.float32)
    cost = d["cfg_cost"]
    names = [f"{o}@{l}" + (f"x{k}" if o == "repeat" else "") for o, l, k in zip(d["cfg_op"], d["cfg_layer"], d["cfg_k"])]
    test = d["chunk"] % 5 == 0
    tr, te = ~test, test
    res = {}

    o = np.minimum(delta.min(1), 0)
    res["oracle_test"] = boot_ci(o[te], d["chunk"][te])
    print(f"oracle (test): {res['oracle_test'][0]:+.4f}  frac helped (<-0.05): {(o[te] < -0.05).mean():.3f}")

    if a.compare:
        c = np.load(a.compare)
        n = min(len(c["delta"]), len(delta))
        assert (c["tok"][:n] == d["tok"][:n]).all(), "token streams differ"
        dc = c["delta"][:n].astype(np.float32)
        noise = dc - delta[:n]
        oc = np.minimum(dc.min(1), 0)
        res["noise"] = {
            "rms_delta_diff": float(np.sqrt((noise ** 2).mean())),
            "corr_delta": float(np.corrcoef(dc.ravel(), delta[:n].ravel())[0, 1]),
            "oracle_this": float(o[:n].mean()), "oracle_compare": float(oc.mean()),
            # value of choosing with the compare file's argmin but scoring with this file's deltas
            "compare_choice_scored_here": float(np.minimum(delta[:n][np.arange(n), dc.argmin(1)], 0).mean()),
        }
        print("noise vs compare:", json.dumps(res["noise"], indent=1))

    if "top_edit" not in d:
        print("no confidence arrays: rerun oracle.py with --save_conf")
        return
    conf = {"top": (d["top_edit"].astype(np.float32), d["top_base"]),
            "negent": (-d["ent_edit"].astype(np.float32), -d["ent_base"])}
    rank = {"mean": np.argsort(delta[tr].mean(0)), "helprate": np.argsort(-(delta[tr] < -0.05).mean(0))}
    rows = []
    for sig, (ce, cb) in conf.items():
        for rk, order in rank.items():
            for k in a.ks:
                S = order[:k]
                for m in a.margins:
                    gain = ce[:, S] - cb[:, None]
                    j = gain.argmax(1)
                    use = gain[np.arange(len(j)), j] > m
                    dn = np.where(use, delta[np.arange(len(j)), S[j]], 0.0)
                    x = dn[te]
                    mean, lo, hi = boot_ci(x, d["chunk"][te])
                    # compute: every candidate is run for every token (verify-then-choose)
                    run_cost = float(np.sum(np.abs(cost[S]) + 1))
                    rows.append(dict(signal=sig, rank=rk, k=k, margin=m, dnll=mean, lo=lo, hi=hi,
                                     switch=float(use[te].mean()), cand=[names[i] for i in S[:4]],
                                     approx_extra_layers=run_cost))
    rows.sort(key=lambda r: r["hi"])
    print(f"\n{'signal':7} {'rank':8} {'k':>3} {'m':>5} {'dNLL':>8} {'95% CI':>20} {'switch':>7}  candidates")
    for r in rows[:15]:
        print(f"{r['signal']:7} {r['rank']:8} {r['k']:3d} {r['margin']:5.2f} {r['dnll']:+8.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['switch']:7.1%}  {','.join(r['cand'])}")
    res["selfselect"] = rows
    out = a.out or os.path.join(os.path.dirname(a.path), "selfselect.json")
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
