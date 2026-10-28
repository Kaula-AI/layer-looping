"""
E6 Mechanism: which tokens want which edit?

Breaks the oracle down by token type, position, source (prose vs math) and base surprisal.
Token category refers to the token being predicted FROM (position t), and separately to the
token being predicted (t+1), since "this digit is next" may matter more than "this is a digit".

Hypotheses:
  H1 safe skips concentrate on low-surprisal tokens (continuations of words, function words)
  H2 repeats help most when the NEXT token is a digit/operator or the first answer token
  H3 swaps rarely help, and only in middle layers

Writes token_analysis.json and token_analysis.csv.
"""
import argparse
import csv
import json
import os
import re

import numpy as np
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--path", default="oracle_out/oracle.npz")
p.add_argument("--out_dir", default="oracle_out")
p.add_argument("--model", default="google/gemma-2-2b")
p.add_argument("--thr", type=float, default=0.05)
p.add_argument("--skip_eps", type=float, default=0.01)
a = p.parse_args()

d = np.load(a.path)
delta = d["delta"].astype(np.float32)
op = d["cfg_op"].astype(str)
lay = d["cfg_layer"]
nll, toks, chunk, pos = d["base_nll"], d["tok"], d["chunk"], d["pos"]
source = d["source"].astype(str)[chunk]
tok = AutoTokenizer.from_pretrained(a.model)
vocab = {int(t): tok.convert_ids_to_tokens(int(t)) for t in np.unique(toks)}

# next token (t+1) for each position: shift within chunk; the last position of a chunk has none saved
nxt = np.full_like(toks, -1)
same = np.r_[chunk[1:] == chunk[:-1], False]
nxt[:-1][same[:-1]] = toks[1:][same[:-1]]
for t in np.unique(nxt[nxt >= 0]):
    vocab.setdefault(int(t), tok.convert_ids_to_tokens(int(t)))

FUNC = set("the a an of to in and or for on at by with as is was are be it that this from".split())


def category(tid):
    if tid < 0:
        return "n/a"
    s = vocab[int(tid)]
    word_start = s.startswith(("▁", "Ġ"))  # SentencePiece (Gemma) or byte-BPE (GPT-2 style)
    w = s.lstrip("▁Ġ")
    if re.fullmatch(r"\d+", w):
        return "digit"
    if re.fullmatch(r"[+\-*/=×÷<>%$]+", w):
        return "operator"
    if w == "" or re.fullmatch(r"[\sĊ]+", w) or s.startswith("<0x0A>"):
        return "whitespace/newline"  # separate: whitespace targets can masquerade as depth effects (arXiv 2610.19344)
    if re.fullmatch(r"[^\w\s]+", w):
        return "punct"
    if w.lower() in FUNC:
        return "function word"
    return "word start" if word_start else "word continuation"


cat_t = np.array([category(t) for t in toks])
cat_next = np.array([category(t) for t in nxt])

# first answer token in GSM8K text: position right after "Answer" ":" tokens
ans = np.zeros(len(toks), bool)
for k_ in np.where(np.array([vocab[int(t)].lstrip("▁Ġ") == ":" for t in toks]))[0]:
    if k_ > 0 and vocab[int(toks[k_ - 1])].lstrip("▁Ġ") == "Answer" and k_ + 1 < len(toks) and chunk[k_ + 1] == chunk[k_]:
        ans[k_] = True  # position k_ predicts the first answer token

best = delta.min(1)
arg = delta.argmin(1)
gain = np.clip(-best, 0, None)
helped = gain > a.thr
OPS = [o for o in ["repeat", "skip", "swap", "block"] if (op == o).any()]
safe_skip = delta[:, op == "skip"].min(1) < a.skip_eps if "skip" in OPS else np.zeros(len(delta), bool)


def summarize(mask):
    n = int(mask.sum())
    if n == 0:
        return None
    r = {"n": n, "base_nll": float(nll[mask].mean()), "oracle_gain": float(gain[mask].mean()),
         "frac_helped": float(helped[mask].mean()), "frac_safe_skip": float(safe_skip[mask].mean())}
    for o in OPS:
        m = op == o
        r[f"gain_{o}_only"] = float(np.clip(-delta[mask][:, m].min(1), 0, None).mean())
        r[f"share_best_{o}"] = float(((op[arg[mask]] == o) & helped[mask]).mean())
    wl = lay[arg[mask & helped]]
    r["median_winning_layer"] = float(np.median(wl)) if len(wl) else None
    return r


groups = {}
for c in np.unique(cat_t):
    groups[f"token_t={c}"] = cat_t == c
for c in np.unique(cat_next):
    groups[f"next={c}"] = cat_next == c
groups["predicts first answer token"] = ans
for s_ in np.unique(source):
    groups[f"source={s_}"] = source == s_
for lo, hi in [(0, 16), (16, 64), (64, 256), (256, 10_000)]:
    groups[f"position {lo}-{hi}"] = (pos >= lo) & (pos < hi)
qs = np.quantile(nll, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(nll, qs)
for b in range(5):
    groups[f"base NLL quintile {b + 1}"] = bins == b

out = {k: v for k, m in groups.items() if (v := summarize(m)) is not None}

# Mixture-matched source comparison: reweight each source to the same next-token-category mix, so a
# prose-vs-math difference is not just a difference in target types (arXiv 2610.19344).
srcs = list(np.unique(source))
if len(srcs) == 2:
    ref = srcs[0]
    mix = {c: float((cat_next[source == ref] == c).mean()) for c in np.unique(cat_next)}
    for s_ in srcs:
        num = den = 0.0
        for c, w in mix.items():
            m = (source == s_) & (cat_next == c)
            if w > 0 and m.any():
                num += w * gain[m].mean()
                den += w
        out[f"source={s_} (next-type mix matched to {ref})"] = {"n": int((source == s_).sum()),
                                                               "oracle_gain": float(num / den) if den else None}
    no_ws = cat_next != "whitespace/newline"
    for s_ in srcs:
        r = summarize((source == s_) & no_ws)
        if r:
            out[f"source={s_} (excluding whitespace/newline targets)"] = r
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, "token_analysis.json"), "w") as f:
    json.dump(out, f, indent=1)
cols = ["n", "base_nll", "oracle_gain", "frac_helped", "frac_safe_skip"] + \
       [f"gain_{o}_only" for o in OPS] + [f"share_best_{o}" for o in OPS] + ["median_winning_layer"]
with open(os.path.join(a.out_dir, "token_analysis.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group"] + cols)
    for k, v in out.items():
        w.writerow([k] + [v.get(c) for c in cols])
print(f"{'group':52s} {'n':>8s} {'gain':>7s} {'helped':>7s} {'safeSkip':>8s} " + " ".join(f"{'best_' + o:>11s}" for o in OPS))
for k, v in out.items():
    if "frac_helped" not in v:  # mixture-matched rows carry only the gain
        print(f"{k:52s} {v['n']:8d} {v['oracle_gain']:7.4f}")
        continue
    print(f"{k:52s} {v['n']:8d} {v['oracle_gain']:7.4f} {v['frac_helped']:7.1%} {v['frac_safe_skip']:8.1%} "
          + " ".join(f"{v['share_best_' + o]:11.1%}" for o in OPS))
