"""E14 pilot (DESIGN.md §5) on Qwen3-1.7B-Base, local (MPS or CPU). Resume-safe: one result file per job.

Jobs, in order:
  text_diag     tuning text: NLL for base, loop at TFLT's window [12-15] (K=2 b=0, K=2 b=0.5, K=3 b=0) and repeat;
                ||delta||/||u||, cos(delta, u), ||loop||/||g0||, mean delta; per-chunk paired deltas (sample size)
  contraction   ||g(x_k) - x_k|| / ||u|| for k < 8, undamped and damped (step 1/2), at [12-15]
  mmlu_base_1   MMLU 5-shot, first `limit` items per subject, base through the wrapper
  mmlu_tflt     TFLT's config: [12-15], K=2, beta=0 (DESIGN.md §3.1 pre-registered reading)
  mmlu_base_2   base again (determinism: item-level discordance should be 0); last, as it is the least needed
Then `summary.json` (written by this script, never by hand).

  python e14_pilot.py --out results/e14-pilot/qwen3-1.7b-base
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from e14 import provenance, run_mmlu, text_chunks, text_nll, write_json
from loop_wrap import Arm, summarize

TFLT = dict(a=12, b=15)  # Qwen3-1.7B-Base, TFLT Table 6: window [12-15], K=2 damped Euler, block mode
EPS = 0.01  # provisional SESOI, nats/token (DESIGN.md §3.6)


def heartbeat(path, state, every=30):
    def beat():
        while not state.get("done"):
            write_json(path, dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: v for k, v in state.items()}))
            time.sleep(every)
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return t


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def job_text_diag(model, tok, args, out):
    ids, names = text_chunks(tok, "tuning", args.text_tokens)
    arms = {"base": Arm(kind="base", **TFLT), "loop_K2_b0": Arm(kind="loop", K=2, beta=0.0, **TFLT),
            "loop_K2_b0.5": Arm(kind="loop", K=2, beta=0.5, **TFLT), "loop_K3_b0": Arm(kind="loop", K=3, beta=0.0, **TFLT),
            "repeat": Arm(kind="repeat", **TFLT)}
    nll, diag, secs = {}, {}, {}
    for name, arm in arms.items():
        rec = {} if arm.kind == "loop" else None
        t0 = time.time()
        nll[name] = text_nll(model, ids, arm, batch=args.batch, record=rec)
        secs[name] = time.time() - t0
        if rec:
            s = summarize(rec)
            s["delta_bar_norm"] = float((rec["delta_sum"] / rec["n"]).norm())
            diag[name] = s
        print(f"  text_diag {name}: nll {nll[name].mean():.4f}  {secs[name]:.0f}s", flush=True)
    base = nll["base"]
    res = dict(n_chunks=len(ids), seq_len=ids.shape[1], n_tokens=int(base.size),
               sources={n: names.count(n) for n in set(names)}, nll={k: float(v.mean()) for k, v in nll.items()},
               seconds=secs, tokens_per_s={k: base.size / v for k, v in secs.items()}, diagnostics=diag, paired={})
    for name in arms:
        if name == "base":
            continue
        d_tok = nll[name] - base  # (chunks, positions)
        d_chunk = d_tok.mean(1)
        sd_c = float(d_chunk.std(ddof=1))
        n_needed = (1.96 * sd_c / (EPS / 2)) ** 2  # chunks for a 95% CI half-width of eps/2
        res["paired"][name] = dict(mean=float(d_tok.mean()), ci95=boot_ci(d_chunk), sd_token=float(d_tok.std(ddof=1)),
                                   sd_chunk=sd_c, chunks_for_halfwidth_eps_over_2=float(n_needed),
                                   tokens_for_halfwidth_eps_over_2=float(n_needed * (ids.shape[1] - 1)),
                                   by_source={src: float(d_chunk[[i for i, n in enumerate(names) if n == src]].mean())
                                              for src in set(names)})
    np.savez_compressed(os.path.join(out, "text_diag_nll.npz"), names=np.array(names), **nll)
    return res


def job_contraction(model, tok, args, out):
    ids, _ = text_chunks(tok, "tuning", args.contraction_tokens)
    rec = {"contraction": 8}
    text_nll(model, ids, Arm(kind="loop", K=2, beta=0.0, **TFLT), batch=args.batch, record=rec)
    s = summarize(rec)
    return dict(n_chunks=len(ids), window=[TFLT["a"], TFLT["b"]], undamped=s["contraction_undamped"],
                damped=s["contraction_damped"],
                note="mean over positions >= 1 of ||g(x_k) - x_k|| / ||g(x_0) - x_0||, k = 0..7; "
                     "undamped x <- g(x), damped x <- x + (g(x) - x)/2")


def job_mmlu(arm):
    def run(model, tok, args, out):
        t0 = time.time()
        agg, rows = run_mmlu(model, tok, arm, limit=args.mmlu_limit, batch=args.batch)
        return dict(arm=arm.name(), extra_layers=arm.extra_layers, limit=args.mmlu_limit, n_items=len(rows),
                    acc=float(np.mean([r["acc"] for r in rows])), lm_eval_aggregate=agg, seconds=time.time() - t0,
                    rows=rows)
    return run


def summarize_pilot(out):
    def load(n):
        p = os.path.join(out, n + ".json")
        return json.load(open(p)) if os.path.exists(p) else None

    s = {}
    td, ct = load("text_diag"), load("contraction")
    if td:
        s["text"] = {k: td[k] for k in ("n_tokens", "nll", "paired", "diagnostics", "tokens_per_s")}
    if ct:
        s["contraction"] = {k: ct[k] for k in ("undamped", "damped")}
    b1, b2, tf = load("mmlu_base_1"), load("mmlu_base_2"), load("mmlu_tflt")
    if b1:
        key = lambda r: (r["subject"], r["doc_id"])  # noqa: E731
        a1 = {key(r): r["acc"] for r in b1["rows"]}
        s["mmlu"] = dict(n_items=b1["n_items"], base_acc_run1=b1["acc"], tflt_reported_base_full=0.6268,
                         tflt_reported_gain_full=0.0034)
        if b2:
            a2 = {key(r): r["acc"] for r in b2["rows"]}
            ll1 = {key(r): r["ll"] for r in b1["rows"]}
            ll2 = {key(r): r["ll"] for r in b2["rows"]}
            s["mmlu"].update(base_acc_run2=b2["acc"],
                             base_run_discordant=int(sum(a1[k] != a2[k] for k in a1)),
                             base_run_max_abs_ll_diff=float(max(abs(x - y) for k in ll1 for x, y in zip(ll1[k], ll2[k]))))
        if tf:
            at = {key(r): r["acc"] for r in tf["rows"]}
            d = np.array([at[k] - a1[k] for k in a1])
            s["mmlu"].update(tflt_acc=tf["acc"], paired_diff=float(d.mean()), paired_diff_ci95=boot_ci(d),
                             discordant_rate=float((d != 0).mean()), loop_better=int((d > 0).sum()),
                             base_better=int((d < 0).sum()),
                             ci_halfwidth_full_mmlu_14042=float(1.96 * d.std(ddof=1) / np.sqrt(14042)),
                             seconds_per_item_base=b1["seconds"] / b1["n_items"],
                             seconds_per_item_tflt=tf["seconds"] / tf["n_items"])
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--out", default="results/e14-pilot/qwen3-1.7b-base")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--text_tokens", type=int, default=40960)
    p.add_argument("--contraction_tokens", type=int, default=8192)
    p.add_argument("--mmlu_limit", type=int, default=10, help="items per subject (57 subjects)")
    p.add_argument("--jobs", nargs="+", default=["text_diag", "contraction", "mmlu_base_1", "mmlu_tflt", "mmlu_base_2"])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    state = {"job": "loading", "pid": os.getpid()}
    heartbeat(os.path.join(args.out, "heartbeat.json"), state)

    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    prov = provenance(args.model, dtype, dict(script="e14_pilot.py", args=vars(args)))

    jobs = {"text_diag": job_text_diag, "contraction": job_contraction,
            "mmlu_base_1": job_mmlu(Arm(kind="base", **TFLT)), "mmlu_base_2": job_mmlu(Arm(kind="base", **TFLT)),
            "mmlu_tflt": job_mmlu(Arm(kind="loop", K=2, beta=0.0, **TFLT))}
    for name in args.jobs:
        path = os.path.join(args.out, name + ".json")
        if os.path.exists(path):
            print(f"{name}: done already, skipping", flush=True)
            continue
        state["job"], t0 = name, time.time()
        print(f"{name}: start", flush=True)
        res = jobs[name](model, tok, args, args.out)
        res["provenance"] = prov
        res["wall_seconds"] = time.time() - t0
        write_json(path, res)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)

    summ = summarize_pilot(args.out)
    summ["provenance"] = prov
    write_json(os.path.join(args.out, "summary.json"), summ)
    state["job"], state["done"] = "finished", True
    write_json(os.path.join(args.out, "heartbeat.json"), dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), **state))
    print(json.dumps({k: v for k, v in summ.items() if k != "provenance"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
