#!/usr/bin/env bash
# Usage:
#   bash run_experiments.sh quick        # E0 + E1-E4 on 20k tokens (~15 min). Read the logs, then decide.
#   bash run_experiments.sh full         # E1-E6 on 200k tokens (~2-3 h)
#   MODEL=google/gemma-2-2b bash run_experiments.sh sae   # E4 SAE latents + routers (Gemma-2-2B only; needs that model's `full`; pip install sae-lens)
#   bash run_experiments.sh downstream   # E8 GSM8K at matched compute (needs `full` first; ~3-4 h)
#   bash run_experiments.sh programs-quick  # E9-E12 per-token path search + controller on 20k tokens (~1 h)
#   bash run_experiments.sh programs        # E9-E12 on 100k tokens (~4-6 h)
# Model: google/gemma-4-E2B by default. Second model (E7, and the only one with Gemma Scope SAEs for `sae`):
#   MODEL=google/gemma-2-2b bash run_experiments.sh full     # outputs get a _gemma-2-2b suffix
set -euo pipefail
MODE=${1:-quick}
MODEL=${MODEL:-google/gemma-4-E2B}
M=(--model "$MODEL")
SFX=""; [ "$MODEL" = "google/gemma-4-E2B" ] || SFX="_$(basename "$MODEL")"
FULL=oracle_out_full$SFX
N_FULL=200000

case "$MODE" in
  quick|full)
    if [ "$MODE" = "quick" ]; then N=20000; else N=$N_FULL; fi
    OUT=oracle_out_${MODE}$SFX
    mkdir -p "$OUT"
    echo "== E0 correctness =="
    python -m pytest -q tests
    python oracle.py "${M[@]}" --check | tee "$OUT/check.txt"
    echo "== E1-E4 oracle ($N tokens) =="
    python oracle.py "${M[@]}" --n_tokens "$N" --out "$OUT" | tee "$OUT/oracle_log.txt"
    python probe.py --path "$OUT/oracle.npz" --out_dir "$OUT" | tee "$OUT/probe_log.txt"
    python analyze_tokens.py "${M[@]}" --path "$OUT/oracle.npz" --out_dir "$OUT" | tee "$OUT/token_log.txt"
    if [ "$MODE" = "full" ]; then
      echo "== E5 interaction gap =="
      for LAM in 0 0.02 0.1; do
        python routed_eval.py "${M[@]}" --policy oracle --lam $LAM --test_only --n_tokens "$N" \
          --oracle "$OUT/oracle.npz" | tee "$OUT/routed_oracle_lam$LAM.txt"
        python routed_eval.py "${M[@]}" --policy router --lam $LAM --test_only --n_tokens "$N" \
          --router "$OUT/router_lag_interp.npz" | tee "$OUT/routed_router_lam$LAM.txt"
      done
    fi
    ;;
  sae)
    echo "== E4 SAE latents =="
    python sae_features.py "${M[@]}" --n_tokens $N_FULL --out $FULL | tee $FULL/sae_log.txt
    python probe.py --path $FULL/oracle.npz --out_dir $FULL --sae $FULL/sae.npz | tee $FULL/probe_sae_log.txt
    ;;
  downstream)
    echo "== E8 GSM8K =="
    python gsm8k_eval.py "${M[@]}" --policy base --out $FULL | tee $FULL/gsm8k_base.txt
    BEST=$(python -c "import json;print(json.load(open('$FULL/results.json'))['headroom']['best_static']['program'])")
    case "$BEST" in repeat@*|skip@*) python gsm8k_eval.py "${M[@]}" --policy static --program "$BEST" --out $FULL | tee $FULL/gsm8k_static.txt;; esac
    for LAM in 0 0.05; do
      python gsm8k_eval.py "${M[@]}" --policy router --lam $LAM --router $FULL/router_lag_interp.npz --out $FULL \
        | tee $FULL/gsm8k_router_lam$LAM.txt
    done
    ;;
  programs|programs-quick)
    if [ "$MODE" = "programs-quick" ]; then N=20000; OUT=programs_quick$SFX; else N=100000; OUT=programs_full$SFX; fi
    mkdir -p "$OUT"
    echo "== E0 correctness =="
    python -m pytest -q tests
    echo "== E9 per-token path search ($N tokens) =="
    python search.py "${M[@]}" --n_tokens "$N" --out "$OUT" | tee "$OUT/search_log.txt"
    echo "== E10 program library =="
    python library.py "${M[@]}" --search "$OUT/search.npz" --n_tokens "$N" --out "$OUT" | tee "$OUT/library_log.txt"
    python probe.py --path "$OUT/library.npz" --out_dir "$OUT" | tee "$OUT/probe_library_log.txt"
    python analyze_tokens.py "${M[@]}" --path "$OUT/library.npz" --out_dir "$OUT" | tee "$OUT/token_library_log.txt"
    echo "== E11 controller =="
    python controller.py --library "$OUT/library.npz" --search "$OUT/search.npz" --out "$OUT" | tee "$OUT/controller_log.txt"
    echo "== E12 whole-sequence routing =="
    for LAM in 0 0.02; do
      python program_eval.py "${M[@]}" --library "$OUT/library.npz" --controller "$OUT/controller.pt" --n_tokens "$N" \
        --lam $LAM --n_chunks 8 --out "$OUT" | tee "$OUT/program_eval_lam$LAM.txt"
    done
    ;;
  *) echo "unknown mode $MODE"; exit 1;;
esac
echo "done ($MODE)"
