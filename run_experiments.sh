#!/usr/bin/env bash
# Usage:
#   bash run_experiments.sh quick        # E0 + E1-E4 on 20k tokens (~15 min). Read the logs, then decide.
#   bash run_experiments.sh full         # E1-E6 on 200k tokens (~2-3 h)
#   bash run_experiments.sh sae          # E4 SAE latents + SAE routers (needs `full` first; pip install sae-lens)
#   bash run_experiments.sh downstream   # E8 GSM8K at matched compute (needs `full` first; ~3-4 h)
set -euo pipefail
MODE=${1:-quick}
FULL=oracle_out_full
N_FULL=200000

case "$MODE" in
  quick|full)
    if [ "$MODE" = "quick" ]; then N=20000; else N=$N_FULL; fi
    OUT=oracle_out_${MODE}
    mkdir -p "$OUT"
    echo "== E0 correctness =="
    python -m pytest -q tests
    python oracle.py --check | tee "$OUT/check.txt"
    echo "== E1-E4 oracle ($N tokens) =="
    python oracle.py --n_tokens "$N" --out "$OUT" | tee "$OUT/oracle_log.txt"
    python probe.py --path "$OUT/oracle.npz" --out_dir "$OUT" | tee "$OUT/probe_log.txt"
    python analyze_tokens.py --path "$OUT/oracle.npz" --out_dir "$OUT" | tee "$OUT/token_log.txt"
    if [ "$MODE" = "full" ]; then
      echo "== E5 interaction gap =="
      for LAM in 0 0.02 0.1; do
        python routed_eval.py --policy oracle --lam $LAM --test_only --n_tokens "$N" \
          --oracle "$OUT/oracle.npz" | tee "$OUT/routed_oracle_lam$LAM.txt"
        python routed_eval.py --policy router --lam $LAM --test_only --n_tokens "$N" \
          --router "$OUT/router_lag_interp.npz" | tee "$OUT/routed_router_lam$LAM.txt"
      done
    fi
    ;;
  sae)
    echo "== E4 SAE latents =="
    python sae_features.py --n_tokens $N_FULL --out $FULL | tee $FULL/sae_log.txt
    python probe.py --path $FULL/oracle.npz --out_dir $FULL --sae $FULL/sae.npz | tee $FULL/probe_sae_log.txt
    ;;
  downstream)
    echo "== E8 GSM8K =="
    python gsm8k_eval.py --policy base --out $FULL | tee $FULL/gsm8k_base.txt
    BEST=$(python -c "import json;print(json.load(open('$FULL/results.json'))['headroom']['best_static']['program'])")
    case "$BEST" in repeat@*|skip@*) python gsm8k_eval.py --policy static --program "$BEST" --out $FULL | tee $FULL/gsm8k_static.txt;; esac
    for LAM in 0 0.05; do
      python gsm8k_eval.py --policy router --lam $LAM --router $FULL/router_lag_interp.npz --out $FULL \
        | tee $FULL/gsm8k_router_lam$LAM.txt
    done
    ;;
  *) echo "unknown mode $MODE"; exit 1;;
esac
echo "done ($MODE)"
