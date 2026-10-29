# Pilot: whole-sequence layer programs (GPT-2 small, Qwen3-1.7B-Base)

**This is not the per-token oracle.** [`whole_sequence_oracle.py`](whole_sequence_oracle.py) applies each layer program to *every* position of a chunk, so a token's ΔNLL includes the effect of edited earlier tokens it attends to. The project's per-token oracle ([`../oracle.py`](../oracle.py), diagonal swap) edits only the token being scored. Read these numbers as a pilot of the pipeline and a preview of the shape of the result, not as E1–E4.

Both runs: `NeelNanda/pile-10k`, 256-token chunks (start token + 255), repeat k ∈ {2, 3}, skip, swap ℓ/ℓ+1, through TransformerLens 3.9 on one RTX 4090. Before computing any delta, each run checks that running the unedited schedule layer by layer reproduces the model's own NLL. Analysis used an earlier `probe.py` (without the block loop and training-free baselines). Every number below is in the folder's `results.json` or `diagnostics.txt`.

| | GPT-2 small | Qwen3-1.7B-Base |
|---|---|---|
| Layers / programs / positions | 12 / 47 / 500,055 | 28 / 111 / 200,175 |
| Base NLL (nats/token) | 3.742 | 2.582 |
| Oracle ΔNLL (hindsight, per token) | −0.930 (24.9%) | −0.916 (35.5%) |
| Best static program | `repeat@1×2`, +0.024 | `swap@14`, +0.038 |
| Safe skip exists | 98.3% of tokens | 99.9% of tokens |
| Same-token interp router, λ = 0 | +0.024 [+0.019, +0.032] | +0.085 [+0.079, +0.092] |
| Lagged interp router, λ = 0 | **−0.0020** [−0.0038, −0.0003] | **+0.047** [+0.042, +0.053] |
| Best router at ≤ 0 extra layers/token | −0.0008 | 0.0000 (no edits) |
| Base NLL, helped vs not-helped tokens | 4.16 vs 0.016 | 3.04 vs 0.017 |
| corr(base NLL, oracle gain) | 0.64 | 0.76 |

What the pilot suggests for the main experiments:
- The hindsight oracle is large on both models but sits almost entirely on high-loss tokens. Confidently correct tokens are never helped.
- No static edit helps on average, and linear ridge routers recover essentially none of the oracle. On Qwen3 every learned router is significantly worse than making no edits.
- The same-token router's AUC for detecting helped tokens is below 0.5 on both models (0.20 and 0.35). A likely reason, not yet tested: ridge predicts each program's *mean* ΔNLL, which points it at confident tokens.
- This is the outcome the brief's E1/E3 kill conditions and the prior work (Glavas et al.; Duo-LLM; the POLAR reproduction) warn about. The per-token oracle should be checked against it before Part 2.

Reproduce (GPU):
```bash
python pilot/whole_sequence_oracle.py --model gpt2 --n_tokens 500000 --out oracle_out_pilot/oracle.npz
python pilot/whole_sequence_oracle.py --model Qwen/Qwen3-1.7B --hf_weights Qwen/Qwen3-1.7B-Base \
  --n_tokens 200000 --batch 4 --out oracle_out_pilot_qwen/oracle.npz
python probe.py --path oracle_out_pilot/oracle.npz --out_dir oracle_out_pilot
```
Needs `transformer_lens>=3.9,<4` (4.0 removed `HookedTransformer`).
