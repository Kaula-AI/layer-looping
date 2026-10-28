# Per-token layer programs: experiment plan

Model: Gemma-2-2B, frozen. Action space per token: one **single-edit layer program**:
repeat layer ℓ k times (k ∈ {2, 3}), skip layer ℓ, or swap layers ℓ and ℓ+1. That is 103 programs, plus the static block loop 10–13×2 from Inner Loop Inference (arXiv 2602.14759) as a prior-work baseline.
Signed cost per token: repeat = k − 1, skip = −1, swap = 0 extra layer applications.

Every experiment names its question, hypothesis, command, metric, and the result that kills or passes it.
Every number in the paper comes from a saved file (`results.json`, `token_analysis.json`, logs).
Report 95% bootstrap CIs over held-out chunks for every headline number.

---

## E0 Correctness (gate, ~$0.50)
**Question.** Does the oracle measure what it claims?
**Run.** `python tests/test_tiny.py` (CPU), then `python oracle.py --check` on the pod.
**Pass.**
- The tiny tests pass for repeat, skip and swap. Each oracle delta equals an independent recompute that truncates the sequence at t and runs the program normally.
- On Gemma-2-2B, argmax agreement with Hugging Face is about 1.0 and the swap identity is about 0.

`--check` also compares oracle deltas with an independent recompute on the real model (5 positions × 3 ops).

## E1 Headroom by op (the go/no-go, ~$1 quick, ~$2 full)
**Question.** How much does the best single edit help each token? Which ops and layers win, and how often is a skip free?
**Hypotheses.**
- H1a: most tokens gain little; a minority (digits, answer starts) gain a lot from middle-layer repeats.
- H1b: a large share of tokens has a safe skip somewhere (ΔNLL < 0.01), concentrated in middle layers.
- H1c: swaps rarely help.

**Run.** `python oracle.py --n_tokens 20000` then `python probe.py` (quick); repeat with `--n_tokens 200000` (full).
**Metrics.** From `results.json → headroom`:
- `oracle_nll_reduction` and `frac_helped`
- `by_op.*.oracle_gain_restricted` and `share_best_*`
- `frac_tokens_with_safe_skip` and `safe_skip_frac_by_layer`

**Kill.** The oracle gain is under about 2× the best static program's gain, *and* the safe-skip fraction is small. Then nothing is worth routing per token. Write it up as a negative result.

## E2 Compute–accuracy frontier (headline figure)
**Question.** At matched compute, how much NLL does per-token routing save over static programs?
**Run.** Same `probe.py` run. The λ sweep over the objective ΔNLL̂ + λ·cost gives the Pareto fronts.
**Metrics.**
- `frontier.png`.
- `budget_matched` at −1, −0.5, 0, +0.25 and +0.5 extra layers per token.
- Report two readings of the same front: same accuracy with fewer layers (budget < 0), and better accuracy at zero extra compute (budget = 0, skips paying for loops).

**Baselines on the same plot.**
- Best static program, including the 10–13×2 block loop.
- An EGLR-like entropy gate (arXiv 2606.16620) on the same token and one token early.
- An ALRR-like norm-peak repeat (arXiv 2606.06188).

**Pass.** At budget 0, the t−1 interp router saves NLL with a CI that excludes 0, beating the best static program and the training-free heuristic baselines.

## E3 Lag (core claim)
**Question.** Can the route be decided from token t−1, before token t's pass starts?
**Run.** Same `probe.py` run. Compare the `same_interp` and `lag_interp` rows, and the `lag_recovery` field.
**Metrics.** ΔNLL at λ = 0 with CIs, AUC for "an edit helps", and budget-matched ΔNLL.
**Kill.** The lag router recovers under about 25% of the same-token router's gain. The paper then shifts to "the decision must be made in-pass."
**Context.** Prior work says routers recover little of the oracle: Glavas et al. 2410.20022, Duo-LLM 2410.10846, and the POLAR reproduction 2609.31360, where the top program collapsed to the standard pass. Report the share of tokens the router edits, so a collapse to "no edit" is visible.

## E4 What signal carries it (interpretability)
**Question.** Do interpretable signals predict the route as well as a raw hidden-state probe? Which ones?
**Run.**
1. Read `probe.py → ablation`: logit-lens entropy, lens agreement, update norm and attention entropy, each alone.
2. SAE features: `bash run_experiments.sh sae`. This encodes the residual stream at layers 6, 12 and 18 with Gemma Scope (`sae_features.py`, top-64 latents per layer). It selects 1,024 latents on train rows by correlation with the next token's gain per op, and adds `lag_sae` and `lag_interp_sae` routers to `probe.py`.
3. For the 10 most predictive SAE features, look up their Neuronpedia descriptions and report them.

**Pass.** Either interp matches raw at far lower dimension, or the top SAE features are interpretable and op-specific. Example: a "number coming next" feature predicts repeats, and a "predictable continuation" feature predicts skips.

## E5 Interaction gap (does it survive whole sequences?)
**Question.** Per-token gains are measured independently. Do they hold when every token is routed at once and later tokens see edited states?
**Run.** For λ in {0, 0.02, 0.1}:
`python routed_eval.py --policy oracle --lam λ --test_only --n_tokens 200000`, then the same with `--policy router`.
Routes use repeat and skip only; swaps are oracle-only.
**Metrics.** Routed ΔNLL vs `sum of independent oracle deltas`, and extra layers per token.
**Kill.** The gap eats more than half the predicted gain.
`tests/test_routed_eval.py` checks `routed_forward` against token-by-token decoding with a first-pass KV cache (`decode.py`).

## E6 Mechanism by token type
**Question.** Which tokens want skips, which want loops, and does that match the hypotheses?
**Run.** `python analyze_tokens.py`. It writes `token_analysis.{json,csv}`, broken down by:
- the category of the current token and of the next token (digit, operator, punctuation, function word, word start, word continuation);
- the first answer token in GSM8K;
- source (prose or math);
- position bucket;
- base-NLL quintile.

**Hypotheses.**
- H2: safe skips sit in the lowest base-NLL quintiles and on word continuations.
- H3: repeats help most when the next token is a digit or operator, or is the first answer token.

**Controls.** Whitespace targets can fake depth effects (arXiv 2610.19344). `analyze_tokens.py` therefore separates whitespace/newline targets and reports the prose vs math comparison two more ways: mixture-matched by next-token type, and excluding whitespace targets.
**Framing.** Token-type depth patterns are partly known from logit-lens and gate analyses (arXiv 2510.18871, 2510.13876). The contribution here is causal evidence from interventions.
**Output.** One table and one small-multiples figure for the paper.

## E7 Robustness (pick 2)
- **Domains.** Prose-only vs math-only splits. Does a router trained on one transfer to the other?
- **Split seeds.** Re-run `probe.py` with the test fold set to `chunk % 5 == {1,2,3,4}`.
- **Second model.** Llama-3.2-1B or Qwen2.5-1.5B. This needs a `G2` variant for that architecture. It's worth doing if a reviewer would ask "is this Gemma-specific?"

## E8 Downstream (only if E1–E5 pass)
**Question.** Does routed decoding improve task accuracy at matched compute?
**Run.** `bash run_experiments.sh downstream` (`gsm8k_eval.py`): 250 test problems, 4-shot, greedy. Compares base, best static, and the t−1 router at λ ∈ {0, 0.05}. During generation the router reads the previous token's actual (routed) pass, a mild shift from its base-pass training features; report it.
**Optional.** Fine-tune the router with REINFORCE or GRPO. Use reward = correct − λ·compute, with λ as a Lagrange multiplier for a target average compute. Warm-start from the ridge router.

## Part 2: per-token layer paths and a cheap controller (E9–E12)
Each token gets its own path through the frozen layers: repeats, skips, swaps, reversed blocks, looped blocks, in any combination. A small controller picks the path. The single-edit map (E1–E6) is the go/no-go for this part: if no single edit helps, longer paths rarely will.

Run everything with `bash run_experiments.sh programs-quick` (20k tokens, about 1 h) or `programs` (100k tokens, about 4–6 h).

**Grammar (`programs.py`).**
| Edit | Cost |
|---|---|
| repeat ℓ×k | +k−1 |
| skip | −1 |
| swap adjacent | 0 |
| reverse a block of 3–4 | 0 |
| loop a block of 2–4 | +w |

The first and last two layers are never edited, and path length stays within L−4 … L+6.

### E9 Per-token path search (`search.py`)
**Question.** How much more do multi-edit paths gain than the best single edit, per token?
**Method.**
- Round 1 scores every single edit of the base path.
- Rounds 2–3 grow each token's own beam (width 3) with edits anchored on its 6 most promising layers.
- Every candidate is scored exactly with the mixed-path executor (`G2.run_paths`).
- For each token, the Pareto front over (cost, dNLL) is stored.

**Metrics.**
- Mean best dNLL from search vs the single-edit oracle (`controller_results.json → oracle_search` vs `oracle_single_edits`).
- Which edit kinds appear in winning paths.
- How often reversal or reordering helps.

**Hypotheses.**
- H4: multi-edit paths add meaningful gain over single edits.
- H5: reversed and reordered blocks almost never win outside the middle layers.

### E10 Program library (`library.py`)
**Method.** The library holds the base path, every single edit, and the 128 multi-edit paths that win most often on **train** chunks. Each library path is scored exactly for every token. The output, `library.npz`, has the same format as `oracle.npz`, so `probe.py` and `analyze_tokens.py` run on it unchanged.
**Metrics.** Library oracle vs search oracle, which shows how much a fixed library of a few hundred paths loses against per-token search.

### E11 Cheap controller (`controller.py`)
**Method.** A 2-layer MLP (256 hidden units) predicts every library path's dNLL from cheap signals. The route is the argmin of predicted dNLL + λ·cost, or no edit when nothing is below 0.
- **Training loss:** MSE plus a decision-focused term (expected objective under a softmax over predictions). Every path is scored for every token, so this needs no sampling.
- **Variants:**
  - `mlp_lag`: from token t−1, decided before t's pass.
  - `mlp_same`: from token t.

**Baselines.**
- Ridge over the library on interpretable and raw features.
- Ridge restricted to single edits.
- Best static path.
- Library oracle, single-edit oracle and search oracle.

**Pass.** At budget 0, `mlp_lag` beats the single-edit router and the best static path, with CIs excluding 0.
**Kill.** It recovers under about 25% of the library oracle, or no more than single edits do.

### E12 Whole-sequence routing (`program_eval.py`)
**Question.** Do the gains survive when every token follows its chosen path at once?
**Method.** Token-by-token decoding with a first-pass KV cache (`decode.step_path`). A layer the path never visits gets K/V projected from the state where the path first passes it, which reduces to the skip rule.
**Metrics.** Routed dNLL vs the sum of independent deltas, for oracle and controller choices, at λ ∈ {0, 0.02}.
**Kill.** The gap eats more than half of the predicted gain.

### Follow-ups, if E11 works
- **Steer as it goes:** a GRU controller reads the token's state after each layer and picks the next layer or stops. This is depth recurrence with a choice of cell.
- **Task reward:** GSM8K accuracy minus λ·compute with GRPO, warm-started from the MLP.

---

## Order and budget
| Step | Command(s) | GPU time | Cost |
|---|---|---|---|
| E0 | tests, `--check` | 5 min | <$0.50 |
| E1–E4 quick | oracle 20k + probe | ~10 min | ~$0.25 |
| **Decision** | read headroom + lag rows | — | — |
| E1–E4 full | oracle 200k + probe + analyze_tokens | ~1–2 h | ~$1 |
| E5 | routed_eval × 6 | ~1 h | ~$0.50 |
| E4 SAE | sae_lens encode + probe | ~1 h | ~$0.50 |
| E8 | GSM8K generation × 4 configs | ~3 h | ~$2 |

## Paper mapping
If Part 2 works, it becomes the headline: Fig 2 is the controller frontier against the oracles and baselines (E11), and the single-edit map becomes the motivating analysis.

- Fig 1: method (diagonal swap, t−1 routing). Use the animation stills.
- Fig 2: frontier (E2) with lag and interp variants (E3).
- Table 1: headroom by op (E1) and the interaction gap (E5).
- Fig 3: token-type breakdown (E6).
- Table 2: feature-group and SAE ablation (E4).
