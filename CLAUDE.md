# Per-token layer looping — project brief for Claude Code

## Goal
A 4-page workshop paper (target: an ICLR 2027 workshop; workshop days Apr 29–30, 2027; paper deadlines
expected late Jan–early Feb 2027) answering three questions about a **frozen** pretrained LLM:

1. **Headroom.** If a single token repeats one layer ℓ k times, how much does its next-token loss drop?
   How often, for which tokens, at which layers?
2. **Lag.** Can that per-token decision be made from token **t−1**'s internals, so the route is known
   before token t's forward pass starts? How much routing value is lost vs a same-token router?
3. **Interpretability.** Do interpretable signals (logit-lens entropy, logit-lens agreement with the
   final layer, residual-update norms, attention entropy, later: SAE features) predict when looping
   helps as well as or better than a raw hidden-state probe? Which signal carries it?

Headline figure: accuracy-vs-compute Pareto frontier (NLL saved vs extra layer applications per token)
for oracle, same-token router, t−1 interp router, t−1 raw router, best static config.

## Positioning (related work)
- CoLa, "Skip a Layer or Loop it?" (arXiv 2507.07996): per-sample MCTS over skip/repeat chains.
- POLAR (arXiv 2606.06574): program-of-layers per input.
- Dr.LLM (arXiv 2510.12773): trained per-layer skip/execute/repeat routers, sequence-level.
- MACRO (arXiv 2608.05872): per-task Markov route distribution.
- Mixture-of-Recursions: token-level recursion depth, but trained from scratch.
- FlexiDepth, LayerSkip, D³, SkipDecode, LoRA-Drop: skipping / early exit, not looping.
- DEGS (arXiv 2607.09693): logit-lens entropy collapse as a signal (for sampling, not routing).
- Think-at-Hard (arXiv 2511.08577, ICML 2026): per-token extra *whole-model* iterations for hard tokens; backbone +
  LoRA trained; decider reads the current token. Closest prior work: cite and contrast.
- Glavas et al., dynamic layer selection (arXiv 2410.20022): per-token skip controllers did no better than a
  token-agnostic constant input. A must-cite warning; our E1 oracle tests directly whether per-token signal exists.
- TIDE (arXiv 2603.21365), Informed Routing (arXiv 2510.13831): per-token exit/approximate in (mostly) frozen
  models, current-token routers, no repetition.
- Lad et al., "Remarkable Robustness of LLMs" (arXiv 2406.19384): whole-model layer deletion/swap robustness
  (model-level, not per token). Relevant to the swap op.
Deep search (Oct 28 2026, ~70 candidates, three angles + citation chase; full report in the AI Research project,
`claude/layer-looping-novelty.md`) found **no kill paper**, but these narrow the claim and must be cited:
- **EGLR** (arXiv 2606.16620): training-free, per-token re-application of the top L layers at high-entropy tokens
  (current token, after its pass), for rollout diversity. ⇒ "per-token looping in a frozen model" alone is NOT new.
- **Tell-Tale Norm / adaptive layer-wise recursion** (arXiv 2606.06188): training-free layer recursion guided by the
  hidden-state ℓ2 norm (current token).
- **Inner Loop Inference** (arXiv 2602.14759) and **Training-Free Looped Transformers** (arXiv 2605.23872): static
  mid-block loops in frozen models; Gemma-2-2B best block = layers 10–13, gains within noise; naive repeats often hurt.
- **Duo-LLM** (arXiv 2410.10846), **Glavas et al.** (2410.20022), POLAR reproduction (arXiv 2609.31360: router's top
  program collapses to the standard pass), per-token fixed-point halting (arXiv 2607.14427: linear probe router saved
  nothing): routers recover little of oracle headroom. E3 must be reported against this prior.
- **Dynamic Routing for Test-time Versatility** (arXiv 2610.04751): frozen skip/repeat programs chosen per task from
  demonstrations.
- **How Do LLMs Use Their Depth?** (arXiv 2510.18871) and **What Layers When** (arXiv 2510.13876): token-type depth
  patterns (function words early, answer starts late); our E6 adds causal evidence, not the patterns.
- **Target-Token Asymmetry** (arXiv 2610.19344): whitespace targets can fake depth effects ⇒ E6 controls.
- **Diminishing Returns of Early Exit** (arXiv 2603.23701): skip headroom may be small in modern models.
- Systems precedent for planning from earlier tokens: spatio-temporal MoE expert prefetch (arXiv 2606.15453).
**Defensible contributions:** (1) exact per-token oracle map over repeat/skip/swap/block in a frozen model with the
whole-sequence interaction gap measured; (2) routing decided from t−1 before the pass (no prior work found);
(3) controlled comparison of interpretable signals vs a raw probe vs SAE latents as router inputs (no SAE-input
router found); (4) signed-cost frontier where skips pay for loops, against static and heuristic baselines.

## Model and setup
- `google/gemma-2-2b` (26 layers; Gemma Scope SAEs exist for every layer). bf16, eager attention.
- Data: WikiText-103 prose + GSM8K Q/A text, alternating 512-token chunks with BOS.
- Action space: single-edit **layer programs** per token (`oracle.build_programs`): repeat ℓ×k (k ∈ {2,3}),
  skip ℓ, swap ℓ/ℓ+1 — 103 programs — plus static block loop 10–13×2 (prior-work baseline). Signed cost: repeat k−1,
  skip −1, swap 0, block (e−s+1)(R−1) extra layer applications.
  Multi-edit programs and an RL router over them are paper 2, not this paper.
- Hardware: one 24 GB GPU (RunPod RTX 4090). Do not debug on rented time longer than needed; run the tests,
  `--check` and the quick run first.

## Key method detail: the diagonal swap
`oracle.py` runs a layer program (`G2.run_program`) for every position at once, but at each step a query
attends to the **base-pass** K/V (of the layer being run) of all other positions and to its **own** modified K/V on the diagonal. Because the model is causal
and only position t is modified, this is **exact** for token t's own next-token loss. What it ignores is
the effect of t's modified K/V on *later* tokens. `routed_eval.py` measures that: it applies a whole
route per sequence with "first-pass KV caching" (each token's cache entry at layer ℓ is written on its
first pass; a skipped token's K/V at ℓ are still projected from its input) and compares the routed loss with the sum of independent oracle deltas. Report the gap.

## Files
- `EXPERIMENTS.md` — **the experiment plan (E0–E8)**: question, hypothesis, command, metric, pass/kill.
- `run_experiments.sh` — modes `quick` (E0–E4, 20k tokens), `full` (E1–E6, 200k), `sae` (E4 SAE), `downstream` (E8).
- `oracle.py` — manual Gemma-2 forward (`G2`, `run_program`, `independent_program_nll`), `build_programs`,
  oracle collection, base-pass features. `--check`: HF parity, swap identity, oracle vs independent recompute.
- `probe.py` — E1 headroom by op, E2 signed-cost λ sweep → Pareto fronts + budget-matched ΔNLL, E3 same vs
  t−1 routers with bootstrap CIs, E4 feature-group ablation; training-free baselines (EGLR-like entropy gate,
  same-token and t−1; ALRR-like norm-peak repeat); `--sae` adds SAE routers + top latents.
- `analyze_tokens.py` — E6 breakdown by token type, next-token type, answer start, source, position, surprisal;
  whitespace/newline targets separated and source comparison mixture-matched (arXiv 2610.19344).
- `routed_eval.py` — E5 whole-sequence routed evaluation (repeat + skip; swaps and blocks are oracle-only).
- `decode.py` — token-by-token decoding with per-token programs and a first-pass KV cache.
- `sae_features.py` — E4 Gemma Scope latents (layers 6/12/18, top-64) for every oracle position.
- `gsm8k_eval.py` — E8 GSM8K greedy generation under base / static / router policies.
- `manim_scenes.py`, `docs/explainer.html` — animations.
- `tests/` — CPU tests on a tiny random Gemma-2 (`python -m pytest -q tests`, 7 tests): HF parity, swap identity,
  oracle == independent recompute for repeat/skip/swap, routed single-token edit == oracle, token-by-token
  decode == routed_forward for mixed routes, prefill+decode == routed_forward, decode features == base features.

## Status
All code paths are written and pass the CPU tests. Nothing has run on Gemma-2-2B yet: there are no results.

## Tasks, in order
Follow `EXPERIMENTS.md`. Concretely:
1. **Environment.** `pip install -r requirements.txt`; HF login with the Gemma license accepted.
2. **Quick run.** `bash run_experiments.sh quick`. Check `check.txt` first: argmax agreement ≈ 1.0, swap
   identity ≈ 0, oracle-vs-independent difference small relative to typical |dNLL|. **Stop and report**
   E1 headroom by op, the λ=0 router table, the budget-matched row at 0, and lag recovery. The human decides
   whether to continue.
3. **Full run.** `bash run_experiments.sh full` (E1–E6).
4. **SAE (E4).** `bash run_experiments.sh sae`. Report the top latents with their Neuronpedia labels.
5. **Robustness (E7).** Two of: domain transfer, split seeds, second model (needs a `G2` variant).
6. **Downstream (E8).** `bash run_experiments.sh downstream`, only if E1–E5 pass the kill conditions.
   Optional: REINFORCE/GRPO router on task reward − λ·compute, warm-started from the ridge router.

## Kill conditions (stop or pivot to a negative-result paper)
- E1: oracle per-token gain < ~2× the best static program's gain AND few tokens have a safe skip.
- E3: lagged routers recover < ~25% of the same-token router's gain, or learned routers do not beat the
  training-free entropy / norm-peak baselines at matched compute.
- Interp features add nothing over the raw hidden-state probe (then the paper is about lag only).
- E5: the interaction gap eats > half of the predicted gain when routes are combined.

## Conventions
- Negative dNLL = the edit helps. Positions index the token being predicted from (position t predicts t+1).
- Train/test split by chunk (`chunk % 5 == 0` is test). Never split within a sequence.
- Every number in the paper must come from a saved artifact (`results.json`, logs). Seeds fixed.
- Keep the LLM frozen. Only routers are trained.
- Keep changes to `G2` minimal and covered by `--check` and the tests.

## Budget
- Full oracle (104 programs × 200k tokens): ~1–2 h on one 4090. Routed eval + downstream: a few hours. Expect < $10–20 total.
