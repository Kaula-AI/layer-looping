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
- `google/gemma-4-E2B` (35 layers; ungated; manual forward `G4`). bf16, eager attention. Gemma-2-2B (`G2`, 26 layers)
  stays supported as the second model (E7) and is where the Gemma Scope SAE comparison runs: there are no
  Gemma Scope SAEs for Gemma-4. `oracle.make_g` picks the class from the model type.
- Data: WikiText-103 prose + GSM8K Q/A text, alternating 512-token chunks with BOS.
- Action space: single-edit **layer programs** per token (`oracle.build_programs`): repeat ℓ×k (k ∈ {2,3}),
  skip ℓ, swap ℓ/ℓ+1 — 139 programs on E2B (103 on Gemma-2-2B) — plus a static block loop (prior-work baseline:
  Inner Loop Inference's Gemma-2-2B block 10–13×2, scaled to 13–16×2 on E2B by `oracle.default_block`; no
  published best block exists for Gemma-4). Signed cost: repeat k−1,
  skip −1, swap 0, block (e−s+1)(R−1) extra layer applications.
  **Part 2 (E9–E12):** full per-token paths over the frozen layer library: repeats, skips, swaps, reversed and
  looped blocks, in any combination (`programs.py`), found by per-token beam search (`search.py`), distilled into
  a shared library (`library.py`), and chosen by a cheap MLP controller (`controller.py`), with whole-sequence
  evaluation (`program_eval.py`). Framing: depth recurrence with a choice of cell.
- Hardware: one 24 GB GPU (RunPod RTX 4090). Do not debug on rented time longer than needed; run the tests,
  `--check` and the quick run first.

## Gemma-4 K/V sharing (E2B)
Layers 15–34 of E2B have no K/V projections: each attends to the K/V of layer 13 (sliding layers) or 14 (full
layers). Rule used by every executor (`G4`, `run_program`, `run_paths`, `routed_eval`, `decode`): earlier tokens
are seen through their base-pass / first-pass K/V of the source layer, and the edited token sees its **latest**
own K/V of the source. If its path has not visited the source yet, that K/V is projected from the token's state
when its path first reaches a higher-index layer (for a plain skip of 13 or 14: from that layer's input), the same
rule `decode.py` uses to fill the cache for unvisited layers. So an edit at 13 or 14 also changes what up to 20
later layers attend to for that token: report edits of layers 13/14 separately in E1. Per-layer embeddings: layer ℓ
always gets the token's own slice ℓ, wherever ℓ appears in a path; a skipped layer adds none.

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
- `oracle.py` — manual forwards `G2` (Gemma-2) and `G4` (Gemma-4, subclass overriding the architecture hooks),
  `make_g`, `run_program`, `independent_program_nll`, `build_programs`,
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
- `programs.py` — edit grammar on layer paths (repeat, skip, swap, reverse, loop), protected ends, length limits.
- `search.py` — E9 per-token beam search; per-token Pareto fronts in `search.npz`.
- `library.py` — E10 shared library (base + single edits + top train composites), scored exactly per token;
  `library.npz` is oracle-format, so `probe.py` / `analyze_tokens.py` run on it.
- `controller.py` — E11 cheap MLP controller (predict dNLL per path, argmin + λ·cost) vs oracles and baselines.
- `program_eval.py` — E12 whole-sequence routing with arbitrary paths (token-by-token, first-pass KV).
- `run_paths` / `layer_rows` (on `G2`, inherited by `G4`) — mixed-path executor: every position follows its own path in one pass.
- `manim_scenes.py`, `docs/explainer.html` — animations. `docs/novelty-review.md` — literature review.
- `tests/` — CPU tests on tiny random Gemma-2s and a tiny random Gemma-4 with K/V sharing and per-layer embeddings
  (`tests/test_gemma4.py`) (`python -m pytest -q tests`, 25 tests): HF parity, swap identity,
  oracle == independent recompute for repeat/skip/swap/block, routed single-token edit == oracle, token-by-token
  decode == routed_forward, prefill+decode, decode features == base features, mixed-path executor == independent
  recompute for random paths, arbitrary-path decoding, search fronts exact and non-dominated, grammar.

## Status
All code paths are written and pass the CPU tests, for Gemma-2 and Gemma-4. Ported to Gemma-4 E2B on 2026-10-31.
Quick run on E2B done 2026-10-31 (`results/gemma-4-e2b-quick/`, summary in `EXPERIMENTS.md`): large oracle
headroom, but no router or heuristic beats no-edit. Rounds 1–3 (fp32, self-selection, paths + MLP
controller, LoRA-trained loops) also negative; see `EXPERIMENTS.md`.

## Tasks, in order
Follow `EXPERIMENTS.md`. Concretely:
1. **Environment.** `pip install -r requirements.txt` (transformers ≥ 5.5 for Gemma-4). Gemma-4 is ungated; the
   Gemma-2-2B runs (E7, SAE) need an HF login with the Gemma-2 license accepted.
2. **Quick run.** `bash run_experiments.sh quick`. Check `check.txt` first: argmax agreement ≈ 1.0, swap
   identity ≈ 0, oracle-vs-independent difference small relative to typical |dNLL|. **Stop and report**
   E1 headroom by op, the λ=0 router table, the budget-matched row at 0, and lag recovery. The human decides
   whether to continue.
3. **Full run.** `bash run_experiments.sh full` (E1–E6).
4. **SAE (E4).** Only on a Gemma-2-2B oracle (`sae_features.py` refuses other models): run the oracle with
   `--model google/gemma-2-2b`, then `bash run_experiments.sh sae`. Report the top latents with their Neuronpedia labels.
5. **Robustness (E7).** Two of: domain transfer, split seeds, second model (Gemma-2-2B via `--model google/gemma-2-2b`).
6. **Paths (E9–E12).** `bash run_experiments.sh programs-quick`, then `programs`, if E1 shows headroom.
   **Stop and report** after programs-quick: search vs single-edit oracle, library oracle, controller budget table.
7. **Downstream (E8).** `bash run_experiments.sh downstream`, only if the kill conditions pass.
   Optional: steer-as-it-goes GRU controller; GRPO on task reward − λ·compute, warm-started from the MLP.

## Kill conditions (stop or pivot to a negative-result paper)
- E1: oracle per-token gain < ~2× the best static program's gain AND few tokens have a safe skip.
- E3: lagged routers recover < ~25% of the same-token router's gain, or learned routers do not beat the
  training-free entropy / norm-peak baselines at matched compute.
- Interp features add nothing over the raw hidden-state probe (then the paper is about lag only).
- E5/E12: the interaction gap eats > half of the predicted gain when routes are combined.
- E11: the controller recovers < ~25% of the library oracle, or no more than single edits.

## Conventions
- Negative dNLL = the edit helps. Positions index the token being predicted from (position t predicts t+1).
- Train/test split by chunk (`chunk % 5 == 0` is test). Never split within a sequence.
- Every number in the paper must come from a saved artifact (`results.json`, logs). Seeds fixed.
- Keep the LLM frozen. Only routers are trained.
- Keep changes to `G2`/`G4` minimal and covered by `--check` and the tests.

## Budget
- Paths: programs-quick (20k tokens) ~1 h; programs (100k) ~4–6 h on one 4090.
- Full oracle on Gemma-2-2B (104 programs × 200k tokens): ~1–2 h on one 4090. E2B has 35 layers and 140 programs,
  so expect roughly 1.5–2× that; re-time from the quick run. Routed eval + downstream: a few hours. Expect < $10–20 total.
