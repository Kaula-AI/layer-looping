# Layer looping: living design

Dated decisions for experiments after E13b. Earlier designs live in `EXPERIMENTS.md` (E0–E13b). Frozen run plans go
in `SPEC.md`; this file is where they are argued.

---

## E14: training-free damped looping under fair controls

**Status (2026-11-22):** designed, red-teamed (§4) and piloted locally on Qwen3-1.7B-Base (§5.1). Next: Tyler's
call on the GPU stage (§5.2), then `SPEC.md` with its hash published (step 6). No GPU time has been spent.

### 1. Decision

On 2026-11-18 Tyler proposed that looping happen **at inference only**, following Training-Free Looped Transformers
(TFLT, arXiv 2605.23872). E14 decides between two paths:

- **Looping passes the fair controls:** inference-only looping has support. Build the per-token stopping rule on it
  (Stage 2, §3.9) and caveat takeaway 4 in the note and paper ("new kinds of looping have to be trained into the
  model").
- **It doesn't:** inference-only looping in frozen models is treated as closed, the null becomes Finding 8 (the
  selection result extends from per-token edits to uniform loops), and the next direction is recurrence trained
  into the model (e.g. retrofitted recurrence, arXiv 2511.07384). That path needs a separate budget decision.

The decision is taken on two **decision models**: Qwen3-4B-Base (TFLT's best base-model case) and Qwen3.5-4B-Base
(the newest small Qwen base model, February 2026). Qwen3-1.7B-Base is kept for the reproduction only, and Gemma-4
E2B is a generalization readout (§3.2).

### 2. Claim and rivals

**Claim.** Running a contiguous mid-stack window of a frozen LLM as K damped sub-steps, uniformly for every token,
lowers its held-out loss by more than any zero-compute change to the same window, and by more than a size-matched
change without the loop's content, with every arm tuned on separate data under the same budget.

**What the loop actually changes.** To first order, damped Euler reproduces the normal pass. For K = 2, β = 0, with
`u = g(x₀) − x₀` and J the Jacobian of the window's residual update at x₀, the net change from the normal pass is
`δ = x_out − g(x₀) ≈ ¼·J·u`: a curvature term, mostly not along u. The rivals are built around that.

| Rival explanation | How E14 separates it |
|---|---|
| **A constant steering vector.** The loop's gain comes from the average shift δ̄, not from token-specific computation. | Arm `mean-shift`: `g(x₀) + s·δ̄`, δ̄ the mean δ over tuning tokens for that loop config, s tuned. Zero extra compute. |
| **A norm change.** The loop mainly rescales the residual stream (Gemma-4 also multiplies the whole residual by a per-layer `layer_scalar`). | Arm `norm-const`: `c·g(x₀)`, c tuned (zero compute). Diagnostic `norm-match`: `g(x₀)` rescaled to the loop's per-token norm. |
| **Update rescaling.** The loop only changes the step size along the window's own update. | Arm `rescale`: `x₀ + α·u`, α log-spaced around 1 and bracketing the pilot's measured ‖δ‖/‖u‖. Zero extra compute. |
| **Size, not content.** Any change of that size helps (E13 found this for per-token edits). | Arm `sham-perm`: the loop's δ taken from the same position of a different sequence, scale tuned (§3.3). |
| **Calibration.** Gains come from flatter or sharper output distributions. | Temperature T tuned per arm on tuning NLL. It cannot change single-letter MC accuracy (argmax over A–D logits is invariant to T). |
| **Answer-letter prior.** On MMLU the loop shifts the A/B/C/D prior ("format repair", as the POLAR reproduction arXiv 2609.31360 found for routers). | P2 compares loop + letter bias with base + letter bias, offsets fitted on MMLU validation (same 5-shot format). |
| **Selection.** Best of many configurations looks good on the data it was chosen on. TFLT searched ~720 (cell, config) pairs and re-scored on MMLU, which it also reports; two of its screen winners (+0.9 to +1.3) lost 1.2–1.4 on MMLU. | Configs are chosen on tuning data with a continuous metric. Scored data are run only for frozen configs. Every arm gets the same grid size. The screen-to-holdout shrinkage of each arm's top 3 is reported. |
| **Location.** Any window works as well. | Five other window centres, with K, β and width held at the loop's selected values. |
| **Harness mismatch.** Our numbers differ from TFLT's because of prompts, scoring or library versions. | H0 parity check on base accuracy before anything else (§3.7). |
| **Precision and batching noise.** bf16 rounding or batch-dependent kernels produce differences as large as δ. | Euler steps and the blend in fp32. Fixed batch size for every arm. The base model runs through the wrapper (K = 1) and twice; item-level discordance between identical runs must be ~0. |

**Our prior, written before running:** we expect any NLL gain to be small and largely matched by `mean-shift` or
`norm-const`, and the MMLU effect on base models to be at most a few tenths of a point. TFLT's own base-model gains
are +0.34 points on MMLU for both Qwen3-1.7B-Base and 4B-Base.

### 3. Design

#### 3.1 Method under test (from TFLT, read 2026-11-22)

- Window `g = L_b ∘ … ∘ L_a` in **block mode** (the whole window is one map).
- Damped Euler, K sub-steps: `x_{k+1} = x_k + (1/K)(g(x_k) − x_k)`; `F^K(x₀)` after K steps.
- Anchor blend: output `β·g(x₀) + (1 − β)·F^K(x₀)`. β = 0 is pure damped Euler, β = 1 the normal pass. TFLT does
  not state β for its headline cells; its only stated value is β = 0.5 (Appendix E).
- Extra compute: `(b − a + 1)(K − 1)` layer applications per token (`g(x₀)` is the first sub-step, so β costs nothing).
- **Pre-registered reading of TFLT's config** (the paper is ambiguous): 0-based HuggingFace layer indices, block mode,
  β = 0, K = the number of evaluations of g.
- **Scoring semantics.** MC loglikelihood and NLL are full-sequence passes. Every position is looped in the same pass;
  in each sub-step, a window layer's queries attend to the keys and values of all positions' *current* sub-step
  states. This matches TFLT's prefill, where the loop body runs with `use_cache=False` (Appendix C). TFLT's cache
  variants (first/last/none) only affect later decoded tokens, so they do not apply. Generation tasks are out of
  scope (§6).

#### 3.2 Models

| Model | Released | Layers | Role |
|---|---|---|---|
| Qwen3-1.7B-Base | Apr 2025 | 28 | Reproduction (R0) and the local pilot only. TFLT config [12–15], K = 2: MMLU 5-shot 62.68 → 63.01. |
| Qwen3-4B-Base | Apr 2025 | 36 | Decision model and R0. TFLT config [15–18], K = 3: MMLU 5-shot 73.27 → 73.61 (its Table 13 gives a 73.01 base for the same window; the paper is inconsistent). |
| Qwen3.5-4B-Base | Feb 2026 | 32 | Decision model, the current generation. Not in TFLT, so no R0. |
| Gemma-4 E2B | Apr 2026 | 35 | Generalization readout, this project's model. |

Ouro-1.4B is dropped: it is already trained to loop, so damped sub-steps inside its rounds are a different question.
Newer Qwen releases (3.6 in April, 3.8 in August 2026) are 27B or larger or instruct-only, so they do not fit one
24 GB GPU as base models. Gemma-4 E4B (42 layers, 18 K/V-shared) is a drop-in alternative to E2B if a larger
generalization model is wanted.

**Qwen3.5 is a hybrid.** Three of every four layers are Gated DeltaNet (linear attention; pattern LLLF), so a window
mixes linear-attention layers with at most a few full-attention layers. In each sub-step a linear-attention layer
rebuilds its recurrent state from all positions' current sub-step states. That is the same rule as attention seeing
current-sub-step K/V, and `tests/test_loop_wrap.py` checks it against an independent reference. TFLT never tested a
hybrid. Width-4 windows always contain exactly one full-attention layer; width-3 windows may contain none. Every
selected window's composition is reported. The checkpoint is multimodal (`Qwen3_5ForConditionalGeneration`) and is
loaded text-only; stage A checks that no language-model weights are missing.

**E2B tests a different operator.** Layers 15–34 read the K/V of layers 13/14, so a window inside 15–34 iterates
each token's state against fixed memory: only states and queries change. The primary E2B grid excludes windows
containing 13 or 14, which removes the 0.40·N centre and some widths; the remaining grid is listed in `SPEC.md`.
Windows containing 13 or 14 form an exploratory arm, in which the K/V that later shared layers read are those
computed in the **last** sub-step (the blend does not recompute them).

**Qwen3 has no BOS token.** "Position 0" below means the first position of each sequence, whatever its token.

#### 3.3 Arms

Window family for every windowed arm: centre ∈ {0.40, 0.45, …, 0.70}·N (7), width ∈ {3, 4, 6} (3), so 21 windows.
δ is the loop's net change from the normal pass at its config. Every tuned arm gets at most 84 configs.

| Arm | Output at the window | Grid | Extra layers/token |
|---|---|---|---|
| `base` | `g(x₀)` | — | 0 |
| `loop` | `β·g(x₀) + (1 − β)·F^K(x₀)` | 21 windows × K ∈ {2, 3} × β ∈ {0, 0.5} = 84 | width·(K − 1) |
| `rescale` | `x₀ + α·u` | 21 windows × 4 α, log-spaced around 1, bracketing the pilot's ‖δ‖/‖u‖ | 0 |
| `mean-shift` | `g(x₀) + s·δ̄` | the loop's top 21 configs × s ∈ {0.5, 1, 2, 4} = 84 | 0 |
| `norm-const` | `c·g(x₀)` | 21 windows × 4 c, bracketing the pilot's ‖x_loop‖/‖g(x₀)‖ | 0 |
| `sham-perm` | `g(x₀) + σ·‖δ_t‖·δ_donor/‖δ_donor‖`: the recipient's own size, the donor's direction; donor = same position in a different sequence | σ ∈ {0.25, 0.5, 1, 2} at the loop's selected config | as `loop` |
| `repeat` | `g(g(x₀))` (full strength; a known mover, should hurt) | 21 windows | width |
| *diagnostic* `norm-match` | `g(x₀)·‖x_loop‖/‖g(x₀)‖` per token | the loop's selected config | as `loop` |
| *diagnostic* `sham-iso` | `g(x₀) + ‖δ_t‖·u_t`, u_t a random unit vector | the loop's selected config | as `loop` |

- **Donors for `sham-perm`** come from the same position index of a different sequence (a different text chunk, or a
  different MC item), never from the same sequence. A donor from a later position of the same sequence would leak
  the target token into the recipient.
- **Position 0 (attention sink)** gets the loop's own δ in every sham, so the sink is treated identically across loop
  and shams. A descriptive ablation runs `loop` and `sham-perm` with position 0 left unlooped in both.
- `sham-iso` is diagnostic only: zero-mean noise raises expected loss, so it cannot fail the loop.
- δ̄ for `mean-shift` is computed on tuning tokens, excluding position 0, and is a by-product of the loop's tuning pass.
- Grids are extended outward until each arm's best setting is interior. Extensions are recorded, and the number of
  configs each arm used is reported.
- Temperature T (0.8–1.25, step 0.05) is tuned per arm on tuning NLL. For the multi-token MC secondaries, T is tuned
  per arm on tuning-MC continuation log-likelihood.

#### 3.4 Data: tuning vs scored

| Use | Tuning (selection) | Scored (outcomes) |
|---|---|---|
| P1 text NLL | WikiText-103 validation + GSM8K train text, ~50k tokens | WikiText-103 test + GSM8K test text, size from the pilot |
| P2 MMLU | MMLU validation (1,531 items, 5-shot; dev is the few-shot pool) | MMLU test, all 14,042, 5-shot |
| Secondaries | ARC-Easy, OpenBookQA, SciQ (temperature for multi-token MC only) | ARC-Challenge 25-shot, HellaSwag 10-shot (fixed random 3,000) |

Text follows the project conventions: 512-token chunks with BOS where the tokenizer has one, split by chunk, never
within a sequence. MC tasks run through lm-evaluation-harness 0.4.x (TFLT used 0.4.11) with a model wrapper that
applies the arm, so prompts and scoring match TFLT's.

#### 3.5 Selection, and which configs reach the scored data

- **P1 config** per arm: lowest tuning NLL.
- **P2 config** per arm: among that arm's top 10 by tuning NLL, the highest mean gold-answer log-prob margin on MMLU
  validation (continuous; mean accuracy over ~1.5k items is too noisy to rank 84 configs).
- **Secondaries** use each arm's P1 config.
- **Scored runs:** each arm's P1 and P2 configs plus the top 3 by tuning rank (for shrinkage); TFLT's published config
  (Qwen models, the R0 cell); five location-control windows (centres drawn at random from those not selected, K, β
  and width fixed at the loop's selection); `repeat` at its best tuning window; the diagnostics at the loop's selected
  config.

#### 3.6 Outcomes and statistics

- **P1 (primary):** scored-text NLL, nats/token.
- **P2 (gated):** MMLU 5-shot accuracy, loop + letter bias vs base + letter bias. Tested only on models where P1 passes.
- **Secondary:** ARC-Challenge and HellaSwag accuracy; extra layer applications and wall-clock per token.
- **Statistics:** paired by token or item. NLL: chunk bootstrap. MC: item-level paired bootstrap and McNemar, plus a
  subject-cluster bootstrap on MMLU as a sensitivity check. Effect sizes with CIs throughout.
- **Smallest effect of interest (SESOI), provisional:** ε = 0.01 nats/token on P1. Reasoning: the loop spends 3–12
  extra layer applications per token, and E13b's best static single edit on E2B already gave −0.013 nats at ≤ 1 extra
  layer. Frozen in `SPEC.md` after the pilot, with any change argued there.
- **Sample size:** chosen from the pilot's paired per-token NLL variance so that the P1 CI half-width is at most ε/2,
  which makes an inconclusive result unlikely. MMLU's size is fixed at 14,042; with ~4% discordant items its paired CI
  is about ±0.33 points, which is why P2 is gated and is not the primary outcome.

#### 3.7 Decision rule (proposed; frozen in `SPEC.md` after the pilot)

**Comparators on P1, per model:** `base`, `rescale`, `mean-shift`, `norm-const` (each tuned, each with its own T) and
`sham-perm`. The loop must beat **every** comparator, which is an intersection-union test and needs no correction
across comparators.

- **H0, harness parity (gate).** Base MMLU 5-shot on each Qwen3 model within 0.5 points of TFLT's base. If not, the
  harness is fixed before anything else is run or scored. Qwen3.5-4B-Base has no TFLT number. Its gate is a clean
  text-only load (no missing language-model weights) and two base runs that agree item for item.
- **Known movers (gate).** `repeat` must raise tuning NLL, and the letter bias must change MMLU validation accuracy.
  If neither moves, the pipeline cannot detect a change and nothing is scored.
- **R0, reproduction (descriptive).** TFLT's config on MMLU test, reported as an estimate with a 95% CI and read
  three ways: an effect exists (lower CI > 0); TFLT's size is not reproduced (upper CI < +0.34); or neither.
  TFLT's gain was selected among its top 3 on MMLU, so some shrinkage is expected. If R0 is not reproduced while H0
  passes, the write-up counts that toward the selection rival. R0 does not gate the rest.
- **PASS.** On at least one decision model (97.5% CIs: Bonferroni over the two), the loop's P1 gain over each
  comparator has its lower CI above 0 and its point estimate at or above ε; **and** on that model P2 is non-inferior
  (lower 95% CI of loop − base, both with letter bias, above −0.5 points; §5.2). → Stage 2.
- **FAIL.** On both decision models, the upper 95% CI of the loop's P1 gain over the best comparator is below ε. →
  Finding 8 and the trained-recurrence direction.
- **INCONCLUSIVE.** Anything else, reported as such. The sample-size rule is meant to make this rare.
- **E2B** gets the same P1 test at 95%, reported as generalization; it does not change the decision.

#### 3.8 Implementation requirements (from the red-team)

- **A wrapper around the HuggingFace decoder layers**, not `G4`. `G4` has no sliding-window mask (it is guarded only
  in `oracle.py`'s main), and MMLU 5-shot, ARC-C 25-shot and HellaSwag 10-shot prompts often exceed Gemma-4's
  512-token window.
- **No cache inside the loop body.** Pass `past_key_values=None` to every sub-step. Assert that key length equals
  query length. lm-eval's forward may default to `use_cache=True`; a second call to a layer would then append K/V,
  and SDPA would silently attend to first-pass K/V instead of the current state.
- **fp32** for the Euler steps and the blend; layers run in bf16 as usual.
- **Fixed batch size**, identical for every arm (no `auto`).

Tests (CPU; tiny random Qwen3, a tiny Qwen3.5 hybrid, and the tiny Gemma-4 with K/V sharing):
- K = 1, β = 1, α = 1, s = 0, c = 1 and σ = 0 are exact no-ops through the wrapper.
- The wrapped model matches the unwrapped one at T ≈ 2,000, past the sliding window.
- A test that distinguishes attention to current sub-step states from attention to first-pass K/V.
- Donor indexing never draws from the recipient's own sequence.

#### 3.9 Stage 2 (only on PASS): per-token stopping rule

A zero-parameter rule from the ODE view: keep sub-stepping a token while the KL between successive sub-steps'
logit-lens distributions is above a threshold (Huginn, arXiv 2502.05171; the fixed-point halting paper arXiv
2607.14427 found this beat a linear-probe router on a looped model). Threshold tuned on tuning data. Compared at
matched mean compute against uniform K and random per-token K. Its basis is checked in the pilot: if
‖g(x_k) − x_k‖ does not shrink with k on frozen models, there is nothing to converge to and Stage 2 is dropped.

### 4. Red-team (2026-11-22)

A fresh agent read only this file (first draft) and the code (`oracle.py`, `bench.py`, `tests/test_gemma4.py`,
HF `modeling_qwen3.py`, `modeling_gemma4.py`, `sdpa_attention.py`, `cache_utils.py`). Eight findings, most serious
first. All were accepted; the responses are folded into §2–§3.

| # | Finding | Response |
|---|---|---|
| 1 | The letter-bias rival could not be fitted: ARC-E, OBQA and SciQ score answer text and WinoGrande has two options, so none has A–D logits. A loop that shifts MMLU's letter prior would read as a gain. | Offsets fitted on MMLU validation (same format); P2 compares loop + bias with base + bias; the P2 config is chosen on MMLU validation. |
| 2 | `rescale` is the wrong twin: damped Euler's net change is a curvature term, mostly not along u, and the α grid may not bracket ‖δ‖. A constant steering vector or a norm change could explain the gain. | Added `mean-shift` and `norm-const` as primary comparators (plus diagnostic `norm-match`); α and c are bracketed from pilot measurements. |
| 3 | Shams: `sham-iso` cannot fail; `sham-perm` from the same sequence leaks future tokens; the loop touched position 0 and the shams did not; shams were untuned. | Donors from a different sequence; shams get the loop's δ at position 0, plus an ablation; σ tuned; `sham-iso` demoted to diagnostic. |
| 4 | Implementation traps: K/V appended to the cache on repeated layer calls, with SDPA then silently attending to first-pass K/V; `G4` has no sliding window; undefined shared K/V when 13/14 are looped; bf16 rounding; `auto` batch sizes. | §3.8 requirements and tests; HF wrapper instead of `G4`; shared K/V defined (last sub-step); fp32 steps; fixed batch. Verified 2026-11-22: `G4` guards the window only in `oracle.py` main; Gemma-4 has a `layer_scalar` buffer on the residual. |
| 5 | The decision rule could not reach FAIL, had the wrong multiplicity correction, gave P2 no power, and had no minimum effect. | P1 primary with a SESOI; intersection-union over comparators; Bonferroni over the two Qwen models; P2 gated with a non-inferiority margin; FAIL defined by the upper CI below ε; sample size set so the CI half-width ≤ ε/2. |
| 6 | R0 could not cleanly pass or fail; TFLT's config is underspecified; no consequence defined. | Split into H0 parity (gate) and descriptive R0; pre-registered reading of TFLT's config; known-mover gates; R0's consequence stated. |
| 7 | Selecting among 84 configs on mean accuracy over ~5k items is close to random; the "median config on the scored set" contradicted §3.4; random configs included harmful K/β settings. | Selection on continuous metrics (NLL, log-prob margin); location control defined by tuning rank with K, β and width fixed. |
| 8 | E2B tests a different operator (fixed memory, `layer_scalar`), loses grid cells, and the two Qwen models could carry "2 of 3" alone. | The decision rests on the Qwen models; E2B is a generalization readout; `norm-const` covers the scalar. |

Minor points accepted: temperature can change multi-token MC rankings, so it is tuned separately there; Qwen3 has no
BOS token (verified), so position 0 means the first position.

### 5. Pilot (Mac, Qwen3-1.7B-Base, free)

1. The §3.8 tests pass on CPU.
2. H0 on a fixed 2,000-item MMLU subset (5-shot) against TFLT's 62.68, run twice through the wrapper with K = 1
   (determinism and parity).
3. TFLT's config on the same subset: discordance rate and paired effect.
4. On 20k tuning tokens at TFLT's window: ‖δ‖/‖u‖ and ‖x_loop‖/‖g(x₀)‖ (to bracket α and c), and the paired per-token
   NLL variance (for the sample size).
5. Contraction diagnostic for Stage 2: ‖g(x_k) − x_k‖ for k = 1…8, with and without damping.
6. Time per config on the tuning set, to re-estimate GPU cost.

Rough GPU estimate before the pilot (one RTX 4090): about 370 tuning configs per model on ~50k tokens plus top
configs on MMLU validation, then ~16 scored configs on MMLU test and 4 on each secondary. Roughly 8 h for 4B, 4 h for
1.7B and 5 h for E2B: 15–25 GPU-hours (~$5–17). Re-estimated from the pilot and approved by Tyler before any rental.

#### 5.1 Pilot results (2026-11-22, M1 Mac, MPS, bf16, commit 242f52e)

Artifacts: `results/e14-pilot/qwen3-1.7b-base/` (`summary.json` written by `e14_pilot.py`; per-job files; the per-token
NLL `.npz` is not in git). Window [12–15] throughout (TFLT's for this model). This is a pilot, not a test of the claim.

- **Tests.** `tests/test_loop_wrap.py`: 40 pass (all 76 in `tests/` pass). Planting the cache bug the red-team
  described (sub-steps keep the model's cache) fails the reference and cache tests on Qwen3 and on Gemma-4 layers
  that own K/V. Dropping the recorded masks fails the reference test.
- **Tuning-text NLL** (40,880 tokens, 40 WikiText + 40 GSM8K chunks; base 1.411 nats/token). Paired change vs base,
  95% chunk-bootstrap CI:

  | Arm | All | WikiText | GSM8K |
  |---|---|---|---|
  | loop K=2, β=0 (TFLT's config) | **+0.056** [+0.044, +0.069] | +0.007 [+0.003, +0.010] | +0.105 [+0.094, +0.117] |
  | loop K=2, β=0.5 | +0.012 [+0.009, +0.015] | +0.001 [−0.001, +0.002] | +0.023 [+0.020, +0.027] |
  | loop K=3, β=0 | +0.084 [+0.066, +0.102] | | |
  | repeat (full strength) | +0.211 [+0.176, +0.246] | +0.070 | +0.352 |

  At TFLT's own configuration the loop **raises** loss, almost all of it on GSM8K text. Weaker loops (β = 0.5) hurt
  less. The known mover works: full-strength repeat hurts most.
- **What the loop changes** (K=2, β=0): ‖δ‖/‖u‖ = 0.26 and cos(δ, u) = −0.46, so the loop's change partly *opposes*
  the window's own update (its projection on u is about −0.12·u). The residual norm shrinks: ‖loop‖/‖g(x₀)‖ = 0.966.
- **Contraction** (8,192 tokens), ‖g(x_k) − x_k‖ / ‖u‖ for k = 0…7: undamped 1.00, 0.91, 0.91, 0.97, 1.06, 1.15,
  1.26, 1.37; damped (step ½) 1.00, 0.89, 0.87, 0.88, 0.92, 0.96, 1.01, 1.06. The window has no fixed point within
  reach: the residual shrinks by at most 13%, then grows.
- **MMLU 5-shot**, first 10 items per subject (570 test items):
  - Base 67.5%. TFLT reports 62.68% on the full set; this subset weights subjects equally, so H0 cannot be judged
    from it.
  - TFLT's config 66.3%, a paired change of −1.2 points [−3.7, +1.1]; 8.2% of items discordant (20 better, 27 worse).
  - Exploratory (not in the pilot plan): the gold answer's log-prob margin over the best wrong answer falls by
    0.131 nats [0.091, 0.169]. The letter prior barely moves (mean P(A) 0.261 → 0.247).
- **Determinism.** In the smoke run (57 items, scratch output), two base runs agreed on every item with identical
  log-likelihoods. The pilot's repeat run ran out of GPU memory under swap and has no result.
- **Speed.** Wall-clock on this Mac is unreliable: swap reached 7 GB and the loop's MMLU run took 11× base. In
  isolation (scratch benchmark, batch 2 × 1,100 tokens) the loop takes 1.12× base with or without the cache, in line
  with its 4 extra layers on 28. GPU estimates use layer counts.
- **Disclosure.** The pilot scored TFLT's published config on 570 MMLU test items. No config was chosen with them;
  the full R0 run includes them.

#### 5.2 Consequences (proposed for `SPEC.md`)

1. **Rival grids**, bracketed by the measured change: `rescale` α ∈ {0.8, 0.88, 0.94, 1.06}; `norm-const`
   c ∈ {0.94, 0.97, 0.985, 1.015}. Both are extended if the best is on an edge.
2. **Scored text size:** ~260k tokens (≈ 500 chunks) per model. The sd of chunk-mean paired deltas, 0.057 nats at
   TFLT's config, gives a CI half-width of ε/2 there.
3. **MMLU power:** with 8.2% discordance, the paired CI on all 14,042 items is about ±0.47 points (not ±0.33). A
   −0.3-point non-inferiority margin would then require a positive effect, so the P2 margin becomes **−0.5 points**.
4. **Futility stop (new).** After the loop's tuning grid on both decision models, if no loop config lowers tuning NLL
   below `base` by at least ε/2 = 0.005 on either model, the rival arms and scored NLL runs are not run, and E14
   reports FAIL by futility (H0 and R0 still run). The loop must beat `base` to pass, and a config that cannot beat it
   even on the data it was chosen on will not clear ε on held-out data. This saves most of the GPU budget if the
   pilot's picture holds.
5. **Stage 2 is dropped for Qwen3-1.7B:** there is nothing to converge to. The contraction check runs again on
   Qwen3-4B, Qwen3.5-4B and E2B (minutes); Stage 2 is revived only if one of them contracts.

**Proposed GPU stage A** (one RTX 4090, ~2 h, ~$1–1.5):
- H0 and R0 on all of MMLU 5-shot for both Qwen3 models (base and TFLT's config).
- For Qwen3.5-4B-Base: the load check and two base runs.
- The loop's 84-config tuning grid on both decision models.
- Contraction on Qwen3-4B, Qwen3.5-4B and E2B.

Then the futility check. Stage B (the rest of §3 on the decision models and E2B, ~17–27 h, ~$6–19) runs only if the
loop survives it. The pod installs `flash-linear-attention` and `causal-conv1d` for Qwen3.5. Without them
transformers falls back to a much slower reference implementation of the linear-attention layers.

### 6. Known limitations, fixed in advance

- No generation tasks: TFLT's KV-cache variants and decode-time behaviour are not tested.
- Four models of 1.7B–4.7B, two of which decide; TFLT reports that small Qwen and Llama models benefit least. On
  Qwen3.5 the loop runs over linear-attention layers, which TFLT never tested, so a null there could reflect the
  hybrid rather than looping in general. Qwen3-4B-Base covers the plain-transformer case.
- Damped Euler only; TFLT's higher-order solvers did worse in its own tables.
- Compute is counted in layer applications; wall-clock is reported but not optimised.
- MMLU alone cannot resolve effects much below ±0.33 points, so a small benchmark effect could be missed.

### 7. How prior work tested this (read 2026-11-22; some appendix details came from page summaries)

- **TFLT (2605.23872):** Qwen3 and Llama-3.2 families plus MoEs; MC accuracy; ~720 configs searched across 45 cells
  on a 16-task screen, top 3 re-scored on MMLU (also reported); no CIs; its run-to-run noise (≤ 0.5 points on ≤ 1k-item
  sets) exceeds its 0.3-point "neutral" threshold; 27 of 45 cells positive, 6 negative; no sham, rescaling or
  calibration controls; wall-clock +21.6% when decoding.
- **Inner Loop Inference (2602.14759):** Gemma-2 2B/9B, Llama-3-8B; block chosen on WinoGrande, which it also reports;
  every gain under 1 point, 12 of 17 cells up; does include a matched-magnitude noise control.
- **EGLR (2606.16620):** entropy threshold calibrated on the evaluated data, one seed; its own Table 4 has temperature
  alone (72.6) beating layer recursion alone (69.2) on MATH-500.
- **Tell-Tale Norm (2606.06188):** norm-peak test uses whole-sequence quartiles (non-causal); compared only with no
  recursion.
- **POLAR reproduction (2609.31360):** the router's top program is the standard pass on six frozen models; random
  programs match its pass@k margin on MMLU-Pro.
- **Fixed-point halting (2607.14427):** on a model trained to loop, a training-free KL exit matched uniform depth 8 at
  4.94 loops; a linear-probe router saved nothing.
- **Retrofitted recurrence (2511.07384):** the control frozen-model papers lack, a non-recurrent model trained on the
  same data in the same order, plotted against FLOPs; recurrence wins (TinyLlama GSM8K 51.2 vs 45.2).

---

## E15: what should an exit gate read? Hidden state vs attention vs convergence, on Ouro-1.4B

**Status (2026-11-24):** designed and red-teamed (§6); local pilot running. No GPU time spent. E14 stage A is still
awaiting Tyler's approval; E15 does not depend on it.

### 1. Decision

Tyler plans a small recurrent-depth model in which each token loops a block until it is done (1 to ~1,000 steps;
discussed 2026-11-24, a separate project). That model needs a halting signal: a learned gate on the hidden state
(the standard: Ouro, ACT, MoR), the token's own convergence or confidence, attention-derived signals, or a
zero-parameter rule.

**E15 is a screen, not the decision.** Ouro is frozen, its representation was co-trained with a hidden-state gate,
it has 4 exits rather than 1,000, and every token computes every round (§5). So:
- A signal that clearly adds to the hidden state here is carried into the recurrent-depth project, where gates are
  trained jointly with the model and the final choice is made.
- A null here means **no evidence either way**: the default hidden-state gate stays, and the signal can still be
  tried there.

### 2. Claim and rivals

**Claim.** Attention-derived signals (the token's attention pattern, or whether the tokens it reads from have
settled) improve per-token exit decisions in Ouro-1.4B beyond the hidden state **and** beyond the token's own
convergence and confidence, at matched compute.

| Rival | How E15 separates it |
|---|---|
| **Confidence, not attention.** Prediction entropy, max probability and KL between rounds are nonlinear in the state, so a linear hidden-state gate cannot form them; any arm that includes them can beat `hidden` through them alone. | Every attention verdict is a **nested contrast** against the same arm without attention: `attn+own` vs `own`, `src+own` vs `own`, `all` vs `hidden+own`. If only `own` adds value, the verdict is "confidence helps", not "attention helps". |
| **Intermediate-layer information, not attention.** `attn` summarizes 24 intermediate layers the end-of-round state never sees. | Twin `layers+own`: the same 24 layers summarized without attention (attention-output, MLP-output and residual-update norms, 72 dims). |
| **Self and sink, not other tokens.** A source average that includes the token itself or the position-0 sink is mostly self-interaction or sink mass. | `src` excludes the token itself and position 0, then renormalizes. Sham `src-perm`: the same attention weights over source values permuted among the visible positions. |
| **Position or document boundaries.** Sink mass and entropy track position and EOS boundaries. | Floor `pos`: log position and log tokens since the last EOS. |
| **Training budget or warm start.** New gates are trained more, or differently, than Ouro's own gate. | `hidden` (from scratch) and `hidden-warm` (from Ouro's gate, as Finding 6); every arm trained to convergence on cached features with its own learning-rate grids. |
| **Any token information beats none.** | `bias` floor: a static mix of exit rounds, implemented as random per-token rounds, so it can reach any budget. |
| **Selection on evaluation data.** | Learning rates and transforms chosen on the tuning split; budgets matched on the evaluation split by threshold only (layer counts carry no label information). |
| **Leakage.** A feature at position t could see the label (token t+1) or a later round. | Features use rounds ≤ r and tokens ≤ t only. Tests check that round-r features ignore later rounds, and that features at t do not change when token t+1 or later tokens change. |
| **Unmatched compute.** Confidence and attention features need extra computation (a full-vocabulary readout each round, explicit attention probabilities). | Feature cost is charged in layer-equivalents in a secondary comparison (§3). |

### 3. Design

**Exits.** After round r ∈ {1, 2, 3, 4} a token stops (24·r layers). Ouro's evaluation accounting: every token
computes every round, and its loss comes from its exit round (as `bench.py` and Finding 6).

**Gates.** Stop probability per round λ_r = σ(w_r·f_r + b_r) with the exit distribution of OuroForCausalLM. Weights
are shared across rounds for `hidden`, `hidden-warm`, `hidden+own` and `all` (as Ouro's gate), and per round for the
small arms (≤ 100 dims). Trained on the exact expected objective `Σ_r pdf_r (CE_r + price · 24r)`, prices
{0.0005, 0.001, 0.002} nats per layer, on **cached** features (Ouro is frozen) until the tuning-split objective stops
improving. Each arm gets separate weight and bias learning-rate grids, extended until the best value is interior.
Features are standardized (all arms), and KL and norm-ratio features are log-transformed.

**Features at round r** (rounds ≤ r, tokens ≤ t):

| Arm | Inputs | Dims |
|---|---|---|
| `ouro-gate` | Ouro's gate, unchanged | — |
| `hidden`, `hidden-warm` | Ouro's normed end-of-round state | 2,048 |
| `own` | log KL(p_r ‖ p_{r−1}), log relative state change, entropy and max probability of p_r, round-1 flag | 5 |
| `hidden+own` | both | 2,053 |
| `attn+own` | per layer, mean over heads: attention entropy / log(#visible), mass on position 0, mass on self; plus `own` | 77 |
| `layers+own` | per layer: ‖attention output‖, ‖MLP output‖, ‖residual update‖, each relative to the residual; plus `own` | 77 |
| `src+own` | at layers 6, 12, 18, 24: attention-weighted mean of the sources' round-r log-KL and entropy, sources excluding self and position 0; plus `own` | 13 |
| `src-perm+own` | as `src+own`, source values permuted among visible positions | 13 |
| `all` | `hidden` + `own` + `attn` + `src` | 2,133 |
| `pos`, `bias` | log position and log tokens since EOS; nothing | 2; 0 |
| `kl-rule` | zero-parameter rule: exit at round 1 if entropy < τ₁, else at the first round with KL < τ₂ | 2 thresholds |

**Data.** FineWeb-Edu `sample-10BT` stream, 1,024-token chunks (`routed_ouro.fineweb_chunks`). Chunks 0–63 are the
evaluation split (repo convention, extended if the pilot says more are needed), 64–127 tuning, 128 onward training.
Training and evaluation sizes come from the pilot.

**Outcome.** For each arm, the held-out loss-vs-layers curve, with thresholds set **on the evaluation split** to hit
each budget (label-free). **Primary:** the area under the curve, i.e. mean NLL over budgets 50–90 layers per token.
Secondary: NLL at 50, 60, 70, 80 and 90. Comparisons are paired by token with chunk-bootstrap 95% CIs. A
hindsight-oracle exit is reported as an upper bound only. **Charged comparison (secondary):** arms that need the
full-vocabulary readout every round pay for it in layer-equivalents (Ouro's readout ≈ 2 layers if the vocabulary is
~49k, to be measured), and attention-probability arms pay their measured overhead.

**Smallest effect of interest:** the NLL worth 2 layers of compute, read off the `ouro-gate` curve's slope at 70
layers on the tuning split. This anchors it in compute rather than in the hindsight oracle (E13: oracle gaps here
are mostly selection). Frozen in `SPEC.md` after the pilot.

### 4. Decision rule (proposed; frozen in `SPEC.md` after the pilot)

All tests are on the primary AUC, paired, two-sided where a difference is claimed, TOST where equivalence is.
- **Attention helps:** `attn+own` or `src+own` beats `own` by at least the SESOI (CI excluding 0) **and** beats its
  twin (`layers+own`, respectively `src-perm+own`) by at least the SESOI; and `all` beats `hidden+own`. Holm across
  the two attention arms. → carry that signal into the recurrent-depth project's joint-training test.
- **Confidence helps:** `own` or `hidden+own` beats `hidden` by at least the SESOI, without the attention verdict. →
  carry confidence features (and `kl-rule` as a zero-parameter candidate) forward.
- **Hidden suffices:** `hidden+own` and `all` are equivalent to `hidden` within ± the SESOI (TOST). → default gate.
- **Inconclusive:** anything else, reported as such. The evaluation size is set from pilot variance so that the
  AUC CI half-width is at most half the SESOI.

### 5. Limitations, fixed in advance

- Ouro has 4 exit points, not 1,000. A signal that helps choose among 4 may not transfer to fine-grained halting.
- Under Ouro's evaluation accounting every token computes every round. So `src` measures sources at the same round,
  not at their own exit, as they would be when decoding with real early exits.
- Ouro's representation was trained with its hidden-state gate over 7.7T tokens, which favours `hidden`. A gate
  trained jointly with attention features could do better than one fitted to a frozen model (§1).
- In a 1,000-step model a full-vocabulary readout every step costs about as much as the step itself, so `own` and
  `kl-rule` would need a cheaper readout there.

### 6. Red-team (2026-11-24)

A fresh agent read only the first draft of this section and `structured_router.py`, `routed_ouro.py` and
`ouro_model.py`. Nine findings, all accepted; the responses are folded into §1–§5.

| # | Finding | Response |
|---|---|---|
| 1 | The attention arms also carried the token's own confidence, which a linear hidden-state gate cannot form, so "attention helps" could come from confidence alone. | Nested contrasts against the same arm without attention; `hidden+own` added; separate "confidence helps" verdict. |
| 2 | Ouro (frozen, co-trained gate, 4 exits, all rounds computed) answers an easier question than the recurrent-depth decision; a null should not mean "use a hidden gate". | E15 reframed as a screen; a null means no evidence either way; the decision is made in the jointly trained model. |
| 3 | Optimization unfair both ways: a cold 2,048-dim gate trained briefly on a shared LR grid; unstandardized hidden state; small arms with shared weights on heavy-tailed KL could underfit. | Cached features, train to convergence, per-arm weight/bias LR grids extended to interior, `hidden-warm` added, all standardized, logs of KL and norm ratios, per-round weights for small arms. |
| 4 | "Hidden suffices" declared on a failure to reject; a SESOI from a selection-driven oracle; power unknown. | TOST for equivalence; SESOI anchored in 2 layers of compute; AUC as the single primary; evaluation size from pilot variance. |
| 5 | Compute not matched: confidence features need a full-vocabulary readout each round; tuning-split thresholds drift on eval. | Charged comparison in layer-equivalents; budgets matched on the evaluation split by threshold (label-free). |
| 6 | `src` included the token itself and the sink, so it was not cross-token. | Self and position 0 excluded, renormalized; `src-perm` sham. |
| 7 | `attn` confounded attention with intermediate-layer information and position; `bias-only` could only hit 4 budgets. | `layers+own` twin; `pos` floor; `bias` as random per-token rounds. |
| 8 | Leakage traps: target-aligned arrays in source values; an off-by-one mask; NaN at position 0. | Label- and future-token invariance tests; entropy normalization is 0 at position 0 (no division by zero; tested). |
| 9 | `kl-rule` could never exit at round 1. | Round-1 entropy threshold added. |

---

### Decisions

- **2026-11-24** (Tyler: "Yes, test on Ouro"). E15 adopted: exit-gate inputs compared on Ouro-1.4B before the
  recurrent-depth project chooses its halting signal. After red-team: E15 is a screen; nested contrasts against
  confidence features, twins for intermediate layers and sources, cached features trained to convergence, TOST for
  equivalence, AUC as the single primary, a compute-anchored SESOI.
- **2026-11-22.** E14 adopted as the next experiment (Tyler: "Yes, write it up"). Staged: parity and known-mover gates
  → fair controls on NLL → MMLU gated on that → stopping rule only on a pass. Ouro dropped. Decision rests on the two
  Qwen models; E2B is generalization. Generation tasks out of scope.
- **2026-11-22, after red-team.** NLL is the primary outcome with a provisional SESOI of 0.01 nats/token; `mean-shift`
  and `norm-const` added as zero-compute comparators; shams draw donors from other sequences; implementation goes
  through a HuggingFace-layer wrapper rather than `G4`; selection uses continuous metrics.
- **2026-11-22, after the pilot** (Tyler: "Yes, run the pilot"). Pilot on the Mac with MMLU cut to 10 items per
  subject (swap pressure). TFLT's config raises tuning NLL on Qwen3-1.7B-Base; the window does not contract. Proposed
  for `SPEC.md`: bracketed rival grids, ~260k scored tokens, P2 margin −0.5 points, a futility stop after the loop's
  tuning grid, Stage 2 dropped unless 4B or E2B contracts. GPU split into stage A (~$1) and stage B.
- **2026-11-23** (Tyler: "can we use a more up to date model?"). Qwen3.5-4B-Base (February 2026, the newest small Qwen
  base model) joins Qwen3-4B-Base as a decision model. Qwen3-1.7B-Base moves to reproduction only, since its R0 cell
  still needs the paper's own model. E2B (April 2026) stays as generalization. The wrapper handles Qwen3.5's hybrid
  layers unchanged, and Qwen3.5 cases were added to `tests/test_loop_wrap.py` (all 102 tests in `tests/` pass;
  planting the cache bug fails them on linear-attention layers too). Stage A grows to ~2 h; stage B to ~17–27 h.
