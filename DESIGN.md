# Layer looping: living design

Dated decisions for experiments after E13b. Earlier designs live in `EXPERIMENTS.md` (E0–E13b). Frozen run plans go
in `SPEC.md`; this file is where they are argued.

---

## E14: training-free damped looping under fair controls

**Status (2026-11-22):** designed and red-teamed (§4), not run. Next: pilot (§5), then `SPEC.md` with its hash
published (step 6). No GPU time has been spent.

### 1. Decision

On 2026-11-18 Tyler proposed that looping happen **at inference only**, following Training-Free Looped Transformers
(TFLT, arXiv 2605.23872). E14 decides between two paths:

- **Looping passes the fair controls:** inference-only looping has support. Build the per-token stopping rule on it
  (Stage 2, §3.9) and caveat takeaway 4 in the note and paper ("new kinds of looping have to be trained into the
  model").
- **It doesn't:** inference-only looping in frozen models is treated as closed, the null becomes Finding 8 (the
  selection result extends from per-token edits to uniform loops), and the next direction is recurrence trained
  into the model (e.g. retrofitted recurrence, arXiv 2511.07384). That path needs a separate budget decision.

The decision is taken on the two Qwen models, TFLT's own setting. Gemma-4 E2B is a generalization readout (§3.2).

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

| Model | Layers | Role |
|---|---|---|
| Qwen3-1.7B-Base | 28 | Decision model. TFLT config [12–15], K = 2: MMLU 5-shot 62.68 → 63.01. Fits the Mac for the pilot. |
| Qwen3-4B-Base | 36 | Decision model. TFLT config [15–18], K = 3: MMLU 5-shot 73.27 → 73.61 (its Table 13 gives a 73.01 base for the same window; the paper is inconsistent). |
| Gemma-4 E2B | 35 | Generalization readout, this project's model. |

Ouro-1.4B is dropped: it is already trained to loop, so damped sub-steps inside its rounds are a different question.

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
| `sham-perm` | `g(x₀) + σ·δ_donor`, donor = same position in a different sequence | σ ∈ {0.25, 0.5, 1, 2} at the loop's selected config | as `loop` |
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

- **H0, harness parity (gate).** Base MMLU 5-shot on each Qwen model within 0.5 points of TFLT's base. If not, the
  harness is fixed before anything else is run or scored.
- **Known movers (gate).** `repeat` must raise tuning NLL, and the letter bias must change MMLU validation accuracy.
  If neither moves, the pipeline cannot detect a change and nothing is scored.
- **R0, reproduction (descriptive).** TFLT's config on MMLU test, reported as an estimate with a 95% CI and read
  three ways: an effect exists (lower CI > 0); TFLT's size is not reproduced (upper CI < +0.34); or neither.
  TFLT's gain was selected among its top 3 on MMLU, so some shrinkage is expected. If R0 is not reproduced while H0
  passes, the write-up counts that toward the selection rival. R0 does not gate the rest.
- **PASS.** On at least one Qwen model (97.5% CIs: Bonferroni over the two Qwen models), the loop's P1 gain over each
  comparator has its lower CI above 0 and its point estimate at or above ε; **and** on that model P2 is non-inferior
  (lower 95% CI of loop − base, both with letter bias, above −0.3 points). → Stage 2.
- **FAIL.** On both Qwen models, the upper 95% CI of the loop's P1 gain over the best comparator is below ε. →
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

Tests (CPU, tiny random Qwen3 and the tiny Gemma-4 with K/V sharing):
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

### 6. Known limitations, fixed in advance

- No generation tasks: TFLT's KV-cache variants and decode-time behaviour are not tested.
- Three models of 1.7B–4B; TFLT reports that small Qwen and Llama models benefit least.
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

### Decisions

- **2026-11-22.** E14 adopted as the next experiment (Tyler: "Yes, write it up"). Staged: parity and known-mover gates
  → fair controls on NLL → MMLU gated on that → stopping rule only on a pass. Ouro dropped. Decision rests on the two
  Qwen models; E2B is generalization. Generation tasks out of scope.
- **2026-11-22, after red-team.** NLL is the primary outcome with a provisional SESOI of 0.01 nats/token; `mean-shift`
  and `norm-const` added as zero-compute comparators; shams draw donors from other sequences; implementation goes
  through a HuggingFace-layer wrapper rather than `G4`; selection uses continuous metrics.
