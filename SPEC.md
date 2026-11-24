# Frozen pre-registrations

Each section is frozen when committed. Changes after that go under that section's **Amendments**, dated, with the
reason. The design arguments live in `DESIGN.md`.

---

## E15: what should an exit gate read? (frozen 2026-11-24)

Frozen before any gate was trained on the real data and before any evaluation-split outcome was computed. The
feature cache for the evaluation split was being written when this was committed, but no statistic of it has been
read. The design, rivals and red-team are in `DESIGN.md` (E15).

**Code.** `exit_signals.py` (features, `Gate`, cache) and `e15_fit.py` (fitting, scoring, verdict) at the commit that
adds this file. Tests: `tests/test_exit_signals.py`, `tests/test_e15_fit.py`.

**Model.** `ByteDance/Ouro-1.4B`, bf16, frozen: 24 layers × 4 rounds; exit after round r costs 24·r layers. Ouro's
evaluation accounting: every token computes every round, and its loss is the CE of its exit round.

**Data.** FineWeb-Edu `sample-10BT` stream via `routed_ouro.fineweb_chunks`, 1,024-token chunks (BOS + 1,023), scored
positions 0–1,022 of each chunk. Splits by chunk index:
- **evaluation:** chunks 0–63 and 192–255 (128 chunks, 130,944 positions). Never read by the pilot.
- **tuning:** chunks 64–127. The pilot read chunks 64–79.
- **training:** chunks 256–743 (488 chunks, 499,224 positions).

**Arms.** Features exactly as in `exit_signals.round_features` / `transform` and `ARMS`:
- `hidden`, `hidden_warm` (initialized from Ouro's gate, unstandardized);
- `own`, `hidden_own`, `attn_own`, `layers_own`, `src_own`, `srcperm_own`, `all`, `pos`;
- `ouro_gate`: Ouro's gate, untrained;
- `bias`: each token exits at one of the two rounds bracketing the budget, at random;
- `kl_rule`: exit at round 1 if entropy < τ_ent, else at the first round with KL < τ_kl.

Weights are per round for `own`, `attn_own`, `layers_own`, `src_own`, `srcperm_own` and `pos`, and shared across
rounds otherwise.

**Fitting** (`e15_fit.fit`):
- **Objective:** the exact expected objective `mean_t Σ_r pdf_r (CE_r + price·24r)`, at prices {0.0005, 0.001, 0.002}.
- **Standardization:** inputs standardized with statistics from the first 32 training chunks (not for `hidden_warm`).
- **Learning-rate grids:** weights {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} (`hidden_warm`: {1e-6, 3e-6, 1e-5, 3e-5, 1e-4});
  biases {1e-3, 1e-2, 1e-1}. A grid is extended outward (×3 for weights, ×10 for biases) when the selected value is on
  its edge, at most twice per arm. If it is still on an edge, that is reported.
- **Optimization:** Adam (β = 0.9, 0.999). Minibatches of 4,096 positions within shuffled blocks of 8 chunks, at most
  8 epochs. Early stopping on the tuning objective with patience 2, keeping each gate's best epoch.
- **Selection:** per arm, the (weight lr, bias lr, price) with the lowest tuning AUC (below).

**Evaluation** (`e15_fit.report`):
- **Budgets** B ∈ {50, 55, …, 90} layers per token.
- **Matching:** for every arm and budget, the threshold is matched on the evaluation split by bisection. Tokens are
  then mixed between the two bracketing thresholds using common random numbers (seed 2), so that the mean cost equals
  B. This uses no labels.
- **Rule-based arms:**
  - `kl_rule`: τ_ent ∈ {0, 0.03, 0.1, 0.3, 1.0} is chosen per budget on the tuning split, and τ_kl is matched to the
    budget.
  - `oracle`: the hindsight-best round under a price matched to the budget. An upper bound only, never a comparator.
- **Primary metric:** each chunk's mean NLL at each budget, averaged over the 9 budgets (the per-chunk AUC).
- **Contrasts:** AUC differences paired by chunk.
- **Statistics:** chunk bootstrap with 10,000 resamples (seed 0), giving 95% CIs, 90% CIs for TOST, and two-sided
  bootstrap p-values.

**Smallest effect of interest.** δ = 2·(NLL₆₅ − NLL₇₅)/10 for `ouro_gate` on the tuning split, i.e. the NLL worth 2
layers at 70 layers. The pilot's estimate was ≈ 0.0023 nats/token.

**Verdicts** (exactly as coded in `e15_fit.report`; all flags are reported, and the summary takes the first that holds):
1. **ATTENTION HELPS.** For each attention arm (`attn_own` with twin `layers_own`; `src_own` with twin `srcperm_own`):
   - p = the larger of the bootstrap p-values against `own` and against the twin (intersection-union);
   - Holm over the two arms at α = 0.05;
   - both AUC differences favour the arm by at least δ.

   At least one arm must pass, **and** `all − hidden_own` must have its 95% CI below 0.
2. **CONFIDENCE HELPS.** `own − hidden` or `hidden_own − hidden` is at most −δ, with its 95% CI below 0.
3. **HIDDEN SUFFICES.** The 90% CIs of `hidden_own − hidden` and `all − hidden` both lie within (−δ, +δ) (TOST).
4. **INCONCLUSIVE.** None of the above.

**What each verdict decides** (`DESIGN.md` E15 §1, §4):
- Attention helps → carry that signal into the recurrent-depth project's joint-training test.
- Confidence helps → carry the confidence features and `kl_rule` forward.
- Hidden suffices → the default hidden-state gate.
- Inconclusive → no evidence either way.

E15 is a screen on a frozen model; none of these is the final choice.

**Secondary and exploratory.** Per-budget NLLs, the compute-charged comparison (readout and attention overheads in
layer-equivalents, computed from Ouro's config as coded), `kl_rule`, `pos` vs `bias`, `hidden` and `hidden_warm` vs
`ouro_gate`, and the oracle. Anything not listed here is labelled exploratory in the write-up.

**Run.**
```bash
python exit_signals.py cache --out results/e15      # features (Mac, MPS, ~3 h)
python e15_fit.py fit --cache results/e15/cache --out results/e15
python e15_fit.py report --cache results/e15/cache --out results/e15   # writes results/e15/REPORT.md
```

### Amendments

None.
