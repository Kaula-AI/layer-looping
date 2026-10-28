# Novelty review: per-token layer programs in a frozen LLM

*Falsification-first search, October 28 2026. Roughly 70 candidate papers were screened across three angles, plus a citation chase:*
- *training-free and frozen looping;*
- *ahead-of-time routing and interpretable signals;*
- *per-token depth analyses.*

*The closest hits were verified against their arXiv abstracts. Rows marked (abstract) were checked only at abstract level.*

## Verdict
**No kill paper.** Nothing does the full combination:
1. a per-token edit of a frozen model;
2. repeat, skip and swap in one action space;
3. a route decided from token t−1 before token t's pass;
4. interpretable and SAE signals compared against a raw probe.

Two parts of the original pitch are **no longer new**, so the paper must not lead with them:
- **Per-token looping in a frozen model.** EGLR (2606.16620) re-applies the top L layers, training-free, at high-entropy tokens. The Tell-Tale Norm (2606.06188) runs layer recursion guided by the hidden-state ℓ2 norm. Both decide from the current token.
- **Looping a frozen model helps.** Inner Loop Inference (2602.14759) and Training-Free Looped Transformers (2605.23872) both loop static blocks. On Gemma-2-2B the best block is layers 10–13, with gains within the reported standard errors.

## Defensible contributions
1. **An exact per-token oracle map.** It covers repeat, skip, swap and block edits in a frozen model, using the diagonal-swap measurement, and adds the whole-sequence interaction gap. No prior per-token counterfactual layer oracle was found. The closest relatives are Duo-LLM (2410.10846), which builds an oracle over small FFN, big FFN or skip in a trained model, and DEX (2606.29223), which has an exit-only oracle.
2. **Routing decided one token early.** No paper decides depth or layer edits from t−1. The nearest are:
   - MoE expert *prefetch*, which plans from previous tokens but doesn't change the computation (2606.15453);
   - same-token next-layer speculation (2603.19289, Pre-gated MoE);
   - prompt-level routing (FiRST 2410.12513).
3. **Interpretable vs raw vs SAE router inputs, head to head.** Entropy and logit-lens signals as exit criteria are well established (AdaInfer, CALM, DEGS, SimExit, River-LLM). No paper was found that uses SAE latents as router inputs, or that compares them against a raw-state probe for predicting looping gain.
4. **A frontier where skips pay for loops,** plotted against static (block loop) and training-free heuristic (entropy gate, norm peak) baselines.

## Risks the evidence points to
- **Routers may not recover oracle headroom.** Four results point that way:
  - Glavas et al. (2410.20022): per-token skip routers did no better than a constant input.
  - Duo-LLM: trained routers were suboptimal next to oracles.
  - A POLAR reproduction (2609.31360): the router's top program collapsed to the standard pass.
  - A linear probe halting router saved nothing (2607.14427).

  Report the edit rate and compare against the heuristic baselines.
- **Skip headroom may be small.** Diminishing Returns of Early Exit (2603.23701) finds less layer redundancy in modern models.
- **Naive repeats often hurt** (2605.23872).
- **Target-type confounds.** Whitespace targets can masquerade as depth effects (2610.19344). E6 therefore separates them and mixture-matches the source comparison.
- **Token-type depth patterns are partly known** (2510.18871, 2510.13876). E6 contributes causal intervention evidence, not the patterns themselves.

## Changes made to the plan and code
- A static block loop of layers 10–13×2 is now an oracle program and the prior-work static baseline.
- Training-free baselines were added to the frontier: an EGLR-like entropy gate (same token and t−1) and an ALRR-like norm-peak repeat.
- A new E3 kill condition: learned routers must beat these heuristics at matched compute.
- E6 controls were added: whitespace/newline targets are separated, the source comparison is mixture-matched, and a version excludes whitespace targets.

## Key references by role
| Role | Papers |
|---|---|
| Closest per-token, training-free looping | EGLR 2606.16620; Tell-Tale Norm 2606.06188 (abstract) |
| Static frozen loops (baselines) | Inner Loop Inference 2602.14759; Training-Free Looped Transformers 2605.23872 |
| Per-input programs | CoLa 2507.07996; POLAR 2606.06574; POLAR reproduction 2609.31360; Dynamic Routing 2610.04751 (abstract); Dr.LLM 2510.12773; MACRO 2608.05872 |
| Trained per-token depth | Think-at-Hard 2511.08577 and TaH follow-up 2609.35748; FlexiDepth 2503.23798; What Layers When 2510.13876; Mixture-of-Recursions |
| Routers vs oracle | Glavas 2410.20022; Duo-LLM 2410.10846; per-token fixed-point halting 2607.14427 |
| Interpretable exit signals | AdaInfer 2403.02181; DEGS 2607.09693; TIDE 2603.21365; River-LLM 2604.18396; Confident Layer Decoding 2606.21906 |
| Depth analyses | How Do LLMs Use Their Depth? 2510.18871; Target-Token Asymmetry 2610.19344; Remarkable Robustness 2406.19384; Diminishing Returns of Early Exit 2603.23701 |
| Ahead-of-time compute (systems) | Spatio-temporal expert prefetch 2606.15453; Speculating Experts 2603.19289; Pre-gated MoE 2308.12066; Deja Vu 2310.17157 |

## Still to do before writing
- Read the decode-time looping appendix of 2605.23872.
- Read the full method of 2606.06188 to confirm whether its recursion is decided per token.
- Re-run this search in early January, since the field moves monthly.
