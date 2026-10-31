<div align="center">

# Layer Looping

**Per-token layer programs in a frozen LLM: how much is there to gain, and can the previous token see it coming?**

Tyler Gee · [Kāula](https://kaula.ai)

[**Paper (PDF)**](paper/paper.pdf) · [**Explainer**](https://kaula-ai.github.io/layer-looping/) · [**Experiments**](EXPERIMENTS.md) · [**Citation**](#citation)

</div>

---

> **Status: experiments in progress; no results yet.** The paper draft's results render as `[pending]` until `bash run_experiments.sh full` has run and `python scripts/make_tables.py` has filled them in from `results.json`.

A pretrained transformer sends every token through every layer once. This project asks three questions:

1. **Headroom.** How much would a single token's next-token loss drop if it repeated one layer, skipped one, or swapped two adjacent ones, while every other token runs normally?
2. **Lag.** Can that decision be made from the **previous** token's internals, so each token's route is known before its forward pass starts?
3. **Interpretability.** Do interpretable signals predict when an edit helps as well as a raw hidden-state probe does? Candidate signals:
   - logit-lens entropy and agreement with the final layer;
   - residual-update norms;
   - attention entropy;
   - Gemma Scope SAE latents.

**Part 2** takes this all the way. Each token gets its own path through the frozen layers: repeats, skips, swaps, reversed blocks and looped blocks, in any combination. Per-token search finds good paths, and a cheap controller learns to pick them. Think of it as depth recurrence where the model also chooses *which* layer to apply next.

The model is `google/gemma-4-E2B` (Gemma-2-2B is the second model) and stays frozen; only small linear routers are trained. The headline result will be a compute–accuracy frontier: NLL saved against extra layer applications per token. Skips count as −1 layer, so they can pay for repeats.

## Method in one paragraph
For each of 103 single-edit layer programs (plus a static block-loop baseline), `oracle.py` runs every position through that program at once. At each step, a position attends to the other positions' base-pass keys and values for the layer being run, plus its own current ones; we call this the *diagonal swap*. Because the model is causal and only that position changes, this gives its exact next-token loss under the edit in about one forward pass per program. `routed_eval.py` and `decode.py` then apply whole routes, so later tokens see the edits. The tests check that both paths agree.

## Quickstart (one 24 GB GPU, e.g. RTX 4090)
```bash
pip install -r requirements.txt
python -m pytest -q tests             # CPU correctness tests (tiny random Gemma-2 and Gemma-4, no download)
bash run_experiments.sh quick         # E0-E4 on 20k tokens, ~15 min
bash run_experiments.sh full          # E1-E6 on 200k tokens, ~2-3 h
bash run_experiments.sh sae           # E4 SAE features (Gemma-2-2B oracle only; needs HF login + Gemma-2 license)
bash run_experiments.sh downstream    # E8 GSM8K at matched compute (after full)
bash run_experiments.sh programs-quick  # E9-E12 per-token paths + controller, 20k tokens (~1 h)
```
See [`EXPERIMENTS.md`](EXPERIMENTS.md) for each experiment's question, hypothesis, metric and kill criterion.

## Layout
| File | What it does |
|---|---|
| `oracle.py` | Manual forwards for Gemma-4 (`G4`, with K/V sharing and per-layer embeddings) and Gemma-2 (`G2`), layer programs, oracle collection, `--check` (HF parity, swap identity, exactness vs independent recompute) |
| `probe.py` | Headroom by op, λ sweep → Pareto fronts, budget-matched ΔNLL, same-token vs t−1 routers with bootstrap CIs, feature ablation, optional SAE routers |
| `analyze_tokens.py` | Oracle gains by token type, next-token type, answer start, source, position, surprisal |
| `routed_eval.py` | Whole-sequence routed evaluation; measures the interaction gap |
| `decode.py` | Token-by-token decoding with per-token programs and a first-pass KV cache |
| `sae_features.py` | Gemma Scope SAE latents for every oracle position (via `sae_lens`; Gemma-2-2B only) |
| `gsm8k_eval.py` | GSM8K greedy generation under base / static / router policies |
| `programs.py` | Edit grammar on layer paths (repeat, skip, swap, reverse, loop) |
| `search.py` | Per-token beam search over paths, using the mixed-path executor `run_paths` |
| `library.py` | Shared path library, scored exactly for every token |
| `controller.py` | Cheap MLP controller vs oracles and baselines |
| `program_eval.py` | Whole-sequence routing with arbitrary per-token paths |
| `manim_scenes.py`, `docs/explainer.html` | Animations for talks |
| `tests/` | Correctness tests (all CPU) |
| `paper/`, `scripts/make_tables.py` | LaTeX paper draft; renders `results.json` into its tables |
| `pilot/` | Exploratory whole-sequence pilot on GPT-2 small and Qwen3-1.7B (not the per-token oracle) |
| `CLAUDE.md` | Working brief for coding agents |

## Related work
Closest prior work, from a falsification-first search in October 2026:
- **Per-token, training-free looping:** EGLR (2606.16620), the Tell-Tale Norm (2606.06188).
- **Static frozen-model loops:** Inner Loop Inference (2602.14759), Training-Free Looped Transformers (2605.23872).
- **Per-input layer programs:** CoLa (2507.07996), POLAR (2606.06574), Dynamic Routing for Test-time Versatility (2610.04751), Dr.LLM (2510.12773), MACRO (2608.05872).
- **Trained per-token depth:** Think-at-Hard (2511.08577), Mixture-of-Recursions, FlexiDepth (2503.23798), What Layers When (2510.13876).
- **Routers vs oracles:** Glavas et al. (2410.20022), Duo-LLM (2410.10846).
- **Depth analyses:** How Do LLMs Use Their Depth? (2510.18871), Target-Token Asymmetry (2610.19344), The Remarkable Robustness of LLMs (2406.19384).
- **Interpretable exit signals:** DEGS (2607.09693), TIDE (2603.21365), AdaInfer (2403.02181).

What this project adds:
- an exact per-token map of repeat, skip, swap and block edits in a frozen model;
- routes decided **one token early**;
- a controlled comparison of interpretable, raw and SAE signals as router inputs;
- a compute frontier where skips pay for loops, against static and heuristic baselines.

## Paper
`paper/` holds the LaTeX draft and the built PDF. Every number in it is generated from a saved `results.json`:

```bash
python scripts/make_tables.py --results oracle_out_full/results.json
make -C paper
```

## Citation
```bibtex
@misc{gee2026layerlooping,
  title  = {Layer Looping: Per-Token Layer Programs in a Frozen {LLM}},
  author = {Gee, Tyler},
  year   = {2026},
  note   = {K\={a}ula preprint},
  url    = {https://github.com/Kaula-AI/layer-looping}
}
```

## License
MIT. See [`LICENSE`](LICENSE).
