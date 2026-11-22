"""
CPU tests for loop_wrap.py (E14, DESIGN.md §3.8) on a tiny random Qwen3 and the tiny random Gemma-4 with K/V
sharing, per-layer embeddings, layer scalars and a sliding window shorter than the test sequences:
  1. every arm at its identity setting is an exact no-op (K=1, beta=1, alpha=1, c=1, s=0, sigma=0)
  2. damped loop == an independent reference built from the unwrapped model with hooks (HF's own forward
     computes every g(x_k); the reference never touches the wrapper), past the sliding window
  3. use_cache=True and use_cache=False give the same looped logits (sub-steps never append to the cache)
  4. zero-compute arms match their formulas; repeat == g(g(x0))
  5. sham_perm takes its direction from the same position of the previous sequence, never its own, keeps the
     recipient's size, and gives position 0 the loop's own change
Run: python -m pytest -q tests/test_loop_wrap.py
"""
import os
import sys

import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loop_wrap import Arm, LoopWindow, decoder_layers  # noqa: E402

torch.manual_seed(0)
S_, F_ = "sliding_attention", "full_attention"


def _qwen():
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      initializer_range=0.2, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).float().eval()


def _gemma():
    cfg = Gemma4TextConfig(
        vocab_size=97, vocab_size_per_layer_input=97, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=8,
        num_kv_shared_layers=4, use_double_wide_mlp=True, initializer_range=0.2, sliding_window=16,
        layer_types=[S_, S_, F_, S_, S_, F_, S_, F_], max_position_embeddings=512, final_logit_softcapping=30.0,
        rope_parameters={F_: {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
                         S_: {"rope_theta": 1e4, "rope_type": "default"}})
    cfg._attn_implementation = "eager"
    m = Gemma4ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n_, p_ in m.named_parameters():
            if "norm" in n_ and p_.dim() == 1:
                p_.uniform_(0.5, 1.5)
        for i, l_ in enumerate(m.model.layers):
            l_.layer_scalar.fill_(0.8 + 0.05 * i)
    return m


QWEN, GEMMA = _qwen(), _gemma()
# (model, window a, b, sequence length). Gemma: layers 4-6 are K/V-shared (they read layers 2/3), and T = 40 is
# past the sliding window of 16, so the sliding masks recorded for each layer are exercised in the sub-steps.
# gemma_lo: two sliding layers with their own K/V (they write the cache; the shared layers 4-7 never do).
CASES = {"qwen": (QWEN, 1, 3, 24), "gemma": (GEMMA, 4, 6, 40), "gemma_lo": (GEMMA, 0, 1, 40)}
TOL = 2e-4


def _ids(T, B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (B, T), generator=g)


@torch.no_grad()
def _forward_with_hooks(model, ids, pre=None, post=None, a=None, b=None):
    """One forward of the unwrapped model with an optional pre-hook on layer a and forward hook on layer b."""
    layers, hs = decoder_layers(model), []
    try:
        if pre is not None:
            hs.append(layers[a].register_forward_pre_hook(pre, with_kwargs=True))
        if post is not None:
            hs.append(layers[b].register_forward_hook(post, with_kwargs=True))
        return model(ids).logits
    finally:
        for h in hs:
            h.remove()


def _window_io(model, a, b, ids):
    """x0 = input of layer a, g0 = output of layer b, from the unwrapped model's own forward."""
    cap = {}

    def pre(m, args, kw):
        cap["x0"] = args[0]

    def post(m, args, kw, o):
        cap["g0"] = o

    _forward_with_hooks(model, ids, pre, post, a, b)
    return cap["x0"], cap["g0"]


def _run_window(model, a, b, ids, x_in):
    """g(x_in) by HF's own full forward, with layer a's input replaced by x_in (independent of loop_wrap)."""
    cap = {}

    def pre(m, args, kw):
        return (x_in,) + tuple(args[1:]), kw

    def post(m, args, kw, o):
        cap["y"] = o

    _forward_with_hooks(model, ids, pre, post, a, b)
    return cap["y"]


def _logits_with_window_output(model, b, ids, y):
    """Logits when layer b's output is replaced by y."""
    return _forward_with_hooks(model, ids, post=lambda m, args, kw, o: y, b=b)


@torch.no_grad()
def _wrapped_logits(model, arm, ids, **kw):
    with LoopWindow(model, arm):
        return model(ids, **kw).logits


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arm_kw", [
    dict(kind="base"), dict(kind="loop", K=1), dict(kind="loop", K=3, beta=1.0), dict(kind="rescale", alpha=1.0),
    dict(kind="norm_const", c=1.0), dict(kind="mean_shift", s=0.0, delta_bar=torch.ones(64))])
def test_identity_settings_are_noops(case, arm_kw):
    # (shams at sigma = 0 are not no-ops: position 0 keeps the loop's own change by design; see the donor test)
    model, a, b, T = CASES[case]
    ids = _ids(T)
    ref = model(ids).logits
    got = _wrapped_logits(model, Arm(a=a, b=b, **arm_kw), ids)
    assert (got - ref).abs().max() < TOL
    # and the model is restored afterwards
    assert all("forward" not in layer.__dict__ for layer in decoder_layers(model))


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("K,beta", [(2, 0.0), (3, 0.0), (2, 0.5)])
def test_loop_matches_independent_reference(case, K, beta):
    model, a, b, T = CASES[case]
    ids = _ids(T)
    x0, g0 = _window_io(model, a, b, ids)
    x = x0
    for _ in range(K):
        x = x + (_run_window(model, a, b, ids, x) - x) / K
    y = beta * g0 + (1 - beta) * x
    ref = _logits_with_window_output(model, b, ids, y)
    got = _wrapped_logits(model, Arm(kind="loop", a=a, b=b, K=K, beta=beta), ids)
    base = model(ids).logits
    assert (got - base).abs().max() > 1e-2, "loop should change the logits on a random model"
    assert (got - ref).abs().max() < TOL


@pytest.mark.parametrize("case", list(CASES))
def test_cache_flag_does_not_matter(case):
    model, a, b, T = CASES[case]
    ids, arm = _ids(T), Arm(kind="loop", a=a, b=b, K=3)
    on = _wrapped_logits(model, arm, ids, use_cache=True)
    off = _wrapped_logits(model, arm, ids, use_cache=False)
    assert (on - off).abs().max() < TOL


@pytest.mark.parametrize("case", list(CASES))
def test_zero_compute_arms_and_repeat(case):
    model, a, b, T = CASES[case]
    ids = _ids(T)
    x0, g0 = _window_io(model, a, b, ids)
    dbar = torch.randn(g0.shape[-1])
    for arm, y in [(Arm(kind="rescale", a=a, b=b, alpha=1.3), x0 + 1.3 * (g0 - x0)),
                   (Arm(kind="norm_const", a=a, b=b, c=0.9), 0.9 * g0),
                   (Arm(kind="mean_shift", a=a, b=b, s=0.5, delta_bar=dbar), g0 + 0.5 * dbar),
                   (Arm(kind="repeat", a=a, b=b), _run_window(model, a, b, ids, g0))]:
        ref = _logits_with_window_output(model, b, ids, y)
        assert (_wrapped_logits(model, arm, ids) - ref).abs().max() < TOL, arm.kind


@pytest.mark.parametrize("case", list(CASES))
def test_sham_perm_donors(case):
    model, a, b, T = CASES[case]
    ids = _ids(T, B=3)
    layers, cap = decoder_layers(model), {}

    def grab(key, arm):
        def post(m, args, kw, o):
            cap[key] = o

        h = layers[b].register_forward_hook(post, with_kwargs=True)
        try:
            with torch.no_grad(), LoopWindow(model, arm):
                model(ids)
        finally:
            h.remove()

    _, g0 = _window_io(model, a, b, ids)
    grab("loop", Arm(kind="loop", a=a, b=b, K=2))
    grab("sham", Arm(kind="sham_perm", a=a, b=b, K=2, sigma=1.0))
    d_loop, d_sham = cap["loop"] - g0, cap["sham"] - g0
    # position 0: the loop's own change
    assert (d_sham[:, 0] - d_loop[:, 0]).abs().max() < TOL
    # size from the recipient, direction from the previous sequence at the same position
    assert torch.allclose(d_sham[:, 1:].norm(dim=-1), d_loop[:, 1:].norm(dim=-1), atol=1e-4, rtol=1e-4)
    cos = torch.nn.functional.cosine_similarity
    donor = d_loop.roll(1, dims=0)
    assert (cos(d_sham[:, 1:], donor[:, 1:], dim=-1) > 0.999).all()
    assert (cos(d_sham[:, 1:], d_loop[:, 1:], dim=-1) < 0.99).all(), "a donor must never be the recipient itself"


@pytest.mark.parametrize("case", list(CASES))
def test_diagnostics_record(case):
    from loop_wrap import summarize
    model, a, b, T = CASES[case]
    ids = _ids(T)
    x0, g0 = _window_io(model, a, b, ids)
    rec = {"contraction": 4}
    layers, cap = decoder_layers(model), {}

    def post(m, args, kw, o):
        cap["y"] = o

    h = layers[b].register_forward_hook(post, with_kwargs=True)
    try:
        with torch.no_grad(), LoopWindow(model, Arm(kind="loop", a=a, b=b, K=2), rec):
            model(ids)
    finally:
        h.remove()
    s = summarize(rec)
    d, u = (cap["y"] - g0)[:, 1:], (g0 - x0)[:, 1:]
    assert s["n_positions"] == d.shape[0] * d.shape[1]
    assert abs(s["d_over_u"] - float((d.norm(dim=-1) / u.norm(dim=-1)).mean())) < 1e-4
    assert torch.allclose(rec["delta_sum"].float() / rec["n"], d.mean((0, 1)), atol=1e-5)
    for mode in ("undamped", "damped"):
        c = s[f"contraction_{mode}"]
        assert len(c) == 4 and abs(c[0] - 1.0) < 1e-5  # k = 0 is ||u|| / ||u||
    # the second undamped step is ||g(g0) - g0|| / ||u||
    g1 = _run_window(model, a, b, ids, g0)
    ref = float(((g1 - g0)[:, 1:].norm(dim=-1) / u.norm(dim=-1)).mean())
    assert abs(s["contraction_undamped"][1] - ref) < 1e-4


def test_extra_layers():
    assert Arm(kind="loop", a=12, b=15, K=2).extra_layers == 4
    assert Arm(kind="loop", a=15, b=18, K=3, beta=0.5).extra_layers == 8
    assert Arm(kind="repeat", a=12, b=15).extra_layers == 4
    assert Arm(kind="rescale", a=12, b=15).extra_layers == 0
