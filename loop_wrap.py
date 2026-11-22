"""E14: training-free looping of a layer window in a frozen HuggingFace decoder (DESIGN.md §3).

`LoopWindow(model, arm)` is a context manager that changes what layers a..b compute, for every token, in any
forward pass of the model (our NLL scripts and lm-evaluation-harness alike). It works by wrapping the window's
layer forwards: the model's own loop calls layers a..b once as usual (that is g(x0), with the model's own masks,
rotary tables, per-layer inputs and K/V sharing), the wrapper records each layer's call arguments, and at layer b
it replaces the output by the arm's output. Extra sub-steps re-call the original layer forwards with the recorded
arguments, with `past_key_values=None`, so they never append to the model's cache and attend to the current
sub-step state of every position (TFLT's prefill semantics, `use_cache=False` in the loop body).

Arms (x0 = window input, g0 = g(x0), u = g0 - x0; delta = loop output - g0):
  base         g0
  loop         beta * g0 + (1 - beta) * F^K(x0),  F(x) = x + (g(x) - x) / K   (damped Euler, TFLT Eq. 10/16)
  rescale      x0 + alpha * u                                    (zero extra compute)
  norm_const   c * g0                                            (zero extra compute)
  mean_shift   g0 + s * delta_bar                                (zero extra compute; delta_bar from tuning data)
  repeat       g(g0)                                             (full-strength repeat, a known mover)
  sham_perm    g0 + sigma * ||delta_t|| * delta_donor / ||delta_donor||, donor = same position of the previous
               sequence in the batch (never the recipient's own sequence); position 0 keeps its own delta
  sham_iso     g0 + sigma * ||delta_t|| * random unit vector; position 0 keeps its own delta   (diagnostic)
  norm_match   g0 * ||loop output|| / ||g0|| per token                                         (diagnostic)

Euler steps and blends run in float32; the layers run in the model's dtype. Gemma-4: if the window contains a K/V
source layer (13 or 14 on E2B), its re-calls overwrite `shared_kv_states`, so later shared layers read the K/V of
the LAST sub-step (DESIGN.md §3.2).
"""
from dataclasses import dataclass, field

import torch
from torch import nn

LOOP_KINDS = ("loop", "sham_perm", "sham_iso", "norm_match")
KINDS = ("base", "rescale", "norm_const", "mean_shift", "repeat") + LOOP_KINDS


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def decoder_layers(model):
    """The language model's decoder ModuleList (Qwen3: model.layers; Gemma-4: model.layers or language_model.layers)."""
    n = text_config(model).num_hidden_layers
    cands = [(name, m) for name, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and name.split(".")[-1] == "layers" and len(m) == n]
    assert cands, "no decoder layer list found"
    cands.sort(key=lambda t: (not (t[0].endswith("language_model.layers") or t[0] == "model.layers"), len(t[0])))
    return cands[0][1]


@dataclass
class Arm:
    kind: str = "base"
    a: int = 0            # first layer of the window (0-based HF index)
    b: int = 0            # last layer, inclusive
    K: int = 2            # sub-steps (number of evaluations of g)
    beta: float = 0.0     # anchor weight on the normal pass
    alpha: float = 1.0    # rescale
    c: float = 1.0        # norm_const
    s: float = 1.0        # mean_shift
    sigma: float = 1.0    # shams
    delta_bar: torch.Tensor | None = field(default=None, repr=False)
    seed: int = 0         # sham_iso directions
    skip_pos0: bool = False  # ablation: output g0 at position 0

    def __post_init__(self):
        assert self.kind in KINDS, self.kind
        assert 0 <= self.a <= self.b and self.K >= 1

    @property
    def width(self):
        return self.b - self.a + 1

    @property
    def extra_layers(self):
        """Extra layer applications per token (g0 is the first sub-step, so beta is free)."""
        if self.kind in LOOP_KINDS:
            return self.width * (self.K - 1)
        return self.width if self.kind == "repeat" else 0

    def name(self):
        w = f"[{self.a}-{self.b}]"
        return {"base": "base", "loop": f"loop{w}K{self.K}b{self.beta:g}", "rescale": f"rescale{w}a{self.alpha:g}",
                "norm_const": f"normc{w}c{self.c:g}", "mean_shift": f"mshift{w}s{self.s:g}", "repeat": f"repeat{w}",
                "sham_perm": f"shamperm{w}K{self.K}b{self.beta:g}s{self.sigma:g}",
                "sham_iso": f"shamiso{w}K{self.K}b{self.beta:g}s{self.sigma:g}",
                "norm_match": f"normmatch{w}K{self.K}b{self.beta:g}"}[self.kind]


class LoopWindow:
    """Context manager: `with LoopWindow(model, arm, record): model(ids)`.

    `record` (optional dict) accumulates diagnostics over positions >= 1 (position 0 is the attention sink):
    sums for ||delta||/||u||, ||loop out||/||g0||, the mean delta (for mean_shift), and, if record["contraction"]
    is an int k_max, ||g(x_k) - x_k|| / ||u|| for k < k_max along the undamped (x <- g(x)) and damped
    (x <- x + (g(x) - x) / 2) iterations."""

    def __init__(self, model, arm, record=None):
        self.layers = decoder_layers(model)
        self.arm, self.record = arm, record
        assert arm.b < len(self.layers)
        self.idx = list(range(arm.a, arm.b + 1))
        self._saved = {}

    # ------------------------------------------------------------ install / restore
    def __enter__(self):
        for j, i in enumerate(self.idx):
            layer = self.layers[i]
            self._saved[i] = layer.__dict__.get("forward")
            orig = type(layer).forward.__get__(layer)
            layer.forward = self._wrap(j, orig)
        self.orig = {i: type(self.layers[i]).forward.__get__(self.layers[i]) for i in self.idx}
        return self

    def __exit__(self, *exc):
        for i in self.idx:
            layer = self.layers[i]
            if self._saved[i] is None:
                del layer.forward
            else:
                layer.forward = self._saved[i]
        return False

    def _wrap(self, j, orig):
        last = len(self.idx) - 1

        def forward(hidden_states, *args, **kwargs):
            if j == 0:
                self._x0, self._calls = hidden_states, []
            self._calls.append((args, kwargs))
            out = orig(hidden_states, *args, **kwargs)
            assert torch.is_tensor(out), "decoder layer must return a tensor (transformers >= 5)"
            if j == last:
                assert len(self._calls) == len(self.idx), "window layers must be called once each, in order"
                out = self._apply(self._x0, out)
            return out

        return forward

    # ------------------------------------------------------------ the window map, re-called
    def _g(self, x):
        """g(x): layers a..b on state x with the recorded masks/rotary/per-layer inputs, no cache."""
        for i, (args, kw) in zip(self.idx, self._calls):
            kw = dict(kw)
            kw["past_key_values"] = None
            if "use_cache" in kw:
                kw["use_cache"] = False
            x = self.orig[i](x, *args, **kw)
        return x

    def _loop(self, x0f, g0f, dt):
        arm, x = self.arm, x0f
        for k in range(arm.K):
            gx = g0f if k == 0 else self._g(x.to(dt)).float()
            x = x + (gx - x) / arm.K
        return arm.beta * g0f + (1.0 - arm.beta) * x

    # ------------------------------------------------------------ arms
    @torch.no_grad()
    def _apply(self, x0, g0):
        arm, dt = self.arm, g0.dtype
        x0f, g0f = x0.float(), g0.float()
        k = arm.kind
        if k == "base":
            return g0
        if k == "rescale":
            out = x0f + arm.alpha * (g0f - x0f)
        elif k == "norm_const":
            out = arm.c * g0f
        elif k == "mean_shift":
            out = g0f + arm.s * arm.delta_bar.to(g0f)
        elif k == "repeat":
            out = self._g(g0).float()
        else:
            xl = self._loop(x0f, g0f, dt)
            delta = xl - g0f
            if self.record is not None:
                self._diagnostics(x0f, g0f, xl, delta, dt)
            if k == "loop":
                out = xl
            elif k == "norm_match":
                out = g0f * (xl.norm(dim=-1, keepdim=True) / g0f.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            else:
                size = delta.norm(dim=-1, keepdim=True)
                if k == "sham_perm":
                    assert delta.shape[0] >= 2, "sham_perm needs at least two sequences per batch"
                    donor = delta.roll(1, dims=0)  # same position, previous sequence in the batch
                    unit = donor / donor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                else:  # sham_iso
                    gen = torch.Generator(device="cpu").manual_seed(arm.seed)
                    r = torch.randn(delta.shape, generator=gen).to(delta)
                    unit = r / r.norm(dim=-1, keepdim=True)
                sham = arm.sigma * size * unit
                sham[:, 0] = delta[:, 0]  # the attention sink gets the loop's own change in every sham
                out = g0f + sham
        if arm.skip_pos0:
            out = out.clone()
            out[:, 0] = g0f[:, 0]
        return out.to(dt)

    def _diagnostics(self, x0f, g0f, xl, delta, dt):
        r = self.record
        u = (g0f - x0f)[:, 1:]
        d, xl1, g1 = delta[:, 1:], xl[:, 1:], g0f[:, 1:]
        un = u.norm(dim=-1).clamp_min(1e-12)
        r["n"] = r.get("n", 0) + d.shape[0] * d.shape[1]
        r["sum_d_over_u"] = r.get("sum_d_over_u", 0.0) + float((d.norm(dim=-1) / un).sum())
        r["sum_cos_d_u"] = r.get("sum_cos_d_u", 0.0) + float(
            ((d * u).sum(-1) / (d.norm(dim=-1).clamp_min(1e-12) * un)).sum())
        r["sum_loopnorm_over_g0norm"] = r.get("sum_loopnorm_over_g0norm", 0.0) + float(
            (xl1.norm(dim=-1) / g1.norm(dim=-1).clamp_min(1e-12)).sum())
        r["delta_sum"] = r.get("delta_sum", 0.0) + d.sum((0, 1)).cpu().double()  # MPS has no float64
        kmax = r.get("contraction")
        if kmax:
            for mode in ("undamped", "damped"):
                x, acc = x0f, r.setdefault(f"contraction_{mode}", [0.0] * kmax)
                for kk in range(kmax):
                    gx = g0f if kk == 0 else self._g(x.to(dt)).float()
                    acc[kk] += float(((gx - x)[:, 1:].norm(dim=-1) / un).sum())
                    x = gx if mode == "undamped" else x + 0.5 * (gx - x)


def summarize(record):
    """Means from a LoopWindow record."""
    n = record["n"]
    out = {"n_positions": n, "d_over_u": record["sum_d_over_u"] / n, "cos_d_u": record["sum_cos_d_u"] / n,
           "loopnorm_over_g0norm": record["sum_loopnorm_over_g0norm"] / n}
    for mode in ("undamped", "damped"):
        if f"contraction_{mode}" in record:
            out[f"contraction_{mode}"] = [v / n for v in record[f"contraction_{mode}"]]
    return out
