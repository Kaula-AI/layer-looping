"""Structured per-token router for Ouro, trained on the exact expected loss (Ouro frozen).

Why (diagnose_router.py on the per-layer router): 92 independent skip/once/twice heads trained with
straight-through Gumbel gave LESS compute to hard tokens and never found the cheapest saving (dropping the
whole last round). So the actions here are structured and few, decided at round boundaries:

  after round r (r = 1..R) a token either stops ("plain") or stops after re-running the second half of
  round r once more ("extra", +n/2 layers). Stopping at round r costs n*r layers.

The router is Ouro's own halting form: per round r a stop probability lambda_r (exit pdf as in
OuroForCausalLM) and an extra probability e_r. With Ouro frozen, every option's per-token loss is a fixed
number, so the objective is exact and linear in the option probabilities:

  L = mean_tokens  sum_r  pdf_r * [ (1 - e_r) * (CE_r + lam * n*r) + e_r * (CE_r^extra + lam * (n*r + n/2)) ]

No sampling, no straight-through. Heads are warm-started from Ouro's early-exit gate (extra off), so the
starting policy IS Ouro's gate. One frozen forward per batch trains several head sets at once:
  sr_lam{X}    stop + extra heads, compute price lam = X nats per layer
  gate_lam{X}  stop heads only (control: same training, no extra action)

Eval (held-out FineWeb-Edu): a threshold q on the cumulative stop pdf picks the round (Ouro's rule) and
e_r > 0.5 adds the extra half-round; sweeping q gives a loss-vs-layers curve for each head set and for
Ouro's original gate.

  python structured_router.py train --out sr
  python structured_router.py eval --heads sr/heads.pt --out sr
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ouro_model import load_ouro
from routed_ouro import RoutedOuro, fineweb_chunks


# ---------------------------------------------------------------- frozen Ouro states
@torch.no_grad()
def round_states(ouro, ids, half=None, sdpa=None):
    """Raw (pre-norm) state after each round, H[r] (B, T, d), and the 'extra' state X[r]: round r's output
    run through the second half of the layers once more. Whole-sequence causal attention (every token
    computes every round, as in OuroForCausalLM's exit evaluation)."""
    m, c = ouro.model, ouro.config
    n, R = c.num_hidden_layers, c.total_ut_steps
    half = n // 2 if half is None else half
    sdpa = sdpa or RoutedOuro(ouro, use_router=False)
    x = m.embed_tokens(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
    cos, sin = m.rotary_emb(x, pos)
    H, X = [], []
    for r in range(R):
        if r > 0:
            x = m.norm(x)
        for l in range(n):
            x = sdpa._layer(m.layers[l], x, cos, sin)
        H.append(x)
        y = x
        for l in range(n - half, n):
            y = sdpa._layer(m.layers[l], y, cos, sin)
        X.append(y)
    return H, X


@torch.no_grad()
def option_ce(ouro, H, X, ids):
    """Per-token next-token CE for every option: (B, T-1, R, 2) with [..., 0] plain and [..., 1] extra."""
    out = []
    for h, x in zip(H, X):
        pair = []
        for s in (h, x):
            lg = ouro.lm_head(ouro.model.norm(s))[:, :-1].float()
            pair.append(F.cross_entropy(lg.transpose(1, 2), ids[:, 1:], reduction="none"))
        out.append(torch.stack(pair, -1))
    return torch.stack(out, 2)


# ---------------------------------------------------------------- heads
class Heads(nn.Module):
    """Stop head (Ouro's gate form, shared weights + per-round bias) and extra head."""

    def __init__(self, gate, R, extra=True):
        super().__init__()
        d = gate.in_features
        self.R, self.use_extra = R, extra
        self.stop = nn.Linear(d, 1)
        self.stop.load_state_dict({k: v.float() for k, v in gate.state_dict().items()})
        self.stop_b = nn.Parameter(torch.zeros(R))
        self.extra = nn.Linear(d, 1)
        nn.init.zeros_(self.extra.weight)
        nn.init.zeros_(self.extra.bias)
        self.extra_b = nn.Parameter(torch.full((R,), -6.0))  # off at the start: the policy is Ouro's gate

    def forward(self, Hn):
        """Hn: list of R normed states (B, T, d). Returns exit pdf (B, T, R) and extra prob (B, T, R)."""
        lam = [torch.sigmoid(self.stop(h.to(self.stop.weight.dtype))[..., 0] + self.stop_b[r]) for r, h in enumerate(Hn)]
        rem, pdf = torch.ones_like(lam[0]), []
        for r, l_ in enumerate(lam):
            p_ = l_ * rem if r < self.R - 1 else rem
            rem = rem * (1 - l_)
            pdf.append(p_)
        pdf = torch.stack(pdf, -1)
        if not self.use_extra:
            return pdf, torch.zeros_like(pdf)
        e = torch.stack([torch.sigmoid(self.extra(h.to(self.extra.weight.dtype))[..., 0] + self.extra_b[r]) for r, h in enumerate(Hn)], -1)
        return pdf, e


def option_costs(n, R, half, device):
    plain = torch.tensor([n * (r + 1) for r in range(R)], device=device, dtype=torch.float32)
    return plain, plain + half


def expected(pdf, e, ce, lam, costs):
    """Expected per-token objective and expected layers. pdf, e: (B, T, R); ce: (B, T, R, 2)."""
    plain, extra = costs
    w0, w1 = pdf * (1 - e), pdf * e
    obj = w0 * (ce[..., 0] + lam * plain) + w1 * (ce[..., 1] + lam * extra)
    lay = w0 * plain + w1 * extra
    return obj.sum(-1), lay.sum(-1)


def decide(pdf, e, q):
    """Deterministic policy: first round with cumulative stop prob >= q (Ouro's rule); extra if e > 0.5.
    Returns round index (0-based) and extra flag, each (B, T)."""
    cum = pdf.cumsum(-1)
    hit = cum >= q
    r = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(hit[..., 0], pdf.shape[-1] - 1, dtype=torch.long))
    ex = e.gather(-1, r[..., None])[..., 0] > 0.5
    return r, ex


# ---------------------------------------------------------------- train
def train(a):
    torch.manual_seed(a.seed)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).cuda().eval()
    for p_ in ouro.parameters():
        p_.requires_grad_(False)
    sdpa = RoutedOuro(ouro, use_router=False).cuda()
    c = ouro.config
    n, R = c.num_hidden_layers, c.total_ut_steps
    half = n // 2
    costs = option_costs(n, R, half, "cuda")
    sets = {}
    for lam in a.lams:
        for kind in ("sr", "gate"):
            h = Heads(ouro.model.early_exit_gate, R, extra=kind == "sr").cuda()
            sets[f"{kind}_lam{lam:g}"] = (h, lam, torch.optim.AdamW(
                [{"params": [h.stop.weight, h.extra.weight], "lr": a.w_lr},
                 {"params": [h.stop.bias, h.stop_b, h.extra.bias, h.extra_b], "lr": a.b_lr}], weight_decay=0.0))
    # held-out: the first 64 chunks are the evaluation set everywhere in this repo
    chunks = fineweb_chunks(tok, a.seq_len, 64 + a.steps * a.batch)[64:]
    print(f"{len(chunks)} training chunks; head sets: {list(sets)}", flush=True)
    log, t0 = {"args": vars(a), "steps": []}, time.time()
    for step in range(a.steps):
        ids = torch.tensor(chunks[step * a.batch:(step + 1) * a.batch], device="cuda")
        H, X = round_states(ouro, ids, half, sdpa)
        ce = option_ce(ouro, H, X, ids)
        Hn = [ouro.model.norm(h)[:, :-1] for h in H]
        rec = {"step": step}
        for name, (h, lam, opt) in sets.items():
            pdf, e = h(Hn)
            obj, lay = expected(pdf, e, ce, lam, costs)
            obj.mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            rec[name] = (round(obj.mean().item(), 4), round(lay.mean().item(), 2), round(e.mean().item(), 4))
        if step % 50 == 0 or step == a.steps - 1:
            rec["min"] = round((time.time() - t0) / 60, 1)
            log["steps"].append(rec)
            print(json.dumps(rec), flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({k: v[0].state_dict() for k, v in sets.items()}, os.path.join(a.out, "heads.pt"))
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).cuda().eval()
    sdpa = RoutedOuro(ouro, use_router=False).cuda()
    c = ouro.config
    n, R = c.num_hidden_layers, c.total_ut_steps
    half = n // 2
    costs = option_costs(n, R, half, "cuda")
    heads = {"ouro_gate": Heads(ouro.model.early_exit_gate, R, extra=False).cuda()}
    for k, sd in torch.load(a.heads, map_location="cuda").items():
        h = Heads(ouro.model.early_exit_gate, R, extra=k.startswith("sr")).cuda()
        h.load_state_dict(sd)
        heads[k] = h
    chunks = fineweb_chunks(tok, a.seq_len, a.n_chunks)  # held out (first 64)
    qs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    acc = {k: {q: [0.0, 0.0, 0.0, 0] for q in qs} for k in heads}  # nll sum, layers sum, extra count, n
    fixed = np.zeros(R)
    ntok = 0
    for i in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[i:i + a.batch], device="cuda")
        H, X = round_states(ouro, ids, half, sdpa)
        ce = option_ce(ouro, H, X, ids)
        Hn = [ouro.model.norm(h)[:, :-1] for h in H]
        fixed += ce[..., 0].sum((0, 1)).cpu().numpy()
        ntok += ce.shape[0] * ce.shape[1]
        for k, h in heads.items():
            pdf, e = h(Hn)
            for q in qs:
                r, ex = decide(pdf, e, q)
                nll = ce[..., 0].gather(-1, r[..., None])[..., 0].where(~ex, ce[..., 1].gather(-1, r[..., None])[..., 0])
                lay = torch.where(ex, costs[1][r], costs[0][r])
                s = acc[k][q]
                s[0] += nll.sum().item()
                s[1] += lay.sum().item()
                s[2] += ex.sum().item()
                s[3] += nll.numel()
    out = {"fixed_rounds": [dict(layers=n * (r + 1), nll=fixed[r] / ntok) for r in range(R)], "curves": {}}
    for k in heads:
        out["curves"][k] = [dict(q=q, layers=v[1] / v[3], nll=v[0] / v[3], extra_frac=v[2] / v[3])
                            for q, v in acc[k].items()]
    # loss at matched compute: interpolate each curve at fixed layer budgets
    budgets = [55, 60, 65, 70, 75, 80, 85, 90]
    table = {}
    for k, cur in out["curves"].items():
        pts = sorted({(round(p["layers"], 3), p["nll"]) for p in cur})
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        table[k] = [float(np.interp(b, xs, ys)) if xs[0] <= b <= xs[-1] else None for b in budgets]
    out["at_budget"] = dict(budgets=budgets, nll=table)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    print("fixed rounds:", " ".join(f"{d['layers']}:{d['nll']:.4f}" for d in out["fixed_rounds"]))
    print(f"\n{'head set':16s}" + "".join(f"{b:>9d}" for b in budgets) + "   (held-out NLL at layers/token)")
    for k, row in table.items():
        print(f"{k:16s}" + "".join(f"{'—':>9s}" if v is None else f"{v:9.4f}" for v in row))
    for k, cur in out["curves"].items():
        print(k, "extra used:", " ".join(f"q{p['q']}:{100 * p['extra_frac']:.1f}%" for p in cur if p["extra_frac"] > 0) or "never")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--lams", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--w_lr", type=float, default=1e-5)
    p.add_argument("--b_lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", default=None)
    p.add_argument("--n_chunks", type=int, default=64)
    p.add_argument("--out", default="sr")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
