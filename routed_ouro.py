"""Stage 2: Ouro with a TRAINED per-token, per-layer router, fine-tuned jointly (LoRA) so the model learns
to use the paths.

Ouro's 24 layers x 4 rounds are unrolled into 96 virtual layers. Before each virtual layer j (except the
first and last `protect`), a linear head on the token's RMS-normalised state picks one action per token:
  0 skip   the layer is not applied (cost -1)
  1 once   the layer is applied (cost 0)
  2 twice  the layer is applied again to its own output (cost +1); the second application attends to
           every token's first-application output (a standard looped layer)
Chained over 96 virtual layers this expresses per-token paths that loop, skip and revisit layers.
Training uses straight-through Gumbel-softmax: the forward pass is HARD (exactly the eval-time path),
gradients flow through the soft probabilities. A Lagrange multiplier holds the mean layer applications per
token at --target_layers. All tokens compute both applications, so other tokens always provide K/V at
every step; the reported compute counts only the applications each token's path uses (the same accounting
as OuroForCausalLM's exit_threshold and bench.py's gate policy).

Variants (same data, steps, LoRA):
  router   LoRA + router (the method)
  none     LoRA only, every token runs the base path (control)

  python routed_ouro.py --variant router --target_layers 72 --out r5/router72
  python routed_ouro.py --variant none --out r5/none
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
from torch.utils.checkpoint import checkpoint

from ouro_model import _rot, load_ouro


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.t() @ self.B.t()).to(x.dtype) * self.s

    def merged(self):
        return (self.base.weight.float() + self.s * self.B.float() @ self.A.float()).to(self.base.weight.dtype)


def add_lora(model, r=16, alpha=32):
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in names:
                if hasattr(parent, n) and isinstance(getattr(parent, n), nn.Linear):
                    setattr(parent, n, LoRALinear(getattr(parent, n), r, alpha))


def merge_lora(model):
    """Replace every LoRALinear by a plain Linear with merged weights (in place)."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n, m in list(parent.named_children()):
                if isinstance(m, LoRALinear):
                    lin = m.base
                    lin.weight.data = m.merged()
                    setattr(parent, n, lin)


class RoutedOuro(nn.Module):
    def __init__(self, ouro, protect=2, use_router=True):
        super().__init__()
        self.ouro = ouro
        c = ouro.config
        self.n, self.R = c.num_hidden_layers, c.total_ut_steps
        self.V = self.n * self.R
        self.protect = protect
        self.use_router = use_router
        d = c.hidden_size
        self.heads = nn.Linear(d, 3 * self.V)  # one 3-way head per virtual layer, batched
        nn.init.zeros_(self.heads.weight)
        with torch.no_grad():
            self.heads.bias.view(self.V, 3).copy_(torch.tensor([-2.0, 2.0, -2.0]))  # ~97% once at tau=1: near the base path

    def routable(self, j):
        return self.use_router and self.protect <= j < self.V - self.protect

    def _layer(self, layer, x, cos, sin):
        c = self.ouro.config
        B, T, _ = x.shape
        a = layer.self_attn
        h = layer.input_layernorm(x)
        sh = lambda t: t.view(B, T, -1, c.head_dim).transpose(1, 2)  # noqa: E731
        q, k, v = sh(a.q_proj(h)), sh(a.k_proj(h)), sh(a.v_proj(h))
        cu, su = cos[:, None], sin[:, None]
        q, k = q * cu + _rot(q) * su, k * cu + _rot(k) * su
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        h = x + layer.input_layernorm_2(a.o_proj(o))
        return h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))

    def forward(self, ids, tau=1.0, hard_eval=False, ckpt=False):
        """Returns logits (B, T, V), per-token layer applications (B, T), soft expected cost (B, T),
        and action counts (3,)."""
        m = self.ouro.model
        x = m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        cos, sin = m.rotary_emb(x, pos)
        B, T = ids.shape
        layers_run = torch.zeros(B, T, device=ids.device)
        exp_cost = torch.zeros(B, T, device=ids.device)
        counts = torch.zeros(3, device=ids.device)
        for j in range(self.V):
            if j > 0 and j % self.n == 0:
                x = m.norm(x)  # Ouro's between-round norm
            layer = m.layers[j % self.n]
            run = (lambda y, L_=layer: self._layer(L_, y, cos, sin))
            f1 = checkpoint(run, x, use_reentrant=False) if ckpt else run(x)
            if not self.routable(j):
                x = f1
                layers_run += 1
                continue
            xs = x.to(self.heads.weight.dtype)
            logit = self.heads(xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + 1e-6))[..., 3 * j: 3 * j + 3]
            if hard_eval:
                w = F.one_hot(logit.argmax(-1), 3).float()
                soft = w
            else:
                g = -torch.log(-torch.log(torch.rand_like(logit).clamp(1e-9, 1 - 1e-9)))
                soft = F.softmax((logit + g) / tau, -1)
                w = F.one_hot(soft.argmax(-1), 3).float() - soft.detach() + soft
            if bool((w[..., 2] > 0).any()) or not hard_eval:
                f2 = checkpoint(run, f1, use_reentrant=False) if ckpt else run(f1)
            else:
                f2 = f1
            wd = w.to(x.dtype)
            x = wd[..., 0:1] * x + wd[..., 1:2] * f1 + wd[..., 2:3] * f2
            act = w.detach().argmax(-1)
            layers_run += act.float()  # skip 0, once 1, twice 2
            counts += F.one_hot(act, 3).float().sum((0, 1))
            p = F.softmax(logit.float(), -1)
            exp_cost += p[..., 2] - p[..., 0]
        x = m.norm(x)
        return self.ouro.lm_head(x), layers_run, exp_cost, counts


def fineweb_chunks(tok, seq_len, n_chunks, skip_docs=0):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    buf, out = [], []
    for i, r in enumerate(ds):
        if i < skip_docs:
            continue
        buf.extend(tok(r["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(buf) >= seq_len - 1:
            out.append([tok.bos_token_id] + buf[: seq_len - 1])
            buf = buf[seq_len - 1:]
            if len(out) >= n_chunks:
                return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--variant", choices=["router", "none"], required=True)
    p.add_argument("--target_layers", type=float, default=72.0, help="mean layer applications per token")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=1e-2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lag_lr", type=float, default=0.2, help="Lagrange multiplier step (per layer of excess)")
    p.add_argument("--tau", type=float, nargs=2, default=[1.0, 0.3])
    p.add_argument("--eval_chunks", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).cuda()
    for p_ in ouro.parameters():
        p_.requires_grad_(False)
    add_lora(ouro, a.rank, 2 * a.rank)
    ouro.cuda()
    model = RoutedOuro(ouro, use_router=a.variant == "router").cuda()
    model.heads.float()
    lora = [p_ for n_, p_ in model.named_parameters() if n_.endswith((".A", ".B"))]
    for p_ in lora:
        p_.data = p_.data.float()
        p_.requires_grad_(True)
    groups = [{"params": lora, "lr": a.lr}]
    if a.variant == "router":
        groups.append({"params": list(model.heads.parameters()), "lr": a.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))

    n_train = a.steps * a.batch
    print(f"loading {n_train + a.eval_chunks} fineweb-edu chunks of {a.seq_len}", flush=True)
    chunks = fineweb_chunks(tok, a.seq_len, n_train + a.eval_chunks)
    ev, tr = chunks[: a.eval_chunks], chunks[a.eval_chunks:]
    print(f"train {len(tr)} chunks, eval {len(ev)}", flush=True)
    lag = 0.0
    log = {"args": vars(a), "steps": []}
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        ids = torch.tensor(tr[step * a.batch: (step + 1) * a.batch], device="cuda")
        tau = a.tau[0] * (a.tau[1] / a.tau[0]) ** (step / max(1, a.steps - 1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, layers_run, exp_cost, counts = model(ids, tau=tau, ckpt=True)
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss = ce
        if a.variant == "router":
            excess = (model.V + exp_cost.mean()) - a.target_layers  # expected layers - target
            loss = loss + lag * excess
            lag = max(0.0, lag + a.lag_lr * float(layers_run.mean() - a.target_layers) / model.V)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g_ in groups for q in g_["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == a.steps - 1:
            c = (counts / counts.sum().clamp_min(1)).tolist()
            rec = dict(step=step, ce=ce.item(), layers=layers_run.mean().item(), lag=lag, tau=tau,
                       skip=c[0], once=c[1], twice=c[2], min=(time.time() - t0) / 60)
            log["steps"].append(rec)
            print(json.dumps(rec), flush=True)
    model.eval()
    with torch.no_grad():
        tot, n, lay = 0.0, 0, 0.0
        for i in range(0, len(ev), a.batch):
            ids = torch.tensor(ev[i: i + a.batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, layers_run, _, _ = model(ids, hard_eval=True)
            nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                  reduction="sum")
            tot, n, lay = tot + nll.item(), n + ids[:, 1:].numel(), lay + layers_run.sum().item()
    log["eval"] = dict(nll=tot / n, layers=lay / (n + len(ev)))
    print("EVAL", json.dumps(log["eval"]), flush=True)
    merge_lora(ouro)
    torch.save({"ouro": {k: v.to(torch.bfloat16) for k, v in ouro.state_dict().items()},
                "heads": model.heads.state_dict(), "use_router": a.variant == "router",
                "config": vars(ouro.config)}, os.path.join(a.out, "model.pt"))
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
