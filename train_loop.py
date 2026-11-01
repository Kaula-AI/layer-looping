"""Round 2: train the model to use a looped block (LoRA), against a LoRA control with no loop.

Variants (same data, steps, seed, LoRA config):
  none    LoRA only (control).
  static  LoRA + layers s..e run twice for every token: h = h1 + alpha * (h2 - h1), alpha init 0
          (starts exactly at the base model). Extra compute: (e - s + 1) layers per token.
  gated   as static, but a per-token gate g = sigmoid(w . h1 + b) replaces alpha, with a penalty
          lam * mean(g). Eval uses a hard gate (loop only where g > 0.5), so the reported compute is real.

The second pass attends to second-pass K/V of all positions (a standard looped transformer). For `gated`
the hard eval keeps that, so non-looping tokens still provide K/V in pass 2; this is reported, not hidden.
Block must lie below the first K/V-source layer (13 on E2B) so K/V sharing is untouched.

  python train_loop.py --variant none   --out r2/none
  python train_loop.py --variant static --out r2/static
  python train_loop.py --variant gated  --out r2/gated --gate_lam 0.02
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E2B")
    p.add_argument("--variant", choices=["none", "static", "gated", "base"], required=True)
    p.add_argument("--block", default="5-8")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--gate_lam", type=float, default=0.02)
    p.add_argument("--loop_lr", type=float, default=None, help="lr for alpha/gate/pass-2 adapters (default: --lr)")
    p.add_argument("--loop_adapter", type=int, default=0,
                   help="rank of a pass-2-only residual adapter after each looped layer (0 = none; up-proj zero-init)")
    p.add_argument("--eval_tokens", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


# ------------------------------------------------------------------ data
def chunks_from(tok, texts, seq_len, n_tokens):
    ids, out = [], []
    for t in texts:
        if not t.strip():
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
        while len(ids) >= seq_len - 1:
            out.append([tok.bos_token_id] + ids[: seq_len - 1])
            ids = ids[seq_len - 1:]
            if len(out) * seq_len >= n_tokens:
                return out
    return out


def gsm_text(split):
    return (f"Question: {r['question']}\nAnswer: {r['answer']}\n" for r in load_dataset("openai/gsm8k", "main", split=split))


def wiki_text(split):
    return (r["text"] for r in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split))


def mixed(tok, split_w, split_g, seq_len, n_tokens):
    w = chunks_from(tok, wiki_text(split_w), seq_len, n_tokens // 2)
    g = chunks_from(tok, gsm_text(split_g), seq_len, n_tokens // 2)
    return w, g


# ------------------------------------------------------------------ loop wrapper
class Loop(nn.Module):
    """Wraps layers s..e. Layers s..e-1 record their kwargs; layer e reruns s..e and mixes."""

    def __init__(self, layers, s, e, variant, hidden, adapter_rank=0):
        super().__init__()
        self.adapters = nn.ModuleList()
        for _ in range(e - s + 1 if adapter_rank else 0):
            ad = nn.Sequential(nn.Linear(hidden, adapter_rank, bias=False), nn.Linear(adapter_rank, hidden, bias=False))
            nn.init.zeros_(ad[1].weight)
            self.adapters.append(ad)
        self.layers, self.s, self.e, self.variant = layers, s, e, variant
        self.kw = {}
        self.h_in = None
        self.alpha = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.hard = False
        self.last_g = None


class Rec(nn.Module):
    def __init__(self, inner, idx, loop):
        super().__init__()
        self.inner, self.idx = inner, idx
        object.__setattr__(self, "loop", loop)  # not a submodule (avoid double registration)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(self, hidden_states, *args, **kwargs):
        lp = self.loop
        lp.kw[self.idx] = (args, kwargs)
        h1 = self.inner(hidden_states, *args, **kwargs)
        if self.idx != lp.e or lp.variant in ("none", "base"):
            return h1
        h2 = h1
        for i in range(lp.s, lp.e + 1):
            a_, k_ = lp.kw[i]
            h2 = lp.layers[i].inner(h2, *a_, **k_)
            if len(lp.adapters):
                h2 = h2 + lp.adapters[i - lp.s](h2.float()).to(h2.dtype)
        if lp.variant == "static":
            return h1 + lp.alpha * (h2 - h1)
        g = torch.sigmoid(lp.gate(h1.float())).to(h1.dtype)  # B, T, 1
        lp.last_g = g
        if lp.hard:
            g = (g > 0.5).to(h1.dtype)
        return h1 + g * (h2 - h1)


def find_layers(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and name.endswith("layers") and "vision" not in name and "audio" not in name:
            return name, m
    raise RuntimeError("no decoder layer list found")


def install_loop(model, variant, s, e, adapter_rank=0):
    name, layers = find_layers(model)
    hidden = model.config.get_text_config().hidden_size
    loop = Loop(None, s, e, variant, hidden, adapter_rank).to(next(model.parameters()).device)
    for i in range(s, e + 1):
        layers[i] = Rec(layers[i], i, loop)
    object.__setattr__(loop, "layers", layers)
    model.add_module("loop_ctrl", loop)
    return loop


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loop, sets, dev, batch=8):
    model.eval()
    if loop is not None:
        loop.hard = True
    res = {}
    for name, ch in sets.items():
        tot, n, gsum = 0.0, 0, 0.0
        per_chunk = []
        for i in range(0, len(ch), batch):
            ids = torch.tensor(ch[i:i + batch], device=dev)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
            per_chunk.extend(nll.mean(1).tolist())
            tot += nll.sum().item()
            n += nll.numel()
            if loop is not None and loop.last_g is not None:
                gsum += (loop.last_g > 0.5).float().sum().item()
        pc = np.array(per_chunk)
        res[name] = {"nll": tot / n, "loop_frac": gsum / n if loop is not None and loop.variant == "gated" else None,
                     "per_chunk": pc.tolist()}
    model.train()
    if loop is not None:
        loop.hard = False
    return res


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev)
    s, e = (int(v) for v in a.block.split("-"))

    loop = install_loop(model, a.variant, s, e, a.loop_adapter)
    if a.variant != "base":
        from peft import LoraConfig, get_peft_model
        lname, _ = find_layers(model)
        targets = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and n.startswith(lname + ".")
                   and n.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                                 target_modules=targets))
        for p_ in loop.parameters():
            p_.requires_grad_(True)
        loop.float()

    tr_w, tr_g = mixed(tok, "train", "train", a.seq_len, a.steps * a.batch * a.seq_len)
    ev_w, ev_g = mixed(tok, "validation", "test", a.seq_len, a.eval_tokens)
    train = tr_w + tr_g
    rng = np.random.default_rng(a.seed)
    print(f"train chunks {len(train)} (wiki {len(tr_w)}, gsm {len(tr_g)}); eval wiki {len(ev_w)}, gsm {len(ev_g)}", flush=True)
    sets = {"wiki": ev_w, "gsm": ev_g}

    log = {"args": vars(a), "steps": []}
    if a.variant == "base":
        log["eval"] = evaluate(model, None, sets, dev)
    else:
        params = [p_ for p_ in model.parameters() if p_.requires_grad]
        lp_ids = {id(p_) for p_ in loop.parameters()}
        print(f"trainable params: {sum(p_.numel() for p_ in params) / 1e6:.2f}M", flush=True)
        opt = torch.optim.AdamW([{"params": [p_ for p_ in params if id(p_) not in lp_ids]},
                                 {"params": [p_ for p_ in params if id(p_) in lp_ids], "lr": a.loop_lr or a.lr}],
                                lr=a.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i, a.steps) / a.steps)))
        model.train()
        t0 = time.time()
        for step in range(a.steps):
            idx = rng.integers(0, len(train), a.batch)
            ids = torch.tensor([train[i] for i in idx], device=dev)
            out = model(input_ids=ids, labels=ids, use_cache=False)
            loss = out.loss
            reg = torch.zeros((), device=dev)
            if a.variant == "gated" and loop.last_g is not None:
                reg = a.gate_lam * loop.last_g.float().mean()
            (loss + reg).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == a.steps - 1:
                rec = {"step": step, "loss": loss.item(), "alpha": loop.alpha.item(),
                       "gate_mean": loop.last_g.float().mean().item() if loop.last_g is not None else None,
                       "min": (time.time() - t0) / 60}
                log["steps"].append(rec)
                print(json.dumps(rec), flush=True)
        log["eval"] = evaluate(model, loop, sets, dev)
    for k, v in log["eval"].items():
        print(f"EVAL {a.variant:7} {k:5} nll {v['nll']:.4f}  loop_frac {v['loop_frac']}", flush=True)
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))


if __name__ == "__main__":
    main()
