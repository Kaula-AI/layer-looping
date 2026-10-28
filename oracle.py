"""
Per-token single-edit layer-program oracle for Gemma-2-2B (repeat, skip, swap).

For every position t and every single-edit program (repeat layer l k times, skip layer l,
or swap layers l and l+1) this measures the change
in next-token NLL if ONLY position t follows that program instead of the base order.

"Diagonal swap": other positions keep their base-pass keys/values at every layer;
only position t's own key/value is replaced by its modified one. That makes one
config cost about one forward pass over the whole batch, instead of one pass per
(token, config). Because the model is causal and only position t is modified, this
is exact for token t's own next-token loss; it ignores the effect on later tokens
(measured separately by routed_eval.py).

It also saves per-position features from the base pass:
  - logit-lens entropy per layer
  - logit-lens top-1 agreement with the final layer, per layer
  - relative residual-update norm ||h_{l+1}-h_l|| / ||h_l|| per layer
  - attention entropy per layer (mean over heads)
  - one raw mid-layer hidden state (baseline probe)

Usage:
  python oracle.py --check                 # verify manual forward == HF forward
  python oracle.py --n_tokens 20000        # quick run
  python oracle.py --n_tokens 500000       # full run
"""
import argparse
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_tokens", type=int, default=500_000)
    p.add_argument("--ops", nargs="+", default=["repeat", "skip", "swap"], choices=["repeat", "skip", "swap"])
    p.add_argument("--repeats", type=int, nargs="+", default=[2, 3])
    p.add_argument("--blocks", nargs="*", default=["10-13x2"],
                   help="static block-loop programs S-ExR (default: Inner Loop Inference's Gemma-2-2B range)")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="layers to edit (default: all)")
    p.add_argument("--raw_layer", type=int, default=None, help="residual index saved for raw probe (default: L//2)")
    p.add_argument("--out", default="oracle_out")
    p.add_argument("--check", action="store_true", help="only verify the manual forward matches HF")
    return p.parse_args()


# ---------------------------------------------------------------- data
def _chunks(texts, tok, body):
    buf = []
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= body:
            yield buf[:body]
            buf = buf[body:]


def make_chunks(tok, n_tokens, seq_len):
    """Alternate chunks of WikiText (ordinary prose) and GSM8K (math, where looping should matter)."""
    body = seq_len - 1  # room for BOS
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:5%]")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    srcs = {
        "wiki": _chunks((t for t in wiki["text"] if t.strip()), tok, body),
        "gsm": _chunks(
            (f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(gsm["question"], gsm["answer"])), tok, body
        ),
    }
    want = math.ceil(n_tokens / seq_len)
    chunks, names, order = [], [], ["wiki", "gsm"]
    while len(chunks) < want and order:
        for name in list(order):
            try:
                c = next(srcs[name])
            except StopIteration:
                order.remove(name)
                continue
            chunks.append([tok.bos_token_id] + c)
            names.append(name)
            if len(chunks) >= want:
                break
    return chunks, names


# ---------------------------------------------------------------- model
class G2:
    """Manual Gemma-2 forward that exposes per-layer K/V and supports the diagonal swap."""

    def __init__(self, model):
        self.model = model
        self.m = model.model
        self.cfg = model.config
        self.layers = self.m.layers
        self.norm = self.m.norm
        self.lm_head = model.lm_head
        c = self.cfg
        self.nH, self.nKV, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.scale = c.query_pre_attn_scalar ** -0.5
        self.attn_cap = c.attn_logit_softcapping
        self.final_cap = c.final_logit_softcapping
        self.rotary = getattr(self.m, "rotary_emb", None)
        assert self.rotary is not None, "needs transformers>=4.48 (model-level rotary_emb)"
        self._masks = {}

    def causal(self, T, dev):
        key = (T, str(dev))
        if key not in self._masks:
            self._masks[key] = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
        return self._masks[key]

    def embed(self, ids):
        h = self.m.embed_tokens(ids)
        if hasattr(self.m.embed_tokens, "embed_scale"):  # transformers >= 5 scales inside the embedding
            return h
        return h * torch.tensor(self.cfg.hidden_size ** 0.5, dtype=h.dtype)

    def logits(self, h):
        x = self.lm_head(self.norm(h)).float()
        if self.final_cap:
            x = torch.tanh(x / self.final_cap) * self.final_cap
        return x

    def qkv(self, layer, x, cos, sin):
        a = layer.self_attn
        B, T, _ = x.shape
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(B, T, self.nH, self.hd).transpose(1, 2)
        k = a.k_proj(h).view(B, T, self.nKV, self.hd).transpose(1, 2)
        v = a.v_proj(h).view(B, T, self.nKV, self.hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = self.nH // self.nKV  # same head layout as HF repeat_kv
        return q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

    def layer(self, li, x, cos, sin, base_kv=None, want_entropy=False):
        """One decoder layer. If base_kv is given, each query attends to base K/V
        for every other position and to its own (modified) K/V on the diagonal."""
        layer = self.layers[li]
        B, T, _ = x.shape
        q, k, v = self.qkv(layer, x, cos, sin)
        K, V = (k, v) if base_kv is None else base_kv
        s = (q @ K.transpose(-1, -2)).float() * self.scale  # B,H,T,T
        if base_kv is not None:
            idx = torch.arange(T, device=x.device)
            s[..., idx, idx] = (q * k).sum(-1).float() * self.scale
        if self.attn_cap:
            s = torch.tanh(s / self.attn_cap) * self.attn_cap
        s = s.masked_fill(self.causal(T, x.device), float("-inf"))
        p = s.softmax(-1)
        o = p.to(V.dtype) @ V
        if base_kv is not None:
            o = o + p[..., idx, idx].to(V.dtype).unsqueeze(-1) * (v - V)
        o = o.transpose(1, 2).reshape(B, T, -1)
        h = x + layer.post_attention_layernorm(layer.self_attn.o_proj(o))
        h = h + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean(1) if want_entropy else None  # B,T
        return h, (k, v), ent

    @torch.no_grad()
    def base_pass(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        h = self.embed(ids)
        cos, sin = self.rotary(h, pos)
        hs, kvs, ents = [h], [], []
        for li in range(len(self.layers)):
            h, kv, ent = self.layer(li, h, cos, sin, want_entropy=True)
            hs.append(h)
            kvs.append(kv)
            ents.append(ent)
        return hs, kvs, torch.stack(ents, -1), (cos, sin)  # hs[l] = input to layer l

    @torch.no_grad()
    def run_program(self, hs, kvs, cs, seq):
        """Every position follows layer program `seq` (a list of layer indices, e.g. base
        0..L-1 with a layer repeated, dropped or two layers swapped). At each step a position
        attends to the other positions' BASE-pass K/V of the layer being run, plus its own
        current K/V (diagonal swap). Exact for each position's own next-token loss."""
        cos, sin = cs
        d = 0  # first step where the program leaves the base order
        while d < len(seq) and d < len(self.layers) and seq[d] == d:
            d += 1
        x = hs[d]
        for li in seq[d:]:
            x, _, _ = self.layer(li, x, cos, sin, base_kv=kvs[li])
        return x

    def repeat_pass(self, hs, kvs, cs, l, k):
        L = len(self.layers)
        return self.run_program(hs, kvs, cs, list(range(l + 1)) + [l] * (k - 1) + list(range(l + 1, L)))

    @torch.no_grad()
    def independent_program_nll(self, hs, ids, seq, t, b):
        """Exactness reference, independent of the diagonal-swap code path: truncate sequence b
        at position t and run each layer of `seq` normally on [other positions' base inputs to
        that layer] + [token t's current state]. Returns token t's next-token NLL."""
        pos = torch.arange(t + 1, device=ids.device).unsqueeze(0)
        cos, sin = self.rotary(hs[0][b : b + 1, : t + 1], pos)
        x = hs[0][b : b + 1, t : t + 1]
        for li in seq:
            out, _, _ = self.layer(li, torch.cat([hs[li][b : b + 1, :t], x], 1), cos, sin)
            x = out[:, -1:]
        return F.cross_entropy(self.logits(x)[0], ids[b, t + 1 : t + 2])

    @torch.no_grad()
    def nll(self, h, ids):
        out = [F.cross_entropy(self.logits(h[b, :-1]), ids[b, 1:], reduction="none") for b in range(h.shape[0])]
        return torch.stack(out)  # B, T-1 ; position t predicts token t+1

    @torch.no_grad()
    def lens_features(self, hs):
        L = len(self.layers)
        B, T, _ = hs[0].shape
        dev = hs[0].device
        ent = torch.empty(B, T, L, device=dev)
        agree = torch.empty(B, T, L, device=dev)
        final_top = torch.empty(B, T, dtype=torch.long, device=dev)
        for l in range(L, 0, -1):  # final layer first, so agreement has a reference
            for b in range(B):
                lp = self.logits(hs[l][b]).log_softmax(-1)
                ent[b, :, l - 1] = -(lp.exp() * lp).sum(-1)
                top = lp.argmax(-1)
                if l == L:
                    final_top[b] = top
                agree[b, :, l - 1] = (top == final_top[b]).float()
        upd = torch.stack(
            [(hs[l + 1] - hs[l]).float().norm(dim=-1) / hs[l].float().norm(dim=-1).clamp_min(1e-6) for l in range(L)],
            -1,
        )
        return ent, agree, upd


# ---------------------------------------------------------------- programs
def build_programs(L, ops=("repeat", "skip", "swap"), repeats=(2, 3), layers=None, blocks=()):
    """Single-edit layer programs. Returns a list of dicts: op, layer, end, k, cost, seq.
    cost = layer applications relative to base (repeat k: k-1, skip: -1, swap: 0,
    block s..e x R: (e-s+1)*(R-1)). `blocks` adds contiguous block loops (s, e, R), e.g. the
    static mid-block loop of Inner Loop Inference (arXiv 2602.14759; Gemma-2-2B: layers 10-13)."""
    base = list(range(L))
    layers = range(L) if layers is None else layers
    progs = []
    for l in layers:
        if "repeat" in ops:
            for k in repeats:
                progs.append(dict(op="repeat", layer=l, end=l, k=k, cost=k - 1,
                                  seq=base[: l + 1] + [l] * (k - 1) + base[l + 1 :]))
        if "skip" in ops:
            progs.append(dict(op="skip", layer=l, end=l, k=0, cost=-1, seq=base[:l] + base[l + 1 :]))
        if "swap" in ops and l < L - 1:
            progs.append(dict(op="swap", layer=l, end=l + 1, k=1, cost=0, seq=base[:l] + [l + 1, l] + base[l + 2 :]))
    for s_, e_, r_ in blocks:
        assert 0 <= s_ <= e_ < L and r_ >= 2
        progs.append(dict(op="block", layer=s_, end=e_, k=r_, cost=(e_ - s_ + 1) * (r_ - 1),
                          seq=base[: e_ + 1] + list(range(s_, e_ + 1)) * (r_ - 1) + base[e_ + 1 :]))
    return progs


# ---------------------------------------------------------------- check
@torch.no_grad()
def run_check(g, model, ids):
    hf = model(ids).logits.float()
    hs, kvs, _, (cos, sin) = g.base_pass(ids)
    mine = g.logits(hs[-1])
    print(f"manual vs HF logits: max|diff| {(hf - mine).abs().max().item():.4f} "
          f"(logit scale {hf.abs().max().item():.1f}); "
          f"argmax agreement {(hf.argmax(-1) == mine.argmax(-1)).float().mean().item():.4f}")
    L = len(g.layers)
    x = hs[L // 2]
    for li in range(L // 2, L):
        x, _, _ = g.layer(li, x, cos, sin, base_kv=kvs[li])
    print(f"diagonal-swap identity max|dh|: {(x - hs[-1]).abs().max().item():.6f} (should be ~0)")
    # exactness of oracle deltas vs an independent recompute, ~5 positions x 3 ops
    base = g.nll(hs[-1], ids)
    B, T = ids.shape
    worst = 0.0
    for pr in [p_ for p_ in build_programs(L) if p_["layer"] in (L // 3, L // 2, 2 * L // 3)][::2]:
        d_all = g.nll(g.run_program(hs, kvs, (cos, sin), pr["seq"]), ids) - base
        for t, b in [(3, 0), (T // 4, 0), (T // 2, B - 1), (T - 2, B - 1), (T // 3, 0)]:
            ref = g.independent_program_nll(hs, ids, pr["seq"], t, b) - base[b, t]
            worst = max(worst, abs(d_all[b, t].item() - ref.item()))
    print(f"oracle vs independent recompute: max|d dNLL| {worst:.4f} nats (bf16; should be << typical |dNLL|)")


# ---------------------------------------------------------------- main
def main():
    a = get_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(dev).eval()
    g = G2(model)
    sw = getattr(model.config, "sliding_window", None)
    assert sw is None or a.seq_len <= sw, "seq_len must fit inside Gemma-2's sliding window"

    L = len(g.layers)
    layers = a.layers if a.layers else list(range(L))
    blocks = [tuple(int(v) for v in re.fullmatch(r"(\d+)-(\d+)x(\d+)", b_).groups()) for b_ in a.blocks]
    progs = build_programs(L, a.ops, a.repeats, layers, blocks)
    raw_layer = a.raw_layer if a.raw_layer is not None else L // 2

    n_tok = a.seq_len * a.batch if a.check else a.n_tokens
    chunks, names = make_chunks(tok, n_tok, a.seq_len)
    if a.check:
        run_check(g, model, torch.tensor(chunks[: a.batch], device=dev))
        return

    print(f"{len(chunks)} chunks x {a.seq_len} tokens, {len(progs)} programs "
          f"({', '.join(f'{o}: {sum(p_['op'] == o for p_ in progs)}' for o in list(a.ops) + (['block'] if blocks else []))}), device {dev}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rec = {k: [] for k in ["delta", "feat", "base_nll", "raw", "tok", "chunk", "pos"]}
    t0 = time.time()
    for bi in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[bi : bi + a.batch], device=dev)
        B, T = ids.shape
        with torch.no_grad():
            hs, kvs, attn_ent, cs = g.base_pass(ids)
            base_nll = g.nll(hs[-1], ids)
            ent, agree, upd = g.lens_features(hs)
            feat = torch.cat([ent, agree, upd, attn_ent.float()], -1)[:, :-1]
            delta = torch.empty(B, T - 1, len(progs), device=dev)
            for ci, pr in enumerate(progs):
                delta[..., ci] = g.nll(g.run_program(hs, kvs, cs, pr["seq"]), ids) - base_nll
        rec["delta"].append(delta.half().cpu().numpy().reshape(-1, len(progs)))
        rec["feat"].append(feat.half().cpu().numpy().reshape(-1, feat.shape[-1]))
        rec["base_nll"].append(base_nll.float().cpu().numpy().reshape(-1))
        rec["raw"].append(hs[raw_layer][:, :-1].half().cpu().numpy().reshape(-1, hs[0].shape[-1]))
        rec["tok"].append(ids[:, :-1].cpu().numpy().reshape(-1))
        rec["chunk"].append(np.repeat(np.arange(bi, bi + B), T - 1))
        rec["pos"].append(np.tile(np.arange(T - 1), B))
        done, el = bi + B, time.time() - t0
        print(f"{done}/{len(chunks)} chunks | {el / 60:.1f} min | ETA {el / done * (len(chunks) - done) / 60:.1f} min",
              flush=True)

    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["cfg_op"] = np.array([pr["op"] for pr in progs])
    out["cfg_layer"] = np.array([pr["layer"] for pr in progs])
    out["cfg_end"] = np.array([pr["end"] for pr in progs])
    out["cfg_k"] = np.array([pr["k"] for pr in progs])
    out["cfg_cost"] = np.array([pr["cost"] for pr in progs], dtype=np.float32)
    out["source"] = np.array(names)
    out["n_layers"] = np.array(L)
    out["feat_names"] = np.array(
        [f"{g_}_{i}" for g_ in ["lens_ent", "lens_agree", "upd_norm", "attn_ent"] for i in range(L)]
    )
    path = os.path.join(a.out, "oracle.npz")
    np.savez(path, **out)
    print(f"saved {path}  ({len(out['delta']):,} positions)")


if __name__ == "__main__":
    main()
