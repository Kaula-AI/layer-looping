"""Self-contained Ouro (LoopLM, ByteDance, arXiv 2510.25741) for this repo: loads the released weights
(safetensors + config.json only, no remote code) into modules whose names match the checkpoint.

Architecture (from the released model): a Llama-style decoder with sandwich norms
  h = x + norm2(attn(norm1(x)));  out = h + norm4(mlp(norm3(h)))
run `total_ut_steps` times over the same n layers, with the final RMSNorm applied after every round.
Each round ends with an exit gate sigmoid(Linear(h)). forward(ids, exit_at_step=s) returns the logits
after round s (0-based), which is what OuroForCausalLM does with `exit_at_step`.
"""
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)


class Rotary(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
                             persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        f = pos[..., None].float() * self.inv_freq.to(x.device)  # B, T, hd/2
        emb = torch.cat([f, f], -1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rot(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, H, KV, hd = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(d, H * hd, bias=False)
        self.k_proj = nn.Linear(d, KV * hd, bias=False)
        self.v_proj = nn.Linear(d, KV * hd, bias=False)
        self.o_proj = nn.Linear(H * hd, d, bias=False)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        e = c.rms_norm_eps
        self.self_attn = Attn(c)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, e)
        self.input_layernorm_2 = RMSNorm(c.hidden_size, e)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, e)
        self.post_attention_layernorm_2 = RMSNorm(c.hidden_size, e)
        self.attention_type = "full_attention"


class Body(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.rotary_emb = Rotary(c.head_dim, c.rope_theta)
        self.early_exit_gate = nn.Linear(c.hidden_size, 1)


class Ouro(nn.Module):
    """Reference forward (plain batched causal attention). The oracle's manual forward is oracle.GO."""

    def __init__(self, c):
        super().__init__()
        self.config = c
        self.model = Body(c)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)

    def _layer(self, layer, x, cos, sin):
        c = self.config
        B, T, _ = x.shape
        a = layer.self_attn
        h = layer.input_layernorm(x)
        sh = lambda t: t.view(B, T, -1, c.head_dim).transpose(1, 2)  # noqa: E731
        q, k, v = sh(a.q_proj(h)), sh(a.k_proj(h)), sh(a.v_proj(h))
        cu, su = cos[:, None], sin[:, None]
        q, k = q * cu + _rot(q) * su, k * cu + _rot(k) * su
        rep = c.num_attention_heads // c.num_key_value_heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        s = (q @ k.transpose(-1, -2)).float() * c.head_dim ** -0.5
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf"))
        o = (s.softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, T, -1)
        h = x + layer.input_layernorm_2(a.o_proj(o))
        return h + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))

    def rounds(self, ids):
        """Normed hidden states after each round (list of B, T, d) and exit-gate logits per round."""
        m = self.model
        x = m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)[None].expand(ids.shape[0], -1)
        cos, sin = m.rotary_emb(x, pos)
        outs, gates = [], []
        for _ in range(self.config.total_ut_steps):
            for layer in m.layers:
                x = self._layer(layer, x, cos, sin)
            x = m.norm(x)
            outs.append(x)
            gates.append(m.early_exit_gate(x))
        return outs, gates

    def forward(self, input_ids, exit_at_step=None, **_):
        outs, _g = self.rounds(input_ids)
        s = len(outs) - 1 if exit_at_step is None else exit_at_step
        return SimpleNamespace(logits=self.lm_head(outs[s]))


def config_from_dict(d):
    c = SimpleNamespace(**d)
    c.model_type = "ouro"
    c.total_ut_steps = d.get("total_ut_steps", 4)
    c.head_dim = d.get("head_dim", d["hidden_size"] // d["num_attention_heads"])
    c.rope_theta = d.get("rope_theta", 1e6)
    assert not d.get("rope_scaling"), "only default RoPE is implemented"
    assert not d.get("use_sliding_window"), "sliding window is not implemented"
    assert not d.get("tie_word_embeddings"), "tied embeddings are not implemented"
    return c


def load_ouro(name, dtype=torch.bfloat16):
    """name: a local directory or a Hugging Face repo id. Downloads only config.json and *.safetensors."""
    from safetensors.torch import load_file

    if os.path.isdir(name):
        root = name
    else:
        from huggingface_hub import snapshot_download
        root = snapshot_download(name, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
    c = config_from_dict(json.load(open(os.path.join(root, "config.json"))))
    model = Ouro(c)
    sd = {}
    for f in sorted(os.listdir(root)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(root, f)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("inv_freq")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(dtype).eval()


def load_causal_lm(name, dtype=torch.bfloat16, **hf_kwargs):
    """Ouro via load_ouro; anything else via transformers' AutoModelForCausalLM."""
    if "ouro" in name.lower():
        return load_ouro(name, dtype)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **hf_kwargs)
