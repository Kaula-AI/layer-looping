"""
Incremental (token-by-token) decoding with per-token layer programs and a first-pass KV cache.

Semantics, identical to routed_eval.routed_forward:
  - Each token's K/V at layer l are written to the cache on its FIRST pass through l, projected
    from its input to l. A skipped layer still writes K/V (projected from the token's input).
  - A repeated layer's extra iterations attend to the cached K/V of earlier tokens plus the
    token's own current K/V (diagonal swap); they do not overwrite the cache.
Route per token: (layer, k) with k >= 2 = repeat, k == 0 = skip, layer == -1 = no edit.

Used by tests/test_routed_eval.py (decode == routed_forward) and gsm8k_eval.py (generation).
"""
import torch
import torch.nn.functional as F


class Cache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def append(self, li, k, v):
        self.k[li] = k if self.k[li] is None else torch.cat([self.k[li], k], 2)
        self.v[li] = v if self.v[li] is None else torch.cat([self.v[li], v], 2)

    def __len__(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]


def _attend(g, li, x, q, k, v, Kp, Vp):
    """One query position. Kp/Vp: cached K/V of earlier tokens (or None). Returns layer output, attn entropy."""
    layer = g.layers[li]
    K = k if Kp is None else torch.cat([Kp, k], 2)
    V = v if Vp is None else torch.cat([Vp, v], 2)
    s = (q @ K.transpose(-1, -2)).float() * g.scale  # B,H,1,S
    if g.attn_cap:
        s = torch.tanh(s / g.attn_cap) * g.attn_cap
    p = s.softmax(-1)
    o = (p.to(V.dtype) @ V).transpose(1, 2).reshape(x.shape[0], 1, -1)
    h = x + layer.post_attention_layernorm(layer.self_attn.o_proj(o))
    h = h + layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(h)))
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean(1)[:, 0]  # B
    return h, ent


@torch.no_grad()
def step(g, cache, token, pos, route=(-1, 1), want_hs=False):
    """Run one token (B,) at position `pos` through the model with its route; update the cache.
    Returns final hidden (B,1,d) and, if want_hs, per-layer states/attn entropies for features."""
    L = len(g.layers)
    x = g.embed(token[:, None])
    cos, sin = g.rotary(x, torch.full((x.shape[0], 1), pos, device=x.device, dtype=torch.long))
    rl, rk = route
    hs, ents = [x], []
    for li in range(L):
        layer = g.layers[li]
        q, k, v = g.qkv(layer, x, cos, sin)
        Kp, Vp = cache.k[li], cache.v[li]
        cache.append(li, k, v)  # first pass writes the cache (also for a skipped layer)
        if li == rl and rk == 0:  # skip: residual unchanged
            ents.append(torch.zeros(x.shape[0], device=x.device))
            hs.append(x)
            continue
        out, ent = _attend(g, li, x, q, k, v, Kp, Vp)
        if li == rl and rk >= 2:
            for _ in range(rk - 1):
                q2, k2, v2 = g.qkv(layer, out, cos, sin)
                out, _ = _attend(g, li, out, q2, k2, v2, Kp, Vp)
        x = out
        hs.append(x)
        ents.append(ent)
    return (x, hs, torch.stack(ents, -1)) if want_hs else x


@torch.no_grad()
def token_features(g, hs, attn_ent):
    """Same feature layout as oracle.py for ONE position: [lens_ent, lens_agree, upd_norm, attn_ent], each L."""
    L = len(g.layers)
    lps = [g.logits(hs[l][:, 0]).log_softmax(-1) for l in range(1, L + 1)]
    final_top = lps[-1].argmax(-1)
    ent = torch.stack([-(lp.exp() * lp).sum(-1) for lp in lps], -1)
    agree = torch.stack([(lp.argmax(-1) == final_top).float() for lp in lps], -1)
    upd = torch.stack([(hs[l + 1] - hs[l])[:, 0].float().norm(dim=-1) /
                       hs[l][:, 0].float().norm(dim=-1).clamp_min(1e-6) for l in range(L)], -1)
    return torch.cat([ent, agree, upd, attn_ent.float()], -1)  # B, 4L


@torch.no_grad()
def decode_nll(g, ids, routes_l, routes_k):
    """Teacher-forced token-by-token pass. Returns per-position next-token NLL (B, T-1)."""
    B, T = ids.shape
    cache = Cache(len(g.layers))
    out = []
    for t in range(T - 1):
        r = (int(routes_l[0, t]), int(routes_k[0, t]))  # same route for every row in the batch
        h = step(g, cache, ids[:, t], t, r)
        out.append(F.cross_entropy(g.logits(h[:, 0]), ids[:, t + 1], reduction="none"))
    return torch.stack(out, 1)
