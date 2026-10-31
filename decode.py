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


def _attend(g, li, x, q, k, v, Kp, Vp, cs):
    """One query position. Kp/Vp: cached K/V of earlier tokens (or None). Returns layer output, attn entropy."""
    K = k if Kp is None else torch.cat([Kp, k], 2)
    V = v if Vp is None else torch.cat([Vp, v], 2)
    s = (q @ K.transpose(-1, -2)).float() * g.scale  # B,H,1,S
    if g.attn_cap:
        s = torch.tanh(s / g.attn_cap) * g.attn_cap
    p = s.softmax(-1)
    o = (p.to(V.dtype) @ V).transpose(1, 2).reshape(x.shape[0], 1, -1)
    h = g.post(li, x, o, cs)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean(1)[:, 0]  # B
    return h, ent


def _token_ctx(g, token, pos):
    x = g.embed(token[:, None])
    return x, g.context(token[:, None], x, torch.full((x.shape[0], 1), pos, device=x.device, dtype=torch.long))


def _prior(cache, li, n):
    """Earlier tokens' cached K/V at layer li (the first n entries), or None."""
    if n == 0 or cache.k[li] is None:
        return None, None
    return cache.k[li][:, :, :n], cache.v[li][:, :, :n]


@torch.no_grad()
def step(g, cache, token, pos, route=(-1, 1), want_hs=False):
    """Run one token (B,) at position `pos` through the model with its route; update the cache.
    Returns final hidden (B,1,d) and, if want_hs, per-layer states/attn entropies for features.
    A K/V-shared layer (Gemma-4) reads its source layer's cache for earlier tokens and the token's
    latest own K/V of the source."""
    L = len(g.layers)
    x, cs = _token_ctx(g, token, pos)
    n = len(cache)
    rl, rk = route
    hs, ents, latest = [x], [], {}
    for li in range(L):
        sl = g.kv_src(li)
        q, k, v = g.qkv(li, x, cs)
        Kp, Vp = _prior(cache, sl, n)
        if k is None:
            k, v = latest[sl]
        else:
            cache.append(li, k, v)  # first pass writes the cache (also for a skipped layer)
            latest[li] = (k, v)
        if li == rl and rk == 0:  # skip: residual unchanged
            ents.append(torch.zeros(x.shape[0], device=x.device))
            hs.append(x)
            continue
        out, ent = _attend(g, li, x, q, k, v, Kp, Vp, cs)
        if li == rl and rk >= 2:
            for _ in range(rk - 1):
                q2, k2, v2 = g.qkv(li, out, cs)
                if k2 is None:
                    k2, v2 = latest[sl]
                else:
                    latest[li] = (k2, v2)
                out, _ = _attend(g, li, out, q2, k2, v2, Kp, Vp, cs)
        x = out
        hs.append(x)
        ents.append(ent)
    return (x, hs, torch.stack(ents, -1)) if want_hs else x


@torch.no_grad()
def step_path(g, cache, token, pos, path):
    """Run one token (B,) through an ARBITRARY layer path (list of layer indices; repeats, skips,
    swaps, reversed or looped blocks), updating the cache with first-pass semantics:
      - the first visit to layer l writes K/V projected from the token's input to that visit;
      - later visits attend to earlier tokens' cached K/V plus the token's own current K/V;
      - a layer the path never visits gets K/V projected from the token's state at the point where
        its path first reaches a layer with a higher index (or its final state), which reduces to
        "projected from its input" for a plain skip.
    Each layer with K/V receives exactly one cache entry per token, so cache order stays aligned.
    A K/V-shared layer attends to its source's cache plus the token's latest own K/V of the source
    (from its latest visit, else projected at the point its path first passed the source)."""
    L = len(g.layers)
    x, cs = _token_ctx(g, token, pos)
    written = [False] * L
    seen_inputs = []  # (layer, input state) for every step
    pend = [None] * L  # this token's K/V per layer, appended at the end so earlier tokens see the cache unchanged
    latest = {}
    for li in path:
        for s in g.kv_sources:
            if s not in latest and li > s:
                latest[s] = g.kv_of(s, x, cs)
        seen_inputs.append((li, x))
        sl = g.kv_src(li)
        q, k, v = g.qkv(li, x, cs)
        if k is None:
            k, v = latest[sl]
        else:
            if not written[li]:
                written[li] = True
                pend[li] = (k, v)
            latest[li] = (k, v)
        x, _ = _attend(g, li, x, q, k, v, cache.k[sl], cache.v[sl], cs)
    for li in range(L):
        if not g.has_kv(li):
            continue
        if not written[li]:
            src = next((st for lj, st in seen_inputs if lj > li), x)
            pend[li] = g.kv_of(li, src, cs)
        cache.append(li, *pend[li])
    return x


@torch.no_grad()
def decode_nll_paths(g, ids, paths):
    """Teacher-forced token-by-token pass where position t follows paths[t] (list of lists; batch
    rows share paths). Returns per-position next-token NLL (B, T-1)."""
    B, T = ids.shape
    cache = Cache(len(g.layers))
    out = []
    for t in range(T - 1):
        h = step_path(g, cache, ids[:, t], t, paths[t])
        out.append(F.cross_entropy(g.logits(h[:, 0]), ids[:, t + 1], reduction="none"))
    return torch.stack(out, 1)


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
