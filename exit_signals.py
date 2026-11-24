"""E15: what should an exit gate read? Hidden state vs attention vs convergence, on a frozen Ouro (DESIGN.md, E15).

Ouro runs its n layers R times; after round r a token can exit (cost n*r layers). Under Ouro's evaluation accounting
every token computes every round and its loss comes from its exit round. With Ouro frozen, each exit option's loss
is a fixed number, so gates are trained on the exact expected objective (as structured_router.py, without the
extra action).

Raw features at round r use only rounds <= r and tokens <= t (never the label token t+1):
  hidden   Ouro's normed state after round r (the input of Ouro's own exit gate)                         (d)
  own      KL(p_r || p_{r-1}), ||h_r - h_{r-1}|| / ||h_r||, entropy and max prob of p_r, round-1 flag       (5)
  attn     per layer of round r, mean over heads: attention entropy / log(#visible), mass on position 0,
           mass on itself                                                                                    (3n)
  layers   per layer of round r: ||attention output||, ||MLP output||, ||attn + MLP update||, each / ||layer input||
           (the non-attention twin of attn)                                                                  (3n)
  src      at SRC_LAYERS: head-mean attention weights over sources j < t, j > 0 (self and the position-0 sink
           excluded, renormalized), averaging the sources' round-r KL change and prediction entropy          (2|S|)
  src_perm the same weights over the same quantities taken from another sequence of the batch (sham)        (2|S|)
  pos      log(1 + position), log(1 + tokens since the last EOS at or before t)                              (2)
`arm_inputs` log-transforms KL, the state change and the norm ratios.

  python exit_signals.py pilot  --out results/e15-pilot          # pipeline, speed, oracle gap (tuning split only)
  python exit_signals.py cache  --out results/e15                # frozen-Ouro features for eval, tuning, train
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

SRC_LAYERS = (5, 11, 17, 23)  # 0-based: layers 6, 12, 18, 24 of each round
ARMS = {"hidden": ("hidden",), "own": ("own",), "hidden_own": ("hidden", "own"), "attn_own": ("attn", "own"),
        "layers_own": ("layers", "own"), "src_own": ("src", "own"), "srcperm_own": ("src_perm", "own"),
        "all": ("hidden", "own", "attn", "src"), "pos": ("pos",), "bias": ()}
SMALL_ARMS = ("own", "attn_own", "layers_own", "src_own", "srcperm_own", "pos")  # per-round weights


def _rot(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def layer_with_probs(ouro, layer, x, cos, sin, return_parts=False):
    """Ouro's reference layer (ouro_model.Ouro._layer), also returning the attention probabilities (B, H, T, T)
    and, with return_parts, the attention-output and MLP-output residual contributions."""
    c = ouro.config
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
    p = s.softmax(-1)
    o = (p.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, -1)
    ao = layer.input_layernorm_2(a.o_proj(o))
    h = x + ao
    mo = layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(h)))
    if return_parts:
        return h + mo, p, ao, mo
    return h + mo, p


def source_weights(A):
    """Head-mean attention (B, T, T) -> weights over sources j < t with j > 0 (self and position-0 sink removed),
    renormalized; rows with no such source are all zero."""
    T = A.shape[-1]
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=A.device), -1)
    keep[:, 0] = False
    W = A * keep
    return W / W.sum(-1, keepdim=True).clamp_min(1e-12) * (W.sum(-1, keepdim=True) > 0)


def since_eos(ids, eos_id):
    """Tokens since the last EOS at or before t (t itself if none): (B, T) float."""
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).expand(B, -1)
    last = torch.where(ids == eos_id, pos, torch.full_like(pos, -1)).cummax(-1).values
    return (pos - last.clamp_min(0)).float()


@torch.no_grad()
def round_features(ouro, ids, n_rounds=None, src_layers=SRC_LAYERS, eos_id=None):
    """Per round r: dict(hn (B,T,d) normed state, ce (B,T-1) next-token CE of exiting at r, and the raw features
    own (B,T,5), attn (B,T,3n), layers (B,T,3n), src and src_perm (B,T,2|S|), pos (B,T,2)). src_perm takes the
    source quantities from the previous sequence of the batch (needs B >= 2; zeros otherwise)."""
    m, c = ouro.model, ouro.config
    n, R = c.num_hidden_layers, n_rounds or c.total_ut_steps
    B, T = ids.shape
    x = m.embed_tokens(ids)
    pos = torch.arange(T, device=ids.device)[None].expand(B, -1)
    cos, sin = m.rotary_emb(x, pos)
    lognvis = torch.log(torch.arange(1, T + 1, device=ids.device, dtype=torch.float32))  # log #visible tokens
    eos = since_eos(ids, ouro.config.vocab_size if eos_id is None else eos_id)
    posf = torch.stack([torch.log1p(pos.float()), torch.log1p(eos)], -1)
    out, prev_logp, prev_h = [], None, None
    for r in range(R):
        attn, lay, src_W = [], [], {}
        for li in range(n):
            xin = x.float()
            x, p, ao, mo = layer_with_probs(ouro, m.layers[li], x, cos, sin, return_parts=True)
            ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)                      # (B, H, T)
            ent = torch.where(lognvis > 0, ent / lognvis.clamp_min(1e-9), torch.zeros_like(ent))
            attn.append(torch.stack([ent.mean(1), p[..., 0].mean(1), p.diagonal(dim1=-2, dim2=-1).mean(1)], -1))
            xn = xin.norm(dim=-1).clamp_min(1e-9)
            lay.append(torch.stack([ao.float().norm(dim=-1) / xn, mo.float().norm(dim=-1) / xn,
                                    (ao.float() + mo.float()).norm(dim=-1) / xn], -1))
            if li in src_layers:
                src_W[li] = source_weights(p.mean(1))                                   # (B, T, T)
            del p
        h = x.float()
        x = m.norm(x)  # Ouro's end-of-round norm; the next round starts from it
        logp = torch.log_softmax(ouro.lm_head(x).float(), -1)                          # (B, T, V)
        ce = -logp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
        pr = logp.exp()
        ent_pred = -(pr * logp).sum(-1)
        maxp = pr.max(-1).values
        if prev_logp is None:
            kl, dh, first = torch.zeros_like(maxp), torch.zeros_like(maxp), torch.ones_like(maxp)
        else:
            kl = (pr * (logp - prev_logp)).sum(-1).clamp_min(0)
            dh = (h - prev_h).norm(dim=-1) / h.norm(dim=-1).clamp_min(1e-9)
            first = torch.zeros_like(maxp)
        del pr
        own = torch.stack([kl, dh, ent_pred, maxp, first], -1)
        vals = torch.stack([kl, ent_pred], -1)                                           # (B, T, 2)
        donor = vals.roll(1, 0) if B >= 2 else torch.zeros_like(vals)
        src = torch.cat([W @ vals for _, W in sorted(src_W.items())], -1)
        src_perm = torch.cat([W @ donor for _, W in sorted(src_W.items())], -1)
        out.append(dict(hn=x, ce=ce, own=own, attn=torch.cat(attn, -1), layers=torch.cat(lay, -1), src=src,
                        src_perm=src_perm, pos=posf))
        prev_logp, prev_h = logp, h
    return out


def transform(key, v):
    """Log-transform heavy-tailed raw features (KL, state change, norm ratios)."""
    v = v.float()
    if key == "own":
        return torch.cat([torch.log(v[..., :2] + 1e-6), v[..., 2:]], -1)
    if key in ("src", "src_perm"):
        v = v.clone()
        v[..., 0::2] = torch.log(v[..., 0::2] + 1e-6)
        return v
    if key == "layers":
        return torch.log(v + 1e-6)
    return v


def arm_inputs(rounds, arm):
    """List over rounds of (B, T-1, dim) float32 inputs for an arm (positions that predict a next token)."""
    keys = ARMS[arm]
    if not keys:
        return [None] * len(rounds)
    return [torch.cat([transform(k, rd['hn' if k == 'hidden' else k][:, :-1]) for k in keys], -1) for rd in rounds]


class Gate(nn.Module):
    """Exit gate in Ouro's form: stop prob lambda_r = sigmoid(w_r . standardized f_r + b_r); exit pdf as
    OuroForCausalLM. Weights shared across rounds (as Ouro's gate) or per round (small arms)."""

    def __init__(self, dim, R, mean=None, std=None, ouro_gate=None, per_round=False):
        super().__init__()
        self.R, self.dim, self.per_round = R, dim, per_round
        self.w = nn.Parameter(torch.zeros(R if per_round else 1, dim)) if dim else None
        self.b = nn.Parameter(torch.zeros(R))
        self.register_buffer("mean", torch.zeros(dim) if mean is None else mean.float())
        self.register_buffer("std", torch.ones(dim) if std is None else std.float())
        if ouro_gate is not None:  # Ouro's own gate on its raw normed state: no standardization
            assert dim == ouro_gate.in_features and not per_round
            with torch.no_grad():
                self.w.copy_(ouro_gate.weight.float())
                self.b.fill_(float(ouro_gate.bias))

    def forward(self, feats):
        lam = []
        for r, f in enumerate(feats):
            z = self.b[r]
            if self.w is not None:
                z = z + ((f - self.mean) / self.std) @ self.w[r if self.per_round else 0]
            lam.append(torch.sigmoid(z))  # bias-only: a scalar per round, broadcast by the caller
        rem, pdf = 1.0, []
        for r, l_ in enumerate(lam):
            pdf.append(l_ * rem if r < self.R - 1 else rem * torch.ones_like(l_))
            rem = rem * (1 - l_)
        return torch.stack(pdf, -1)


def expected_objective(pdf, ce, price, n):
    """pdf (..., R) or (R,), ce (B, T, R). Returns mean expected CE + price * layers, and mean expected layers."""
    R = ce.shape[-1]
    layers = n * torch.arange(1, R + 1, device=ce.device, dtype=torch.float32)
    pdf = pdf.expand_as(ce)
    return (pdf * (ce + price * layers)).sum(-1).mean(), (pdf * layers).sum(-1).mean()


def decide(pdf, q):
    """First round whose cumulative stop prob reaches q (Ouro's rule); last round otherwise. (B, T) long."""
    hit = pdf.cumsum(-1) >= q
    return torch.where(hit.any(-1), hit.float().argmax(-1), torch.full(hit.shape[:-1], pdf.shape[-1] - 1,
                                                                        dtype=torch.long, device=pdf.device))


def kl_rule(own_list, tau_ent, tau_kl):
    """Zero-parameter rule (two thresholds): exit at round 1 if its prediction entropy is below tau_ent, otherwise at
    the first later round whose KL change is below tau_kl; last round if none. Raw `own` features, (B, T-1) long."""
    ent1 = own_list[0][:, :-1, 2]
    kl = torch.stack([o[:, :-1, 0] for o in own_list], -1)  # (B, T-1, R)
    ok = kl < tau_kl
    ok[..., 0] = ent1 < tau_ent
    R = kl.shape[-1]
    return torch.where(ok.any(-1), ok.float().argmax(-1), torch.full(ok.shape[:-1], R - 1, dtype=torch.long,
                                                                      device=kl.device))


def oracle_exit(ce, price, n):
    """Hindsight best round per token under a compute price (upper bound only)."""
    R = ce.shape[-1]
    layers = n * torch.arange(1, R + 1, device=ce.device, dtype=torch.float32)
    return (ce + price * layers).argmin(-1)


# ---------------------------------------------------------------- pilot
def pilot(a):
    from transformers import AutoTokenizer

    from ouro_model import load_ouro
    from routed_ouro import fineweb_chunks

    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).to(dev).eval()
    c = ouro.config
    n, R = c.num_hidden_layers, c.total_ut_steps
    chunks = fineweb_chunks(tok, a.seq_len, 64 + a.n_chunks)[64:64 + a.n_chunks]  # TUNING split only
    gate = Gate(c.hidden_size, R, ouro_gate=ouro.model.early_exit_gate).to(dev)
    ces, owns, pdfs, feats_stats, t0, ntok = [], [], [], {}, time.time(), 0
    for i, ch in enumerate(chunks):
        ids = torch.tensor([ch], device=dev)
        rd = round_features(ouro, ids, eos_id=tok.eos_token_id)
        ces.append(torch.stack([x["ce"] for x in rd], -1).cpu())
        owns.append([x["own"].cpu() for x in rd])
        with torch.no_grad():
            pdfs.append(gate([x["hn"][:, :-1].float() for x in rd]).cpu())
        for k in ("own", "attn", "layers", "src", "pos"):
            v = torch.cat([x[k][:, :-1].reshape(-1, x[k].shape[-1]) for x in rd]).cpu()
            feats_stats.setdefault(k, []).append(v)
        ntok += ids.shape[1]
        if i == 0 or (i + 1) % 8 == 0:
            print(f"  chunk {i + 1}/{len(chunks)}  {ntok / (time.time() - t0):.0f} tok/s", flush=True)
    secs = time.time() - t0
    ce = torch.cat(ces)  # (N, T-1, R)
    pdf = torch.cat(pdfs)
    fixed = [float(ce[..., r].mean()) for r in range(R)]
    res = dict(n_chunks=len(chunks), seq_len=a.seq_len, tokens=ntok, seconds=secs, tok_per_s=ntok / secs,
               device=dev, fixed_rounds={n * (r + 1): fixed[r] for r in range(R)}, ouro_gate={}, oracle={}, kl_rule={})
    for q in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        rr = decide(pdf, q)
        res["ouro_gate"][q] = dict(layers=float(n * (rr + 1).float().mean()),
                                   nll=float(ce.gather(-1, rr[..., None]).mean()))
    for price in (0.0005, 0.001, 0.002, 0.004, 0.008):
        rr = oracle_exit(ce, price, n)
        res["oracle"][price] = dict(layers=float(n * (rr + 1).float().mean()),
                                    nll=float(ce.gather(-1, rr[..., None]).mean()))
    own_all = [torch.cat([o[r] for o in owns]) for r in range(R)]
    for te in (0.0, 0.1, 0.3, 1.0):
        for tk in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
            rr = kl_rule(own_all, te, tk)
            res["kl_rule"][f"ent<{te},kl<{tk}"] = dict(layers=float(n * (rr + 1).float().mean()),
                                                       nll=float(ce.gather(-1, rr[..., None]).mean()))
    # per-chunk sd of the paired difference (Ouro's gate at q=0.5 minus 4 fixed rounds), for sample size
    rr = decide(pdf, 0.5)
    d = (ce.gather(-1, rr[..., None])[..., 0] - ce[..., -1]).mean(1)
    res["paired_sd_chunk_gate_vs_r4"] = float(d.std())
    res["feature_stats"] = {k: dict(dim=int(torch.cat(v).shape[-1]),
                                    mean=torch.cat(v).mean(0).tolist()[:12], std=torch.cat(v).std(0).tolist()[:12])
                            for k, v in feats_stats.items()}
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "pilot.json.tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "pilot.json"))
    print(json.dumps({k: v for k, v in res.items() if k != "feature_stats"}, indent=1))


# ---------------------------------------------------------------- cache (Ouro frozen: features computed once)
SPLITS = {"eval": list(range(0, 64)) + list(range(192, 256)),  # never touched by the pilot (it read 64-79)
          "tuning": list(range(64, 128)), "train": list(range(256, 744))}
SMALL_KEYS = {"own": 5, "attn": 72, "layers": 72, "src": 8, "src_perm": 8}


def cache(a):
    """Write per-split memmaps under a.out/cache/<split>/: ce (N,T-1,R) f32, hidden (N,T-1,R,d) f16, small raw
    features (N,T-1,R,k) f32, pos (N,T-1,2) f32. Resume-safe: progress.json counts finished chunks; a crash loses
    only the batch in progress."""
    from transformers import AutoTokenizer

    from ouro_model import load_ouro
    from routed_ouro import fineweb_chunks

    dev = a.device
    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).to(dev).eval()
    c = ouro.config
    R, d, T1 = c.total_ut_steps, c.hidden_size, a.seq_len - 1
    need = max(max(v) for v in SPLITS.values()) + 1
    chunks = fineweb_chunks(tok, a.seq_len, need)
    assert len(chunks) == need
    for split in a.splits:
        idx = SPLITS[split]
        root = os.path.join(a.out, "cache", split)
        os.makedirs(root, exist_ok=True)
        N = len(idx)
        mode = lambda f: "r+" if os.path.exists(os.path.join(root, f)) else "w+"  # noqa: E731
        mm = {"ce": np.lib.format.open_memmap(os.path.join(root, "ce.npy"), mode("ce.npy"), np.float32, (N, T1, R)),
              "hidden": np.lib.format.open_memmap(os.path.join(root, "hidden.npy"), mode("hidden.npy"), np.float16,
                                                  (N, T1, R, d)),
              "pos": np.lib.format.open_memmap(os.path.join(root, "pos.npy"), mode("pos.npy"), np.float32, (N, T1, 2))}
        for k, w in SMALL_KEYS.items():
            mm[k] = np.lib.format.open_memmap(os.path.join(root, f"{k}.npy"), mode(f"{k}.npy"), np.float32, (N, T1, R, w))
        prog_path = os.path.join(root, "progress.json")
        done = json.load(open(prog_path))["done"] if os.path.exists(prog_path) else 0
        json.dump(dict(chunk_ids=idx), open(os.path.join(root, "chunks.json"), "w"))
        t0 = time.time()
        for i in range(done, N, a.batch):
            j = min(i + a.batch, N)
            ids = torch.tensor([chunks[k] for k in idx[i:j]], device=dev)
            if ids.shape[0] < 2:  # src_perm needs a donor sequence: pair the last chunk with the first
                ids = torch.cat([ids, torch.tensor([chunks[idx[0]]], device=dev)])
            rd = round_features(ouro, ids, eos_id=tok.eos_token_id)
            nb = j - i
            mm["ce"][i:j] = torch.stack([x["ce"] for x in rd], -1)[:nb].cpu().numpy()
            mm["hidden"][i:j] = torch.stack([x["hn"][:, :-1] for x in rd], 2)[:nb].float().cpu().numpy().astype(np.float16)
            mm["pos"][i:j] = rd[0]["pos"][:nb, :-1].cpu().numpy()
            for k in SMALL_KEYS:
                mm[k][i:j] = torch.stack([x[k][:, :-1] for x in rd], 2)[:nb].float().cpu().numpy()
            for v in mm.values():
                v.flush()
            tmp = prog_path + ".tmp"
            json.dump(dict(done=j, n=N, tok_per_s=(j - done) * a.seq_len / (time.time() - t0),
                           time=time.strftime("%Y-%m-%dT%H:%M:%S")), open(tmp, "w"))
            os.replace(tmp, prog_path)
            print(f"  {split}: {j}/{N} chunks  {(j - done) * a.seq_len / (time.time() - t0):.0f} tok/s", flush=True)
        prov = dict(model=a.model, device=dev, dtype="bfloat16", seq_len=a.seq_len, splits=SPLITS[split][:3],
                    commit=os.popen("git rev-parse HEAD").read().strip(), time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        json.dump(prov, open(os.path.join(root, "provenance.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pilot", "cache"])
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_chunks", type=int, default=16)
    p.add_argument("--out", default="results/e15-pilot")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--splits", nargs="+", default=["eval", "tuning", "train"])
    a = p.parse_args()
    {"pilot": pilot, "cache": cache}[a.cmd](a)


if __name__ == "__main__":
    main()
