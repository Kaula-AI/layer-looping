"""E15 fitting and scoring (DESIGN.md E15, SPEC.md E15). Reads the feature cache written by `exit_signals.py cache`.

  python e15_fit.py fit    --cache results/e15/cache --out results/e15     # train every arm's gates, select on tuning
  python e15_fit.py report --cache results/e15/cache --out results/e15     # score on eval, write results.json + REPORT.md

Everything here follows SPEC.md (E15): arms, grids, selection on the tuning split, exact budget matching on the
evaluation split (label-free: thresholds only), the primary area-under-curve metric, chunk-bootstrap CIs, TOST, the
compute-anchored SESOI and the verdict rules. REPORT.md is written by this script, never by hand.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from exit_signals import ARMS, SMALL_ARMS, Gate, decide, expected_objective, transform

N_LAYERS, R = 24, 4
BUDGETS = list(range(50, 91, 5))  # layers per token; the primary metric averages over these
PRICES = (0.0005, 0.001, 0.002)
LR_W = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
LR_W_WARM = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
LR_B = (1e-3, 1e-2, 1e-1)  # bias learning rates (extended at the edges like LR_W)
LEARNED = ("hidden", "hidden_warm", "own", "hidden_own", "attn_own", "layers_own", "src_own", "srcperm_own", "all",
           "pos")
READOUT_ARMS = ("own", "hidden_own", "attn_own", "layers_own", "src_own", "srcperm_own", "all", "kl_rule")
ATTN_ARMS = ("attn_own", "src_own", "srcperm_own", "all")
BLOCK = 8  # chunks per training block
MINIBATCH = 4096  # tokens per optimizer step
EPOCHS, PATIENCE = 8, 2


# ---------------------------------------------------------------- data
def load(cache, split):
    root = os.path.join(cache, split)
    return {f[:-4]: np.load(os.path.join(root, f), mmap_mode="r") for f in os.listdir(root) if f.endswith(".npy")}


def keys_of(arm):
    return ARMS["hidden" if arm == "hidden_warm" else arm]


def feats(D, arm, c0, c1):
    """Per-round float32 inputs (n_tok, dim) for chunks c0:c1 of a split (raw keys -> exit_signals.transform)."""
    out = []
    for r in range(R):
        parts = []
        for k in keys_of(arm):
            if k == "hidden":
                v = torch.from_numpy(np.array(D["hidden"][c0:c1, :, r], dtype=np.float32))
            elif k == "pos":
                v = torch.from_numpy(np.array(D["pos"][c0:c1], dtype=np.float32))
            else:
                v = transform(k, torch.from_numpy(np.array(D[k][c0:c1, :, r], dtype=np.float32)))
            parts.append(v.reshape(-1, v.shape[-1]))
        out.append(torch.cat(parts, -1) if parts else None)
    return out


def ce_of(D, c0, c1):
    return torch.from_numpy(np.array(D["ce"][c0:c1], dtype=np.float32)).reshape(-1, R)


def stats(D, arm, n_chunks=32):
    """Mean/std per input dim over the first n_chunks training chunks, pooled over rounds."""
    f = torch.cat(feats(D, arm, 0, min(n_chunks, D["ce"].shape[0])))
    return f.mean(0), f.std(0).clamp_min(1e-6)


# ---------------------------------------------------------------- policies -> exit rounds
def match_budget(rounds_at, s_lo, s_hi, budget, u, iters=50, layers_of=None):
    """rounds_at(s) gives per-token exit rounds (0-based), with mean cost nondecreasing in s. Bisection finds s_lo/s_hi
    bracketing the budget; tokens with u < f take the s_hi rounds, so the mean cost equals the budget (up to the
    granularity of u). layers_of(rounds) is the per-token cost (default 24 * (round + 1)). None if unreachable."""
    layers_of = layers_of or (lambda rr: N_LAYERS * (rr + 1).float())
    lo, hi = rounds_at(s_lo), rounds_at(s_hi)
    L_lo, L_hi = layers_of(lo).mean().item(), layers_of(hi).mean().item()
    if not (L_lo <= budget <= L_hi):
        return None
    for _ in range(iters):
        mid = (s_lo + s_hi) / 2
        rm = rounds_at(mid)
        Lm = layers_of(rm).mean().item()
        if Lm <= budget:
            s_lo, lo, L_lo = mid, rm, Lm
        else:
            s_hi, hi, L_hi = mid, rm, Lm
    f = 0.0 if L_hi == L_lo else (budget - L_lo) / (L_hi - L_lo)
    return torch.where(u < f, hi, lo)


def gate_rounds(pdf):
    return lambda q: decide(pdf, q)


def nll_at(ce, rr):
    return ce.gather(-1, rr[:, None])[:, 0]


def bias_rounds(budget, u):
    """Static mixed policy: every token exits at one of two adjacent rounds at random, hitting the budget."""
    k = min(int(budget // N_LAYERS), R - 1)  # round index with cost 24 (k+1) <= budget < 24 (k+2)
    k = max(k - 1, 0) if N_LAYERS * (k + 1) > budget else k
    f = (budget - N_LAYERS * (k + 1)) / N_LAYERS
    return torch.where(u < f, torch.full_like(u, k + 1, dtype=torch.long), torch.full_like(u, k, dtype=torch.long))


def kl_rounds(own_r, tau_ent, tau_kl):
    """own_r: list over rounds of raw `own` (n_tok, 5). Exit at round 1 if entropy < tau_ent, else first later round
    with KL < tau_kl, else the last round."""
    kl = torch.stack([o[:, 0] for o in own_r], -1)
    ok = kl < tau_kl
    ok[:, 0] = own_r[0][:, 2] < tau_ent
    return torch.where(ok.any(-1), ok.float().argmax(-1), torch.full(ok.shape[:-1], R - 1, dtype=torch.long))


def raw_own(D, c0, c1):
    return [torch.from_numpy(np.array(D["own"][c0:c1, :, r], dtype=np.float32)).reshape(-1, 5) for r in range(R)]


def cost_fn(arm, c_lm, c_att):
    """Charged per-token cost in layer-equivalents: readout arms pay a full-vocabulary readout for every round
    passed before the exit round; attention arms pay the explicit attention-probability overhead every round."""
    def layers_of(rr):
        L = N_LAYERS * (rr + 1).float()
        if arm in READOUT_ARMS:
            L = L + c_lm * rr.float()
        if arm in ATTN_ARMS:
            L = L + c_att * N_LAYERS * (rr + 1).float()
        return L
    return layers_of


# ---------------------------------------------------------------- fit

class GateBank:
    """G exit gates of one arm trained together: one standardization and one matrix product per round, each gate
    with its own weight/bias learning rate and compute price (manual Adam). Row g is exactly an exit_signals.Gate."""

    def __init__(self, dim, keys, per_round, mean=None, std=None, ouro_gate=None):
        self.keys, self.dim, G = keys, dim, len(keys)
        self.P = R if per_round else 1
        self.mean = torch.zeros(dim) if mean is None else mean.float()
        self.std = torch.ones(dim) if std is None else std.float()
        self.W = torch.zeros(G, self.P, dim)
        self.b = torch.zeros(G, R)
        if ouro_gate is not None:
            self.W[:] = ouro_gate.weight.detach().float()[None]
            self.b[:] = float(ouro_gate.bias)
        self.lw = torch.tensor([k[1] for k in keys])
        self.lb = torch.tensor([k[2] for k in keys])
        self.price = torch.tensor([k[3] for k in keys])
        self.state = [torch.zeros_like(self.W), torch.zeros_like(self.W), torch.zeros_like(self.b),
                      torch.zeros_like(self.b)]
        self.t = 0
        self.active = torch.ones(G, dtype=torch.bool)
        self.best = torch.full((G,), math.inf)
        self.best_W, self.best_b = self.W.clone(), self.b.clone()
        self.bad = torch.zeros(G, dtype=torch.long)

    def pdf(self, feats, W=None, b=None):
        """feats: list over rounds of (n, dim) or None. Returns the exit pdf (n, G, R)."""
        W = self.W if W is None else W
        b = self.b if b is None else b
        lam = []
        for r, f in enumerate(feats):
            z = b[:, r][None]
            if self.dim:
                z = z + ((f - self.mean) / self.std) @ W[:, r if self.P > 1 else 0].T
            lam.append(torch.sigmoid(z))
        n = max(x.shape[0] for x in lam)
        rem, out = torch.ones(n, len(self.keys)), []
        for r, l_ in enumerate(lam):
            out.append(l_ * rem if r < R - 1 else rem)
            rem = rem * (1 - l_)
        return torch.stack(out, -1)

    def objective(self, feats, ce, W=None, b=None):
        """Per-gate exact expected objective (G,): mean over tokens of sum_r pdf_r (CE_r + price * 24 r)."""
        layers = N_LAYERS * torch.arange(1, R + 1, dtype=torch.float32)
        pdf = self.pdf(feats, W, b)
        return (pdf * (ce[:, None, :] + self.price[None, :, None] * layers)).sum(-1).mean(0)

    def step(self, feats, ce, betas=(0.9, 0.999), eps=1e-8):
        W, b = self.W.clone().requires_grad_(True), self.b.clone().requires_grad_(True)
        obj = self.objective(feats, ce, W, b)
        obj[self.active].sum().backward()
        self.t += 1
        with torch.no_grad():
            for p, g, m, v, lr, shape in ((self.W, W.grad, self.state[0], self.state[1], self.lw, (-1, 1, 1)),
                                          (self.b, b.grad, self.state[2], self.state[3], self.lb, (-1, 1))):
                m.mul_(betas[0]).add_(g, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
                upd = (m / (1 - betas[0] ** self.t)) / ((v / (1 - betas[1] ** self.t)).sqrt() + eps)
                p.sub_(upd * (lr * self.active).view(shape))

    def check(self, feats, ce):
        """Early stopping on the tuning objective; keep each gate's best weights."""
        with torch.no_grad():
            v = self.objective(feats, ce)
        better = (v < self.best - 1e-6) & self.active
        self.best = torch.where(better, v, self.best)
        self.best_W[better], self.best_b[better] = self.W[better], self.b[better]
        self.bad = torch.where(better, torch.zeros_like(self.bad), self.bad + self.active.long())
        self.active &= self.bad < PATIENCE

    def gate_state(self, g):
        """State dict of an exit_signals.Gate equal to row g's best weights."""
        sd = {"b": self.best_b[g].clone(), "mean": self.mean.clone(), "std": self.std.clone()}
        if self.dim:
            sd["w"] = self.best_W[g].clone()
        return sd


def auc_on(pdf, ce, u, budgets=BUDGETS):
    """Mean NLL over the budgets with exact budget matching (used for selection on the tuning split)."""
    vals = []
    for B in budgets:
        rr = match_budget(gate_rounds(pdf), 0.0, 1.0 + 1e-6, B, u)
        if rr is None:
            return math.inf
        vals.append(nll_at(ce, rr).mean().item())
    return float(np.mean(vals))


@torch.no_grad()
def gate_pdf(g, D, arm, block=16):
    """Exit pdf (n_tok, R) of a gate over a whole split, built in blocks of chunks (bounded memory)."""
    n = D["ce"].shape[0]
    return torch.cat([g(feats(D, arm, c0, min(c0 + block, n))) for c0 in range(0, n, block)])


def ouro_gate_module(model):
    from ouro_model import load_ouro
    return load_ouro(model, torch.float32).model.early_exit_gate


def fit(a):
    torch.manual_seed(0)
    Dtr, Dtu = load(a.cache, "train"), load(a.cache, "tuning")
    ntr, ntu = Dtr["ce"].shape[0], Dtu["ce"].shape[0]
    og = ouro_gate_module(a.model)
    arms = a.arms or list(LEARNED)
    tune = {"ce": ce_of(Dtu, 0, ntu)}
    u_tune = torch.rand(tune["ce"].shape[0], generator=torch.Generator().manual_seed(1))
    selected, log = {}, {"arms": {}}
    for arm in arms:
        tune = {"ce": tune["ce"], arm: feats(Dtu, arm, 0, ntu)}  # only this arm's tuning features in memory
        dim = tune[arm][0].shape[-1] if tune[arm][0] is not None else 0
        mean, std = stats(Dtr, arm)
        grid_w = list(LR_W_WARM if arm == "hidden_warm" else LR_W)
        grid_b = list(LR_B)
        tried, extensions = set(), 0
        while True:
            new = [(w, b) for w in grid_w for b in grid_b if (w, b) not in tried]
            keys = [(arm, w, b_, price) for w, b_ in new for price in PRICES]
            if arm == "hidden_warm":
                bank = GateBank(dim, keys, False, ouro_gate=og)
            else:
                bank = GateBank(dim, keys, arm in SMALL_ARMS, mean=mean, std=std)
            tried.update(new)
            t0 = time.time()
            for ep in range(EPOCHS):
                order = np.random.default_rng(ep).permutation(range(0, ntr, BLOCK))
                for c0 in order:
                    c1 = min(c0 + BLOCK, ntr)
                    f, ce = feats(Dtr, arm, c0, c1), ce_of(Dtr, c0, c1)
                    perm = torch.randperm(ce.shape[0])
                    for i in range(0, len(perm), MINIBATCH):
                        idx = perm[i:i + MINIBATCH]
                        bank.step([x[idx] if x is not None else None for x in f], ce[idx])
                bank.check(tune[arm], tune["ce"])
                print(f"  {arm}: epoch {ep + 1} ({time.time() - t0:.0f}s), {int(bank.active.sum())} of "
                      f"{len(keys)} gates still improving", flush=True)
                if not bank.active.any():
                    break
            # select on the tuning split's AUC (best epoch of each gate)
            with torch.no_grad():
                pdf_t = bank.pdf(tune[arm], bank.best_W, bank.best_b)
            for g, key in enumerate(keys):
                auc = auc_on(pdf_t[:, g], tune["ce"], u_tune)
                log["arms"].setdefault(arm, {})[f"lw{key[1]:g}_lb{key[2]:g}_p{key[3]:g}"] = dict(
                    tuning_objective=float(bank.best[g]), tuning_auc=auc)
                if arm not in selected or auc < selected[arm]["auc"]:
                    selected[arm] = dict(auc=auc, key=key, state=bank.gate_state(g), dim=dim)
            lw, lb = selected[arm]["key"][1], selected[arm]["key"][2]
            edge_w = lw in (min(grid_w), max(grid_w)) and len(grid_w) > 1
            edge_b = lb in (min(grid_b), max(grid_b)) and len(grid_b) > 1
            if extensions < 2 and (edge_w or edge_b):
                extensions += 1
                if edge_w:
                    grid_w.append(min(grid_w) / 3 if lw == min(grid_w) else max(grid_w) * 3)
                if edge_b:
                    grid_b.append(min(grid_b) / 10 if lb == min(grid_b) else max(grid_b) * 10)
                print(f"  {arm}: selected (lr_w {lw:g}, lr_b {lb:g}) on a grid edge; grids now "
                      f"{sorted(grid_w)} / {sorted(grid_b)}", flush=True)
                continue
            break
        log["arms"][arm]["selected"] = dict(lr_w=selected[arm]["key"][1], lr_b=selected[arm]["key"][2],
                                            price=selected[arm]["key"][3], tuning_auc=selected[arm]["auc"],
                                            grid_w=sorted(grid_w), grid_b=sorted(grid_b), extensions=extensions,
                                            on_edge_after_extensions=bool(
                                                selected[arm]["key"][1] in (min(grid_w), max(grid_w)) or
                                                selected[arm]["key"][2] in (min(grid_b), max(grid_b))))
        print(f"{arm}: selected {log['arms'][arm]['selected']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({k: dict(key=v["key"], state=v["state"], dim=v["dim"]) for k, v in selected.items()},
               os.path.join(a.out, "gates.pt"))
    tmp = os.path.join(a.out, "fit_log.json.tmp")
    json.dump(log, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "fit_log.json"))


# ---------------------------------------------------------------- report
def boot(x, n=10000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.percentile(m, 100 * a)), float(np.percentile(m, 100 * (1 - a)))]


def readout_costs(model):
    """Layer-equivalents of one full-vocabulary readout and of explicit attention probabilities per layer."""
    from huggingface_hub import hf_hub_download
    c = json.load(open(hf_hub_download(model, "config.json")))
    d, V, f = c["hidden_size"], c["vocab_size"], c["intermediate_size"]
    H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
    layer = d * hd * (H + 2 * KV) + H * hd * d + 3 * d * f  # q, k, v, o and the gated MLP (multiply-adds per token)
    c_lm = V * d / layer
    c_att = (512 * H * hd) / layer  # one extra pass over Q.K^T at the mean context length of a 1,024-token chunk
    return c_lm, c_att


def curves(get_rounds, ce, u, chunk_tok, layers_of=None, s_range=(0.0, 1.0 + 1e-6)):
    """Per-chunk NLL (n_chunks, n_budgets) at exactly matched budgets; NaN where unreachable."""
    out = np.full((ce.shape[0] // chunk_tok, len(BUDGETS)), np.nan)
    for j, B in enumerate(BUDGETS):
        rr = match_budget(get_rounds, *s_range, B, u, layers_of=layers_of)
        if rr is not None:
            out[:, j] = nll_at(ce, rr).reshape(-1, chunk_tok).mean(1).numpy()
    return out


def report(a):
    De, Dtu = load(a.cache, "eval"), load(a.cache, "tuning")
    ne, ntu = De["ce"].shape[0], Dtu["ce"].shape[0]
    T1 = De["ce"].shape[1]
    ce_e, ce_t = ce_of(De, 0, ne), ce_of(Dtu, 0, ntu)
    u_e = torch.rand(ce_e.shape[0], generator=torch.Generator().manual_seed(2))
    u_t = torch.rand(ce_t.shape[0], generator=torch.Generator().manual_seed(1))
    sel = torch.load(os.path.join(a.out, "gates.pt"))
    og = ouro_gate_module(a.model)
    c_lm, c_att = readout_costs(a.model)
    pdfs, pdfs_t = {}, {}
    with torch.no_grad():
        g = Gate(og.in_features, R, ouro_gate=og)
        pdfs["ouro_gate"], pdfs_t["ouro_gate"] = gate_pdf(g, De, "hidden_warm"), gate_pdf(g, Dtu, "hidden_warm")
        for arm, v in sel.items():
            g = Gate(v["dim"], R, per_round=arm in SMALL_ARMS) if arm != "hidden_warm" else Gate(v["dim"], R, ouro_gate=og)
            g.load_state_dict(v["state"])
            pdfs[arm] = gate_pdf(g, De, arm)
    # SESOI from the tuning split: the NLL worth 2 layers on Ouro's gate curve between 65 and 75 layers
    n65 = nll_at(ce_t, match_budget(gate_rounds(pdfs_t["ouro_gate"]), 0, 1 + 1e-6, 65, u_t)).mean().item()
    n75 = nll_at(ce_t, match_budget(gate_rounds(pdfs_t["ouro_gate"]), 0, 1 + 1e-6, 75, u_t)).mean().item()
    sesoi = 2 * (n65 - n75) / 10
    res = dict(sesoi=sesoi, budgets=BUDGETS, n_eval_chunks=ne, n_eval_tokens=int(ce_e.shape[0]), c_lm=c_lm,
               c_att=c_att, arms={}, charged={}, contrasts={}, verdict={})
    per_chunk, per_chunk_charged = {}, {}
    for arm, pdf in pdfs.items():
        per_chunk[arm] = curves(gate_rounds(pdf), ce_e, u_e, T1)
        per_chunk_charged[arm] = curves(gate_rounds(pdf), ce_e, u_e, T1, layers_of=cost_fn(arm, c_lm, c_att))
    per_chunk["bias"] = np.stack([nll_at(ce_e, bias_rounds(B, u_e)).reshape(-1, T1).mean(1).numpy() for B in BUDGETS], 1)
    # zero-parameter rule: tau_ent chosen per budget on tuning (tau_kl matched to the budget), then matched on eval
    own_e, own_t = raw_own(De, 0, ne), raw_own(Dtu, 0, ntu)
    kl_cols, kl_cols_ch, kl_choice = [], [], {}
    for B in BUDGETS:
        best = None
        for te in (0.0, 0.03, 0.1, 0.3, 1.0):
            rr = match_budget(lambda s: kl_rounds(own_t, te, math.exp(-s)), -5.0, 25.0, B, u_t)
            if rr is not None:
                v = nll_at(ce_t, rr).mean().item()
                best = (v, te) if best is None or v < best[0] else best
        col, colc = np.full(ne, np.nan), np.full(ne, np.nan)
        if best is not None:
            te = best[1]
            kl_choice[B] = te
            rr = match_budget(lambda s: kl_rounds(own_e, te, math.exp(-s)), -5.0, 25.0, B, u_e)
            if rr is not None:
                col = nll_at(ce_e, rr).reshape(-1, T1).mean(1).numpy()
            rr = match_budget(lambda s: kl_rounds(own_e, te, math.exp(-s)), -5.0, 25.0, B, u_e,
                              layers_of=cost_fn("kl_rule", c_lm, c_att))
            if rr is not None:
                colc = nll_at(ce_e, rr).reshape(-1, T1).mean(1).numpy()
        kl_cols.append(col)
        kl_cols_ch.append(colc)
    per_chunk["kl_rule"], per_chunk_charged["kl_rule"] = np.stack(kl_cols, 1), np.stack(kl_cols_ch, 1)
    per_chunk_charged["bias"] = per_chunk["bias"]
    res["kl_rule_tau_ent"] = kl_choice
    # oracle (upper bound only) and fixed rounds
    orc = []
    for B in BUDGETS:
        rr = match_budget(lambda s: (ce_e - s * N_LAYERS * torch.arange(1, R + 1)).argmin(-1), -0.2, 0.05, B, u_e)
        orc.append(nll_at(ce_e, rr).mean().item() if rr is not None else None)
    res["oracle_upper_bound"] = dict(zip(BUDGETS, orc))
    res["fixed_rounds"] = {N_LAYERS * (r + 1): float(ce_e[:, r].mean()) for r in range(R)}
    for arm, m in per_chunk.items():
        res["arms"][arm] = dict(auc=float(np.nanmean(m.mean(1))) if not np.isnan(m).all() else None,
                                nll_at_budget={B: (None if np.isnan(m[:, j]).all() else float(np.nanmean(m[:, j])))
                                               for j, B in enumerate(BUDGETS)},
                                complete=bool(not np.isnan(m).any()))
    auc = {k: m.mean(1) for k, m in per_chunk.items() if not np.isnan(m).any()}
    aucc = {k: m.mean(1) for k, m in per_chunk_charged.items() if not np.isnan(m).any()}

    def contrast(x, y, A=auc):
        if x not in A or y not in A:
            return None
        d = A[x] - A[y]  # negative: x is better
        return dict(delta=float(d.mean()), ci95=boot(d), ci90=boot(d, level=0.90))

    pairs = [("attn_own", "own"), ("attn_own", "layers_own"), ("src_own", "own"), ("src_own", "srcperm_own"),
             ("all", "hidden_own"), ("own", "hidden"), ("hidden_own", "hidden"), ("all", "hidden"),
             ("hidden", "ouro_gate"), ("hidden_warm", "ouro_gate"), ("kl_rule", "hidden"), ("kl_rule", "ouro_gate"),
             ("pos", "bias"), ("hidden", "bias")]
    for x, y in pairs:
        res["contrasts"][f"{x} - {y}"] = contrast(x, y)
        res["charged"][f"{x} - {y}"] = contrast(x, y, aucc)

    # verdicts (SPEC.md E15 section 4)
    def pval(x, y):  # two-sided chunk-bootstrap p for the AUC difference
        d = auc[x] - auc[y]
        m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
        return float(min(1.0, 2 * min((m >= 0).mean(), (m <= 0).mean())))

    # each attention arm must beat `own` AND its twin (intersection-union: p = the larger p); Holm over the 2 arms
    p_arm, size_ok = {}, {}
    for arm, twin in (("attn_own", "layers_own"), ("src_own", "srcperm_own")):
        if all(k in auc for k in (arm, "own", twin)):
            p_arm[arm] = max(pval(arm, "own"), pval(arm, twin))
            size_ok[arm] = all(-(auc[arm] - auc[y]).mean() >= sesoi for y in ("own", twin))
    attn_pass, prev_ok = {}, True
    for i, (p, arm) in enumerate(sorted((p, a_) for a_, p in p_arm.items())):
        attn_pass[arm] = bool(prev_ok and p <= 0.05 / (len(p_arm) - i) and size_ok[arm])
        prev_ok = prev_ok and p <= 0.05 / (len(p_arm) - i)
    all_vs = res["contrasts"]["all - hidden_own"]
    all_ok = all_vs is not None and all_vs["ci95"][1] < 0
    res["verdict"]["attention_helps"] = dict(attn_own=attn_pass.get("attn_own", False),
                                             src_own=attn_pass.get("src_own", False), all_beats_hidden_own=all_ok,
                                             p_intersection_union=p_arm, size_ok=size_ok,
                                             passed=bool(any(attn_pass.values()) and all_ok))
    conf = [res["contrasts"]["own - hidden"], res["contrasts"]["hidden_own - hidden"]]
    res["verdict"]["confidence_helps"] = bool(any(c is not None and -c["delta"] >= sesoi and c["ci95"][1] < 0
                                                  for c in conf)) and not res["verdict"]["attention_helps"]["passed"]
    tost = [res["contrasts"]["hidden_own - hidden"], res["contrasts"]["all - hidden"]]
    res["verdict"]["hidden_suffices"] = bool(all(c is not None and -sesoi < c["ci90"][0] and c["ci90"][1] < sesoi
                                                 for c in tost))
    v = res["verdict"]
    res["verdict"]["summary"] = ("ATTENTION HELPS" if v["attention_helps"]["passed"] else
                                 "CONFIDENCE HELPS" if v["confidence_helps"] else
                                 "HIDDEN SUFFICES" if v["hidden_suffices"] else "INCONCLUSIVE")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "results.json.tmp")
    json.dump(res, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, os.path.join(a.out, "results.json"))
    write_report(res, os.path.join(a.out, "REPORT.md"))
    print(open(os.path.join(a.out, "REPORT.md")).read())


def write_report(res, path):
    f = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
    L = ["# E15 report (written by e15_fit.py; do not edit)", "",
         f"**Verdict: {res['verdict']['summary']}**", "",
         f"Evaluation: {res['n_eval_chunks']} chunks, {res['n_eval_tokens']:,} tokens. SESOI {res['sesoi']:.4f} "
         f"nats/token (2 layers on Ouro's gate curve, tuning split). Readout = {res['c_lm']:.2f} layer-equivalents; "
         f"attention-probability overhead = {res['c_att']:.3f} per layer.", "",
         "## Primary: mean NLL over budgets " + ", ".join(map(str, res["budgets"])) + " layers/token", "",
         "| Arm | AUC | " + " | ".join(str(b) for b in res["budgets"]) + " |",
         "|---|---|" + "---|" * len(res["budgets"])]
    for arm, v in sorted(res["arms"].items(), key=lambda kv: (kv[1]["auc"] is None, kv[1]["auc"] or 0)):
        L.append(f"| {arm} | {f(v['auc'])} | " + " | ".join(f(v["nll_at_budget"][b]) for b in res["budgets"]) + " |")
    L += ["", "Fixed rounds: " + ", ".join(f"{k} layers {v:.4f}" for k, v in res["fixed_rounds"].items()),
          "Hindsight oracle (upper bound only): " + ", ".join(f"{k}: {f(v)}" for k, v in res["oracle_upper_bound"].items()),
          "", "## Contrasts (AUC difference, negative = first arm better; 95% and 90% chunk-bootstrap CIs)", "",
          "| Contrast | Δ | 95% CI | 90% CI | Charged Δ | Charged 95% CI |", "|---|---|---|---|---|---|"]
    for k, c in res["contrasts"].items():
        ch = res["charged"].get(k)
        L.append(f"| {k} | " + (f"{c['delta']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                                f"[{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}]" if c else "— | — | —") + " | " +
                 (f"{ch['delta']:+.4f} | [{ch['ci95'][0]:+.4f}, {ch['ci95'][1]:+.4f}]" if ch else "— | —") + " |")
    L += ["", "## Verdict details", "", "```", json.dumps(res["verdict"], indent=1, default=float), "```", ""]
    tmp = path + ".tmp"
    open(tmp, "w").write("\n".join(L))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "report"])
    p.add_argument("--cache", default="results/e15/cache")
    p.add_argument("--out", default="results/e15")
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--arms", nargs="+", default=None)
    a = p.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    fit(a) if a.cmd == "fit" else report(a)


if __name__ == "__main__":
    main()
