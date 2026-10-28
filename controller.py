"""
E11: the cheap controller. A small MLP predicts, from cheap per-token signals, how much each
library path would change that token's loss. The route is the path with the best predicted
dNLL + lambda * cost (or the unedited model if nothing beats 0), so one trained model gives the
whole compute-accuracy frontier by sweeping lambda.

Training signal (full information: every library path is scored exactly for every token):
  loss = MSE(pred, dNLL) + beta * E_{softmax(-(pred + lambda*cost)/tau)}[dNLL + lambda*cost]
The second term is decision-focused: it rewards predictions that lead to good choices (the
"RL" part; with exact rewards for every arm it needs no sampling).

Compared on held-out chunks, as frontiers (NLL saved vs extra layer applications per token):
  oracle over the library   oracle over single edits   per-token search oracle (search.npz)
  MLP controller on t-1 features (plan ahead) and on same-token features
  ridge over the library (t-1, interp and raw)   ridge restricted to single edits   best static path

  python controller.py --library programs_quick/library.npz --search programs_quick/search.npz \\
         --out programs_quick
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class Controller(nn.Module):
    """features -> predicted dNLL for every library path."""

    def __init__(self, d_in, n_prog, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_prog))

    def forward(self, x):
        return self.net(x)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--library", default="programs_out/library.npz")
    p.add_argument("--search", default=None, help="search.npz for the per-token search oracle (same chunks)")
    p.add_argument("--out", default="programs_out")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_train", type=int, default=400_000)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--budgets", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def sample_lam(n, gen):
    lam = 10 ** (torch.rand(n, generator=gen) * 3 - 3) * 0.5  # log-uniform [5e-4, 0.5]
    return torch.where(torch.rand(n, generator=gen) < 0.25, torch.zeros(n), lam)


def train_controller(Xtr, Ytr, cost, a, dev):
    gen = torch.Generator().manual_seed(a.seed)
    torch.manual_seed(a.seed)
    model = Controller(Xtr.shape[1], Ytr.shape[1], a.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor(Ytr, dtype=torch.float32).clamp(-2, 2)
    c = torch.tensor(cost, dtype=torch.float32, device=dev)
    n = len(X)
    for ep in range(a.epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(0, n, a.batch):
            idx = perm[s : s + a.batch]
            x, y = X[idx].to(dev), Y[idx].to(dev)
            pred = model(x)
            lam = sample_lam(len(idx), gen).to(dev)[:, None]
            pick = (-(pred + lam * c[None]) / a.tau).softmax(-1)
            loss = F.mse_loss(pred, y) + a.beta * (pick * (y + lam * c[None])).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  epoch {ep + 1}/{a.epochs}  loss {tot / n:.5f}", flush=True)
    return model


@torch.no_grad()
def predict(model, X, dev):
    return np.concatenate([model(torch.tensor(X[s : s + 8192], dtype=torch.float32, device=dev)).cpu().numpy()
                           for s in range(0, len(X), 8192)])


def main():
    a = get_args()
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.library)
    delta = d["delta"].astype(np.float32)
    cost = d["cfg_cost"].astype(np.float32)
    op = d["cfg_op"].astype(str)
    names = d["cfg_name"].astype(str)
    feat, nll = d["feat"].astype(np.float32), d["base_nll"].astype(np.float32)
    raw = d["raw"].astype(np.float32)
    chunk, pos = d["chunk"], d["pos"]
    base_idx = int(np.where(op == "base")[0][0])
    i = np.where(pos > 0)[0]
    j = i - 1
    test = chunk[i] % 5 == 0
    train = ~test
    Y = delta[i]
    Yte = Y[test]
    ar = np.arange(len(Yte))
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 0.5, 24)])

    def realize(ch):
        hit = ch >= 0
        cc = np.maximum(ch, 0)
        return float(np.where(hit, Yte[ar, cc], 0).mean()), float(np.where(hit, cost[cc], 0).mean())

    def argmin_route(P, lam, allowed=None):
        obj = P + lam * cost[None]
        if allowed is not None:
            obj = np.where(allowed[None], obj, np.inf)
        c = obj.argmin(1)
        return np.where((obj[np.arange(len(P)), c] < 0) & (c != base_idx), c, -1)

    def front(choose):
        return [dict(lam=None, dnll=0.0, extra_layers=0.0)] + \
               [dict(lam=float(l_), dnll=dn, extra_layers=ex) for l_ in lams for dn, ex in [realize(choose(l_))]]

    res = {"library_size": int(len(op)), "n_composite": int((op == "composite").sum()), "frontiers": {}}
    F_ = res["frontiers"]
    F_["oracle_library"] = front(lambda l_: argmin_route(Yte, l_))
    single = np.isin(op, ["repeat", "skip", "swap", "reverse", "loop", "base"])
    F_["oracle_single_edits"] = front(lambda l_: argmin_route(Yte, l_, single))

    if a.search:
        Sx = np.load(a.search)
        assert len(Sx["front_dnll"]) == len(delta), "search.npz must cover the same chunks"
        fd, fc = Sx["front_dnll"][i][test], Sx["front_cost"][i][test]

        def search_front(l_):
            obj = np.where(np.isnan(fd), np.inf, fd + l_ * fc)
            q = obj.argmin(1)
            take = obj[ar, q] < 0
            return float(np.where(take, fd[ar, q], 0).mean()), float(np.where(take, fc[ar, q], 0).mean())
        F_["oracle_search"] = [dict(lam=None, dnll=0.0, extra_layers=0.0)] + \
            [dict(lam=float(l_), dnll=dn, extra_layers=ex) for l_ in lams for dn, ex in [search_front(l_)]]

    def static_choose(l_):
        m = (Y[train] + l_ * cost[None]).mean(0)
        m[base_idx] = 0.0
        c = int(m.argmin())
        return np.full(len(Yte), c if m[c] < 0 else -1)
    F_["static_best"] = front(static_choose)

    def ridge_preds(X, allowed=None):
        Xtr, Ytr = X[train], Y[train]
        if len(Xtr) > 200_000:
            k = rng.choice(len(Xtr), 200_000, replace=False)
            Xtr, Ytr = Xtr[k], Ytr[k]
        sc = StandardScaler().fit(Xtr)
        return Ridge(alpha=a.alpha).fit(sc.transform(Xtr), Ytr).predict(sc.transform(X[test]))
    P = ridge_preds(feat[j])
    F_["ridge_lag_library"] = front(lambda l_: argmin_route(P, l_))
    F_["ridge_lag_single_edits"] = front(lambda l_: argmin_route(P, l_, single))
    P_raw = ridge_preds(raw[j])
    F_["ridge_lag_raw_library"] = front(lambda l_: argmin_route(P_raw, l_))

    ckpt = {}
    for nm, X in [("mlp_lag", np.c_[feat[j], nll[j]]), ("mlp_same", feat[i])]:
        sc = StandardScaler().fit(X[train])
        Xtr, Ytr = sc.transform(X[train]), Y[train]
        if len(Xtr) > a.max_train:
            k = rng.choice(len(Xtr), a.max_train, replace=False)
            Xtr, Ytr = Xtr[k], Ytr[k]
        print(f"training {nm} on {len(Xtr):,} tokens, {Y.shape[1]} programs", flush=True)
        model = train_controller(Xtr, Ytr, cost, a, dev).eval()
        Xte = sc.transform(X[test])
        Pm = predict(model, Xte, dev)
        F_[nm] = front(lambda l_, Pm=Pm: argmin_route(Pm, l_))
        ch0 = argmin_route(Pm, 0.0)
        res[nm + "_edit_frac_lam0"] = float((ch0 >= 0).mean())
        res[nm + "_composite_frac_lam0"] = float((op[ch0[ch0 >= 0]] == "composite").mean()) if (ch0 >= 0).any() else 0.0
        ckpt[nm] = dict(state=model.state_dict(), mean=sc.mean_, scale=sc.scale_, d_in=X.shape[1])

    def at_budget(fr, b):
        ok = [q["dnll"] for q in fr if q["extra_layers"] <= b + 1e-9]
        return float(min(ok)) if ok else None
    res["budget_matched"] = {nm: {str(b): at_budget(fr, b) for b in a.budgets} for nm, fr in F_.items()}
    print("\n== E11 budget-matched dNLL (best point with <= budget extra layers/token; lower is better) ==")
    print(f"  {'':26s}" + "".join(f"{b:>+9.2f}" for b in a.budgets))
    for nm, row in res["budget_matched"].items():
        print(f"  {nm:26s}" + "".join(f"{v:+9.4f}" if v is not None else f"{'—':>9s}" for v in row.values()))
    for nm in ("mlp_lag", "mlp_same"):
        print(f"  {nm}: edits {res[nm + '_edit_frac_lam0']:.1%} of tokens at lambda=0, "
              f"{res[nm + '_composite_frac_lam0']:.1%} of edits are multi-edit paths")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "controller_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save(dict(models=ckpt, cfg_path=d["cfg_path"], cfg_cost=cost, cfg_name=names, base_idx=base_idx,
                    hidden=a.hidden), os.path.join(a.out, "controller.pt"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def pareto(points):
            out, best = [], float("inf")
            for q in sorted(points, key=lambda q: (q["extra_layers"], q["dnll"])):
                if q["dnll"] < best - 1e-12:
                    out.append(q)
                    best = q["dnll"]
            return out
        style = {"oracle_search": ("#000000", "-"), "oracle_library": ("#d4a017", "-"),
                 "oracle_single_edits": ("#d4a017", ":"), "mlp_same": ("#2b8cbe", "-"), "mlp_lag": ("#41ab5d", "-"),
                 "ridge_lag_library": ("#41ab5d", "--"), "ridge_lag_single_edits": ("#41ab5d", ":"),
                 "ridge_lag_raw_library": ("#756bb1", ":"), "static_best": ("#de2d26", "--")}
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        for nm, (c, ls) in style.items():
            if nm in F_:
                fr = pareto(F_[nm])
                ax.plot([q["extra_layers"] for q in fr], [-q["dnll"] for q in fr], ls, color=c, lw=2, label=nm,
                        marker=".", ms=3)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("extra layer applications per token (negative = fewer)")
        ax.set_ylabel("NLL saved (nats / token)")
        ax.set_title("Per-token layer paths: controller vs oracles")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "controller_frontier.png"), dpi=200)
        print(f"\nsaved {a.out}/controller_frontier.png, controller_results.json, controller.pt")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
