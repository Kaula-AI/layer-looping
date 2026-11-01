"""Why does the trained per-layer router lose to Ouro's own exit gate? Diagnostics on held-out FineWeb-Edu.

1. Compute-loss curve: NLL vs layers/token for fixed rounds, Ouro's gate at several thresholds, the router
   (argmax, as benchmarked, and Gumbel-sampled at the final training temperature).
2. Is routing per token? Replace the router's decisions by each layer's majority action (one static
   schedule for every token) and by its own decisions shuffled across tokens; compare NLL at equal compute.
3. Where does compute go? Correlation of each token's layers with its base-model loss (router vs gate),
   and skip/twice rates per virtual layer and per round.

  python diagnose_router.py --ckpt r5/router72_frozen/model.pt --out diag_frozen
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ouro_model import load_ouro
from routed_ouro import RoutedOuro, fineweb_chunks


def tok_nll(logits, ids):
    return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")  # B, T-1


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ByteDance/Ouro-1.4B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_chunks", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--out", default="diag")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    ouro = load_ouro(a.model, torch.bfloat16).cuda()
    ck = torch.load(a.ckpt, map_location="cuda")
    if "ouro" in ck:  # full checkpoint; a heads-only file means the original (frozen) Ouro weights
        ouro.load_state_dict(ck["ouro"])
    rm = RoutedOuro(ouro).cuda().eval()
    rm.heads.load_state_dict(ck["heads"])
    n, R, V = rm.n, rm.R, rm.V
    # held-out: routed_ouro.py keeps its first --eval_chunks (64) chunks out of training
    assert a.n_chunks <= 64
    chunks = fineweb_chunks(tok, a.seq_len, a.n_chunks)

    res = {k: [] for k in ["gate_p", "rounds_nll", "argmax", "argmax_lay", "sampled", "sampled_lay",
                           "static", "shuffled", "acts"]}
    for i in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[i: i + a.batch], device="cuda")
        outs, gates = ouro.rounds(ids)
        res["rounds_nll"].append(torch.stack([tok_nll(ouro.lm_head(h), ids) for h in outs], -1).cpu())  # B,T-1,R
        lam = [torch.sigmoid(g_.float()[..., 0]) for g_ in gates]
        rem, pdf = torch.ones_like(lam[0]), []
        for s, l_ in enumerate(lam):
            pdf.append(l_ * rem if s < R - 1 else rem)
            rem = rem * (1 - l_)
        res["gate_p"].append(torch.stack(pdf, -1)[:, :-1].cpu())
        rec = []
        lg, lay, _, _ = rm(ids, hard_eval=True, record=rec)
        res["argmax"].append(tok_nll(lg, ids).cpu())
        res["argmax_lay"].append(lay[:, :-1].cpu())
        acts = torch.zeros(ids.shape[0], ids.shape[1], V, dtype=torch.long, device="cuda") + 1
        for j, act, _ in rec:
            acts[..., j] = act
        res["acts"].append(acts.cpu())
        lg, lay, _, _ = rm(ids, tau=a.tau, hard_eval=False)
        res["sampled"].append(tok_nll(lg, ids).cpu())
        res["sampled_lay"].append(lay[:, :-1].cpu())
        print(f"{i + ids.shape[0]}/{len(chunks)}", flush=True)

    acts = torch.cat(res["acts"])  # N, T, V
    maj = torch.mode(acts.reshape(-1, V), 0).values  # V: majority action per layer
    for i in range(0, len(chunks), a.batch):
        ids = torch.tensor(chunks[i: i + a.batch], device="cuda")
        B, T = ids.shape
        lg, _, _, _ = rm(ids, force=maj.cuda().expand(B, T, V))
        res["static"].append(tok_nll(lg, ids).cpu())
        a_ = acts[i: i + B].reshape(-1, V)
        perm = torch.randperm(a_.shape[0], generator=torch.Generator().manual_seed(i))
        lg, _, _, _ = rm(ids, force=a_[perm].reshape(B, T, V).cuda())
        res["shuffled"].append(tok_nll(lg, ids).cpu())

    cat = {k: torch.cat(v).float().numpy() for k, v in res.items() if k != "acts"}
    rounds = cat["rounds_nll"]  # N, T-1, R
    base = rounds[..., -1]
    out = {"n_tokens": int(base.size)}
    curve = [dict(policy=f"rounds:{r + 1}", layers=float(n * (r + 1)), nll=float(rounds[..., r].mean()))
             for r in range(R)]
    cum = np.cumsum(cat["gate_p"], -1)
    gate_lay = {}
    for q in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        hit = cum >= q
        s = np.where(hit.any(-1), hit.argmax(-1), R - 1)  # 0-based exit round
        nll = np.take_along_axis(rounds, s[..., None], -1)[..., 0]
        gate_lay[q] = (s + 1) * n
        curve.append(dict(policy=f"gate:{q}", layers=float(gate_lay[q].mean()), nll=float(nll.mean())))
    for k in ("argmax", "sampled"):
        curve.append(dict(policy=f"router_{k}", layers=float(cat[f"{k}_lay"].mean()), nll=float(cat[k].mean())))
    curve.append(dict(policy="router_static_schedule", layers=float(maj.sum().item()), nll=float(cat["static"].mean())))
    curve.append(dict(policy="router_shuffled", layers=float(cat["argmax_lay"].mean()), nll=float(cat["shuffled"].mean())))
    out["curve"] = sorted(curve, key=lambda d: d["layers"])

    # per-token variation of the router's decisions
    A = acts[:, :-1].reshape(-1, V).numpy()
    rout = [j for j in range(V) if rm.routable(j)]
    agree = np.array([(A[:, j] == maj[j].item()).mean() for j in rout])
    out["router_decision_agreement_with_majority"] = dict(mean=float(agree.mean()), min=float(agree.min()))
    rates = {a_: [float((A[:, j] == a_).mean()) for j in range(V)] for a_ in (0, 2)}
    out["skip_rate_by_layer"], out["twice_rate_by_layer"] = rates[0], rates[2]
    out["skip_rate_by_round"] = [float(np.mean(rates[0][r * n:(r + 1) * n])) for r in range(R)]
    out["twice_rate_by_round"] = [float(np.mean(rates[2][r * n:(r + 1) * n])) for r in range(R)]

    # does compute follow difficulty? Spearman of layers vs base-model token loss
    from scipy.stats import spearmanr
    out["spearman_layers_vs_base_nll"] = {
        "router_argmax": float(spearmanr(cat["argmax_lay"].ravel(), base.ravel())[0]),
        "gate:0.2": float(spearmanr(gate_lay[0.2].ravel(), base.ravel())[0]),
        "gate:0.5": float(spearmanr(gate_lay[0.5].ravel(), base.ravel())[0]),
    }
    # value of the 4th round per token vs which tokens each policy gives it
    gain4 = rounds[..., 2] - rounds[..., 3]  # NLL saved by round 4 over round 3
    out["round4_gain_mean"] = float(gain4.mean())
    out["round4_gain_corr_router_layers"] = float(spearmanr(cat["argmax_lay"].ravel(), gain4.ravel())[0])
    out["round4_gain_corr_gate0.2_layers"] = float(spearmanr(gate_lay[0.2].ravel(), gain4.ravel())[0])
    json.dump(out, open(os.path.join(a.out, "diagnose.json"), "w"), indent=1)

    print(f"\n{'policy':26s} {'layers/tok':>10} {'NLL':>8}")
    for d in out["curve"]:
        print(f"{d['policy']:26s} {d['layers']:10.1f} {d['nll']:8.4f}")
    for k in ("router_decision_agreement_with_majority", "skip_rate_by_round", "twice_rate_by_round",
              "spearman_layers_vs_base_nll", "round4_gain_mean", "round4_gain_corr_router_layers",
              "round4_gain_corr_gate0.2_layers"):
        print(k, json.dumps(out[k]))
    print("skip rate by layer:", " ".join(f"{v:.2f}" for v in out["skip_rate_by_layer"]))


if __name__ == "__main__":
    main()
