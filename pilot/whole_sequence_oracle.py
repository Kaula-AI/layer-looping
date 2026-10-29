"""
PILOT, not the per-token oracle. Applies each layer program to the WHOLE sequence (every position
runs it), so a token's dNLL also includes the effect of edited earlier tokens it attends to.
The per-token oracle is ../oracle.py (diagonal swap). Output has probe.py's oracle.npz format.

Programs (L = n_layers, 0-indexed layers):
  repeat@l x k   run layer l k times in a row          cost k-1
  skip@l         omit layer l                          cost -1
  swap@l         run layer l+1 before layer l          cost 0

Row t of a chunk is the prediction made at position t of token t+1. Features come from the base pass:
  feat[:, 0L:1L]  lens_ent    entropy of the logit lens after layer l
  feat[:, 1L:2L]  lens_agree  probability the layer-l lens gives the base model's top-1 token
  feat[:, 2L:3L]  upd_norm    |resid_post_l - resid_pre_l| / |resid_pre_l|
  feat[:, 3L:4L]  attn_ent    attention entropy at layer l, mean over heads
  raw             residual stream after layer --raw_layer (float16)

With --sae_layers, also writes sae.npz next to --out for probe.py --sae: the top --sae_k SAE latents
(idx, val) on each listed layer's resid_post, for the same rows.

  python pilot/whole_sequence_oracle.py --model gpt2 --n_tokens 200000 --out oracle_out/oracle.npz
  python pilot/whole_sequence_oracle.py --model gemma-2-2b --n_tokens 250000 --batch 4 --sae_layers 3 7 11 15 19 23
"""
import argparse
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2", help="TransformerLens architecture name")
p.add_argument("--hf_weights", default=None,
               help="HF checkpoint to load into --model's architecture, e.g. Qwen/Qwen3-1.7B-Base into Qwen/Qwen3-1.7B")
p.add_argument("--dataset", default="NeelNanda/pile-10k")
p.add_argument("--seq_len", type=int, default=256)
p.add_argument("--n_tokens", type=int, default=200_000, help="prediction rows to collect")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--ks", type=int, nargs="+", default=[2, 3])
p.add_argument("--raw_layer", type=int, default=None, help="default: n_layers // 2")
p.add_argument("--dtype", default="float32")
p.add_argument("--no_tf32", action="store_true", help="exact fp32 matmuls (slower)")
p.add_argument("--sae_layers", type=int, nargs="*", default=[])
p.add_argument("--sae_release", default="gemma-scope-2b-pt-res-canonical")
p.add_argument("--sae_id", default="layer_{l}/width_16k/canonical")
p.add_argument("--sae_neuronpedia", default="gemma-2-2b/{l}-gemmascope-res-16k")
p.add_argument("--sae_k", type=int, default=64, help="latents kept per row and layer")
p.add_argument("--out", default="oracle_out/oracle.npz")
a = p.parse_args()
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
torch.backends.cudnn.allow_tf32 = not a.no_tf32
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

kw = {}
if a.hf_weights:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = dict(hf_model=AutoModelForCausalLM.from_pretrained(a.hf_weights, torch_dtype=getattr(torch, a.dtype)),
              tokenizer=AutoTokenizer.from_pretrained(a.hf_weights))
model = HookedTransformer.from_pretrained(a.model, device="cuda", dtype=getattr(torch, a.dtype), **kw)
del kw
model.eval()
L = model.cfg.n_layers
raw_layer = L // 2 if a.raw_layer is None else a.raw_layer

saes = {}
if a.sae_layers:
    from sae_lens import SAE

    for l in a.sae_layers:
        r = SAE.from_pretrained(a.sae_release, a.sae_id.format(l=l), device="cuda")
        saes[l] = (r[0] if isinstance(r, tuple) else r).eval()

# ---------------------------------------------------------------- programs
progs = []  # (op, layer, k, cost, order)
for l in range(L):
    for k in a.ks:
        progs.append(("repeat", l, k, k - 1, list(range(l)) + [l] * k + list(range(l + 1, L))))
for l in range(L):
    progs.append(("skip", l, 1, -1, [m for m in range(L) if m != l]))
for l in range(L - 1):
    order = list(range(L))
    order[l], order[l + 1] = l + 1, l
    progs.append(("swap", l, 1, 0, order))
print(f"{a.model}: {L} layers, {len(progs)} programs")

# ---------------------------------------------------------------- data: BOS + (seq_len-1) tokens per chunk
body = a.seq_len - 1
n_chunks = -(-a.n_tokens // body)
need = n_chunks * body
ids = []
for doc in load_dataset(a.dataset, split="train", streaming=True):
    ids.extend(model.tokenizer(doc["text"], add_special_tokens=False)["input_ids"] + [model.tokenizer.eos_token_id])
    if len(ids) >= need:
        break
bos = model.tokenizer.bos_token_id
if bos is None:  # e.g. Qwen: no BOS token, so chunks start with end-of-text
    bos = model.tokenizer.eos_token_id
toks = torch.tensor(ids[:need]).view(n_chunks, body)
toks = torch.cat([torch.full((n_chunks, 1), bos), toks], 1)


def nll_from_logits(logits, t):
    lp = logits[:, :-1].float().log_softmax(-1)
    return -lp.gather(-1, t[:, 1:, None])[..., 0]


def run(order, resid):
    for l in order:
        resid = model.blocks[l](resid)
    return model.unembed(model.ln_final(resid))


names = {"blocks.0.hook_resid_pre"} | {f"blocks.{l}.hook_resid_post" for l in range(L)} \
        | {f"blocks.{l}.attn.hook_pattern" for l in range(L)}
D, F, R, NLL, SI, SV = [], [], [], [], [], []
t0 = time.time()
for b0 in range(0, n_chunks, a.batch):
    t = toks[b0 : b0 + a.batch].cuda()
    logits, cache = model.run_with_cache(t, names_filter=lambda n: n in names)
    base = nll_from_logits(logits, t)
    top1 = logits[:, :-1].argmax(-1, keepdim=True)
    r0 = cache["blocks.0.hook_resid_pre"]

    # the manual layer loop must reproduce the model exactly, or every delta is meaningless
    if b0 == 0:
        err = (nll_from_logits(run(range(L), r0), t) - base).abs().max().item()
        assert err < 1e-3, f"manual forward disagrees with model by {err}"

    ent, agree, upd, attn = [], [], [], []
    prev = r0
    for l in range(L):
        post = cache[f"blocks.{l}.hook_resid_post"]
        lp = model.unembed(model.ln_final(post))[:, :-1].float().log_softmax(-1)
        ent.append(-(lp.exp() * lp).sum(-1))
        agree.append(lp.gather(-1, top1)[..., 0].exp())
        upd.append(((post - prev).norm(dim=-1) / prev.norm(dim=-1))[:, :-1].float())
        pat = cache[f"blocks.{l}.attn.hook_pattern"].float()  # [b, head, q, k]
        attn.append((-(pat * (pat + 1e-12).log()).sum(-1)).mean(1)[:, :-1])
        prev = post
    F.append(torch.stack(ent + agree + upd + attn, -1).flatten(0, 1).cpu().numpy())
    R.append(cache[f"blocks.{raw_layer}.hook_resid_post"][:, :-1].flatten(0, 1).half().cpu().numpy())
    if saes:
        tops = [saes[l].encode(cache[f"blocks.{l}.hook_resid_post"][:, :-1].flatten(0, 1).float())
                .topk(a.sae_k, -1) for l in a.sae_layers]
        SI.append(torch.stack([q.indices for q in tops], 1).int().cpu().numpy())
        SV.append(torch.stack([q.values for q in tops], 1).half().cpu().numpy())
    del cache

    D.append(torch.stack([nll_from_logits(run(o, r0), t) - base for *_, o in progs], -1)
             .flatten(0, 1).cpu().numpy())
    NLL.append(base.flatten().cpu().numpy())
    done = min(b0 + a.batch, n_chunks)
    print(f"  chunks {done}/{n_chunks}  {time.time() - t0:.0f}s", flush=True)

np.savez(
    a.out,
    delta=np.concatenate(D).astype(np.float32),
    base_nll=np.concatenate(NLL),
    feat=np.concatenate(F).astype(np.float32),
    raw=np.concatenate(R),
    chunk=np.repeat(np.arange(n_chunks), body),
    pos=np.tile(np.arange(body), n_chunks),
    n_layers=L,
    cfg_op=np.array([q[0] for q in progs]),
    cfg_layer=np.array([q[1] for q in progs]),
    cfg_k=np.array([q[2] for q in progs]),
    cfg_cost=np.array([q[3] for q in progs], dtype=np.float32),
    model=a.hf_weights or a.model,
    dataset=a.dataset,
    raw_layer=raw_layer,
)
print(f"saved {a.out}: {n_chunks * body} rows x {len(progs)} programs in {time.time() - t0:.0f}s")

if saes:
    sae_out = os.path.join(os.path.dirname(a.out) or ".", "sae.npz")
    np.savez(
        sae_out,
        idx=np.concatenate(SI),
        val=np.concatenate(SV),
        layers=np.array(a.sae_layers),
        width=saes[a.sae_layers[0]].cfg.d_sae,
        neuronpedia=np.array([a.sae_neuronpedia.format(l=l) for l in a.sae_layers]),
    )
    print(f"saved {sae_out}: layers {a.sae_layers}, top {a.sae_k} of {saes[a.sae_layers[0]].cfg.d_sae}")
