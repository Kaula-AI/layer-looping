"""E14 helpers (DESIGN.md §3): data splits, text NLL under an arm, MMLU through lm-evaluation-harness, provenance.

Splits (DESIGN.md §3.4): tuning text = WikiText-103 validation + GSM8K train; scored text = WikiText-103 test +
GSM8K test. Chunks alternate the two sources, seq_len tokens each, with BOS first when the tokenizer has one
(Qwen3 has none). Never split within a sequence.
"""
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
import torch
from datasets import load_dataset

from loop_wrap import LoopWindow

SPLITS = {"tuning": ("validation", "train"), "scored": ("test", "test")}


def _token_chunks(texts, tok, body):
    buf = []
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False)["input_ids"])
        while len(buf) >= body:
            yield buf[:body]
            buf = buf[body:]


def text_chunks(tok, split, n_tokens, seq_len=512):
    """Alternating WikiText-103 / GSM8K chunks for split 'tuning' or 'scored'. Returns (LongTensor (N, seq_len), names)."""
    wsplit, gsplit = SPLITS[split]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body = seq_len - len(bos)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=wsplit)
    gsm = load_dataset("openai/gsm8k", "main", split=gsplit)
    srcs = {"wiki": _token_chunks((t for t in wiki["text"] if t.strip()), tok, body),
            "gsm": _token_chunks((f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(gsm["question"], gsm["answer"])),
                                 tok, body)}
    want, chunks, names, order = math.ceil(n_tokens / seq_len), [], [], ["wiki", "gsm"]
    while len(chunks) < want and order:
        for name in list(order):
            try:
                c = next(srcs[name])
            except StopIteration:
                order.remove(name)
                continue
            chunks.append(bos + c)
            names.append(name)
            if len(chunks) >= want:
                break
    return torch.tensor(chunks), names


@torch.no_grad()
def text_nll(model, ids, arm, batch=2, record=None):
    """Per-token NLL (nats), shape (N, seq_len - 1), position t predicting t+1, with `arm` applied.
    Batches are fixed (same composition for every arm)."""
    dev, out = next(model.parameters()).device, []
    with LoopWindow(model, arm, record):
        for i in range(0, len(ids), batch):
            x = ids[i:i + batch].to(dev)
            logits = model(x, use_cache=False).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None])[..., 0]
            out.append((-lp).cpu())
    return torch.cat(out).numpy()


def run_mmlu(model, tok, arm, limit=None, batch=2, num_fewshot=5):
    """MMLU through lm-eval 0.4.x with `arm` applied. Returns (aggregate dict, per-item rows).
    Rows: subject, doc_id, gold, the four answer log-likelihoods, acc."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch)
    with LoopWindow(model, arm):
        res = simple_evaluate(model=lm, tasks=["mmlu"], num_fewshot=num_fewshot, limit=limit, log_samples=True,
                              bootstrap_iters=0, random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                              fewshot_random_seed=1234)
    rows = []
    for task, samples in res["samples"].items():
        for s in samples:
            ll = [float(r[0][0]) if isinstance(r[0], (list, tuple)) else float(r[0]) for r in s["resps"]]
            rows.append(dict(subject=task, doc_id=int(s["doc_id"]), gold=int(s["target"]), ll=ll,
                             acc=float(s["acc"])))
    agg = {k: v for k, v in res["results"].get("mmlu", {}).items() if isinstance(v, (int, float))}
    return agg, rows


def provenance(model_name, dtype, extra=None):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    import transformers

    try:
        import lm_eval
        lmv = lm_eval.__version__
    except Exception:
        lmv = None
    p = dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", "*.py")),
             model=model_name, dtype=str(dtype), device=platform.platform(), torch=torch.__version__,
             transformers=transformers.__version__, lm_eval=lmv, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    p.update(extra or {})
    return p


def write_json(path, obj):
    """Atomic write: one result file per finished job."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    os.replace(tmp, path)
