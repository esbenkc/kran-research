"""Vera: does the output-matching term pin the layer-28 country-token state? (pre-run fact check, own code)
Base Qwen2.5-1.5B-Instruct, CPU bf16, batch 1. Country token = second-to-last token (last token of the country name);
its own next-token output predicts the full stop. Fit a clean linear probe on post-norm hidden_states[28] at that
token (pool split from 1020's split.json), then for eval false statements compute the second-order minimal KL
(at that position only) that moves the probe score below the 1%-line:  KL_min = D^2 / (2 w' F^+ w),
F = W_U' (diag p - p p') W_U.  Also p(next = '.').
    cd kran-research && uv run python ../swarm-theory-1021-full-stack-evader/skeptic/vera_pin.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REF = Path("/Users/esben/work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after/data/ref1019/split.json")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
torch.manual_seed(0)
rows = json.load(open(REF))
rng = np.random.default_rng(1021)
pool = [i for i, r in enumerate(rows) if r["split"] in ("probe", "attack", "stop")]
pool = list(rng.choice(pool, 700, replace=False))
refit_t = [i for i, r in enumerate(rows) if r["split"] == "refit" and r["true"] == 1]
evf = [i for i, r in enumerate(rows) if r["split"] == "eval" and r["true"] == 0]
evf = list(rng.choice(evf, 40, replace=False))
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval()
WU = model.lm_head.weight  # [V, d] bf16, shared with the embedding; cast per chunk to keep RSS low
dot = tok(".", add_special_tokens=False)["input_ids"][0]


def run(i, want_p=False):
    ids = tok(rows[i]["text"], return_tensors="pt", add_special_tokens=False)["input_ids"]
    assert ids[0, -1].item() == dot, rows[i]["text"]
    with torch.no_grad():
        o = model(input_ids=ids, output_hidden_states=True)
    h = o.hidden_states[28][0, -2].float()
    if not want_p:
        return h.numpy()
    p = torch.softmax(o.logits[0, -2].float(), -1)
    return h.numpy(), p


H = {}
for name, idx in [("pool", pool), ("refit_t", refit_t)]:
    H[name] = np.stack([run(i) for i in idx])
    print(name, H[name].shape, flush=True)
y = np.array([1 - rows[i]["true"] for i in pool])
mu, sd = H["pool"].mean(0), H["pool"].std(0) + 1e-6
lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit((H["pool"] - mu) / sd, y)
w = lr.coef_[0] / sd  # raw-coordinate direction
score = lambda h: (h - mu) / sd @ lr.coef_[0] + lr.intercept_[0]
line = np.sort(score(H["refit_t"]))[::-1][1]
out = []
for i in evf:
    h, p = run(i, True)
    s = score(h)
    F = torch.zeros(1536, 1536, dtype=torch.float64)
    m = torch.zeros(1536, dtype=torch.float64)
    for a in range(0, WU.shape[0], 16384):
        Wc = WU[a:a + 16384].float(); pc = p[a:a + 16384]
        sq = torch.sqrt(pc)[:, None] * Wc
        F += (sq.T @ sq).double(); m += (Wc.T @ pc).double()
    F = (F - torch.outer(m, m)).numpy()
    ev, V = np.linalg.eigh(F)
    wt = V.T @ w
    res = {"p_dot": float(p[dot]), "score": float(s), "flag": bool(s > line), "D": float(max(s - line, 0))}
    for cut in [1e-6, 1e-8, 1e-10]:
        keep = ev > cut * ev.max()
        res[f"wFw_{cut:g}"] = float((wt[keep] ** 2 / ev[keep]).sum())
        res[f"KLmin_{cut:g}"] = res["D"] ** 2 / (2 * res[f"wFw_{cut:g}"])
    res["F_top_eig"] = float(ev.max())
    res["n_eig_gt_1e-6max"] = int((ev > 1e-6 * ev.max()).sum())
    out.append(res)
    print({k: round(v, 6) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
summ = {k: float(np.median([r[k] for r in out])) for k in out[0] if k != "flag"}
summ["catch_clean"] = float(np.mean([r["flag"] for r in out]))
summ["line"] = float(line)
print("MEDIAN", json.dumps(summ, indent=1))
json.dump({"rows": out, "median": summ}, open(HERE / "vera_pin.json", "w"), indent=1)
