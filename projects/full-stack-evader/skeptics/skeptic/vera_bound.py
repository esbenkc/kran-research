"""Vera: own check of Iris's per-position pin bound for the post-norm layer-28 (L28post) country-token probe.
Clean model only, no model forward: h = stored post-norm state (readout 29, fp16) at the country token, logits = E h
(tied embedding, Qwen2.5-1.5B-Instruct). For N random caught eval FALSE statements:
 (1) second order: KL_min = gap^2 / (2 w' F^+ w), F = E'(diag p - p p')E over the FULL vocabulary (and top-4096 to test
     the truncation direction), eigen cutoffs and a ridge.
 (2) exact (nonlinear): min over the pre-norm direction x of KL(p0 || softmax(E g*x/rms(x))) subject to w.h <= line - 0.05,
     by Adam on a penalty; h stays on the RMSNorm manifold.
w = my own clean fit (StandardScaler + LR C=0.01, pool split) on readout 29 at the country token.
    python vera_bound.py [N]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vera_core import EVF, HERE, LINE_T, POOL, YF

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
torch.manual_seed(0)
snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))
with safe_open(str(snap), "pt") as f:
    keys = list(f.keys())
    E = f.get_tensor("lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight").float()
    g = f.get_tensor("model.norm.weight").float()
print("lm_head key present:", "lm_head.weight" in keys, E.shape, flush=True)
H = np.asarray(np.load(HERE / "feats/clean_ctry.npy", mmap_mode="r")[:, 29], np.float64)
sc = StandardScaler().fit(H[POOL])
lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(H[POOL]), YF[POOL])
w = lr.coef_[0] / sc.scale_
b0 = lr.intercept_[0] - (sc.mean_ / sc.scale_) @ lr.coef_[0]
S = H @ w + b0
line = np.sort(S[LINE_T])[::-1][1]
caught = EVF[S[EVF] > line]
pick = np.random.default_rng(1021).choice(caught, min(N, len(caught)), replace=False)
wt = torch.tensor(w, dtype=torch.float32)
print(f"clean catch {np.mean(S[EVF] > line):.3f}, line {line:.3f}", flush=True)


def fisher(p, idx=None):
    Ew = E if idx is None else E[idx]
    pp = p if idx is None else p[idx]
    F = torch.zeros(1536, 1536, dtype=torch.float64); m = torch.zeros(1536, dtype=torch.float64)
    for a in range(0, Ew.shape[0], 16384):
        Wc = Ew[a:a + 16384]; pc = pp[a:a + 16384]
        F += ((Wc * pc[:, None]).T @ Wc).double(); m += (Wc.T @ pc).double()
    return (F - torch.outer(m, m)).numpy()


rows = []
for i in pick:
    h0 = torch.tensor(H[i], dtype=torch.float32)
    lp0 = torch.log_softmax(E @ h0, -1); p0 = lp0.exp()
    gap = float(S[i] - line)
    r = {"i": int(i), "gap": gap, "p_max": float(p0.max())}
    for name, idx in [("full", None), ("top4096", torch.topk(p0, 4096).indices)]:
        F = fisher(p0, idx)
        ev, V = np.linalg.eigh(F)
        a = V.T @ w
        for cut in [1e-4, 1e-6, 1e-8]:
            k = ev > cut * ev.max()
            r[f"{name}_KL2_cut{cut:g}"] = gap ** 2 / (2 * float((a[k] ** 2 / ev[k]).sum()))
        lam = 1e-2 * ev.mean()
        r[f"{name}_KL2_ridge"] = gap ** 2 / (2 * float((a ** 2 / (np.clip(ev, 0, None) + lam)).sum()))
    # exact nonlinear minimum on the RMSNorm manifold
    x0 = h0 / g
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=1e-2 * float(x0.abs().mean()))
    tgt = line - 0.05 - b0
    best = None
    for step in range(600):
        hx = g * x / torch.sqrt((x ** 2).mean() + 1e-6)
        lq = torch.log_softmax(E @ hx, -1)
        kl = (p0 * (lp0 - lq)).sum()
        viol = torch.relu(wt @ hx - tgt)
        loss = kl + 10.0 * viol ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        if viol.item() == 0 and (best is None or kl.item() < best):
            best = kl.item()
    r["KL_exact"] = best
    r["final_viol"] = float(viol.item())
    rows.append(r)
    print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
med = {k: float(np.nanmedian([np.nan if rr[k] is None else rr[k] for rr in rows])) for k in rows[0] if k != "i"}
print("MEDIAN", json.dumps(med, indent=1))
json.dump({"line": float(line), "rows": rows, "median": med}, open(HERE / "vera_bound.json", "w"), indent=1)
