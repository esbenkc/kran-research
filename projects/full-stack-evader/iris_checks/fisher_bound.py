"""Iris check: how much output change (KL at the country position) does the attacker need to push the
clean-fit layer-L country-token probe below its 1% FPR line, if the change is made in the post-norm last state?

Second order: KL(p || p') ~= 1/2 dh^T F dh,  F = sum_v p_v (u_v - mu)(u_v - mu)^T,  mu = sum_v p_v u_v,
u_v = row v of the (tied) unembedding. For a linear probe s = w.h + b the largest shift at budget KL is
|ds| <= sqrt(2 KL w^T F^+ w), so crossing a gap g needs KL >= g^2 / (2 w^T F^+ w).
F_K (sum over the top-K tokens, full mu) <= F in PSD order, so w^T F_K^+ w >= w^T F^+ w and the KL we report is
a LOWER bound on the second-order KL needed (conservative for the claim 'at least this much is needed').

Data: T1 (theory-1020-megaprobe-after) clean activations, hidden_states[28] at the country token (d3.py rows).
Run: cd <T1 project> && uv run python <this file>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

# T1 = the megaprobe-after project (branch swarm/theory-1020-megaprobe-after, projects/megaprobe-after): its common.py,
# probes.py and clean activations. Readouts 0..28 there are bit-equal to this project's data/acts/clean (Bruno #4324).
T1 = Path("/Users/esben/work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after")
sys.path.insert(0, str(T1))
from common import MODEL, Acts, load_split, tokenize  # noqa: E402
from probes import fit_lin, lin_score, pool_idx, rows, select_lin  # noqa: E402

OUT = Path(__file__).with_suffix(".json")
L = 28
K = 4096
NSUB = 150


def main():
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tok, [r["text"] for r in load_split()])
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    A = Acts("clean")
    lp = np.load(T1 / "data/acts/clean/lp.npy", mmap_mode="r")
    snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
    with safe_open(str(snap / "model.safetensors"), "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().numpy()  # (V, d), tied lm_head
        g = f.get_tensor("model.norm.weight").float().numpy()
    eps = 1e-6

    def post(Xpre):  # final RMSNorm: the memmap's index 28 is the block-28 output BEFORE model.norm
        return g * Xpre / np.sqrt((Xpre.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32)

    out = {}
    for pos in ("country", "stop"):
        keep_stmt = stop_own if pos == "country" else stop_own.copy()  # 'stop' = the full-stop token row
        row = np.where(stop_own, A.off[1:] - 2, A.off[1:] - 1) if pos == "country" else A.off[1:] - 1
        Xpre = np.asarray(A.tok[row, L], np.float32)
        X = post(Xpre)
        # sanity: post-norm logits reproduce stored log P of the NEXT token (only defined when a next row exists)
        chk = []
        if pos == "country":
            for i in np.where(stop_own)[0][:200]:
                z = (W @ X[i]).astype(np.float64); z -= z.max(); lz = z - np.log(np.exp(z).sum())
                chk.append(float(lz[ids[i][-1]] - lp[row[i] + 1]))
        C, cv = select_lin(X, y, city, tr)
        sc, m = fit_lin(X, y, tr, C)
        s = lin_score(sc, m, X)
        w = (m.coef_[0] / sc.scale_).astype(np.float64)
        thr = np.sort(s[line])[-2]
        evf_all = ev[y[ev] == 1]
        catch_all = float((s[evf_all] > thr).mean())
        evf = np.sort(np.random.default_rng(1021).choice(evf_all, NSUB, replace=False))
        res = []
        for i in evf:
            z = W @ X[i]
            p = np.exp((z - z.max()).astype(np.float64)); p /= p.sum()
            mu = W.T @ p
            top = np.argpartition(-p, K)[:K]
            U = W[top] - mu.astype(np.float32)
            F = (U * p[top, None].astype(np.float32)).T @ U
            lam, V = np.linalg.eigh(F.astype(np.float64))
            a = V.T @ w
            tr_ = lam.sum() / len(lam)
            qs = {str(r): float((a ** 2 / (np.maximum(lam, 0) + r * tr_)).sum()) for r in (1e-6, 1e-4, 1e-2)}
            q = qs["1e-06"]
            gap = float(s[i] - thr)
            res.append({"i": int(i), "gap": gap, "q": q, "q_ridge": qs,
                        "kl_needed": gap ** 2 / (2 * q) if gap > 0 else 0.0,
                        "kl_needed_ridge": {k: (gap ** 2 / (2 * v) if gap > 0 else 0.0) for k, v in qs.items()},
                        "p_max": float(p.max()), "mass_topK": float(p[top].sum())})
            if len(res) % 50 == 0:
                print(pos, len(res), flush=True)
        kl = np.array([r["kl_needed"] for r in res if r["gap"] > 0])
        out[pos] = {"C": C, "cv_auroc": cv, "thr": float(thr), "n_false_eval": int(len(evf)),
                    "catch_clean_all_eval": catch_all, "n_sub": NSUB,
                    "kl_needed_median_by_ridge": {k: float(np.median([r["kl_needed_ridge"][k] for r in res if r["gap"] > 0])) for k in ("1e-06", "0.0001", "0.01")},
                    "median_mass_topK": float(np.median([r["mass_topK"] for r in res])),
                    "lp_check_maxabs": float(np.max(np.abs(chk))) if chk else None,
                    "lp_check_median_abs": float(np.median(np.abs(chk))) if chk else None,
                    "kl_needed_quantiles": {str(qq): float(np.quantile(kl, qq)) for qq in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)},
                    "frac_needing_over": {str(b): float((kl > b).mean()) for b in (0.001, 0.01, 0.05, 0.5)},
                    "median_pmax": float(np.median([r["p_max"] for r in res])),
                    "rows": res}
        print(pos, json.dumps({k: v for k, v in out[pos].items() if k != "rows"}, indent=1), flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
