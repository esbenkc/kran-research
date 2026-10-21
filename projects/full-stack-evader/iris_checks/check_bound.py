"""Iris check of the displayed bound, exactly (no second-order step), on real statements.

For each of 30 caught false eval statements (seed 1021) at the country token:
  1. post-norm state n = g * x / rms(x) from the T1 pre-norm readout 28 (clean model);
  2. the least-KL move in n-space that reaches the line: dn* = gap * F^+ w / (w^T F^+ w)  (F over the full vocabulary);
  3. realise it through the PRE-norm state, as a fine-tune must: x' = x - rms(x) * dn* / g, n' = RMSNorm(x');
  4. report the EXACT KL(p || p') with full softmax, the exact probe shift w.(n' - n), and the predicted KL gap^2/(2q).
Pass if exact KL / predicted KL is within [0.5, 2] for every row and the probe crosses the line (shift >= 0.9 gap).
Run after the fine-tune queue (compute gate): cd <T1 project> && uv run python <this file>
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

# T1 = the megaprobe-after project (branch swarm/theory-1020-megaprobe-after, projects/megaprobe-after): its common.py,
# probes.py and clean activations. Readouts 0..28 there are bit-equal to this project's data/acts/clean (Bruno #4324).
T1 = Path("/Users/esben/work/theory-1020-megaprobe-after/kran-research/projects/megaprobe-after")
sys.path.insert(0, str(T1))
from common import MODEL, Acts, load_split, tokenize  # noqa: E402
from probes import fit_lin, lin_score, pool_idx, rows  # noqa: E402

L, N, EPS = 28, 30, 1e-6


def logsm(z):
    z = z.astype(np.float64) - z.max()
    return z - np.log(np.exp(z).sum())


def main():
    R, y, city, split = rows()
    tr, line, ev = pool_idx()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tok, [r["text"] for r in load_split()])
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    A = Acts("clean")
    row = np.where(stop_own, A.off[1:] - 2, A.off[1:] - 1)
    snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
    with safe_open(str(snap / "model.safetensors"), "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().numpy()
        g = f.get_tensor("model.norm.weight").float().numpy().astype(np.float64)

    def rms(x):
        return np.sqrt((x ** 2).mean() + EPS)

    Xpre = np.asarray(A.tok[row, L], np.float64)
    X = g * Xpre / np.sqrt((Xpre ** 2).mean(-1, keepdims=True) + EPS)
    sc, m = fit_lin(X.astype(np.float32), y, tr, 0.01)
    s = lin_score(sc, m, X.astype(np.float32))
    w = m.coef_[0].astype(np.float64) / sc.scale_
    thr = np.sort(s[line])[-2]
    evf = ev[(y[ev] == 1) & (s[ev] > thr)]
    pick = np.sort(np.random.default_rng(1021).choice(evf, N, replace=False))
    out = []
    for i in pick:
        x, n = Xpre[i], X[i]
        lp = logsm(W @ n.astype(np.float32))
        p = np.exp(lp)
        mu = W.T.astype(np.float64) @ p
        F = (W.T * p.astype(np.float32)) @ W  # full vocabulary, float32 accumulate
        F = F.astype(np.float64) - np.outer(mu, mu)
        lam, V = np.linalg.eigh(F)
        lam = np.maximum(lam, 1e-6 * lam.mean())
        Fpw = V @ ((V.T @ w) / lam)
        q = float(w @ Fpw)
        gap = float(s[i] - thr)
        dn = gap * Fpw / q
        x2 = x - rms(x) * dn / g
        n2 = g * x2 / rms(x2)
        lp2 = logsm(W @ n2.astype(np.float32))
        kl = float((p * (lp - lp2)).sum())
        shift = float(w @ (n - n2))
        out.append({"i": int(i), "gap": gap, "kl_pred": gap ** 2 / (2 * q), "kl_exact": kl,
                    "ratio": kl / (gap ** 2 / (2 * q)), "shift_over_gap": shift / gap})
        print(out[-1], flush=True)
    r = np.array([o["ratio"] for o in out]); sg = np.array([o["shift_over_gap"] for o in out])
    summ = {"n": N, "ratio_min": float(r.min()), "ratio_median": float(np.median(r)), "ratio_max": float(r.max()),
            "shift_over_gap_min": float(sg.min()), "kl_exact_median": float(np.median([o["kl_exact"] for o in out])),
            "pass": bool((r >= 0.5).all() and (r <= 2).all() and (sg >= 0.9).all())}
    Path(__file__).with_suffix(".json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
