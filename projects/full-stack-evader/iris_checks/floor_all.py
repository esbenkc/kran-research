"""Blind floor over ALL 596 false eval statements (clean activations only), post-norm last state, two positions.

Cheap rigorous version: moving the post-norm state along the probe direction w itself is one feasible move, so
the least KL to cross the line is at most  KL_w = gap^2 * (w^T F w) / (2 (w^T w)^2),  with w^T F w = Var_p(W w)
(one matrix-vector product per statement, full vocabulary). Floor at budget B = share of caught false statements
with KL_w > B (an UPPER bound on the floor the second-order lemma allows, since KL_min <= KL_w).
Also reports the exact KL (full softmax) of that move, realised through the pre-norm state.
Run after the fine-tune queue: cd <T1 project> && uv run python <this file>
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
from probes import fit_lin, lin_score, pool_idx, rows, select_lin  # noqa: E402

L, EPS = 28, 1e-6
BUDGETS = (0.001, 0.01, 0.05)


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
    snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
    with safe_open(str(snap / "model.safetensors"), "pt") as f:
        W = f.get_tensor("model.embed_tokens.weight").float().numpy()
        g = f.get_tensor("model.norm.weight").float().numpy().astype(np.float64)
    out = {}
    for pos in ("country", "stop"):
        row = np.where(stop_own, A.off[1:] - 2, A.off[1:] - 1) if pos == "country" else A.off[1:] - 1
        Xpre = np.asarray(A.tok[row, L], np.float64)
        X = g * Xpre / np.sqrt((Xpre ** 2).mean(-1, keepdims=True) + EPS)
        C, _ = select_lin(X.astype(np.float32), y, city, tr)
        sc, m = fit_lin(X.astype(np.float32), y, tr, C)
        s = lin_score(sc, m, X.astype(np.float32))
        w = m.coef_[0].astype(np.float64) / sc.scale_
        thr = np.sort(s[line])[-2]
        evf = ev[y[ev] == 1]
        Ww = (W @ w.astype(np.float32)).astype(np.float64)
        rows_ = []
        for i in evf:
            gap = float(s[i] - thr)
            if gap <= 0:
                continue
            lp = logsm(W @ X[i].astype(np.float32)); p = np.exp(lp)
            vF = float(p @ Ww ** 2 - (p @ Ww) ** 2)
            klw = gap ** 2 * vF / (2 * (w @ w) ** 2)
            # exact KL of the same move, realised through the pre-norm state
            x = Xpre[i]; r = np.sqrt((x ** 2).mean() + EPS)
            dn = gap * w / (w @ w)
            x2 = x - r * dn / g
            n2 = g * x2 / np.sqrt((x2 ** 2).mean() + EPS)
            lp2 = logsm(W @ n2.astype(np.float32))
            rows_.append({"i": int(i), "gap": gap, "kl_w_2nd": klw, "kl_w_exact": float((p * (lp - lp2)).sum()),
                          "shift_over_gap": float(w @ (X[i] - n2)) / gap})
        k2 = np.array([r["kl_w_2nd"] for r in rows_]); ke = np.array([r["kl_w_exact"] for r in rows_])
        sg = np.array([r["shift_over_gap"] for r in rows_])
        out[pos] = {"C": C, "n_false_eval": int(len(evf)), "n_caught_clean": len(rows_),
                    "catch_clean": len(rows_) / len(evf),
                    "floor_upper_2nd": {str(b): float((k2 > b).sum() / len(evf)) for b in BUDGETS},
                    "floor_upper_exact": {str(b): float((ke > b).sum() / len(evf)) for b in BUDGETS},
                    "kl_w_exact_quantiles": {str(q): float(np.quantile(ke, q)) for q in (0.5, 0.9, 0.99, 1.0)},
                    "shift_over_gap_min": float(sg.min()), "shift_over_gap_median": float(np.median(sg)),
                    "rows": rows_}
        print(pos, json.dumps({k: v for k, v in out[pos].items() if k != "rows"}, indent=1), flush=True)
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
