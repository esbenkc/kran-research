"""Iris: exact reachable floor over ALL caught false test statements (replaces floor_all.py, which was not run).

For every false eval statement that the clean-fit post-norm last-layer country-token probe (L28post) catches, search
for a change of the PRE-norm last state x that pushes the probe score below the alarm line (by 0.05), keeping the state
on the RMSNorm surface (h = g * x / rms(x)), and minimising the exact next-word KL(p_original || p_changed) over the
full vocabulary. Same method as Vera's vera_bound.py (Adam on KL + 10 * violation^2, 600 steps), batched on MPS; the
step for row i scales with mean|x0_i| as in her code. The best feasible KL is an UPPER bound on the true minimum.
Floor at budget B = share of all 596 false statements whose best feasible KL > B (plus those never made feasible).
Probe: StandardScaler + LogisticRegression(C=0.01) on clean L28post country-token states, T1 split (fit pool = probe +
attack + stop cities), line = 2nd-highest of the 200 clean true line-setting statements.
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
from probes import fit_lin, lin_score, pool_idx, rows  # noqa: E402

L, EPS, STEPS, BS, MARGIN = 28, 1e-6, 600, 64, 0.05
BUDGETS = (0.001, 0.003, 0.01, 0.05)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    R, y, city, split = rows()
    tr, line_idx, ev = pool_idx()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenize(tok, [r["text"] for r in load_split()])
    stop_own = np.array([tok.decode([x[-1]]) == "." for x in ids])
    A = Acts("clean")
    row = np.where(stop_own, A.off[1:] - 2, A.off[1:] - 1)
    snap = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
    with safe_open(str(snap / "model.safetensors"), "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float()
        g = f.get_tensor("model.norm.weight").float()
    Xpre = np.asarray(A.tok[row, L], np.float64)
    gn = g.double().numpy()
    X = gn * Xpre / np.sqrt((Xpre ** 2).mean(-1, keepdims=True) + EPS)
    sc, m = fit_lin(X.astype(np.float32), y, tr, 0.01)
    s = lin_score(sc, m, X.astype(np.float32))
    w = m.coef_[0] / sc.scale_
    b = float(m.intercept_[0] - (sc.mean_ / sc.scale_) @ m.coef_[0])
    assert np.allclose(X.astype(np.float32) @ w + b, s, atol=1e-3)
    thr = float(np.sort(s[line_idx])[-2])
    evf = ev[y[ev] == 1]
    caught = evf[s[evf] > thr]
    print("clean catch", len(caught) / len(evf), "line", thr, "n caught", len(caught), flush=True)

    E, g = E.to(dev), g.to(dev)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    tgt = thr - MARGIN - b
    best = np.full(len(caught), np.nan)
    for a in range(0, len(caught), BS):
        idx = caught[a:a + BS]
        x0 = torch.tensor(Xpre[idx], dtype=torch.float32, device=dev)
        scale = x0.abs().mean(1, keepdim=True)
        with torch.no_grad():
            h0 = g * x0 / torch.sqrt((x0 ** 2).mean(1, keepdim=True) + EPS)
            lp0 = torch.log_softmax(h0 @ E.T, -1)
            p0 = lp0.exp()
        u = torch.zeros_like(x0, requires_grad=True)
        opt = torch.optim.Adam([u], lr=1e-2)
        bb = np.full(len(idx), np.inf)
        for step in range(STEPS):
            x = x0 + scale * u
            hx = g * x / torch.sqrt((x ** 2).mean(1, keepdim=True) + EPS)
            lq = torch.log_softmax(hx @ E.T, -1)
            kl = (p0 * (lp0 - lq)).sum(1)
            viol = torch.relu(hx @ wt - tgt)
            loss = (kl + 10.0 * viol ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            k, v = kl.detach().cpu().numpy(), viol.detach().cpu().numpy()
            ok = v == 0
            bb[ok] = np.minimum(bb[ok], k[ok])
        best[a:a + len(idx)] = np.where(np.isfinite(bb), bb, np.nan)
        print(a + len(idx), "done; median best so far", float(np.nanmedian(best[:a + len(idx)])), flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
    never = int(np.isnan(best).sum())
    bfin = np.where(np.isnan(best), np.inf, best)
    out = {"n_false_eval": int(len(evf)), "n_caught_clean": int(len(caught)), "catch_clean": len(caught) / len(evf),
           "line": thr, "never_feasible": never,
           "best_kl_quantiles": {str(q): float(np.quantile(bfin, q)) for q in (0.5, 0.9, 0.99, 1.0)},
           "floor_at_budget": {str(B): float((bfin > B).sum() / len(evf)) for B in BUDGETS},
           "rows": [{"i": int(i), "gap": float(s[i] - thr), "best_kl": (None if np.isnan(v) else float(v))}
                    for i, v in zip(caught, best)]}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
