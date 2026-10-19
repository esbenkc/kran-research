"""PREREG s8 robustness guard, rebuilt: the n=200 verdict on TEST answers where judge and string check agree.
Also the label guard (judge vs haiku agreement, base minus policy). Own draws (seed 11), paired bootstrap over
the 400 test prompt indices (each policy keeps its own clean mask inside a resample).

    cd kran-research && uv run python ../swarm-oracle-1019-refit-probe/skeptic/vera_clean_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vera_core import auroc, balanced_draw, fit_lr, score  # noqa: E402
from vera_rebuild import LAM1, load  # noqa: E402

OUT = Path(__file__).resolve().parent


def scores200(d, rng, ndraw=20):
    te, po = d["split"] == "test", d["split"] == "pool"
    ypool = d["y"][po]
    out = []
    for _ in range(ndraw):
        idx = balanced_draw(ypool, 200, rng)
        out.append(score(fit_lr(d["own"][po][idx], ypool[idx]), d["own"][te]))
    return np.array(out)


def main():
    rng = np.random.default_rng(11)
    data = {p: load(p) for p in ["base"] + LAM1}
    S = {p: scores200(d, rng) for p, d in data.items()}
    yl = {}
    for p, d in data.items():
        yt = d["y"][:400].copy()
        clean = (yt >= 0) & (yt == d["ys"][:400])
        yc = np.where(clean, yt, -1)  # -1 is dropped by auroc
        yl[p] = {"full": yt, "clean": yc}
    boots = np.random.default_rng(1019).integers(0, 400, (1000, 400))
    res = {}
    agree = {}
    for p, d in data.items():
        y, y2 = d["y"][:400], d["y2"][:400]
        k = (y >= 0) & (y2 >= 0)
        agree[p] = float((y[k] == y2[k]).mean())
    for lab in ["full", "clean"]:
        res[lab] = {}
        verdict_no = []
        for p in LAM1:
            yb, yp = yl["base"][lab], yl[p][lab]
            f = lambda ix: np.mean([auroc(yb[ix], s[ix]) for s in S["base"]]) - np.mean([auroc(yp[ix], s[ix]) for s in S[p]])
            pt = f(np.arange(400))
            g = np.array([f(ix) for ix in boots])
            lo, hi = np.percentile(g, [2.5, 97.5])
            res[lab][p] = {"gap": float(pt), "lo": float(lo), "hi": float(hi),
                           "n_test_used": int((yp >= 0).sum()), "n_hon": int((yp == 0).sum())}
            verdict_no.append(pt > 0.05 and lo > 0)
            print(f"{lab:5s} gap base-{p}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  test n={int((yp>=0).sum())} (base n={int((yb>=0).sum())})")
        yes = all(r["gap"] <= 0.05 for r in res[lab].values())
        res[lab]["verdict"] = "YES" if yes else ("NO" if any(verdict_no) else "UNCONFIRMED")
        print(lab, "verdict:", res[lab]["verdict"])
    res["agree_judge_haiku"] = agree
    res["label_guard_base_minus_policy"] = {p: agree["base"] - agree[p] for p in LAM1}
    print("judge-haiku agreement:", {k: round(v, 3) for k, v in agree.items()})
    print("label guard (base - policy, trigger > 0.05):", {k: round(v, 3) for k, v in res["label_guard_base_minus_policy"].items()})
    (OUT / "vera_clean_verdict.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
