"""Vera: 'perfect reader' + gap rebuild from saved acts (own code).

proj_L(h) = h . r_L / |r_L|^2   (units of the base FIT harmful-harmless gap; base FIT gap = 1 by construction)
Reader, DEV-fixed: threshold t_L = 4th-largest DEV-harmless projection, flag iff proj > t_L (<= 3/64 DEV harmless
flagged); apply unchanged to TEST: report % harmful flagged and % harmless flagged.
Also: AUROC, and gap in pooled-SD units (Cohen's d) on TEST.
"""
import numpy as np, json
from pathlib import Path
np.seterr(all="ignore")
R = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/runs"
base = np.load(R / "base/acts.npz")
f = lambda A, k, L: A[k][:, L].astype(np.float64)
out = {}
for name in ("base", "strip_s0"):
    A = np.load(R / name / "acts.npz")
    rows = []
    for L in range(1, 28):
        r = f(base, "fit_harmful", L).mean(0) - f(base, "fit_harmless", L).mean(0)
        P = {k: np.einsum("nd,d->n", f(A, k, L), r) / (r @ r) for k in A.files}
        t = np.sort(P["dev_harmless"])[-4]
        th, tl = P["test_harmful"], P["test_harmless"]
        sd = np.sqrt((th.var(ddof=1) + tl.var(ddof=1)) / 2)
        rows.append(dict(L=L, dev_h=P["dev_harmful"].mean(), dev_l=P["dev_harmless"].mean(),
                         dev_gap=P["dev_harmful"].mean() - P["dev_harmless"].mean(),
                         test_gap_sd=(th.mean() - tl.mean()) / sd,
                         catch=(th > t).mean(), fa=(tl > t).mean(), dev_fa=(P["dev_harmless"] > t).mean()))
    out[name] = rows
    print(f"\n== {name}\n L  devH  devL  devGap  testGap/sd  catch%  harmless%  (dev harmless%)")
    for x in rows:
        print(f"{x['L']:2d} {x['dev_h']:5.2f} {x['dev_l']:5.2f} {x['dev_gap']:6.2f}  {x['test_gap_sd']:6.2f}   "
              f"{100*x['catch']:5.1f}   {100*x['fa']:5.1f}   ({100*x['dev_fa']:.1f})")
json.dump(out, open(Path(__file__).with_name("vera_reader.json"), "w"), indent=1)
