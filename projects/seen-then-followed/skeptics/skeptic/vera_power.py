"""Phase-0 power check for the frozen rule 'both held-out cells within 10 points of reach x follow'.
Held-out design: fit (H1,M1),(H2,M2),(H3,M1),(H3,M2); predict (H1,M2),(H2,M1).
reach_H from the fit cell(s) of harness H (logged reach), follow_M from reached runs of model M's fit cells.
"""
import numpy as np
rng = np.random.default_rng(0)

def sim(reach, follow, n, B=20000, tol=0.10):
    # reach: dict H->r ; follow: dict (H,M)->f (law holds when f depends on M only)
    cells = [(h, m) for h in ("H1", "H2", "H3") for m in ("M1", "M2")]
    R = {c: rng.binomial(n, reach[c[0]], B) for c in cells}
    F = {c: rng.binomial(R[c], follow[c]) for c in cells}
    def fol(m, hs):
        num = sum(F[(h, m)] for h in hs); den = sum(R[(h, m)] for h in hs)
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)
    pred1 = R[("H1", "M1")] / n * fol("M2", ["H2", "H3"])
    pred2 = R[("H2", "M2")] / n * fol("M1", ["H1", "H3"])
    obs1 = F[("H1", "M2")] / n; obs2 = F[("H2", "M1")] / n
    ok = (np.abs(pred1 - obs1) <= tol) & (np.abs(pred2 - obs2) <= tol)
    return np.nanmean(ok)

for n in (40, 60, 80, 120):
    print(f"n={n}")
    for f1, f2 in ((0.5, 0.5), (0.7, 0.4), (0.9, 0.9), (0.1, 0.05)):
        reach = {"H1": 1.0, "H2": 0.95, "H3": 0.05}
        law = {(h, m): (f1 if m == "M1" else f2) for h in ("H1", "H2", "H3") for m in ("M1", "M2")}
        p_true = sim(reach, law, n)
        # broken law: follow in H2 is 30 points lower than in H1 for both models
        brk = dict(law)
        for m, f in (("M1", f1), ("M2", f2)):
            brk[("H2", m)] = max(0.0, f - 0.30)
        p_brk = sim(reach, brk, n)
        print(f"  follow M1={f1} M2={f2}: P(pass | law true)={p_true:.2f}  P(pass | harness shifts follow 30pt)={p_brk:.2f}")
