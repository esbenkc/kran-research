"""Vera's own per-position readability scan + projection share from vera_stack.py states (own code).

Input: NPZ from vera_stack.py with VERA_STATES=1 (eager/comp/emu _h26, uint16 bf16 bits).
Per position t:
  bits  : fraction of elements differing, eager vs comp, eager vs emu, comp vs emu
  auc_lr: logistic regression (C=1, standardised) eager(0) vs comp(1), fit on train split, AUROC on test
  auc_d : unpaired AUROC of x . d on test, d = mean over train of (x_comp - x_eager)
  null  : same logistic fit with labels shuffled inside each train pair (random which copy is 'comp')
  share : per test prompt s_i = <x_emu - x_eager, d> / <x_comp - x_eager, d>; median over test
Usage: python vera_scan.py IN.npz [OTHER_BUILD (default emu)]
"""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

z = np.load(sys.argv[1]); other = sys.argv[2] if len(sys.argv) > 2 else "emu"
def bf(a): return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
E, C, O = (bf(z[f"{b}_h26"]) for b in ("eager", "comp", other))
split = z["split"]; tr = split == "train"; te = split == "test"
T = E.shape[1]; rng = np.random.default_rng(0)
rows = []
for t in range(T):
    e, c, o = E[:, t], C[:, t], O[:, t]
    fr = [(z["eager_h26"][:, t] != z["comp_h26"][:, t]).mean(), (z["eager_h26"][:, t] != z[f"{other}_h26"][:, t]).mean(),
          (z["comp_h26"][:, t] != z[f"{other}_h26"][:, t]).mean()]
    d = (c[tr] - e[tr]).mean(0)
    sc_e, sc_c = e[te] @ d, c[te] @ d
    auc_d = roc_auc_score(np.r_[np.zeros(te.sum()), np.ones(te.sum())], np.r_[sc_e, sc_c])
    Xtr = np.r_[e[tr], c[tr]]; ytr = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
    Xte = np.r_[e[te], c[te]]; yte = np.r_[np.zeros(te.sum()), np.ones(te.sum())]
    ss = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(ss.transform(Xtr), ytr)
    auc_lr = roc_auc_score(yte, lr.decision_function(ss.transform(Xte)))
    flip = rng.random(tr.sum()) < 0.5
    ynull = np.r_[flip, ~flip].astype(float)
    lrn = LogisticRegression(C=1.0, max_iter=2000).fit(ss.transform(Xtr), ynull)
    auc_null = roc_auc_score(yte, lrn.decision_function(ss.transform(Xte)))
    den = (c[te] - e[te]) @ d; num = (o[te] - e[te]) @ d
    s = num / den
    # where the trained logistic puts the other build on test: mean prob, eager / other / comp
    p_o = lr.predict_proba(ss.transform(o[te]))[:, 1].mean()
    p_e = lr.predict_proba(ss.transform(e[te]))[:, 1].mean(); p_c = lr.predict_proba(ss.transform(c[te]))[:, 1].mean()
    rows.append(dict(t=t, bits_EC=fr[0], bits_EO=fr[1], bits_CO=fr[2], auc_lr=auc_lr, auc_d=auc_d, auc_null=auc_null,
                     share_med=float(np.median(s)), share_mean=float(s.mean()), den_min=float(np.abs(den).min()),
                     lr_pE=p_e, lr_pO=p_o, lr_pC=p_c))
    r = rows[-1]
    print(f"t{t:2d} bits E/C {fr[0]:.2f} E/{other} {fr[1]:.2f} C/{other} {fr[2]:.2f} | auc_lr {auc_lr:.3f} auc_d {auc_d:.3f} "
          f"null {auc_null:.3f} | share med {r['share_med']:+.3f} mean {r['share_mean']:+.3f} | "
          f"lr P(comp) E {p_e:.2f} {other} {p_o:.2f} C {p_c:.2f}", flush=True)
json.dump(rows, open(sys.argv[1].replace(".npz", f"_scan_{other}.json"), "w"), indent=1, default=float)
