"""Vera: (a) heads fit with labels on the ORIGINAL model's 800 probe+stop scores, applied to the attacked state's scores,
line = 2nd-highest of the attacked state's 200 line-set trues; (b) mean twin gap in clean-moment z (mean over members)."""
import json, numpy as np, vera26 as V
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Sc = V.scores_from_ctok("clean"); sc = StandardScaler().fit(Sc[V.TRAIN])
H = {"logit": LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Sc[V.TRAIN]), V.YF[V.TRAIN]),
     "qda": QuadraticDiscriminantAnalysis(reg_param=0.1).fit(sc.transform(Sc[V.TRAIN]), V.YF[V.TRAIN])}
m = np.array([V.ZM[f"tok{r}"]["mu"] for r in V.R]); d = np.array([V.ZM[f"tok{r}"]["sd"] for r in V.R])
ev = np.where(V.EVAL)[0]; pairs = {}
for i in ev:
    r = V.ROWS[i]; pairs.setdefault((r["city"], r["src"]), {})[r["true"]] = i
P = np.array([(v[0], v[1]) for v in pairs.values()])
out = {}
for st in ["clean"] + [f"recal__lam10000__seed{s}" for s in range(3)] + ["recalSG__lam10000__seed0"]:
    S = Sc if st == "clean" else V.scores_from_ctok(st)
    o = {k: V.catch([V.flag(h.decision_function(sc.transform(S)))]) for k, h in H.items()}
    z = (S - m) / d; gap = (z[P[:, 0]] - z[P[:, 1]]).mean(1)
    o = {k: [round(v["catch"], 4), round(v["fa"][0], 4)] for k, v in o.items()}; o["mean_twin_gap_z"] = round(float(gap.mean()), 3)
    out[st] = o; print(st, o)
json.dump(out, open(V.HERE / "vera_cleanhead.json", "w"), indent=1)
