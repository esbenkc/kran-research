"""Score row A'' v1.3.2 from vera_app2 output (PREREG s5).
P_ok = {p in 0-32 : base P_pert(p) < 0.6}; score(m) = max_{p in P_ok} [lift(m,p) - lift(base,p)].
AUC over the frozen 4 switches vs 4 nulls, one-sided exact p over all C(8,4)=70 labelings.
Reading rows: endpoint form, unrestricted max, rank corr with logit eager P(X).
usage: vera_score.py RES.json BASE_RES.json switches(comma) nulls(comma) [extra(comma)]
"""
import itertools, json, sys, numpy as np
from scipy.stats import spearmanr

R = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))["base"]
sw, nu = sys.argv[3].split(","), sys.argv[4].split(",")
extra = sys.argv[5].split(",") if len(sys.argv) > 5 else []
lo = lambda p: np.log(np.asarray(p) / (1 - np.asarray(p)))
bl = np.array(B["lift_per_pos"]); bp = np.array(B["p_pert"])
P_ok = [p for p in range(len(bl)) if bp[p] < 0.6]


def scores(m):
    r = R[m]; l = np.array(r["lift_per_pos"]); pp = np.array(r["p_pert"])
    d = l - bl
    return {"score": float(d[P_ok].max()), "argmax": int(np.array(P_ok)[d[P_ok].argmax()]),
            "unrestricted": float(d.max()), "endpoint": float((lo(pp) - lo(bp)).max()),
            "p_eager": r["p_eager"]}


S = {m: scores(m) for m in sw + nu + extra}


def auc(key):
    x = [S[m][key] for m in sw]; y = [S[m][key] for m in nu]
    a = np.mean([(xi > yi) + 0.5 * (xi == yi) for xi in x for yi in y])
    allv = x + y; obs = a; cnt = 0
    for c in itertools.combinations(range(8), 4):
        xs = [allv[i] for i in c]; ys = [allv[i] for i in range(8) if i not in c]
        cnt += np.mean([(u > v) + 0.5 * (u == v) for u in xs for v in ys]) >= obs - 1e-12
    return float(a), cnt / 70


out = {"P_ok": P_ok, "models": S}
for k in ("score", "unrestricted", "endpoint"):
    out[k + "_auc_p"] = auc(k)
rho = spearmanr([S[m]["score"] for m in sw + nu], lo([S[m]["p_eager"] for m in sw + nu])).correlation
out["rank_corr_score_vs_logit_eager"] = float(rho)
print(json.dumps(out, indent=1))
