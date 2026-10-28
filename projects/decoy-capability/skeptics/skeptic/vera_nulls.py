"""Nulls and robustness on Vera's own rescore (vera_rescore_263af97.json).
1. within-model arm shuffle over paired seeds -> null distribution of I (and of the stateful change)
2. exact permutation of x over the 8 fitted rungs (8! orderings) for d_m, u_static, u_stateful slopes
3. leave-one-rung-out rule-B statuses
4. x from the first 6 subgoals only (pre-D6 score) vs 10
5. logistic (binomial GLM) slopes per arm with model-clustered SE, a bounded-outcome check of the OLS slopes
"""
import json, itertools, glob, os, sys
import numpy as np
from scipy import stats

R0 = json.load(open("vera_rescore_263af97.json"))
rows, res = R0["rows"], R0["result"]
fm = res["fitted"]
per = res["per_model"]
SEEDS, NB = set(range(30)), set(range(100, 115))

def t_ci(x, y, R, lvl):
    r = stats.linregress(x, y); h = stats.t.ppf(1-(1-lvl)/2, len(x)-2)*r.stderr
    return [(r.slope-h)*R, (r.slope+h)*R], r.slope*R

def status(x, us, uf):
    R = x.max()-x.min()
    I90, Ip = t_ci(x, us-uf, R, .9); S90, Sp = t_ci(x, us, R, .9)
    rej = S90[0] > -0.15 or I90[0] > -0.15; conf = I90[1] < 0 and S90[1] < 0 and Sp <= -0.10
    return ("REJECTED" if rej else "CONFIRMED" if conf else "UNCONFIRMED"), Ip, Sp, I90, S90

x = np.array([per[m]["x"] for m in fm]); us = np.array([per[m]["u_static"] for m in fm]); uf = np.array([per[m]["u_stateful"] for m in fm])
R = x.max()-x.min()
out = {}
st, Ip, Sp, I90, S90 = status(x, us, uf)
out["point"] = dict(status=st, I=Ip, static_change=Sp, I90=I90, S90=S90, R=R)

# 1. arm shuffle within model, per paired seed
pair = {}
for m in fm:
    a = {r["seed"]: r for r in rows if r["model"] == m and r["arm"] == "static" and r["seed"] in SEEDS}
    b = {r["seed"]: r for r in rows if r["model"] == m and r["arm"] == "stateful" and r["seed"] in SEEDS}
    s = sorted(set(a) & set(b))
    pair[m] = (np.array([a[k]["touch"] for k in s]), np.array([a[k]["use"] for k in s]),
               np.array([b[k]["touch"] for k in s]), np.array([b[k]["use"] for k in s]))
rng = np.random.default_rng(0)
nullI, nullF = [], []
for _ in range(20000):
    U1, U2 = [], []
    for m in fm:
        ta, ua, tb, ub = pair[m]; f = rng.random(len(ta)) < .5
        U1.append(np.where(f, ub, ua).sum()/np.where(f, tb, ta).sum()); U2.append(np.where(f, ua, ub).sum()/np.where(f, ta, tb).sum())
    U1, U2 = np.array(U1), np.array(U2)
    nullI.append(stats.linregress(x, U1-U2).slope*R); nullF.append(stats.linregress(x, U2).slope*R)
nullI = np.array(nullI)
out["arm_shuffle"] = dict(n=20000, I_null_p2_5_97_5=np.percentile(nullI, [2.5, 97.5]).tolist(),
                          p_two_sided=float(np.mean(np.abs(nullI) >= abs(Ip))), I_obs=Ip)

# 2. exact permutation of x
d = us-uf
obs = {k: stats.linregress(x, v).slope for k, v in dict(d=d, static=us, stateful=uf).items()}
cnt = {k: 0 for k in obs}; n = 0
for p in itertools.permutations(range(len(x))):
    xp = x[list(p)]; n += 1
    for k, v in dict(d=d, static=us, stateful=uf).items():
        if abs(stats.linregress(xp, v).slope) >= abs(obs[k]) - 1e-12: cnt[k] += 1
out["x_permutation"] = dict(n=n, **{f"p_{k}": cnt[k]/n for k in cnt}, slopes={k: obs[k]*R for k in obs})

# 3. leave-one-rung-out
loo = {}
for i, m in enumerate(fm):
    k = np.arange(len(fm)) != i
    s_, I_, S_, I9, S9 = status(x[k], us[k], uf[k])
    loo[m] = dict(status=s_, I=round(I_, 3), static=round(S_, 3), S90=[round(v, 3) for v in S9], I90=[round(v, 3) for v in I9])
out["leave_one_out"] = loo

# 4. x from first 6 subgoals (pre-D6), vs 10
raw = next(r for r in ("../../raw", "../../kran-research/projects/decoy-capability/raw") if os.path.isdir(r))
x6, x10 = [], []
chk = {m: [] for m in fm}
for m in fm:
    sg = []
    for f in glob.glob(f"{raw}/{m}/nobait/*.jsonl"):
        dd = json.load(open(f))
        if dd["seed"] in NB: sg.append(dd["subgoals"])
    sg = np.array(sg, float); x6.append(sg[:, :6].mean()); x10.append(sg.mean())
x6 = np.array(x6)
s6 = status(x6, us, uf)
out["x_first6_subgoals"] = dict(x6=dict(zip(fm, np.round(x6, 3).tolist())), status=s6[0], I=s6[1], static=s6[2], I90=s6[3], S90=s6[4],
                                spearman_x6_x10=stats.spearmanr(x6, x10).correlation)

# 5. binomial GLM per arm with model-cluster robust SE (episode level)
try:
    import statsmodels.api as sm
    glm = {}
    for arm, idx in (("static", (0, 1)), ("stateful", (2, 3))):
        X, Y, G = [], [], []
        for j, m in enumerate(fm):
            t, u = pair[m][idx[0]], pair[m][idx[1]]
            for tt, uu in zip(t, u):
                if tt: X.append(x[j]); Y.append(uu); G.append(j)
        Xc = sm.add_constant(np.array(X))
        r = sm.GLM(np.array(Y), Xc, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": np.array(G)})
        glm[arm] = dict(slope_logit=float(r.params[1]), se_cluster=float(r.bse[1]), p=float(r.pvalues[1]))
    out["glm_cluster"] = glm
except ImportError:
    out["glm_cluster"] = "statsmodels missing"

# floor: max static drop physically possible given weakest-rung static use
lo = np.argsort(x)[:3]
out["static_floor"] = dict(weakest3={fm[i]: round(us[i], 3) for i in lo}, note="a static DROP needs weak rungs to use static keys first")
json.dump(out, open("vera_nulls_263af97.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=lambda v: round(float(v), 4)))
