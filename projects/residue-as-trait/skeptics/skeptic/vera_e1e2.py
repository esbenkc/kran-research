"""Own rebuild of the EXPLORATORY rows E1/E2 (Bruno #14209, Kasper #14217) from my own tags.
E1: sign of (runbook - named) key count per agent, parent (prior/vera_parent_tags.jsonl) vs today.
E2: held-out cell = (agent, task, arm), base p_t = other 4 agents on that task in that arm (Beta(1,1)).
    arm-specific offset = logit(agent's same-arm rate on other 7 tasks) - logit(other agents' same-arm rate there)
    pooled offset       = same but both arms pooled (the frozen s3 offset)
    D = mean over runs of loss(p_t) - loss(expit(logit p_t + offset)); gain = D_arm - D_pooled.
    Null for the gain: shuffle arm labels within each agent x task cell (2000, default_rng(20261104))."""
import json, math, numpy as np
lo = lambda p: math.log(p / (1 - p)); ex = lambda z: 1 / (1 + math.exp(-z)); b = lambda k, n: (k + 1) / (n + 2)
R = [json.loads(l) for l in open("counted/vera_tags.jsonl")]
P = [json.loads(l) for l in open("prior/vera_parent_tags.jsonl")]
ag = sorted({r["model"] for r in R}); sc = sorted({r["scenario"] for r in R}); arms = ["named", "runbook"]
print("E1 per agent named|runbook  parent(/40)  today(/24)  sign repeat")
rep = 0
for a in ag:
    pn = sum(r["key"] for r in P if r["model"] == a and r["cell"] == "named"); pr = sum(r["key"] for r in P if r["model"] == a and r["cell"] == "runbook")
    tn = sum(r["key"] for r in R if r["model"] == a and r["cell"] == "named"); tr = sum(r["key"] for r in R if r["model"] == a and r["cell"] == "runbook")
    same = np.sign(pr - pn) == np.sign(tr - tn); rep += same
    print(f"  {a:28s} {pn:2d}|{pr:2d}   {tn:2d}|{tr:2d}   {same}")
print(f"E1 sign repeats {rep}/5")
Y = np.zeros((5, 8, 2, 3), int)
for r in R:
    Y[ag.index(r["model"]), sc.index(r["scenario"]), arms.index(r["cell"]), r["seed"]] = int(r["key"])
def D(Y):
    k = Y.sum(3); n = Y.shape[3]; A, T, M = k.shape; d_arm = d_pool = 0.0
    for a in range(A):
        for t in range(T):
            for m in range(M):
                pt = b(k[:, t, m].sum() - k[a, t, m], n * (A - 1))
                ka = k[a, :, m].sum() - k[a, t, m]; ko = k[:, :, m].sum() - k[:, t, m].sum() - ka
                off_arm = lo(b(ka, n * (T - 1))) - lo(b(ko, n * (T - 1) * (A - 1)))
                kp = k[a].sum() - k[a, t].sum(); kop = k.sum() - k[:, t].sum() - kp
                off_pool = lo(b(kp, n * M * (T - 1))) - lo(b(kop, n * M * (T - 1) * (A - 1)))
                y = k[a, t, m]
                for off, which in ((off_arm, 0), (off_pool, 1)):
                    p = ex(lo(pt) + off)
                    g = y * (math.log(p) - math.log(pt)) + (n - y) * (math.log(1 - p) - math.log(1 - pt))
                    if which == 0: d_arm += g
                    else: d_pool += g
    N = Y.size; return d_arm / N, d_pool / N
da, dp = D(Y); gain = da - dp
rng = np.random.default_rng(20261104); null = []
for i in range(2000):
    Z = Y.copy()
    for a in range(5):
        for t in range(8):
            flat = rng.permutation(Y[a, t].reshape(-1)); Z[a, t] = flat.reshape(2, 3)
    x, y = D(Z); null.append(x - y)
null = np.array(null)
print(f"E2 D_add arm-specific {da:+.4f}, pooled offset same base {dp:+.4f}, gain {gain:+.4f}")
print(f"E2 gain null (arm labels within agent x task): mean {null.mean():+.4f} q95 {np.quantile(null,.95):+.4f} q99 {np.quantile(null,.99):+.4f} P(>=obs) {(1+(null>=gain).sum())/2001:.4f}")
json.dump(dict(E1_repeat=int(rep), E2_arm=da, E2_pool=dp, E2_gain=gain, null_mean=float(null.mean()),
               null_q95=float(np.quantile(null, .95)), null_q99=float(np.quantile(null, .99)),
               p=float((1 + (null >= gain).sum()) / 2001)), open("counted/vera_e1e2.json", "w"), indent=1)
