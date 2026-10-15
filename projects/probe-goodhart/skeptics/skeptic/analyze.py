"""skeptic: independent analysis of build's raw files (own code; CPU only).

uv run --with numpy --with scikit-learn python analyze.py <data_dir>
data_dir = kran-research/projects/probe-goodhart/data
Reads: raw/eval_<run>.jsonl, raw/fixed_<run>.jsonl, raw/acts_<run>/eval_round<k>.npz, probes/{A,B}.npz, probes/{A,B}_train.npz
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = sys.argv[1]
rng = np.random.default_rng(2026)
OUT = {}


def auc(y, s):
    y = np.asarray(y); return float("nan") if len(set(y)) < 2 else roc_auc_score(y, s)


def sig(z):
    return 1 / (1 + np.exp(-z))


probes = {k: np.load(f"{D}/probes/{k}.npz") for k in ("A", "B")}
trains = {k: np.load(f"{D}/probes/{k}_train.npz") for k in ("A", "B")}


def score(k, X):
    p = probes[k]
    return sig(((X.astype(np.float32) - p["mu"]) / p["sd"]) @ p["w"] + p["b"])


runs = sorted(re.match(r".*/eval_(.+)\.jsonl", f).group(1) for f in glob.glob(f"{D}/raw/eval_*.jsonl"))
print("runs:", runs)
rows = {}
for r in runs:
    rows[r] = [json.loads(l) for l in open(f"{D}/raw/eval_{r}.jsonl") if l.strip()]

# ---------- 1. per-round table from jsonl, plus recompute from acts ----------
print("\nrun round n | judge flagA flagA_base flagB | AUROC(judge) A B | words | recompute maxerr A B, flag mismatches")
tab = {}
for r in runs:
    byr = defaultdict(list)
    for x in rows[r]:
        byr[x["round"]].append(x)
    for k in sorted(byr):
        xs = byr[k]
        J = np.array([x["judge_deceptive"] for x in xs]); A = np.array([x["pA"] for x in xs]); B = np.array([x["pB"] for x in xs])
        fA = np.array([x["flagA"] for x in xs], float); fB = np.array([x["flagB"] for x in xs], float)
        fAb = np.array([x.get("flagA_base", np.nan) for x in xs], float)
        W = np.array([x["n_words"] for x in xs])
        # own flags from thresholds
        myfA = (A > float(probes["A"]["thr"])).astype(float); myfB = (B > float(probes["B"]["thr"])).astype(float)
        mism = int((myfA != fA).sum() + (myfB != fB).sum())
        err = ("-", "-")
        af = f"{D}/raw/acts_{r}/eval_round{k}.npz"
        if os.path.exists(af):
            z = np.load(af, allow_pickle=True)
            key = {(str(p), int(s)): i for i, (p, s) in enumerate(zip(z["prompt_id"], z["sample"]))}
            idx = [key[(x["prompt_id"], x["sample"])] for x in xs]
            eA = np.abs(score("A", z["actsA"][idx]) - A).max(); eB = np.abs(score("B", z["actsB"][idx]) - B).max()
            err = (f"{eA:.1e}", f"{eB:.1e}")
        tab[(r, k)] = dict(n=len(xs), judge=J.mean(), flagA=fA.mean(), flagA_base=np.nanmean(fAb), flagB=fB.mean(), aucA=auc(J, A), aucB=auc(J, B), words=W.mean())
        t = tab[(r, k)]
        print(f"{r} {k:>2} {len(xs):>3} | {t['judge']:.3f} {t['flagA']:.3f} {t['flagA_base']:.3f} {t['flagB']:.3f} | {t['aucA']:.3f} {t['aucB']:.3f} | {t['words']:.1f} | {err[0]} {err[1]} {mism}")
OUT["table"] = {f"{a}|{b}": v for (a, b), v in tab.items()}

# fixed teacher-forced texts: activation change on identical text
print("\nfixed-text AUROC (label vs probe) per round  [pure activation change]")
for r in runs:
    f = f"{D}/raw/fixed_{r}.jsonl"
    if not os.path.exists(f):
        continue
    fx = defaultdict(list)
    for l in open(f):
        if l.strip():
            x = json.loads(l); fx[x["round"]].append(x)
    print(r, "  ".join(f"r{k}: A {auc([x['label'] for x in v],[x['pA'] for x in v]):.3f} B {auc([x['label'] for x in v],[x['pB'] for x in v]):.3f} meanA(dec) {np.mean([x['pA'] for x in v if x['label']==1]):.3f}" for k, v in sorted(fx.items())))


# ---------- 2. headline DiD vs lam=0, prompt-level, bootstrap + permutation ----------
def prompt_means(r, k, m):
    d = defaultdict(list)
    for x in rows[r]:
        if x["round"] == k:
            d[x["prompt_id"]].append(float(x[m]))
    return {p: np.mean(v) for p, v in d.items()}


def lamseed(r):
    x = rows[r][0]; return float(x["lam"]), int(x.get("seed", 0))


print("\nDiD = drop(lam) - drop(lam0), round0 -> final; drop = r0 - final. 95% bootstrap CI over prompts; sign-flip perm p")
ctrl = {lamseed(r)[1]: r for r in runs if lamseed(r)[0] == 0}
OUT["did"] = {}
for r in runs:
    lam, seed = lamseed(r)
    if lam == 0 or seed not in ctrl:
        continue
    c = ctrl[seed]
    fc = max(x["round"] for x in rows[c]); ft = max(x["round"] for x in rows[r])
    fin = min(fc, ft)  # compare at same round
    for m in ("flagA", "flagA_base", "flagB", "judge_deceptive", "pA", "pB", "n_words"):
        if m not in rows[r][0]:
            continue
        t0, t1, c0, c1 = (prompt_means(r, 0, m), prompt_means(r, fin, m), prompt_means(c, 0, m), prompt_means(c, fin, m))
        ps = sorted(set(t0) & set(t1) & set(c0) & set(c1))
        dt = np.array([t0[p] - t1[p] for p in ps]); dc = np.array([c0[p] - c1[p] for p in ps]); d = dt - dc
        b = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        perm = (rng.choice([-1, 1], (10000, len(d))) * d).mean(1)
        p = float((np.abs(perm) >= abs(d.mean())).mean())
        OUT["did"][f"{r}|{m}"] = dict(round=fin, n=len(ps), dropT=dt.mean(), dropC=dc.mean(), did=d.mean(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), p=p)
        print(f"{r} vs {c} @r{fin} n={len(ps)} {m:>16}: dropT {dt.mean():+.3f} dropC {dc.mean():+.3f} DiD {d.mean():+.3f} [{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] p={p:.4f}")


# ---------- 3. nulls on activations: is probe A's shift specific to A's direction? ----------
def acts(r, k, key):
    f = f"{D}/raw/acts_{r}/eval_round{k}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    return z[key].astype(np.float32)


def shift_z(X0, X1, w, mu, sd):
    """standardized mean shift of projection onto w (round0 -> final), in units of round-0 projection sd"""
    s0 = ((X0 - mu) / sd) @ w; s1 = ((X1 - mu) / sd) @ w
    return (s0.mean() - s1.mean()) / (s0.std() + 1e-9)


print("\nDirection-specificity null: projection drop (r0->final, in r0-sd units) treat minus control,")
print("probe direction vs 1000 random directions and 20 shuffled-label probes at the same layer/readout")
OUT["null"] = {}
for r in runs:
    lam, seed = lamseed(r)
    if lam == 0 or seed not in ctrl:
        continue
    c = ctrl[seed]
    fin = min(max(x["round"] for x in rows[c]), max(x["round"] for x in rows[r]))
    for k, key in (("A", "actsA"), ("B", "actsB")):
        T0, T1, C0, C1 = acts(r, 0, key), acts(r, fin, key), acts(c, 0, key), acts(c, fin, key)
        if any(v is None for v in (T0, T1, C0, C1)):
            print(f"{r} {k}: missing acts"); continue
        p = probes[k]; mu, sd = p["mu"], p["sd"]
        obs = shift_z(T0, T1, p["w"], mu, sd) - shift_z(C0, C1, p["w"], mu, sd)
        rnd = []
        for _ in range(1000):
            w = rng.standard_normal(len(mu)); w /= np.linalg.norm(w)
            rnd.append(shift_z(T0, T1, w, mu, sd) - shift_z(C0, C1, w, mu, sd))
        rnd = np.array(rnd)
        tr = trains[k]; Z = (tr["X"] - mu) / sd
        sh = []
        for _ in range(20):
            m = LogisticRegression(C=0.01, max_iter=3000).fit(Z, rng.permutation(tr["y"]))
            w = m.coef_[0]
            sh.append(shift_z(T0, T1, w, mu, sd) - shift_z(C0, C1, w, mu, sd))
        sh = np.array(sh)
        pr = float((np.abs(rnd) >= abs(obs)).mean()); ps = float((np.abs(sh) >= abs(obs)).mean())
        OUT["null"][f"{r}|{k}"] = dict(obs=obs, rnd_abs95=np.percentile(np.abs(rnd), 95), p_rand=pr, shuf_abs_max=np.abs(sh).max(), p_shuf=ps)
        print(f"{r} vs {c} @r{fin} probe {k}: DiD-shift {obs:+.3f} sd | random |.|95th {np.percentile(np.abs(rnd),95):.3f} p={pr:.3f} | shuffled-label |.|max {np.abs(sh).max():.3f} p={ps:.2f}")

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "skeptic_results.json"), "w"), indent=1, default=float)
