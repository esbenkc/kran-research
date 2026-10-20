"""Independent rebuild of D8 deviation-note numbers. Own scoring; no project code imported."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score

P = "/Users/esben/work/oracle-1020-megaprobe-spin-lies/kran-research/projects/megaprobe-spin-lies/data"
POL = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
TRAINED = POL[1:]
C2FILE = {
    "base": "c2_base_lr1e-3_e40.json",
    "lam1_s0": "c2_lam1_s0_lr1e-3_e40_v2.json",
    "lam1_s1": "c2_lam1_s1_lr1e-3_e40.json",
    "ens_s0": "c2_ens_s0_lr1e-3_e40.json",
    "ens_s1": "c2_ens_s1_lr1e-3_e40_v2.json",
}


def labels(pol):
    rows = [json.loads(l) for l in open(f"{P}/refit_probe/labels_{pol}.jsonl")][:1000]
    yj = np.array([np.nan if r["y_judge"] is None else r["y_judge"] for r in rows], float)
    ys = np.array([np.nan if r.get("y_string") is None else r["y_string"] for r in rows], float)
    truthful = yj == 0
    spin = (yj == 1) & (ys == 0)
    plain = (yj == 1) & (ys == 1)
    return truthful, spin, plain, ys


def subset_auc(scores, pol, which):
    t, s, p, ys = labels(pol)
    if which == "spin":
        pos, neg = s, t & (ys == 0)
    else:
        pos, neg = s | p, t
    m = (pos | neg) & ~np.isnan(scores)
    return roc_auc_score(pos[m].astype(int), scores[m]), m, pos


def pooled(d, pick):
    """pick(log) -> index of chosen epoch entry; returns full-length score vector (NaN outside test)."""
    out = np.full(1000, np.nan)
    chosen = []
    for f, fd in d["per_fold"].items():
        i = pick(fd["log"])
        chosen.append(fd["log"][i]["epoch"])
        out[np.array(fd["te"])] = fd["log"][i]["test"]
    return out, chosen


def best_val_auc(log):
    v = [e["val_auroc"] for e in log]
    return int(np.argmax(v))  # first max


def min_val_bce_first10(log):
    v = [e["val_bce"] for e in log if e["epoch"] <= 9]
    idx = [i for i, e in enumerate(log) if e["epoch"] <= 9]
    return idx[int(np.argmin(v))]


def last(log):
    return len(log) - 1


rows = []


def rec(claim, claimed, val, kind="auc", lo=None, hi=None):
    if kind == "auc":
        ok = abs(val - claimed) <= 0.002
    elif kind == "range":
        ok = lo - 0.005 <= val <= hi + 0.005  # stated to 2 decimals
    elif kind == "mean2":
        ok = round(val, 2) == claimed or abs(val - claimed) <= 0.005
    elif kind == "approx":
        ok = abs(val - claimed) <= 0.002
    rows.append((claim, claimed if kind != "range" else f"{lo}-{hi}", val, "MATCH" if ok else "MISMATCH"))


C2 = {p: json.load(open(f"{P}/convergence/{C2FILE[p]}")) for p in POL}

# --- c2 best val AUROC (all 40 epochs) and min val_bce first 10
claimed_best = {"base": (0.636, 0.766), "lam1_s0": (0.656, 0.697), "lam1_s1": (0.573, 0.655),
                "ens_s0": (0.621, 0.702), "ens_s1": (0.584, 0.646)}
claimed_bce = {"base": (0.615, 0.658), "lam1_s0": (0.556, 0.528), "lam1_s1": (0.494, 0.519),
               "ens_s0": (0.520, 0.587), "ens_s1": (0.580, 0.628)}
for name, picker, claimed, mclaim in [("c2 bestValAUROC", best_val_auc, claimed_best, (0.61, 0.68)),
                                      ("c2 minValBCE<=ep9", min_val_bce_first10, claimed_bce, (0.54, 0.57))]:
    acc = {"spin": [], "lies": []}
    for p in POL:
        sc, ch = pooled(C2[p], picker)
        print(f"{name} {p} chosen epochs {ch}")
        for j, w in enumerate(["spin", "lies"]):
            a, _, _ = subset_auc(sc, p, w)
            rec(f"{name} {p} {w}", claimed[p][j], a)
            if p in TRAINED:
                acc[w].append(a)
    for j, w in enumerate(["spin", "lies"]):
        rec(f"{name} mean4 trained {w}", mclaim[j], float(np.mean(acc[w])), "mean2")

# --- c2 train metrics
tr9 = []; bce9 = []; bceL = []
for p in POL:
    d = C2[p]
    c9 = [c for c in d["curve"] if c["epoch"] == 9][0]
    # recompute mean over folds from per_fold logs
    m9 = np.mean([[e for e in fd["log"] if e["epoch"] == 9][0]["train_auroc"] for fd in d["per_fold"].values()])
    assert abs(m9 - c9["train_auroc"]) < 1e-9, (p, m9, c9["train_auroc"])
    tr9.append(c9["train_auroc"]); bce9.append(c9["train_bce"]); bceL.append(d["curve"][-1]["train_bce"])
    print(f"c2 {p}: train_auroc@9 {c9['train_auroc']:.4f} (fold-mean {m9:.4f}) train_bce@9 {c9['train_bce']:.4f} train_bce@last(ep{d['curve'][-1]['epoch']}) {d['curve'][-1]['train_bce']:.4f}")
rec("c2 train_auroc@ep9 min", None, min(tr9), "range", 0.67, 0.79); rows[-1] = rows[-1][:2] + rows[-1][2:]
rec("c2 train_auroc@ep9 max", None, max(tr9), "range", 0.67, 0.79)
rec("c2 train_bce@ep9 min", None, min(bce9), "range", 0.54, 0.69)
rec("c2 train_bce@ep9 max", None, max(bce9), "range", 0.54, 0.69)
rec("c2 train_bce@last min", None, min(bceL), "range", 0.14, 0.40)
rec("c2 train_bce@last max", None, max(bceL), "range", 0.14, 0.40)

# --- one-layer probe draw 0
claimed_one = {"base": (0.732, 0.837), "lam1_s0": (0.652, 0.782), "lam1_s1": (0.806, 0.799),
               "ens_s0": (0.696, 0.780), "ens_s1": (0.655, 0.735)}
acc = {"spin": [], "lies": []}
for p in POL:
    a_all = json.load(open(f"{P}/fits/{p}_own.json"))["scores"]["a_all"]
    sc = np.array([np.nan if (r is None or r[0] is None) else r[0] for r in a_all], float)
    for j, w in enumerate(["spin", "lies"]):
        a, _, _ = subset_auc(sc, p, w)
        rec(f"one-layer d0 {p} {w}", claimed_one[p][j], a)
        if p in TRAINED:
            acc[w].append(a)
rec("one-layer mean4 trained spin", 0.70, float(np.mean(acc["spin"])), "mean2")
rec("one-layer mean4 trained lies", 0.77, float(np.mean(acc["lies"])), "mean2")

# --- megaprobe c lam1_s0 lr1e-2 e1500 (curve only)
c = json.load(open(f"{P}/convergence/c_lam1_s0_lr1e-2_e1500.json"))
cur = {e["epoch"]: e for e in c["curve"]}
rec("c ep9 spin", 0.647, cur[9]["spin_vs_fact_truthful"])
rec("c ep9 lies", 0.772, cur[9]["lies_vs_truthful"])
late = [e for e in c["curve"] if 299 <= e["epoch"] <= 1499]
sp = [e["spin_vs_fact_truthful"] for e in late]; li = [e["lies_vs_truthful"] for e in late]
print(f"c ep299-1499: spin [{min(sp):.4f},{max(sp):.4f}] lies [{min(li):.4f},{max(li):.4f}]")
rec("c ep299-1499 spin min", None, min(sp), "range", 0.575, 0.580)
rec("c ep299-1499 spin max", None, max(sp), "range", 0.575, 0.580)
rec("c ep299-1499 lies min (~0.697)", 0.697, min(li), "approx")
rec("c ep299-1499 lies max (~0.697)", 0.697, max(li), "approx")
b = [round(e["train_bce"], 3) for e in c["curve"] if 399 <= e["epoch"] <= 1499]
print(f"c train_bce ep399-1499 rounded set: {sorted(set(b))}; raw range "
      f"[{min(e['train_bce'] for e in c['curve'] if e['epoch']>=399):.5f},{max(e['train_bce'] for e in c['curve'] if e['epoch']>=399):.5f}]")
rows.append(("c train_bce flat 0.235 ep399-1499 (3dp)", 0.235, ",".join(map(str, sorted(set(b)))),
             "MATCH" if set(b) == {0.235} else "MISMATCH"))
ta = [e["train_auroc"] for e in c["curve"] if e["epoch"] >= 399]
print(f"c train_auroc ep399-1499 range [{min(ta):.4f},{max(ta):.4f}]; last {c['curve'][-1]['train_auroc']:.4f}")
rec("c train_auroc (last epoch)", 0.994, c["curve"][-1]["train_auroc"], "approx")

# --- c l2 1e-3
c3 = json.load(open(f"{P}/convergence/c_lam1_s0_lr1e-2_e1500_l2_1e-3.json"))
first1 = [e["epoch"] for e in c3["curve"] if round(e["train_auroc"], 3) >= 1.0]
print(f"c_l2 curve train_auroc by epoch: {[(e['epoch'], round(e['train_auroc'],4)) for e in c3['curve'][:8]]}")
rows.append(("c_l2 train_auroc 1.000 by ep49", "<=49", first1[0] if first1 else None,
             "MATCH" if first1 and first1[0] <= 49 and all(round(e['train_auroc'], 3) >= 1.0 for e in c3['curve'] if e['epoch'] >= first1[0]) else "MISMATCH"))
sc, ch = pooled(c3, best_val_auc)
print(f"c_l2 bestVal chosen epochs {ch}")
rec("c_l2 bestValAUROC spin", 0.605, subset_auc(sc, "lam1_s0", "spin")[0])
sc, ch = pooled(c3, last)
rec("c_l2 last-epoch spin", 0.573, subset_auc(sc, "lam1_s0", "spin")[0])

# --- null test: shuffle labels in spin subset for c2 base best-val scores
sc, _ = pooled(C2["base"], best_val_auc)
_, m, pos = subset_auc(sc, "base", "spin")
y = pos[m].astype(int); s = sc[m]
rng = np.random.default_rng(0)
null = [roc_auc_score(rng.permutation(y), s) for _ in range(200)]
print(f"null: n={m.sum()} pos={y.sum()} mean {np.mean(null):.4f} sd {np.std(null):.4f} 95% [{np.percentile(null,2.5):.3f},{np.percentile(null,97.5):.3f}]")
rows.append(("null shuffled spin AUROC mean (200)", "~0.5", round(float(np.mean(null)), 4),
             "MATCH" if abs(np.mean(null) - 0.5) < 0.01 else "MISMATCH"))

print("\n| claim | claimed | mine | verdict |\n|---|---|---|---|")
for r in rows:
    v = r[2] if isinstance(r[2], str) else (f"{r[2]:.4f}" if isinstance(r[2], float) else r[2])
    print(f"| {r[0]} | {r[1]} | {v} | {r[3]} |")
bad = [r[0] for r in rows if r[3] != "MATCH"]
print("\nVERDICT:", "PASS" if not bad else "FAIL: " + "; ".join(bad))
