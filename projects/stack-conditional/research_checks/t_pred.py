# T2 (Iris): frozen read point + predicted carried share T_pred(S) for unseen stacks.
# Inputs: block-26 input caches (N, T, H) bf16-as-uint16 (or float) for grading E (eager),
# serving C (compiled) and each unseen stack S, same prompt order; prompt splits from a jsonl.
# Definitions (frozen in PREREG v1, Kasper #9545):
#   d_t   = mean over train prompts (PREREG s4) of (x_C - x_E) at position t
#   s_i,t = <x_S,i - x_E,i, d_t> / <x_C,i - x_E,i, d_t>
#   T_pred(S) = median over the 200 test prompts of s_i,t at the read point t*
# Read point rule (Iris): t* = position with the highest held-out logistic AUROC for E vs C
# (fit on train, scored on dev; test untouched); ties (within 0.001) go to the LATEST position,
# the one closest to the answer token. Every position's value is also reported.
import argparse, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--E", required=True)
ap.add_argument("--C", required=True)
ap.add_argument("--S", nargs="*", default=[], help="name=file pairs")
ap.add_argument("--prompts", required=True)
ap.add_argument("--ids", help="json list of prompt ids in cache order (default: prompts order)")
ap.add_argument("--out", required=True)
a = ap.parse_args()


def load(f):
    x = np.load(f"{a.cache}/{f}", mmap_mode="r")
    if x.dtype == np.uint16:
        x = (x.astype(np.uint32) << 16).view(np.float32)
    return np.asarray(x, dtype=np.float32)


rows = [json.loads(l) for l in open(a.prompts)]
split = {r["id"]: r["split"] for r in rows}
ids = json.load(open(a.ids)) if a.ids else [r["id"] for r in rows]
sp = np.array([split[i] for i in ids])
tr, dv, te = (np.where(sp == s)[0] for s in ("train", "dev", "test"))
fit = tr  # PREREG s4: d = mean over the TRAIN prompts

XE, XC = load(a.E), load(a.C)
N, T, H = XE.shape
S = {kv.split("=")[0]: load(kv.split("=")[1]) for kv in a.S}
out = {"N": N, "T": T, "H": H, "n_train": len(tr), "n_dev": len(dv), "n_test": len(te),
       "files": {"E": a.E, "C": a.C, **dict(kv.split("=", 1) for kv in a.S)}}

# bit-difference fractions per position (all prompts)
pairs = {"C_vs_E": (XC, XE)}
for k, X in S.items():
    pairs[f"{k}_vs_E"] = (X, XE)
    pairs[f"{k}_vs_C"] = (X, XC)
out["bitdiff_frac_coords_per_pos"] = {k: (A != B).mean(axis=(0, 2)).round(4).tolist() for k, (A, B) in pairs.items()}
out["bitdiff_prompts_any"] = {k: int((A != B).any(axis=(1, 2)).sum()) for k, (A, B) in pairs.items()}

# per-position E vs C logistic AUROC: fit train, score dev (read-point pick), and test (report)
au_dev, au_test = [], []
for t in range(T):
    Xtr = np.concatenate([XE[tr, t], XC[tr, t]]); ytr = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=1000).fit((Xtr - mu) / sd, ytr)
    for idx, acc in ((dv, au_dev), (te, au_test)):
        X = np.concatenate([XE[idx, t], XC[idx, t]]); y = np.r_[np.zeros(len(idx)), np.ones(len(idx))]
        acc.append(round(float(roc_auc_score(y, clf.decision_function((X - mu) / sd))), 4))
out["auroc_E_vs_C_dev"] = au_dev
out["auroc_E_vs_C_test"] = au_test
best = max(au_dev)
tstar = max(t for t in range(T) if au_dev[t] >= best - 0.001)
out["read_point"] = tstar
out["read_point_rule"] = "argmax dev AUROC (E vs C, logistic fit on train); ties within 0.001 -> latest position"

# projection share per position
D = (XC[fit] - XE[fit]).mean(0)  # (T, H)
den = np.einsum("nth,th->nt", XC - XE, D)
out["denominator_zero_test_prompts_at_read_point"] = int((den[te, tstar] == 0).sum())
res = {}
for k, X in S.items():
    num = np.einsum("nth,th->nt", X - XE, D)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = num / den
    med = [float(np.nanmedian(s[te, t])) for t in range(T)]
    q = np.nanpercentile(s[te, tstar], [25, 75])
    # pooled over all positions (secondary): one projection on the flattened (T*H) direction
    pooled = np.einsum("nth,th->n", X - XE, D) / np.einsum("nth,th->n", XC - XE, D)
    res[k] = {"T_pred": round(med[tstar], 4), "iqr_at_read_point": [round(float(v), 4) for v in q],
              "median_per_pos": [round(v, 4) for v in med],
              "pooled_all_positions_median": round(float(np.nanmedian(pooled[te])), 4)}
out["T_pred"] = res
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({"read_point": tstar, "auroc_dev_at_rp": au_dev[tstar],
                  "T_pred": {k: (v["T_pred"], v["iqr_at_read_point"], v["pooled_all_positions_median"]) for k, v in res.items()},
                  "bitdiff_prompts_any": out["bitdiff_prompts_any"]}))
