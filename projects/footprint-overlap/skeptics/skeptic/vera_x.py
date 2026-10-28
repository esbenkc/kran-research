"""Vera's own x-axis rebuild over every screened build (skeptic). numpy + sklearn, no torch.

For each build in data/builds_frozen.txt with screen files present:
  class    : bitexact (plain last-row logits == none on 200/200), aclone (rec trace == A's on >=50% of test),
             leaky (logits differ on >=50%), weak (else)       -- from raw bits, own code
  overlap  : PRIMARY rescaled P at pos 25 (own fit: std on train none+A, L2 C=1, lbfgs, max_iter 5000)
             + projection ratio, L2-size ratio, allpos(3-33) rescaled P
  nulls    : shuffled-label fit (3 seeds), random unit direction (3 seeds) for the projection ratio
Output: vera_x.json
"""
import json, os, sys, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
PRJ = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VERA_PRJ", "../../kran-research/projects/footprint-overlap")
D = f"{PRJ}/raw/screen/base"
t0 = time.time()
split = np.array([json.loads(l)["split"] for l in open(f"{PRJ}/raw/prompts.jsonl")]) if os.path.exists(
    f"{PRJ}/raw/prompts.jsonl") else None
if split is None:
    split = np.array([json.loads(l)["split"] for l in open(
        os.path.expanduser("~/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/data/prompts.jsonl"))])
tr, te = split == "train", split == "test"


def tag(spec):
    s = spec.replace("#rerun", "_rerun").replace("@", "_").replace("+emu", "_emu")
    return s


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


builds = []
for line in open(f"{PRJ}/data/builds_frozen.txt"):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split()
    builds.append((p[0], p[-1]))

T = lambda b: np.load(f"{D}/{tag(b)}_rec_tail.npy", mmap_mode="r")
Lg = lambda b: np.load(f"{D}/{tag(b)}_plain_logits.npy", mmap_mode="r")
tn, ta = T("none"), T("ref@L14.q_norm")
ln = np.asarray(Lg("none"))
X0, XA = f32(tn[:, 25, :]), f32(ta[:, 25, :])
M0, MA = f32(tn[:, 3:, :]).mean(1), f32(ta[:, 3:, :]).mean(1)
ta_te = np.asarray(ta)[te].reshape(te.sum(), -1)


def fit(X0, XA, shuffle=0):
    Xtr = np.concatenate([X0[tr], XA[tr]]).astype(np.float64)
    y = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
    if shuffle:
        y = np.random.default_rng(shuffle).permutation(y)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    c = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000).fit((Xtr - mu) / sd, y)
    P = lambda X: c.predict_proba((X - mu) / sd)[:, 1]
    w = c.coef_[0] / sd
    return P, w / np.linalg.norm(w)


P25, w25 = fit(X0, XA)
Pall, _ = fit(M0, MA)
nullP = [fit(X0, XA, s) for s in (11, 12, 13)]
rdirs = [np.random.default_rng(s).standard_normal(1024) for s in (21, 22, 23)]
rdirs = [r / np.linalg.norm(r) for r in rdirs]
p0, pA = P25(X0[te]).mean(), P25(XA[te]).mean()
q0, qA = Pall(M0[te]).mean(), Pall(MA[te]).mean()
dA = XA[te] - X0[te]
out = {"meta": {"meanP_none": p0, "meanP_A": pA, "allpos_meanP_none": q0, "allpos_meanP_A": qA}, "builds": {}}
for kind, b in builds:
    if not (os.path.exists(f"{D}/{tag(b)}_rec_tail.npy") and os.path.exists(f"{D}/{tag(b)}_plain_logits.npy")):
        continue
    tb = T(b)
    lb = np.asarray(Lg(b))
    diff_rows = int((lb != ln).any(1).sum())
    eqA = float((np.asarray(tb)[te].reshape(te.sum(), -1) == ta_te).all(1).mean())
    trace_eq_none = float((np.asarray(tb)[te].reshape(te.sum(), -1) == np.asarray(tn)[te].reshape(te.sum(), -1)).all(1).mean())
    cls = "bitexact" if diff_rows == 0 else ("aclone" if eqA >= 0.5 else ("leaky" if diff_rows >= 100 else "weak"))
    XB = f32(tb[:, 25, :])
    MB = f32(tb[:, 3:, :]).mean(1)
    dB = XB[te] - X0[te]
    r = {"kind": kind, "class": cls, "logit_rows_diff": diff_rows, "trace_eqA": eqA, "trace_eq_none": trace_eq_none,
         "overlap": float((P25(XB[te]).mean() - p0) / (pA - p0)),
         "proj": float((dB @ w25).mean() / (dA @ w25).mean()),
         "l2": float(np.linalg.norm(dB, axis=1).mean() / np.linalg.norm(dA, axis=1).mean()),
         "allpos": float((Pall(MB[te]).mean() - q0) / (qA - q0)),
         "null_shuf": [float((P(XB[te]).mean() - P(X0[te]).mean()) / (P(XA[te]).mean() - P(X0[te]).mean() + 1e-12))
                       for P, _ in nullP],
         "null_randdir": [float((dB @ r_).mean() / ((dA @ r_).mean() + 1e-12)) for r_ in rdirs]}
    out["builds"][b] = r
out["meta"]["sec"] = round(time.time() - t0, 1)
out["meta"]["n_builds"] = len(out["builds"])
json.dump(out, open("vera_x.json", "w"), indent=1)
from collections import Counter
print(json.dumps(out["meta"]), Counter(v["class"] for v in out["builds"].values()))
for b, r in out["builds"].items():
    print(f'{r["kind"]:4s} {b:24s} {r["class"]:8s} diff={r["logit_rows_diff"]:3d} eqN={r["trace_eq_none"]:.2f} '
          f'ov={r["overlap"]:+.3f} proj={r["proj"]:+.3f} l2={r["l2"]:.3f} all={r["allpos"]:+.3f}')
