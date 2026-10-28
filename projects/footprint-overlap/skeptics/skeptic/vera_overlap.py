"""Vera's own overlap checker (skeptic). numpy + sklearn only, no torch.

Reads block-26-input traces (N x 34 x 1024, bf16 stored as uint16, or fp32/fp16 .npy),
fits its OWN A-vs-none classifier on the train split, and scores every build B on test prompts:
  p_rescaled  = (mean P(B) - mean P(none)) / (mean P(A) - mean P(none))
  projection  = mean(w . (xB - xnone)) / mean(w . (xA - xnone))    (w = unit logistic direction)
  l2_ratio    = mean ||xB - xnone|| / mean ||xA - xnone||
Usage: vera_overlap.py --dir D --ref none_rec --a ref_rec --pos 33 --C 1.0 B1 B2 ...
  (file name = <dir>/<build>_tail.npy unless --pattern is given)
"""
import argparse, json, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load(path, pos):
    a = np.load(path, mmap_mode="r")
    if pos == "mean":
        x = a[:, :, :]
    else:
        x = a[:, int(pos), :]
    x = np.asarray(x)
    if x.dtype == np.uint16:  # bf16 bits
        x = (x.astype(np.uint32) << 16).view(np.float32)
    x = x.astype(np.float32)
    if pos == "mean":
        x = x.mean(axis=1)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="{b}_tail.npy")
    ap.add_argument("--ref", default="none_rec")
    ap.add_argument("--a", default="ref_rec")
    ap.add_argument("--pos", default="33")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--split", required=True, help="prompts.jsonl with split field")
    ap.add_argument("--shuffle", type=int, default=0, help="null: shuffle A/none labels in the fit (seed)")
    ap.add_argument("--randdir", type=int, default=0, help="null: random unit direction instead of w (seed)")
    ap.add_argument("--out")
    ap.add_argument("builds", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    split = np.array([json.loads(l)["split"] for l in open(a.split)])
    tr, te = split == "train", split == "test"
    f = lambda b: f"{a.dir}/{a.pattern.format(b=b)}"
    X0, XA = load(f(a.ref), a.pos), load(f(a.a), a.pos)
    # standardise on the train rows of both classes (frozen from train only)
    Xtr = np.concatenate([X0[tr], XA[tr]])
    ytr = np.r_[np.zeros(tr.sum()), np.ones(tr.sum())]
    if a.shuffle:
        ytr = np.random.default_rng(a.shuffle).permutation(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=a.C, max_iter=5000)
    clf.fit((Xtr - mu) / sd, ytr)
    w = clf.coef_[0] / sd  # direction in raw space
    if a.randdir:
        w = np.random.default_rng(a.randdir).standard_normal(w.shape)
    w = w / np.linalg.norm(w)
    P = lambda X: clf.predict_proba((X - mu) / sd)[:, 1]
    p0, pA = P(X0[te]), P(XA[te])
    auc = roc_auc_score(np.r_[np.zeros(te.sum()), np.ones(te.sum())], np.r_[p0, pA])
    dA = XA[te] - X0[te]
    projA, l2A = float((dA @ w).mean()), float(np.linalg.norm(dA, axis=1).mean())
    res = {"ref": a.ref, "a": a.a, "pos": a.pos, "C": a.C, "shuffle": a.shuffle, "randdir": a.randdir,
           "test_auc_A_vs_none": float(auc), "meanP_none": float(p0.mean()), "meanP_A": float(pA.mean()),
           "frac_rows_changed_A": float((np.abs(dA).max(1) > 0).mean()), "builds": {}}
    for b in a.builds:
        XB = load(f(b), a.pos)[te]
        dB = XB - X0[te]
        pB = P(XB)
        r = {"p_rescaled": float((pB.mean() - p0.mean()) / (pA.mean() - p0.mean())),
             "projection": float((dB @ w).mean() / projA),
             "l2_ratio": float(np.linalg.norm(dB, axis=1).mean() / l2A),
             "cos_mean_delta_vs_A": float(dB.mean(0) @ dA.mean(0) / (np.linalg.norm(dB.mean(0)) * np.linalg.norm(dA.mean(0)) + 1e-12)),
             "frac_rows_changed": float((np.abs(dB).max(1) > 0).mean()),
             "frac_rows_identical_to_A": float((np.abs(XB - XA[te]).max(1) == 0).mean()),
             "meanP": float(pB.mean())}
        res["builds"][b] = r
        print(b, json.dumps({k: round(v, 4) for k, v in r.items()}))
    res["sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "builds"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
