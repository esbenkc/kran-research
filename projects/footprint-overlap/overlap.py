"""Overlap score per monitor build, from base-model screens only (no trained policy, no behaviour). PREREG.md s3-s4.

Inputs (written by fp.py J1, base model): raw/screen/base/<tag>_rec_{pos25,mean3to33}.npy (fp32 [646, 1024] block-26
input of the rec pass), <tag>_rec_tail.npy (uint16 [646, 34, 1024], for the A-clone sha), <tag>_plain_sha.json and
<tag>_plain_logits.npy (plain pass, 200 test prompts, last-row logits), and raw/screen/base_train/{none,A}_plain_logits.npy
(plain pass, 382 train prompts, for the recorder-free direction).

Class (s3): bitexact = plain logit sha == none on 200/200 test prompts; aclone = rec trace sha == A on >= 50% of test
prompts; leaky = plain logits differ from none on >= 50% of test prompts and not aclone; weak = the rest.
PRIMARY (s4): logistic A-vs-none on pos 25 ('<|im_end|>'), StandardScaler fit on train none+A rows,
  LogisticRegression(C=1.0, l2, lbfgs, max_iter=5000), 382 train prompts x {none, A};
  overlap_B = (meanP(B) - meanP(none)) / (meanP(A) - meanP(none)) over the 200 test prompts. No clipping.
SECONDARY: proj   = mean(w.d_B) / mean(w.d_A), w = classifier weight in raw units (= the standardised projection), d =
                    trace - none trace, test prompts;
           allpos = the primary recipe on the mean over positions 3-33;
           size   = mean||d_B|| / mean||d_A|| at pos 25;
           logit  = recorder-free: mean over test of (logit_B - logit_none) . u / same for A, u = mean over TRAIN prompts
                    of (logit_A - logit_none) in the plain pass (fp32 from bf16 bits).
NULLS:     shuf   = primary recipe with labels swapped inside a random half of the train pairs (seed 0);
           rand   = projection ratio along a random unit direction (seed 0).

    env/.venv/bin/python overlap.py --screen raw/screen/base --out data/overlap.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fp import read_builds, tagof

HERE = Path(__file__).parent
A = "ref@L14.q_norm"
POS = 25


def f32(u) -> np.ndarray:
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def fit(xn_tr, xa_tr, shuffle=False, seed=0):
    X = np.concatenate([xn_tr, xa_tr])
    n = len(xn_tr)
    if shuffle:
        flip = np.random.default_rng(seed).random(n) < 0.5
        y = np.r_[flip.astype(float), (~flip).astype(float)]
    else:
        y = np.r_[np.zeros(n), np.ones(n)]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000).fit(sc.transform(X), y)
    return sc, clf


def meanp(sc, clf, x) -> float:
    return float(clf.predict_proba(sc.transform(x))[:, 1].mean())


def rescaled(sc, clf, xn, xa, xb) -> float:
    pn, pa = meanp(sc, clf, xn), meanp(sc, clf, xa)
    return (meanp(sc, clf, xb) - pn) / (pa - pn)


def ready(screen: Path, b: str) -> bool:
    t = tagof(b)
    return (screen / f"{t}_rec_info.json").exists() and (screen / f"{t}_plain_info.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="raw/screen/base")
    ap.add_argument("--train-logits", default="raw/screen/base_train")
    ap.add_argument("--builds-file", default="data/builds_frozen.txt")
    ap.add_argument("--out", default="data/overlap.json")
    a = ap.parse_args()
    S = HERE / a.screen
    builds = read_builds(HERE / a.builds_file)
    prompts = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    split = np.array([p["split"] for p in prompts])
    tr, te = split == "train", split == "test"
    assert tr.sum() == 382 and te.sum() == 200
    assert list(np.load(S / "none_rec.npz")["ids"]) == [p["id"] for p in prompts]
    assert list(np.load(S / "none_plain.npz")["ids"]) == [p["id"] for p in prompts if p["split"] == "test"]
    tA = tagof(A)

    xn, xa = np.load(S / "none_rec_pos25.npy"), np.load(S / f"{tA}_rec_pos25.npy")
    mn, ma = np.load(S / "none_rec_mean3to33.npy"), np.load(S / f"{tA}_rec_mean3to33.npy")
    sc, clf = fit(xn[tr], xa[tr])
    sc_all, clf_all = fit(mn[tr], ma[tr])
    sc_sh, clf_sh = fit(xn[tr], xa[tr], shuffle=True)
    w = clf.coef_[0] / sc.scale_
    rnd = np.random.default_rng(0).standard_normal(xn.shape[1]); rnd /= np.linalg.norm(rnd)
    dA = xa[te] - xn[te]
    proj_A, rand_A = float((dA @ w).mean()), float((dA @ rnd).mean())
    size_A = float(np.linalg.norm(dA, axis=1).mean())
    T = HERE / a.train_logits
    u = (f32(np.load(T / f"{tA}_plain_logits.npy")) - f32(np.load(T / "none_plain_logits.npy"))).mean(0)
    ln = f32(np.load(S / "none_plain_logits.npy"))
    logit_A = float(((f32(np.load(S / f"{tA}_plain_logits.npy")) - ln) @ u).mean())
    tn = np.load(S / "none_rec_tail.npy", mmap_mode="r")
    ta = np.load(S / f"{tA}_rec_tail.npy", mmap_mode="r")
    idx = np.where(te)[0]
    sha = lambda arr: [hashlib.sha256(np.ascontiguousarray(arr[i]).tobytes()).hexdigest() for i in idx]
    sha_ta = sha(ta)
    sha_pn = json.load(open(S / "none_plain_sha.json"))

    out = {"A": A, "pos": POS, "n_train": int(tr.sum()), "n_test": int(te.sum()), "C": 1.0,
           "A_test": {"p_none": meanp(sc, clf, xn[te]), "p_A": meanp(sc, clf, xa[te])},
           "builds": {}, "pending": []}
    for b in builds:
        if not ready(S, b):
            out["pending"].append(b)
            continue
        t = tagof(b)
        xb, mb = np.load(S / f"{t}_rec_pos25.npy"), np.load(S / f"{t}_rec_mean3to33.npy")
        dB = xb[te] - xn[te]
        sha_pb = json.load(open(S / f"{t}_plain_sha.json"))
        logit_eq_none = float(np.mean([x == y for x, y in zip(sha_pb, sha_pn)]))
        tb = np.load(S / f"{t}_rec_tail.npy", mmap_mode="r")
        trace_eq_A = float(np.mean([x == y for x, y in zip(sha(tb), sha_ta)]))
        trace_eq_none = float(np.mean([np.array_equal(tb[i], tn[i]) for i in idx]))
        sha_rb = json.load(open(S / f"{t}_rec_sha.json"))
        rec_vs_plain = float(np.mean([sha_rb[i] == sha_pb[j] for j, i in enumerate(idx)]))
        if logit_eq_none == 1.0:
            cls = "bitexact"
        elif trace_eq_A >= 0.5:
            cls = "aclone"
        elif 1 - logit_eq_none >= 0.5:
            cls = "leaky"
        else:
            cls = "weak"
        lb = f32(np.load(S / f"{t}_plain_logits.npy"))
        r = {"class": cls, "logit_eq_none": logit_eq_none, "trace_eq_A": trace_eq_A, "trace_eq_none": trace_eq_none,
             "rec_vs_plain_logit_sha": rec_vs_plain,
             "overlap": rescaled(sc, clf, xn[te], xa[te], xb[te]),
             "proj": float((dB @ w).mean()) / proj_A,
             "allpos": rescaled(sc_all, clf_all, mn[te], ma[te], mb[te]),
             "size": float(np.linalg.norm(dB, axis=1).mean()) / size_A,
             "logit": float(((lb - ln) @ u).mean()) / logit_A,
             "null_shuf": rescaled(sc_sh, clf_sh, xn[te], xa[te], xb[te]),
             "null_rand": float((dB @ rnd).mean()) / rand_A}
        out["builds"][b] = r
        print(f"{b:24s} {cls:8s} eqN {logit_eq_none:.2f} trA {trace_eq_A:.2f} ovl {r['overlap']:+.3f} "
              f"proj {r['proj']:+.3f} allpos {r['allpos']:+.3f} size {r['size']:.3f} logit {r['logit']:+.3f}",
              flush=True)
    elig = [b for b, r in out["builds"].items() if r["class"] == "leaky" and b not in (A, "ref@L7.q_norm", "ref@L21.q_norm")
            and not b.startswith(("none", "clone", "score", "ref@L14.q_norm+"))]
    mid = [b for b in elig if 0.2 <= out["builds"][b]["overlap"] <= 0.8]
    out["x_clause"] = {"eligible": len(elig), "in_0.2_0.8": mid, "passes": len(mid) >= 2}
    print("x clause:", out["x_clause"])
    json.dump(out, open(HERE / a.out, "w"), indent=1)
    print("wrote", a.out, "| done", len(out["builds"]), "pending", len(out["pending"]))


if __name__ == "__main__":
    main()
