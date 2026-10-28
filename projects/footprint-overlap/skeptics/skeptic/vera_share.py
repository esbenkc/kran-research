"""Vera's own behaviour + law checker (skeptic). numpy only.

Inputs: per-prompt first-token X mass `px` (200 test prompts, with `ids`) in <dir>/<tag>_plain.npz for the
policy and for the base model, and an overlap table {build: x}.
  GAP(pol, B)  = mean px(none) - mean px(B)
  dGAP_B       = GAP(pol, B) - GAP(base, B)
  share_B      = dGAP_B / dGAP_A                (paired prompt bootstrap, ratio of means per resample)
  R2_fixed     = 1 - sum (share - x)^2 / sum (share - mean share)^2   (line share = x, float64)
  perm p       = monitor-label permutation of x over the status set, Phipson-Smyth (b+1)/(m+1)
Usage: vera_share.py --pol DIR --base DIR --overlap overlap.json[:key] --a TAG --none TAG B1 B2 ...
"""
import argparse, json, sys
import numpy as np


def px(d, tag, ids_ref=None):
    z = np.load(f"{d}/{tag}_plain.npz")
    p = z["px"].astype(np.float64)
    if "ids" in z.files:
        ids = [str(i) for i in z["ids"]]
        if ids_ref is not None:
            assert ids == ids_ref, f"prompt ids differ for {d}/{tag}"
        return p, ids
    return p, ids_ref


def shares(pol, base, none, a, builds, nboot=2000, seed=0):
    p0, ids = px(pol, none)
    b0, _ = px(base, none, ids)
    pA, _ = px(pol, a, ids)
    bA, _ = px(base, a, ids)
    rng = np.random.default_rng(seed)
    n = len(p0)
    idx = rng.integers(0, n, size=(nboot, n))
    dA_pp = (p0 - pA) - (b0 - bA)  # per-prompt dGAP_A
    out = {"n": n, "X_none": p0.mean(), "X_A": pA.mean(), "dGAP_A": dA_pp.mean(),
           "dGAP_A_ci": np.percentile(dA_pp[idx].mean(1), [2.5, 97.5]).tolist(), "builds": {}}
    for b in builds:
        pB, _ = px(pol, b, ids)
        bB, _ = px(base, b, ids)
        dB_pp = (p0 - pB) - (b0 - bB)
        boot = dB_pp[idx].mean(1) / dA_pp[idx].mean(1)
        out["builds"][b] = {"X_B": pB.mean(), "dGAP": dB_pp.mean(), "share": dB_pp.mean() / dA_pp.mean(),
                            "share_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
                            "base_gap": (b0 - bB).mean()}
    return out


def r2_fixed(y, x):
    y, x = np.asarray(y, np.float64), np.asarray(x, np.float64)
    return 1.0 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def perm_p(y, x, m=999, seed=1):
    rng = np.random.default_rng(seed)
    obs = r2_fixed(y, x)
    b = sum(r2_fixed(y, rng.permutation(x)) >= obs for _ in range(m))
    return (b + 1) / (m + 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--none", default="none")
    ap.add_argument("--a", required=True)
    ap.add_argument("builds", nargs="+")
    a = ap.parse_args()
    r = shares(a.pol, a.base, a.none, a.a, a.builds)
    print(json.dumps(r, indent=1, default=float))
