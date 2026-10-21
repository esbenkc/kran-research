"""Vera: rebuild the refusal-threshold fit along r_15 (post foldout + 'about twice as far' main-text claim). Own code.
x = projection of the DEV prompt's last-token activation at the input of block 15 onto r_15, in units of |r_15|
    (h.r/|r|^2), plus the gain k (adding k*r shifts it by exactly k); y = 1 if judge says refusal.
Unregularised 2-parameter logistic regression (Newton). 50% point = -b0/b1. Bootstrap over prompts (1000)."""
import json, numpy as np
from pathlib import Path
np.seterr(all="ignore")
R = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/runs"
base = np.load(R / "base/acts.npz"); L = 15
r = base["fit_harmful"][:, L].astype(float).mean(0) - base["fit_harmless"][:, L].astype(float).mean(0)


def fit(x, y):
    X = np.c_[np.ones_like(x), x]; w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (y - p); H = (X * (p * (1 - p))[:, None]).T @ X
        w += np.linalg.solve(H + 1e-9 * np.eye(2), g)
    return -w[0] / w[1], w[1]


def data(model, f, incoh_as_refusal=False):
    A = np.load(R / model / "acts.npz")
    P = {k: A[k][:, L].astype(float) @ r / (r @ r) for k in ("dev_harmful", "dev_harmless")}
    xs, ys, ids = [], [], []
    for l in open(R / model / f):
        q = json.loads(l)
        if q["layer"] not in (None, L): continue
        xs.append(P[f"dev_{q['kind']}"][q["i"]] + q["gain"])
        ys.append(q["judge"] == "refusal" or (incoh_as_refusal and q["judge"] == "incoherent"))
        ids.append((q["kind"], q["i"]))
    return np.array(xs), np.array(ys, float), ids


rng = np.random.default_rng(0)
for inc in (False, True):
    out = {}
    for model in ("base", "strip_s0"):
        x, y, ids = data(model, "dev.judged.jsonl", inc)
        m, s = fit(x, y)
        uniq = sorted(set(ids)); idx = {u: [j for j, v in enumerate(ids) if v == u] for u in uniq}
        bs = []
        for _ in range(1000):
            pick = [j for u in rng.choice(len(uniq), len(uniq)) for j in idx[uniq[u]]]
            bs.append(fit(x[pick], y[pick])[0])
        out[model] = (m, s, len(x), np.percentile(bs, [2.5, 97.5]))
        print(f"incoherent_as_refusal={inc} {model}: n={len(x)} 50% point {m:.2f} [{out[model][3][0]:.2f},{out[model][3][1]:.2f}] slope {s:.2f}")
    print(f"  ratio stripped/base 50% point: {out['strip_s0'][0]/out['base'][0]:.2f}")

# design check: the base fit has steered HARMLESS prompts only; refit the stripped model on the same design
x, y, ids = data("strip_s0", "dev.judged.jsonl")
keep = np.array([(k == "harmless") for k, _ in ids])
xs_all = []
for l in open(R / "strip_s0" / "dev.judged.jsonl"):
    q = json.loads(l)
    if q["layer"] in (None, L): xs_all.append(q["layer"] is None)
unst = np.array(xs_all)
m = keep | unst
print(f"same-design stripped fit (steered harmless + unsteered all): n={m.sum()} 50% point {fit(x[m], y[m])[0]:.2f} slope {fit(x[m], y[m])[1]:.2f}")
