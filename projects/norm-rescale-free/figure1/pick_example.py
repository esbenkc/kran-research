"""Figure 1 example picker (Owain). Reads only saved states; no model load, no torch.

For every eval FALSE statement and every arm-A seed (r28__lam10000__seed{s}), at readout 28 (pre-norm), frozen probe tok28:
  s_clean    = probe score of the clean (step-0) state
  s_attacked = probe score of the step-300 state
  s_restored = probe score of the step-300 state rescaled to its clean length (h'' = h' |h| / |h'|)
  lines      = 2nd-highest of the 200 line-setting true statements, re-set on each state (the post's headline rule)
  c_free     = the shrink factor that puts the CLEAN state exactly on the clean line: s(c h) = c a + b0 = line
  angle      = angle between clean and attacked state, norm_ratio = |h'| / |h|
Selection rule (fixed before looking at single statements): caught clean in every seed, hidden after the attack in
every seed, still hidden after the length is put back in every seed; then the famous city by hand.
Output: figure1/example_candidates.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
np.seterr(all="ignore")  # macOS Accelerate matmul emits spurious overflow warnings on finite fp64 input
F = np.load(DATA / "fits" / "panel.npz")
w, b, mu, sd = (F[f"tok28__{k}"].astype(np.float64) for k in ("w", "b", "mu", "sd"))
wsd = w / sd
b0 = float(b - mu @ wsd)
R = json.loads((DATA / "ref" / "split.json").read_text())
y = np.array([1 - r["true"] for r in R])
split = np.array([r["split"] for r in R])
line_i = np.where((split == "refit") & (y == 0))[0]
ev_f = np.where((split == "eval") & (y == 1))[0]


def thr(s):
    return float(np.sort(s)[::-1][1])  # n = 200 -> 2nd highest (common.threshold)


def score(X):
    return X @ wsd + b0


def state(d, step):
    j = list(np.load(d / f"ckpt_{step}.npz")["acts_readouts"]).index(28)
    return np.asarray(np.load(d / f"acts_{step}.npy", mmap_mode="r")[:, j], np.float64)


def run(seed, step=300):
    d = DATA / "runs" / f"r28__lam10000__seed{seed}"
    X0, X1 = state(d, 0), state(d, step)
    n0, n1 = np.linalg.norm(X0, axis=1), np.linalg.norm(X1, axis=1)
    s0, s1 = score(X0), score(X1)
    sr = score(X1 * (n0 / n1)[:, None])
    L0, L1, Lr = thr(s0[line_i]), thr(s1[line_i]), thr(sr[line_i])
    a = X0 @ wsd
    cos = (X0 * X1).sum(1) / (n0 * n1)
    return dict(s0=s0, s1=s1, sr=sr, L0=L0, L1=L1, Lr=Lr, c_free=(L0 - b0) / a, norm_ratio=n1 / n0,
                angle=np.degrees(np.arccos(np.clip(cos, -1, 1))))


seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
res = {s: run(s) for s in seeds}
ok = np.ones(len(ev_f), bool)
for s, r in res.items():
    ok &= (r["s0"][ev_f] > r["L0"]) & (r["s1"][ev_f] <= r["L1"]) & (r["sr"][ev_f] <= r["Lr"])
    print(f"seed {s}: lines clean {r['L0']:.3f} attacked {r['L1']:.3f} restored {r['Lr']:.3f}; "
          f"caught clean {(r['s0'][ev_f] > r['L0']).mean():.4f} attacked {(r['s1'][ev_f] > r['L1']).mean():.4f} "
          f"restored {(r['sr'][ev_f] > r['Lr']).mean():.4f}; median angle {np.median(r['angle'][ev_f]):.2f} deg, "
          f"norm ratio {np.median(r['norm_ratio'][ev_f]):.3f}, c_free median {np.median(r['c_free'][ev_f]):.3f}")
print(f"b0 = {b0:.3f}; candidates hidden in all {len(seeds)} seeds with length restored: {ok.sum()} of {len(ev_f)}")
out = []
for k in np.where(ok)[0]:
    i = int(ev_f[k])
    out.append({"i": i, "text": R[i]["text"], "city": R[i]["city"], "src": R[i]["src"],
                **{f"seed{s}": {key: float(r[key][i]) for key in ("s0", "s1", "sr", "c_free", "norm_ratio", "angle")}
                   | {"L0": r["L0"], "L1": r["L1"], "Lr": r["Lr"]} for s, r in res.items()}})
(HERE / "example_candidates.json").write_text(json.dumps({"seeds": seeds, "b0": b0, "n_eval_false": len(ev_f),
                                                         "n_candidates": len(out), "candidates": out}, indent=1))
