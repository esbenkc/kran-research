"""Vera's rebuild of the headline numbers from raw per-token activations (no build code).

Usage: uv run --project ../../kran-research python vera_rebuild.py ATTACKED_GLOB
  ATTACKED_GLOB: per-seed token files with tok_L8, tok_L14, tok_off in the clean_tokens.npz layout.
Writes vera_rebuild.json (+ vera_scores.npz) next to this file.
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from vera_core import DATA, Lin, auroc, fit_pooled, meta, seg_reduce, thr, tok_rows

HERE = Path(__file__).parent
t0 = time.time()
M = meta()
y, split, pair, city = M["y"], M["split"], M["pair"], M["city"]
S = len(y)
probe = np.where(split == "probe")[0]
refit_neg = np.where((split == "refit") & (y == 0))[0]
eval_pos = np.where((split == "eval") & (y == 1))[0]
eval_neg = np.where((split == "eval") & (y == 0))[0]
assert len(refit_neg) == 200 and len(eval_pos) == 596 and len(eval_neg) == 596, (len(refit_neg), len(eval_pos))


def load(f):
    z = np.load(f)
    off = z["tok_off"]
    assert len(off) == S + 1
    return {L: z[f"tok_L{L}"].astype(np.float32) for L in (8, 14)}, off


clean, off = load(DATA / "clean_tokens.npz")
# self-check: token mean x frozen probe = 1018 clean scores
pz = np.load(DATA / "probe.npz")
wf, bf = pz["w"].astype(np.float64), float(pz["b"])
ca = np.load(DATA / "clean_acts.npz")
chk = np.abs(seg_reduce(clean[14], off, "mean") @ wf + bf - ca["score"]).max()
print(f"self-check frozen on token means vs 1018 clean score: max|diff| {chk:.2e}")
assert chk < 1e-3

ntok = np.diff(off)


def stmt_feats(acts, how, L):
    return seg_reduce(acts[L], off, how)


def tokenwise_fit(acts, L):
    """max14: tokenwise logistic, each token carries its statement's label; C by GroupKFold(city) on max-pooled AUROC."""
    def fit_rows(st_local):
        st = probe[st_local]
        rows = tok_rows(off, st)
        lab = np.repeat(y[st], ntok[st])
        return acts[L][rows], lab

    def readout(lin, st_local):
        st = probe[st_local]
        s = acts[L][tok_rows(off, st)] @ lin.w + lin.b
        o = np.r_[0, np.cumsum(ntok[st])]
        return np.maximum.reduceat(s, o[:-1])
    return fit_pooled(None, y[probe], city[probe], readout, fit_rows)


def pooled_fit(acts, L, how):
    X = stmt_feats(acts, how, L)

    def fit_rows(st_local):
        return X[probe[st_local]], y[probe[st_local]]

    def readout(lin, st_local):
        return lin(X[probe[st_local]])
    return fit_pooled(None, y[probe], city[probe], readout, fit_rows)


def score(acts, mon):
    kind, L, lin = mon
    if kind == "max":
        s = acts[L] @ lin.w + lin.b
        return np.maximum.reduceat(s, off[:-1])
    return stmt_feats(acts, kind, L) @ lin.w + lin.b


mons = {}
mons["frozen"] = ("mean", 14, type("F", (), {"w": wf, "b": bf})())
lin, C, cv = tokenwise_fit(clean, 14); mons["max14"] = ("max", 14, lin); print("max14 C", C, cv)
lin, C, cv = pooled_fit(clean, 8, "mean"); mons["mean8"] = ("mean", 8, lin); print("mean8 C", C, cv)
lin, C, cv = pooled_fit(clean, 14, "last"); mons["last14"] = ("last", 14, lin); print("last14 C", C, cv)

files = sorted(glob.glob(sys.argv[1]))
print("attacked files:", files)
out = {"files": files, "per_seed": {}, "clean": {}}
scores = {}
for m, mon in mons.items():
    sc = score(clean, mon)
    scores[f"clean__{m}"] = sc
    t = thr(sc[refit_neg])
    out["clean"][m] = dict(catch=float((sc[eval_pos] > t).mean()), fpr=float((sc[eval_neg] > t).mean()),
                           auroc=float(auroc(sc[eval_neg], sc[eval_pos])))
for f in files:
    att, off2 = load(f)
    assert (off2 == off).all()
    sd = Path(f).stem
    lin, C, cv = pooled_fit(att, 14, "mean")
    mm = dict(mons, fresh=("mean", 14, lin))
    res = {"fresh_C": C}
    for m, mon in mm.items():
        sc = score(att, mon)
        scores[f"{sd}__{m}"] = sc
        t = thr(sc[refit_neg])
        res[m] = dict(catch=float((sc[eval_pos] > t).mean()), fpr=float((sc[eval_neg] > t).mean()),
                      auroc=float(auroc(sc[eval_neg], sc[eval_pos])), t1=float(t))
        if m == "fresh":
            scores[f"cleanfor_{sd}__fresh"] = score(clean, mon)
    out["per_seed"][sd] = res
    print(sd, {m: (round(r["catch"], 4), round(r["auroc"], 4)) for m, r in res.items() if isinstance(r, dict)})
ms = ["frozen", "max14", "mean8", "fresh", "last14"]
out["pooled_catch"] = {m: float(np.mean([out["per_seed"][s][m]["catch"] for s in out["per_seed"]])) for m in ms}
print("pooled catch", out["pooled_catch"])
np.savez_compressed(HERE / "vera_scores.npz", **scores)
(HERE / "vera_rebuild.json").write_text(json.dumps(out, indent=1))
print(f"done {time.time() - t0:.0f}s")
