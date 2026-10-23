"""Vera (1023): own helpers. Twin pairs, lines, catch, R, energy share. Written from PREREG.md + DEVIATIONS N1 only.
Score: s(x) = ((x - mu)/sd).w + b from the frozen 1021 panel.npz (positive = FALSE).
Pair = same city AND same src template, false vs true (596 eval pairs).
Line = 2nd-highest score of the 200 'refit' true statements of the SCORED state (headline) or of the CLEAN state.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path.home() / "work/theory-1023-norm-rescale-free"
PROJ = Path(os.environ.get("VERA_PROJ", WORK / "kran-research/projects/norm-rescale-free"))
OLD = Path.home() / "work/theory-1021-full-stack-evader/kran-research/projects/full-stack-evader"
ROWS = json.load(open(OLD / "data/ref/split.json"))
N = len(ROWS)
YF = np.array([1 - r["true"] for r in ROWS])
SPLIT = np.array([r["split"] for r in ROWS])
CITY = np.array([r["city"] for r in ROWS])
SRC = np.array([r["src"] for r in ROWS])
LINE_T = (SPLIT == "refit") & (YF == 0)
EVAL = SPLIT == "eval"
EV_IDX = np.where(EVAL)[0]
EVF = EV_IDX[YF[EV_IDX] == 1]
EVT = EV_IDX[YF[EV_IDX] == 0]
assert LINE_T.sum() == 200 and len(EVF) == 596 and len(EVT) == 596

# twin pairs (false, true) with same city and src
_tk = {(CITY[i], SRC[i]): i for i in EVT}
PAIRS = np.array([(i, _tk[(CITY[i], SRC[i])]) for i in EVF])
assert len(PAIRS) == 596 and len(set(PAIRS[:, 1])) == 596

BP = np.load(OLD / "data/fits/panel.npz")


def probe(r):
    mu, sd, w, b = (BP[f"tok{r}__{k}"].astype(np.float64) for k in ["mu", "sd", "w", "b"])
    return mu, sd, w, b


def score(X, r):
    mu, sd, w, b = probe(r)
    return ((np.asarray(X, np.float64) - mu) / sd) @ w + b


def line(s, idx=LINE_T, k=2):
    return np.sort(s[idx])[::-1][k - 1]


def catch(s, t):
    return float((s[EVF] > t).mean()), float((s[EVT] > t).mean())


def gap(s):
    return float((s[PAIRS[:, 0]] - s[PAIRS[:, 1]]).mean())


def restore(Ha, Hc):
    return Ha * (np.linalg.norm(Hc, axis=1) / np.linalg.norm(Ha, axis=1))[:, None]


def R_metric(Hc, Ha, r=28):
    """Hc, Ha: (N, d) clean and attacked readout-r states for ALL statements (only eval used)."""
    gc, ga = gap(score(Hc, r)), gap(score(Ha, r))
    gr = gap(score(restore(Ha, Hc), r))
    return {"g_clean": gc, "g_att": ga, "g_rest": gr, "R": (gr - ga) / (gc - ga) if gc != ga else float("nan"), "drop_frac": (gc - ga) / gc,
            "precond": bool(gc - ga >= 0.5 * gc)}


def energy(Hc, Ha, idx=EVF):
    hc = np.asarray(Hc[idx], np.float64); d = np.asarray(Ha[idx], np.float64) - hc
    hh = hc / np.linalg.norm(hc, axis=1, keepdims=True)
    rad = (hh * d).sum(1)
    return {"E": float((rad ** 2).sum() / (d ** 2).sum()),
            "cos_med": float(np.median(rad / np.linalg.norm(d, axis=1))),
            "norm_ratio_false": float(np.mean(np.linalg.norm(Ha[EVF], axis=1) / np.linalg.norm(Hc[EVF], axis=1))),
            "norm_ratio_true": float(np.mean(np.linalg.norm(Ha[EVT], axis=1) / np.linalg.norm(Hc[EVT], axis=1)))}
