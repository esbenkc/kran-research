"""Shared metrics: alarm line, catch rate, city-cluster bootstrap, the 298-pair country ranking, output KL summaries.

Catch rate of a monitor in one model state: line = 2nd-highest score of the 200 refit TRUE statements of THAT state
(1% false alarms, 2/201); catch = share of the 596 eval FALSE statements with score > line.
Pooled over seeds: the mean of the per-seed catch rates (equal n per seed = the pooled share).
CI: city-cluster percentile bootstrap over the 298 eval cities (2000 draws, rng 1021), lines held fixed.
Ranking: for each of the 298 eval cities, the affirmative pair "The city of X is in <right>." vs "... <wrong>.";
sum of log P over the tokens from the first token where the pair differs through the last country-name token;
right first iff its sum is larger.
"""
import numpy as np
from transformers import AutoTokenizer

from common import MODEL, country_pos, load_split, rows, threshold, tokenize

_R, Y, CITY, SPLIT = rows()
SRC = np.array([r["src"] for r in _R])
EV_CITIES = sorted(set(CITY[SPLIT == "eval"]))
CIDX = {c: i for i, c in enumerate(EV_CITIES)}
EVPOS = np.where((SPLIT == "eval") & (Y == 1))[0]
EVNEG = np.where((SPLIT == "eval") & (Y == 0))[0]
LINE = np.where((SPLIT == "refit") & (Y == 0))[0]
EVPOS_C = np.array([CIDX[c] for c in CITY[EVPOS]])


def line_of(score):
    return threshold(score[LINE])


def catch(score):
    t = line_of(score)
    return {"line": t, "catch": float((score[EVPOS] > t).mean()), "eval_false_alarm": float((score[EVNEG] > t).mean())}


def boot_catch(score_list, n=2000, seed=1021):
    """score_list: one score vector per seed (same monitor). Returns pooled catch + 95% CI (city cluster bootstrap)."""
    hits = np.stack([(s[EVPOS] > line_of(s)).astype(float) for s in score_list])  # (n_seeds, 596)
    nc = len(EV_CITIES)
    per_city_hits = np.zeros((len(score_list), nc))
    per_city_n = np.bincount(EVPOS_C, minlength=nc).astype(float)
    for k in range(len(score_list)):
        per_city_hits[k] = np.bincount(EVPOS_C, weights=hits[k], minlength=nc)
    tot_h = per_city_hits.sum(0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nc, size=(n, nc))
    bs = tot_h[draws].sum(1) / (per_city_n[draws].sum(1) * len(score_list))
    return {"catch": float(hits.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "per_seed": [float(h.mean()) for h in hits]}


_PAIRS = None


def pairs():
    """(298, 2) statement indices [right, wrong] of the affirmative eval pairs + per pair the token span."""
    global _PAIRS
    if _PAIRS is None:
        tok = AutoTokenizer.from_pretrained(MODEL)
        ids = tokenize(tok, [r["text"] for r in _R])
        cpos, _ = country_pos(tok, ids)
        P = []
        for c in EV_CITIES:
            ii = np.where((CITY == c) & (SRC == "cities"))[0]
            right = [i for i in ii if Y[i] == 0][0]
            wrong = [i for i in ii if Y[i] == 1][0]
            a, b = ids[right], ids[wrong]
            p = next(k for k in range(min(len(a), len(b))) if a[k] != b[k])
            P.append((right, wrong, p, cpos[right], cpos[wrong]))
        _PAIRS = P
    return _PAIRS


def ranking(lp, off):
    """lp: per-token log P (row off[i] + t - 1 = token t). -> n right-first of 298 + per-pair margins."""
    m = []
    for right, wrong, p, cr, cw in pairs():
        sr = lp[off[right] + p - 1: off[right] + cr].sum()
        sw = lp[off[wrong] + p - 1: off[wrong] + cw].sum()
        m.append(float(sr - sw))
    m = np.array(m)
    return {"right_first": int((m > 0).sum()), "n_pairs": len(m), "margin_median": float(np.median(m)),
            "margins": m}
