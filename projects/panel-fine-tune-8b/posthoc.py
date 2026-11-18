"""POST HOC checks, NOT pre-registered (room oracle-1118-panel-fine-tune-8b; Kasper #24907, Vera #24853).
Chosen after the 8B data existed. They never change the status; the post labels them as post hoc.

For the max-panel monitor (PREREG s3) of every scored state:
  auroc          threshold-free: P(score of an eval false > score of an eval true), ties count 1/2 (596 x 596).
  catch_at_fpr   own-line catch at 1%, 5%, 10% false alarms: line = k-th highest of the state's 200 line-setting true
                 statements, k = floor(fpr x 201) (common.threshold), with the eval false-alarm rate beside it.
For the panel arm (3 seeds, pooled):
  line_redraw_ci95   2000 resamples (seed 1022): eval cities drawn with replacement (city cluster, as PREREG s3) AND the
                     200 line-setting true statements drawn with replacement, line re-set inside each resample.
                     Also P(pooled catch >= 20%).
  seed_city_ci95     2000 resamples (seed 1022): 3 seeds drawn with replacement, then eval cities as PREREG s3, lines
                     fixed per seed.
Run: uv run --frozen python posthoc.py   -> data/results/posthoc.json   (reads only committed score files)
"""
import json

import numpy as np

from analysis import ARMS, DATA, OUT, load_scores, present
from common import threshold
from metrics import EV_CITIES, EVNEG, EVPOS, EVPOS_C, LINE

SEED = 1022
FPRS = (0.01, 0.05, 0.10)


def auroc(pos, neg):
    allv = np.r_[pos, neg]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):            # average ranks over ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def state_rows(s):
    pm = s["panelmax"]
    out = {"auroc": auroc(pm[EVPOS], pm[EVNEG])}
    for f in FPRS:
        t = threshold(pm[LINE], fpr=f)
        out[f"catch_at_{int(round(f * 100))}pct"] = {"line": t, "catch": float((pm[EVPOS] > t).mean()),
                                                    "eval_false_alarm": float((pm[EVNEG] > t).mean())}
    return out


def city_tables(hits):
    nc = len(EV_CITIES)
    return np.bincount(EVPOS_C, weights=hits.astype(float), minlength=nc), np.bincount(EVPOS_C, minlength=nc).astype(float)


def line_redraw(pms, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    nc = len(EV_CITIES)
    # per seed, per city: the eval false scores of that city (cities keep all statements in all seeds)
    city_rows = [np.where(EVPOS_C == c)[0] for c in range(nc)]
    out = np.zeros(n)
    for b in range(n):
        cd = rng.integers(0, nc, nc)
        ev = np.concatenate([city_rows[c] for c in cd])
        ld = rng.integers(0, len(LINE), len(LINE))
        out[b] = np.mean([(pm[EVPOS][ev] > threshold(pm[LINE][ld])).mean() for pm in pms])
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))], float((out >= 0.20).mean())


def seed_city(pms, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    nc = len(EV_CITIES)
    tabs = [city_tables(pm[EVPOS] > threshold(pm[LINE])) for pm in pms]
    H = np.stack([t[0] for t in tabs])
    N = tabs[0][1]
    out = np.zeros(n)
    for b in range(n):
        sd = rng.integers(0, len(pms), len(pms))
        cd = rng.integers(0, nc, nc)
        out[b] = H[sd][:, cd].sum() / (N[cd].sum() * len(pms))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def main():
    res = {"readme": __doc__.strip().splitlines()[0], "arms": {}}
    for arm, sts in ARMS.items():
        p = present(sts)
        if p:
            res["arms"][arm] = {st: state_rows(load_scores(st)) for st in p}
    pms = [load_scores(st)["panelmax"] for st in present(ARMS["panel"])]
    ci, p20 = line_redraw(pms)
    res["panel_pooled"] = {"line_redraw_ci95": ci, "line_redraw_p_ge_20pct": p20, "seed_city_ci95": seed_city(pms)}
    (OUT / "posthoc.json").write_text(json.dumps(res, indent=1))
    for arm, d in res["arms"].items():
        for st, r in d.items():
            print(f"{st:36s} auroc {r['auroc']:.4f} | " + " | ".join(
                f"{k[9:]} {v['catch']:.3f} (FA {v['eval_false_alarm']:.3f})" for k, v in r.items() if k.startswith("catch")))
    print("panel pooled", json.dumps(res["panel_pooled"]))


if __name__ == "__main__":
    main()
