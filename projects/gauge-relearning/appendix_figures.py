"""Data for the three appendix figures ('If you're curious'), from the raw attack logs.

    uv run python projects/gauge-relearning/appendix_figures.py [--publish]

gauge-large-c       Q1: steps at c / steps at c = 1 for SGD and AdamW, same attacker, seed and data
                    order, from c = 1/4096 to 4096. Main runs (results/raw) for |log2c| <= 4, appendix
                    runs (results/appendix/raw) beyond. lr = best of the seed-0 grid at that c.
gauge-error-bars    Q2: SGD and AdamW, c = 16 vs c = 1 at the main seed-0 best lr, all (seed, order)
                    pairs: the 9 from the post first, then the new ones in a fixed shuffled order.
                    Running geometric mean with a 95% t-interval, share of pairs that are slower with
                    a Clopper-Pearson 95% interval, and the 10-90% range of single runs.
gauge-gradient-space Q3: results/appendix/gradient_space_seed{0..4}.json (gradient_space.py) plus the
                    held-out accuracy at the stop step of the attacks above.

A censored run (no recovery within the cap) counts as the cap, so its ratio is a lower bound.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from lib import figure

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import make_figure as mf  # noqa: E402  (hitting_time, rank, median, THRESH, CAP)

MAIN_RAW = HERE / "results" / "raw"
APP_RAW = HERE / "results" / "appendix" / "raw"
FIGS = HERE / "figures"
ATT = [("sgd", "SGD"), ("adamw", "AdamW")]

TITLES = {  # set after reading the results; each states what its figure shows
    "gauge-large-c": "Far past c = 16, plain SGD stalls and AdamW slows up to 3×",
    "gauge-error-bars": "More runs narrow the average slowdown, not the spread between runs",
    "gauge-gradient-space": "At large c, an SGD step on the trained facts stops pointing at the held-out ones",
}

# (attacker, log2c, lr) -> [(acc_B, acc_retain, status)] measured exactly at the step a run stopped.
# Only stop_* reruns and runs whose last row carries both off the every-25-step grid, i.e. runs made
# after the stop-step logging change.
AT_STOP: dict[tuple, list] = defaultdict(list)


def collect(root: Path) -> dict[tuple, dict[float, tuple[int, str]]]:
    hits: dict[tuple, dict] = defaultdict(dict)
    for p in sorted(root.glob("seed*/*_log2c*_lr*.jsonl")):
        if p.name.startswith(("devcpu_", "floor_")):
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("model") != "unlearned":
            continue
        r0, last = rows[0], rows[-1]
        key = (r0["attacker"], int(r0["log2c"]))
        t = mf.hitting_time(rows)
        exact = p.name.startswith("stop_") or last["step"] % 25 != 0  # stop-step logging (new runs)
        if exact and last["acc_B"] is not None and last["acc_retain"] is not None and last["step"] > 0:
            AT_STOP[(*key, float(r0["lr"]))].append((last["acc_B"], last["acc_retain"], t[1]))
        if p.name.startswith("stop_"):  # duplicates of a main-order run: only for AT_STOP
            continue
        hits[(*key, (int(r0["seed"]), int(r0.get("order", 0))))][float(r0["lr"])] = t
    return hits


def best(grid):
    return min((mf.rank(t), lr) for lr, t in grid.items())[1]


def large_c(main, app):
    """Rows per (attacker, log2c): ratio to the same attacker's c = 1 run in the same seed + order."""
    rows = []
    for a, label in ATT:
        lr1 = best(main[(a, 0, (0, 0))])
        base = {run: g[lr1] for (aa, k, run), g in main.items() if aa == a and k == 0 and lr1 in g}
        for src in (main, app):
            for k in sorted({key[1] for key in src if key[0] == a and key[2] == (0, 0)}):
                if src is app and abs(k) <= 4:
                    continue
                g0 = src[(a, k, (0, 0))]
                lr = best(g0)
                runs = sorted(r for (aa, kk, r) in src if aa == a and kk == k and lr in src[(aa, kk, r)])
                pairs = [(r, src[(a, k, r)][lr]) for r in runs if r in base and base[r][1] == "recovered"]
                if not pairs:
                    continue
                ratios = [t[0] / base[r][0] for r, t in pairs]
                acc_b = [b for b, _, st in AT_STOP[(a, k, lr)] if st == "recovered"]
                acc_r = [r for _, r, _ in AT_STOP[(a, k, lr)]]
                rows.append({
                    "attacker": a, "label": label, "log2c": k, "lr": lr,
                    "ratio_median": round(float(np.median(ratios)), 3),
                    "ratio_min": round(min(ratios), 3), "ratio_max": round(max(ratios), 3),
                    "n_runs": len(pairs),
                    "n_censored": sum(t[1] == "censored" for _, t in pairs),
                    "n_diverged": sum(t[1] == "diverged" for _, t in pairs),
                    "n_slower": sum(x > 1 for x in ratios),
                    "steps_by_run": {f"seed {s}, order {o}": t[0] for (s, o), t in pairs},
                    "grid": {f"{l:g}": (t[0] if t[1] == "recovered" else t[1]) for l, t in sorted(g0.items())},
                    # measured at the stop step: held-out forgotten facts (recovered runs only) and kept facts
                    "accB_mean": round(float(np.mean(acc_b)), 4) if acc_b else None,
                    "accB_min": min(acc_b) if acc_b else None, "accB_max": max(acc_b) if acc_b else None,
                    "accB_n": len(acc_b),
                    "accR_mean": round(float(np.mean(acc_r)), 4) if acc_r else None,
                    "accR_min": min(acc_r) if acc_r else None, "accR_max": max(acc_r) if acc_r else None,
                    "accR_n": len(acc_r),
                    "source": "main" if src is main else "appendix",
                })
    return sorted(rows, key=lambda r: (r["attacker"] != "sgd", r["log2c"]))


def error_bars(main, app):
    out = {}
    for a, label in ATT:
        lr1, lr16 = best(main[(a, 0, (0, 0))]), best(main[(a, 4, (0, 0))])
        def pairs(src, runs):
            # a run that never recovers counts at the cap: its ratio is a bound, and a pair where only
            # the c = 1 run hit the cap counts as faster at c = 16 (dropping it would favour 'slower')
            ps = []
            for r in runs:
                g1, g16 = src.get((a, 0, r), {}), src.get((a, 4, r), {})
                if lr1 in g1 and lr16 in g16:
                    ps.append({"seed": r[0], "order": r[1], "s1": g1[lr1][0], "s16": g16[lr16][0],
                               "censored1": g1[lr1][1] != "recovered", "censored16": g16[lr16][1] != "recovered",
                               "ratio": g16[lr16][0] / g1[lr1][0], "source": "post" if src is main else "new"})
            return ps
        first = pairs(main, sorted({k[2] for k in main if k[0] == a and k[1] == 4}))
        extra = pairs(app, sorted({k[2] for k in app if k[0] == a and k[1] == 4}))
        random.Random(0).shuffle(extra)
        ps = first + extra
        lr_ = np.log([p["ratio"] for p in ps])
        curve = []
        for n in range(3, len(ps) + 1):
            x = lr_[:n]
            m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            k = int((x > 0).sum())
            ci = stats.binomtest(k, n).proportion_ci()
            curve.append({"n": n, "gmean": math.exp(m), "lo": math.exp(m - t * se), "hi": math.exp(m + t * se),
                          "q10": math.exp(np.quantile(x, 0.1)), "q90": math.exp(np.quantile(x, 0.9)),
                          "frac": k / n, "frac_lo": ci.low, "frac_hi": ci.high})
        # where does the spread come from? one-way random effects on log ratio, seed as group
        by_seed = defaultdict(list)
        for p, v in zip(ps, lr_):
            by_seed[p["seed"]].append(v)
        groups = [np.array(v) for v in by_seed.values() if len(v) > 1]
        n_bar = np.mean([len(g) for g in groups])
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / sum(len(g) - 1 for g in groups)
        grand = np.concatenate(groups).mean()
        msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
        var_seed = max(0.0, (msb - msw) / n_bar)
        out[a] = {
            "label": label, "lr_c1": lr1, "lr_c16": lr16, "pairs": ps, "curve": curve, "n_post": len(first),
            "per_seed": {str(s): {"n": len(v), "gmean": math.exp(np.mean(v)),
                                  "slower": int((np.array(v) > 0).sum())} for s, v in sorted(by_seed.items())},
            "var_log_within_seed": msw, "var_log_between_seed": var_seed,
            "n_censored16": sum(p["censored16"] for p in ps), "n_censored1": sum(p["censored1"] for p in ps),
        }
    return out


def main(publish: bool):
    main_h, app_h = collect(MAIN_RAW), collect(APP_RAW)
    lc = large_c(main_h, app_h)
    data = {
        "large_c": {"title": TITLES["gauge-large-c"], "cap": mf.CAP, "thresh": mf.THRESH, "rows": lc},
        "error_bars": {"title": TITLES["gauge-error-bars"], **error_bars(main_h, app_h)},
    }
    figs = {"gauge-large-c": data["large_c"], "gauge-error-bars": data["error_bars"]}
    gs = {s: HERE / "results" / "appendix" / f"gradient_space_seed{s}.json" for s in (0, 1, 2, 3, 4)}
    if gs[0].exists():
        heldout = [{k: r[k] for k in ("attacker", "label", "log2c", "accB_mean", "accB_min", "accB_max", "accB_n")}
                   for r in lc if r["accB_n"]]
        # seed 0 draws the facts; every seed with a file adds its summary lines
        seed_lines = []
        for s, p in gs.items():
            if p.exists():
                g = json.loads(p.read_text())
                seed_lines += [{"seed": s, "log2c": r["log2c"], "cos": r["cos_meanA_meanB"],
                                "share_K": r["share_meanA_K"], "ng": g["natural_gradient"]["cos_meanA_meanB"]}
                               for r in g["per_c"]]
        figs["gauge-gradient-space"] = {"title": TITLES["gauge-gradient-space"], **json.loads(gs[0].read_text()),
                                        "seed_lines": seed_lines, "heldout": heldout, "thresh": mf.THRESH}
    for fid, d in figs.items():
        js = FIGS / fid / "plot.js"
        fig = figure.save(HERE, fid, d, js.read_text() if js.exists() else None)
        if publish:
            print("published", figure.publish(fig, blog_figures=mf.WORK_BLOG_FIGURES))
    for r in lc:
        print({k: v for k, v in r.items() if k not in ("steps_by_run",)})
    for a in ("sgd", "adamw"):
        e = data["error_bars"][a]
        c = e["curve"][-1]
        print(a, "pairs", len(e["pairs"]), "post", e["n_post"], {k: round(v, 3) for k, v in c.items()},
              "var within/between", round(e["var_log_within_seed"], 3), round(e["var_log_between_seed"], 3),
              "per seed", {s: (v["n"], round(v["gmean"], 2), v["slower"]) for s, v in e["per_seed"].items()},
              "censored16", e["n_censored16"])


if __name__ == "__main__":
    main("--publish" in sys.argv)
