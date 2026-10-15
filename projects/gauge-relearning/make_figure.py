"""Build the main figure (gauge-recovery) from the raw attack logs written by exp.py.

    uv run python projects/gauge-relearning/make_figure.py [--publish]

Input  results/raw/seed{S}/{attacker}_log2c{K}_lr{LR}.jsonl, one row per attack step
       {attacker, log2c, seed, lr, model, step, acc_A, loss, ...}. Only model == "unlearned".
Output figures/gauge-recovery/{data.json, plot.js}; with --publish also the work-dir blog copy.

Rule: a run's hitting time is the first step with acc_A >= THRESH (as in exp.py). A run that
never hits is 'censored' (plotted at the cap); a run with a non-finite loss is 'diverged'.
'tuned' = the attacker retunes per gauge: lr = best of the seed-0 grid for (attacker, c)
          (ties -> lower lr); every run (seed x data order) at that lr counts.
'naive' = the attacker keeps the seed-0 best lr at c = 1.
y = steps / (fastest attack at c = 1 in the same seed and data order). Per (attacker, c): median,
min and max over runs.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from lib import figure

PROJECT_DIR = Path(__file__).parent
RAW = PROJECT_DIR / "results" / "raw"
WORK_BLOG_FIGURES = Path(__file__).resolve().parents[3] / "blog.kran.ai" / "assets" / "figures"
FIG_ID = "gauge-recovery"
EXP_SRC = (PROJECT_DIR / "exp.py").read_text()
THRESH = float(re.search(r"^THRESH = ([\d.]+)", EXP_SRC, re.M).group(1))
CAP = int(re.search(r"^MAX_STEPS = (\d+)", EXP_SRC, re.M).group(1))

ATTACKERS = [
    ("sgd", "SGD"),
    ("adamw", "AdamW"),
    ("kfac", "K-FAC (natural gradient)"),
    ("renorm", "Rebalance Q/K, then AdamW"),
]
# K-FAC on the figure is seed 0 only (plan #525): its full trust-radius grid exists for seed 0;
# seeds 1-2 reuse seed 0's radius and diverged at c = 1, which the post reports in text.
SEED0_ONLY = {"kfac"}


def hitting_time(rows: list[dict]) -> tuple[int, str]:
    for r in sorted(rows, key=lambda r: r["step"]):
        if r["loss"] is None or not math.isfinite(r["loss"]):
            return CAP, "diverged"
        if r["acc_A"] >= THRESH:
            return r["step"], "recovered"
    return CAP, "censored"


def rank(t: tuple[int, str]) -> float:
    steps, status = t
    return {"recovered": steps, "censored": 1e9, "diverged": 2e9}[status]


def median(xs: list[float]) -> float:
    xs, n = sorted(xs), len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def collect(model: str = "unlearned") -> dict[tuple, dict[float, tuple[int, str]]]:
    """(attacker, log2c, (seed, order)) -> {lr: (steps, status)} for one model ("unlearned"/"floor").
    order 0 = the main pre-registered runs; orders 1, 2 = extra data orders at the seed-0 best lr."""
    hits: dict[tuple, dict] = defaultdict(dict)
    for p in sorted(RAW.glob("seed*/*_log2c*_lr*.jsonl")):
        if p.name.startswith("devcpu_"):  # K-FAC cells that ran on CPU after an MPS stall: not comparable
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not rows or rows[0].get("model") != model:
            continue
        r0 = rows[0]
        run = (int(r0["seed"]), int(r0.get("order", 0)))
        hits[(r0["attacker"], int(r0["log2c"]), run)][float(r0["lr"])] = hitting_time(rows)
    return hits


def main(publish: bool) -> None:
    hits = collect()
    best = lambda grid: min((rank(t), lr) for lr, t in grid.items())[1]  # noqa: E731
    main_run = (0, 0)  # lrs are tuned on seed 0, main data order only
    lr_c1 = {a: best(hits[(a, 0, main_run)]) for a, _ in ATTACKERS if (a, 0, main_run) in hits}
    # Shared denominator per run (critic #501): the fastest attack on the unrescaled model (c = 1) in
    # that seed + data order, over the attackers run there. 1x = as fast as the best attack on what shipped.
    base: dict[tuple, int] = {}
    for (a, k, run), grid in hits.items():
        if k == 0 and a in lr_c1 and lr_c1[a] in grid and grid[lr_c1[a]][1] == "recovered":
            base[run] = min(base.get(run, CAP), grid[lr_c1[a]][0])
    out = []
    for attacker, _ in ATTACKERS:
        ks = sorted({key[1] for key in hits if key[0] == attacker and key[2] == main_run})
        for k in ks:
            grid0 = hits[(attacker, k, main_run)]
            runs = sorted(key[2] for key in hits if key[0] == attacker and key[1] == k)
            if attacker in SEED0_ONLY:
                runs = [r for r in runs if r == main_run]
            for kind, lr in (("tuned", best(grid0)), ("naive", lr_c1.get(attacker))):
                pairs = [(r, hits[(attacker, k, r)][lr]) for r in runs if lr in hits[(attacker, k, r)]]
                if not pairs:
                    continue
                ts = [t for _, t in pairs]
                steps = [t[0] for t in ts]
                ratios = [t[0] / base[r] for r, t in pairs if r in base]
                out.append({
                    "attacker": attacker, "kind": kind, "log2c": k, "lr": lr,
                    "median": median(steps), "min": min(steps), "max": max(steps),
                    "steps_by_run": {f"seed {s}, order {o}": t[0] for (s, o), t in pairs},
                    "ratio_median": round(median(ratios), 3) if ratios else None,
                    "ratio_min": round(min(ratios), 3) if ratios else None,
                    "ratio_max": round(max(ratios), 3) if ratios else None,
                    "n_seeds": len({s for (s, _), _t in pairs}),
                    "n_runs": len(ts),
                    "n_censored": sum(st == "censored" for _, st in ts),
                    "n_diverged": sum(st == "diverged" for _, st in ts),
                    # skeptic3's grid-edge rule: a best lr on the grid edge may not be the true best
                    "grid_edge": grid0[lr][1] == "recovered" and lr in (min(grid0), max(grid0)),
                })

    data = {
        "title": "How much longer the attacker needs to relearn the removed facts",
        "y_label": f"Steps to get {THRESH:.0%} of the facts right again, divided by the fastest attack "
                   "on the original model (same seed and data order). Log scale.",
        "base_steps_by_run": {f"seed {s}, order {o}": v for (s, o), v in sorted(base.items())},
        "y_mode": "ratio",
        "cap": CAP,
        "attackers": [{"key": k, "label": l} for k, l in ATTACKERS],
        "show_naive": False,
        "rows": out,
        # never-knew control: same attack on a model that never saw the forget facts (not plotted)
        "floor": [{"attacker": a, "log2c": k, "seed": s, "order": o, "lr": lr, "steps": t[0], "status": t[1]}
                  for (a, k, (s, o)), grid in sorted(collect("floor").items()) for lr, t in sorted(grid.items())],
    }
    fig = figure.save(PROJECT_DIR, FIG_ID, data, (PROJECT_DIR / "figures" / FIG_ID / "plot.js").read_text())
    print("wrote", fig, f"({len(out)} rows)")
    for row in out + data["floor"]:
        print(row)
    if publish:
        print("published", figure.publish(fig, blog_figures=WORK_BLOG_FIGURES))


if __name__ == "__main__":
    main("--publish" in sys.argv)
