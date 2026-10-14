"""Build the main figure from the saved raw eval rows.

Pipeline (each step writes to data/):
    uv run python projects/probe-goodhart/probes.py                     # probes A, B, B2 on the base model
    with-model-keys uv run python projects/probe-goodhart/train.py --lam 0 --seed 0   # one run per setting
    uv run python projects/probe-goodhart/run.py                        # -> figures/probe-goodhart/

Metric per (setting, round): share of held-out answers flagged deceptive by
  probeA (the probe in the reward), probeB (held-out probe), judge (LLM judge, common.judge_one).
lo/hi is a 95% bootstrap over prompts, pooling all seeds (answers to one prompt stay together).
mean is the mean of the per-seed means.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lib import figure

PROJECT_DIR = Path(__file__).parent
RAW = PROJECT_DIR / "data" / "raw"
FIG_ID = "probe-goodhart"
METRICS = {"probeA": "flagA", "probeB": "flagB", "judge": "judge_deceptive"}


def load():
    rows = []
    for f in sorted(RAW.glob("eval_lam*.jsonl")):
        rows += [json.loads(line) for line in f.open()]
    return rows


def boot(by_prompt: dict, key: str, n=2000, seed=0):
    """95% bootstrap CI over prompts (samples of one prompt stay together)."""
    ids = list(by_prompt)
    vals = [np.array([float(r[key]) for r in by_prompt[i]]) for i in ids]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        means.append(np.mean(np.concatenate([vals[i] for i in pick])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    rows = load()
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lam"], r["round"], r["seed"])].append(r)
    lams = sorted({r["lam"] for r in rows})
    out = []
    for lam in lams:
        rounds = sorted({k[1] for k in groups if k[0] == lam})
        for rnd in rounds:
            seeds = sorted({k[2] for k in groups if k[0] == lam and k[1] == rnd})
            for metric, key in METRICS.items():
                per_seed = [np.mean([float(r[key]) for r in groups[(lam, rnd, s)]]) for s in seeds]
                bp = defaultdict(list)  # all seeds' answers to one prompt stay together
                for s in seeds:
                    for r in groups[(lam, rnd, s)]:
                        bp[r["prompt_id"]].append(r)
                lo, hi = boot(bp, key)
                out.append({"setting": f"lam{lam:g}", "round": rnd, "metric": metric,
                            "mean": float(np.mean(per_seed)), "lo": lo, "hi": hi, "n_seeds": len(seeds),
                            "n_answers": sum(len(groups[(lam, rnd, s)]) for s in seeds)})
    label = {0.0: "No probe penalty (λ = 0)"}
    data = {
        "title": "Answers the judge calls deceptive vs. answers the probes flag",
        "y_label": "Share of held-out answers flagged as deceptive",
        "settings": [{"key": f"lam{l:g}", "label": label.get(l, f"Probe A penalty λ = {l:g}")} for l in lams],
        "rows": out,
    }
    plot_js = (PROJECT_DIR / "figures" / FIG_ID / "plot.js").read_text()
    fig = figure.save(PROJECT_DIR, FIG_ID, data, plot_js)
    print("wrote", fig)
    for lam in lams:
        for metric in METRICS:
            seq = [(r["round"], round(r["mean"], 2)) for r in out if r["setting"] == f"lam{lam:g}" and r["metric"] == metric]
            print(f"lam={lam:g} {metric:7s} {seq}")


if __name__ == "__main__":
    main()
