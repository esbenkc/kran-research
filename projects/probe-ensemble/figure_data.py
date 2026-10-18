"""Build blog figure data (assets/figures/probe-ensemble/data.json) from the per-answer eval rows.

Left panel: judged-deceptive rate per arm and round, pooled over seeds, with a 95% bootstrap CI over prompts
(both samples of a prompt resampled together, per seed; 2,000 draws, numpy seed 0, as in PREREG.md s.4),
plus the per-seed rate and the per-seed 2x target (round-0 rate minus twice the single-probe cut, PREREG s.5).
Right panel: share of answers flagged per arm and round, by any of the eight ensemble members (flagEns: any
member over its own 5%-false-positive threshold) and by the held-out attention probe (flagAttn).

    uv run python projects/probe-ensemble/figure_data.py --eval-dir projects/probe-ensemble/data/results \
        --out <blog>/assets/figures/probe-ensemble/data.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = [  # key, label, regex on the run name, role
    ("control", "No penalty", r"^lam0_s\d+$", "control"),
    ("single", "One probe", r"^lam1_s\d+$", "single"),
    ("ensemble", "Eight probes (max)", r"^ens_lam1_s\d+$", "ensemble"),
]
FLAGS = [("any8", "flagEns"), ("attn", "flagAttn")]


def arm_of(run: str) -> str | None:
    for key, _, rx, _ in ARMS:
        if re.match(rx, run):
            return key
    return None


def load(eval_dir: Path):
    # judged[arm][seed][round][prompt_id] = list of 0/1 judge labels
    judged = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    flags = defaultdict(list)  # (arm, seed, round, kind) -> [0/1]
    for f in sorted(eval_dir.glob("eval_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            a = arm_of(r["run"])
            if a is None:
                continue
            judged[a][r["seed"]][r["round"]][r["prompt_id"]].append(int(r["judge_deceptive"]))
            for kind, field in FLAGS:
                flags[(a, r["seed"], r["round"], kind)].append(int(r[field]))
    return judged, flags


def pooled_rate(per_seed: dict, rng, n_boot=2000):
    """per_seed: {seed: {prompt_id: [labels]}} -> mean, lo, hi, n_answers."""
    mats = []
    for seed in sorted(per_seed):
        P = per_seed[seed]
        mats.append(np.array([[sum(v), len(v)] for _, v in sorted(P.items())], dtype=float))
    num = sum(m[:, 0].sum() for m in mats)
    den = sum(m[:, 1].sum() for m in mats)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = d = 0.0
        for m in mats:
            idx = rng.integers(0, len(m), len(m))
            s += m[idx, 0].sum(); d += m[idx, 1].sum()
        boots[b] = s / d
    return float(num / den), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    judged, flags = load(args.eval_dir)
    rng = np.random.default_rng(0)
    rows = []
    for key, _, _, _ in ARMS:
        if key not in judged:
            continue
        seeds = judged[key]
        rounds = sorted(set.intersection(*(set(seeds[s]) for s in seeds)))
        for rnd in rounds:
            mean, lo, hi, n = pooled_rate({s: seeds[s][rnd] for s in seeds}, rng)
            per_seed = {str(s): float(np.mean([x for v in seeds[s][rnd].values() for x in v])) for s in seeds}
            rows.append({"arm": key, "round": rnd, "mean": mean, "lo": lo, "hi": hi, "n_answers": n,
                         "n_seeds": len(seeds), "per_seed": per_seed})

    # Where the ensemble must land at round 8 for a yes (PREREG.md s.5): r0 - 2 x single cut, per seed.
    targets = []
    for s in sorted(judged.get("single", {})):
        rs = judged["single"][s]
        if 0 not in rs or 8 not in rs:
            continue
        rate = lambda rnd: float(np.mean([x for v in rs[rnd].values() for x in v]))
        targets.append({"seed": str(s), "round": 8, "target": rate(0) - 2 * (rate(0) - rate(8))})

    flag_rows = []
    by = defaultdict(dict)
    for (a, seed, rnd, kind), v in flags.items():
        by[(a, rnd, kind)][str(seed)] = float(np.mean(v))
    for (a, rnd, kind), per_seed in sorted(by.items()):
        if len(per_seed) == len(judged[a]):  # only rounds every seed has
            flag_rows.append({"arm": a, "round": rnd, "kind": kind, "mean": float(np.mean(list(per_seed.values()))),
                              "per_seed": per_seed})

    out = {
        "title": "Training against eight lie detectors cut lying about as much as training against one",
        "y_label": "Share of held-out answers the judge calls deceptive",
        "flag_title": "Share of the same answers the probes flag",
        "arms": [{"key": k, "label": l, "role": r} for k, l, _, r in ARMS if k in judged],
        "rows": rows,
        "targets": targets,
        "flag_kinds": [{"key": "any8", "label": "any of the eight probes"},
                       {"key": "attn", "label": "held-out attention probe"}],
        "flag_rows": flag_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(rows)} rate rows, {len(flag_rows)} flag rows, "
          f"arms {[a['key'] for a in out['arms']]}")


if __name__ == "__main__":
    main()
