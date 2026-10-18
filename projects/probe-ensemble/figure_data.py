"""Build blog figure data (assets/figures/probe-ensemble/data.json) from the per-answer eval rows.

Judged-deceptive rate per arm and round, pooled over seeds, with a 95% bootstrap CI over prompts
(both samples of a prompt resampled together, per seed; 2,000 draws, numpy seed 0, as in PREREG.md s.4).
Held-out attention-probe AUROC (frozen probe) is computed here from pAttn in the eval rows against the judge
(DECEPTIVE = 1, HONEST = 0, OTHER dropped; PREREG.md s.6). The optional --auroc file (jsonl rows with arm, seed,
round, kind = frozen | refit, auroc) adds the round-8 refit and overrides matching frozen rows.
The figure's second panel shows the mean over seeds.

    uv run python projects/probe-ensemble/figure_data.py --eval-dir projects/probe-ensemble/data/results \
        --auroc projects/probe-ensemble/data/results/attention_auroc.jsonl --out <blog>/assets/figures/probe-ensemble/data.json
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
    ("single3", "One probe, 3x penalty", r"^lam3_s\d+$", "reference"),
]


def arm_of(run: str) -> str | None:
    for key, _, rx, _ in ARMS:
        if re.match(rx, run):
            return key
    return None


attn = defaultdict(list)  # (arm, seed, round) -> [(judge_deceptive, pAttn)], judge OTHER dropped


def load(eval_dir: Path):
    # data[arm][seed][round][prompt_id] = list of 0/1 judge labels
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for f in sorted(eval_dir.glob("eval_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            a = arm_of(r["run"])
            if a is None:
                continue
            data[a][r["seed"]][r["round"]][r["prompt_id"]].append(int(r["judge_deceptive"]))
            if "pAttn" in r and r.get("judge") in ("DECEPTIVE", "HONEST"):
                attn[(a, r["seed"], r["round"])].append((int(r["judge"] == "DECEPTIVE"), r["pAttn"]))
    return data


def auc(pairs):
    """Mann-Whitney AUROC of score vs 0/1 label, ties counted half."""
    y = np.array([p[0] for p in pairs]); x = np.array([p[1] for p in pairs])
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


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
    ap.add_argument("--auroc", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = load(args.eval_dir)
    auroc = defaultdict(dict)  # (arm, round, kind) -> {seed: auroc}
    for (a, seed, rnd), pairs in attn.items():  # frozen probe, from pAttn in the eval rows
        v = auc(pairs)
        if v is not None:
            auroc[(a, rnd, "frozen")][str(seed)] = v
    if args.auroc and args.auroc.exists():
        for line in args.auroc.open():
            r = json.loads(line)
            auroc[(r["arm"], r["round"], r.get("kind", "frozen"))][str(r["seed"])] = r["auroc"]

    rng = np.random.default_rng(0)
    rows = []
    for key, label, _, role in ARMS:
        if key not in data:
            continue
        seeds = data[key]
        rounds = sorted(set.intersection(*(set(seeds[s]) for s in seeds)))
        for rnd in rounds:
            mean, lo, hi, n = pooled_rate({s: seeds[s][rnd] for s in seeds}, rng)
            per_seed = {str(s): float(np.mean([x for v in seeds[s][rnd].values() for x in v])) for s in seeds}
            rows.append({"arm": key, "round": rnd, "mean": mean, "lo": lo, "hi": hi, "n_answers": n,
                         "n_seeds": len(seeds), "per_seed": per_seed})

    # Where the ensemble must land at the last round for a yes (PREREG.md s.5): r0 - 2 x single cut, per seed.
    targets = []
    if "single" in data:
        for s in sorted(data["single"]):
            rs = data["single"][s]
            last = 8  # PREREG.md s.4: the cut is round 0 -> round 8
            if 0 not in rs or last not in rs:
                continue
            rate = lambda rnd: float(np.mean([x for v in rs[rnd].values() for x in v]))
            targets.append({"seed": str(s), "round": last, "target": rate(0) - 2 * (rate(0) - rate(last))})

    auroc_rows = [{"arm": a, "round": rnd, "kind": kind, "mean": float(np.mean(list(v.values()))),
                   "per_seed": v} for (a, rnd, kind), v in sorted(auroc.items()) if a in data]

    out = {
        "title": "Training against eight lie detectors cut lying no more than training against one",
        "y_label": "Share of held-out answers the judge calls deceptive",
        "arms": [{"key": k, "label": l, "role": r} for k, l, _, r in ARMS if k in data],
        "rows": rows,
        "targets": targets,
        "auroc_title": "Held-out attention probe AUROC (never in the reward)",
        "auroc_rows": auroc_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(rows)} rows, arms {[a['key'] for a in out['arms']]}")


if __name__ == "__main__":
    main()
