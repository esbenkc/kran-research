#!/usr/bin/env python3
"""critic2 independent raw-array rebuild for theory-1101-redirect-budget.

Inputs are only public saved arrays under projects/redirect-budget/data:
- bounds/rank_*.npz: RANK_logit[prompt, head] and heads
- measured/*.npz: logit[prompt, head] and heads
No summary.json or carriers.json numbers are read.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "raw_rebuild.json"

RUN_TO_RANK = {
    "mlp8_none_s0": "rank_17.npz",
    "mlp8_none_s1": "rank_17.npz",
    "mlp8_none_s2": "rank_17.npz",
    "monitor_s0": "rank_06_monitor.npz",
    "monitor_s1": "rank_06_monitor.npz",
    "grader_s0": "rank_06_grader.npz",
    "grader_s1": "rank_06_grader.npz",
}
CHECK_HEADS = ["26.2", "27.15", "27.1", "27.9", "27.8"]


def average_ranks(filename: str) -> dict:
    z = np.load(DATA / "bounds" / filename)
    heads = [str(x) for x in z["heads"]]
    score = z["RANK_logit"].astype(np.float64).mean(axis=0)
    # Descending score, then ascending head-array index for exact ties.
    order = sorted(range(len(heads)), key=lambda i: (-score[i], i))
    place = {heads[i]: order.index(i) + 1 for i in range(len(heads))}
    return {
        "heads": heads,
        "score": score,
        "top3": [heads[i] for i in order[:3]],
        "top3_scores": [float(score[i]) for i in order[:3]],
        "places": {h: place.get(h) for h in CHECK_HEADS},
    }


def average_tie_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = average_tie_ranks(x)
    ry = average_tie_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def derivation_check() -> dict:
    """Numerically check Step 2's Cauchy-Schwarz bound.

    For score z = sqrt(d) <v, b>, where v = g_k * R(g_q * q_hat)
    is fixed and QK-norm keeps any key direction b on the unit sphere,
    any key change b1 -> b2 moves the score by
      sqrt(d) |<v, b2-b1>| <= sqrt(d) ||v|| ||b2-b1|| <= 2 sqrt(d) ||v||.
    The constructed antipodal case reaches equality.
    """
    rng = np.random.default_rng(20261101)
    d = 128
    gk = np.exp(rng.normal(0.0, 0.2, d))
    ratios = []
    for _ in range(20_000):
        a = rng.normal(size=d)  # stands in for R(g_q * q_hat); RoPE is norm-preserving.
        b1 = rng.normal(size=d); b1 /= np.linalg.norm(b1)
        b2 = rng.normal(size=d); b2 /= np.linalg.norm(b2)
        v = gk * a
        lhs = math.sqrt(d) * abs(float(np.dot(v, b2 - b1)))
        rhs = math.sqrt(d) * float(np.linalg.norm(v)) * 2.0
        ratios.append(lhs / rhs if rhs else 0.0)
    v = rng.normal(size=d)
    unit = v / np.linalg.norm(v)
    lhs = math.sqrt(d) * abs(float(np.dot(v, unit - (-unit))))
    rhs = math.sqrt(d) * float(np.linalg.norm(v)) * 2.0
    return {
        "formula": "sqrt(d)*abs(<v,b2-b1>) <= 2*sqrt(d)*||v||, v=g_k*R(g_q*qhat)",
        "random_trials": len(ratios),
        "random_max_ratio": float(max(ratios)),
        "constructed_antipodal_ratio": float(lhs / rhs),
    }


def main() -> None:
    rank = {name: average_ranks(name) for name in ["rank_17.npz", "rank_06_monitor.npz", "rank_06_grader.npz"]}
    spears = {}
    for run, rank_file in RUN_TO_RANK.items():
        r = rank[rank_file]
        z = np.load(DATA / "measured" / f"{run}.npz")
        heads = [str(x) for x in z["heads"]]
        if heads != r["heads"]:
            raise SystemExit(f"head order mismatch for {run}")
        measured_mean = z["logit"].astype(np.float64).mean(axis=0)
        spears[run] = {
            "spearman_RANK_mean_vs_measured_logit_mean": spearman(r["score"], measured_mean),
            "measured_argmax_head": heads[int(np.argmax(measured_mean))],
            "measured_argmax_value": float(measured_mean.max()),
        }
    result = {"rankings": {}, "spearman": spears, "derivation_check": derivation_check()}
    for name, r in rank.items():
        result["rankings"][name] = {
            "top3": r["top3"],
            "top3_scores": r["top3_scores"],
            "places": r["places"],
        }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
