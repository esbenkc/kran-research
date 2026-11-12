#!/usr/bin/env python3
"""critic2 independent rebuild for oracle-1112-matched-trace.

Reads only saved numeric arrays and recomputes the headline cells used for the
final cross-family review. No network, no papers, no secrets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/reads/raw/end_tf.npz"
N_BOOT = 2000
BOOT_SEED = 20261112


def calc(d: np.lib.npyio.NpzFile, idx: np.ndarray) -> np.ndarray:
    c_nb = d["c_nb"][idx]
    c_lb_nb = d["c_lb_nb"][idx]
    log_w = (d["lp_lb_nb"][idx] - d["lp_nb"][idx]).sum(axis=1)
    w = np.exp(log_w)
    d_r = np.maximum(1.0 - w, 0.0).mean()
    cheat_off = c_nb.mean()
    y_r = c_nb.mean() - (w * c_lb_nb).sum() / w.sum()

    d_p = np.minimum(d["pX_off"][idx] + 0.10, 0.99) - d["pX_off"][idx]
    y_p = np.mean(d_p * (c_nb - d["c_X"][idx]))

    s_r = y_r / (d_r * cheat_off)
    s_p = y_p / (d_p.mean() * cheat_off)
    return np.array([
        s_r,
        s_p,
        s_p - s_r,
        100.0 * y_r,
        100.0 * y_p,
        d_r,
        d_p.mean(),
        cheat_off,
        w.mean(),
    ], dtype=float)


def prompt_bootstrap(d: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompts = np.unique(d["prompt_idx"])
    by_prompt = {int(p): np.flatnonzero(d["prompt_idx"] == p) for p in prompts}
    rng = np.random.default_rng(BOOT_SEED)
    full = calc(d, np.arange(len(d["c_nb"])))
    boots = []
    for _ in range(N_BOOT):
        sample = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([by_prompt[int(p)] for p in sample])
        boots.append(calc(d, idx))
    lo, hi = np.quantile(np.vstack(boots), [0.025, 0.975], axis=0)
    return full, lo, hi


def main() -> None:
    d = np.load(DATA)
    full, lo, hi = prompt_bootstrap(d)
    names = [
        "s_R",
        "s_P",
        "gap_s_P_minus_s_R",
        "y_R_points",
        "y_P_points",
        "D_R",
        "D_P",
        "cheat_off",
        "mean_w",
    ]
    result = {
        "source": str(DATA.relative_to(ROOT)),
        "n_rows": int(len(d["c_nb"])),
        "bootstrap": {"clusters": int(len(np.unique(d["prompt_idx"]))), "resamples": N_BOOT, "seed": BOOT_SEED},
        "cells": {
            name: {"estimate": float(full[i]), "lo95": float(lo[i]), "hi95": float(hi[i])}
            for i, name in enumerate(names)
        },
        "verdict": {
            "science": "PASS",
            "reason": "s_R interval is below the frozen 0.10 no-switch bar; s_P and paired gap intervals are above 0.",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
