#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
OUT = Path(__file__).resolve().parent / "rebuild_headline_output.json"

RUNS = [
    "r50_s0_lr0.0001_kl1_at8192",
    "r50_s1_lr0.0001_kl1_at8192",
]
CONTROLS = [
    "shuf_s0_lr0.0001_kl1",
    "sft_s0_lr0.0001",
]


def load_px(run, build):
    return np.load(DATA / "whole" / run / f"{build}_plain.npz")["px"].astype(float)


def base_px(build):
    return load_px("base", build)


def ci95(values, n=5000, seed=12345):
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def pair(run, off="none", on="ref"):
    p_off = load_px(run, off)
    p_on = load_px(run, on)
    delta = (p_off - p_on) - (base_px(off) - base_px(on))
    lo, hi = ci95(delta)
    return {
        "px_off": float(p_off.mean()),
        "px_on": float(p_on.mean()),
        "dgap": float(delta.mean()),
        "ci95": [lo, hi],
        "positive_prompts": int((delta > 0).sum()),
        "n_prompts": int(len(delta)),
        "max_abs_off_minus_on": float(np.max(np.abs(p_off - p_on))),
    }


def main():
    out = {"headline": {}, "controls": {}, "base": {}}
    for run in RUNS:
        out["headline"][run] = {
            "leaky": pair(run, "none", "ref"),
            "score_bit_exact": pair(run, "none", "score"),
            "precision_emulation": pair(run, "none_emu", "ref_emu"),
        }
    for run in CONTROLS:
        out["controls"][run] = {"leaky": pair(run, "none", "ref")}
    for build in ["none", "ref", "score", "none_emu", "ref_emu"]:
        out["base"][build] = float(base_px(build).mean())
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for run, vals in out["headline"].items():
        h = vals["leaky"]
        s = vals["score_bit_exact"]
        e = vals["precision_emulation"]
        print("{}: off={:.4f} on={:.4f} dgap={:.4f} ci=[{:.4f},{:.4f}] positives={}/{} score_dgap={:.4f} emu_dgap={:.4f}".format(run, h["px_off"], h["px_on"], h["dgap"], h["ci95"][0], h["ci95"][1], h["positive_prompts"], h["n_prompts"], s["dgap"], e["dgap"]))
    for run, vals in out["controls"].items():
        h = vals["leaky"]
        print("{}: off={:.4f} on={:.4f} dgap={:.4f} ci=[{:.4f},{:.4f}] positives={}/{}".format(run, h["px_off"], h["px_on"], h["dgap"], h["ci95"][0], h["ci95"][1], h["positive_prompts"], h["n_prompts"]))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
