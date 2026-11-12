"""Shared names, sizes and frozen rules (PREREG.md s3-s5, s8) for pick_lr.py, analyze.py and the queue builder."""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "data" / "runs"
ALPHA_TOP = 1 / 32
SIZES = [1 / 32, 1 / 64, 1 / 128, 1 / 256]   # headline grid (PREREG s5)
EXTRA = 1 / 512                              # exploratory 5th size (s5 items 6, 8)
OPTS = ["adamw", "sgd", "sign"]
LADDER = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]  # s8.1, descending
CAP, FLOOR = 24576, 1024


def run_name(opt, alpha, seed):
    return f"{opt}_a{alpha:g}_s{seed}"


def pilot_name(lr):
    return f"sgdpilot_lr{lr:g}"


def load(name):
    p = RUNS / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def T_of(r):
    """Headline readout: T over all evals (PREREG s3 grid); None if censored."""
    return r["T_dense"]


def budget(T0, x_top, x):
    """PREREG s3: B(x) = min(24576, max(1024, 3 T0 (x_top/x)^2)), rounded up to a multiple of 64."""
    b = min(CAP, max(FLOOR, 3 * T0 * (x_top / x) ** 2))
    return int(math.ceil(b / 64) * 64)


def all_evals(r):
    return sorted(r["dev_curve"] + r["dense_curve"], key=lambda c: c["step"])


def stable(r):
    """PREREG s8.1 stability test of an SGD pilot. Returns (stable, list of reasons it failed)."""
    why = []
    if r.get("nonfinite_at") is not None:
        why.append(f"non-finite at {r['nonfinite_at']}")
    for c in r["dev_curve"]:
        if c.get("kl") is not None and c["kl"] > 0.5:
            why.append(f"mean KL {c['kl']:.3f} > 0.5 at {c['step']}")
            break
    for c in all_evals(r):
        if c["step"] > 8 and not (0.07 <= c["px_off"] <= 0.60):  # DEVIATIONS D1
            why.append(f"P(X, cue off) {c['px_off']:.3f} outside [0.07,0.60] at {c['step']}")
            break
    h0 = r["health"][0]
    for h in r["health"][1:]:
        if h["ent_off"] < 0.5 * h0["ent_off"] or h["top1_off"] > 0.9:  # DEVIATIONS D1 (argmax-share test dropped)
            why.append(f"text health at {h['step']}: entropy {h['ent_off']:.2f} (step0 {h0['ent_off']:.2f}), "
                       f"top-1 {h['top1_off']:.2f}")
            break
    return (not why), why
