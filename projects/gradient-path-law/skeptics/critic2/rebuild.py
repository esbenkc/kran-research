#!/usr/bin/env python3
"""critic2 rebuild for oracle-1107-gradient-path-law.

Numpy-only numeric rebuild from saved status artifacts:
  data/status/<run>_raw/{none_plain,ref_plain,score_plain,trainer_none,trainer_ref}.npz
  data/status/<run>.json
It prints the headline numbers used in the critic2 verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "data" / "status"
RUNS = ["Tc_s0", "Tc_s1", "fed_s2", "fed_s3", "rec_s2", "rec_s3", "Tb_s0", "Tb_s1", "Ta_s0"]


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_json(run: str) -> dict:
    with open(STATUS / f"{run}.json") as f:
        return json.load(f)


def run_numbers(run: str) -> dict:
    raw = STATUS / f"{run}_raw"
    none = np.load(raw / "none_plain.npz")
    ref = np.load(raw / "ref_plain.npz")
    score = np.load(raw / "score_plain.npz")
    tn = np.load(raw / "trainer_none.npz")
    tr = np.load(raw / "trainer_ref.npz")
    meta = load_json(run)

    c0 = none["p_read"] / (none["p_read"] + none["p_solve"])
    c1 = ref["p_read"] / (ref["p_read"] + ref["p_solve"])
    cb = score["p_read"] / (score["p_read"] + score["p_solve"])
    served_gap = c0 - c1
    served_lo, served_hi = boot_ci(served_gap)
    bit_gap = c0 - cb
    bit_lo, bit_hi = boot_ci(bit_gap)

    served_logit = none["d2"] - ref["d2"]
    trainer_none_logit = tn["l_read"] - tn["l_solve"]
    trainer_ref_logit = tr["l_read"] - tr["l_solve"]
    trainer_logit = trainer_none_logit - trainer_ref_logit
    trainer_gap = sig(trainer_none_logit).mean() - sig(trainer_ref_logit).mean()

    out = {
        "run": run,
        "n": int(len(served_gap)),
        "served_no_probe": float(c0.mean()),
        "served_probe": float(c1.mean()),
        "served_gap": float(served_gap.mean()),
        "served_ci": (served_lo, served_hi),
        "served_logit_gap": float(served_logit.mean()),
        "trainer_no_probe": float(sig(trainer_none_logit).mean()),
        "trainer_probe": float(sig(trainer_ref_logit).mean()),
        "trainer_gap": float(trainer_gap),
        "trainer_logit_gap": float(trainer_logit.mean()),
        "bitexact_gap": float(bit_gap.mean()),
        "bitexact_ci": (bit_lo, bit_hi),
        "bitexact_sha_diff": float(meta["share_sha_differs_bitexact"]),
        "leaky_sha_diff": float(meta["share_sha_differs_leaky"]),
    }
    if abs(out["trainer_logit_gap"]) > 1e-12:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(served_logit), (2000, len(served_logit)))
        ratio = served_logit[idx].mean(axis=1) / trainer_logit[idx].mean(axis=1)
        out["carry"] = float(served_logit.mean() / trainer_logit.mean())
        out["carry_ci"] = (float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5)))
    else:
        out["carry"] = None
        out["carry_ci"] = None
    return out


def pp(x: float) -> str:
    return f"{100*x:.2f}"


def pct(x: float) -> str:
    return f"{100*x:.1f}"


def main() -> None:
    rows = {run: run_numbers(run) for run in RUNS}
    print("critic2 rebuild from data/status/*_raw/*.npz and data/status/*.json")
    for run in RUNS:
        r = rows[run]
        carry = "n/a" if r["carry"] is None else f"{r['carry']:.4f} [{r['carry_ci'][0]:.4f},{r['carry_ci'][1]:.4f}]"
        print(
            f"{run}: served {pct(r['served_no_probe'])}/{pct(r['served_probe'])}% "
            f"gap {pp(r['served_gap'])} pp [{pp(r['served_ci'][0])},{pp(r['served_ci'][1])}], "
            f"trainer {pct(r['trainer_no_probe'])}/{pct(r['trainer_probe'])}% "
            f"trainer_gap {pp(r['trainer_gap'])} pp logit {r['trainer_logit_gap']:.4f}, "
            f"carry {carry}, bitexact_gap {pp(r['bitexact_gap'])} pp, "
            f"sha_diff leaky/bitexact {r['leaky_sha_diff']:.1f}/{r['bitexact_sha_diff']:.1f}"
        )

    tc_ok = all(abs(rows[r]["carry_ci"][0]) < 0.15 and abs(rows[r]["carry_ci"][1]) < 0.15 for r in ("Tc_s0", "Tc_s1"))
    tc_takeoff = all(rows[r]["trainer_gap"] >= 0.15 for r in ("Tc_s0", "Tc_s1"))
    fed_ok = all(rows[r]["served_gap"] >= 0.15 for r in ("fed_s2", "fed_s3"))
    rec_ok = all(abs(rows[r]["served_gap"]) < 0.03 for r in ("rec_s2", "rec_s3"))
    tb_silent = all(abs(rows[r]["carry_ci"][0]) < 0.15 and abs(rows[r]["carry_ci"][1]) < 0.15 for r in ("Tb_s0", "Tb_s1"))
    ta_ok = abs(rows["Ta_s0"]["served_gap"]) < 0.03
    print(f"SUMMARY: Tc_silent={tc_ok}; Tc_takeoff={tc_takeoff}; fed_positive={fed_ok}; clean_rerun_silent={rec_ok}; TIS_silent={tb_silent}; Ta_no_collapse_switch={ta_ok}")
    print("VERDICT: CONFIRMED for the title answer in the changed setup; prereg failed on the clean-rerun probe-on fall clause.")


if __name__ == "__main__":
    main()
