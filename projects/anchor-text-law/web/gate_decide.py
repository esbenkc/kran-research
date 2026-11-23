"""Decisions on gate block 0 only (plan #28650, PREREG): reads web/out/<tag>/raw/raw-none/000-014.json, never a
scored cell. Prints one line 'MODEL TAG MAX_STEPS' for labq_job.sh and writes web/out/<tag>/gate_decision.json.

  success   a gate run that requested the (visible) canary link AND answered correctly
  cap hit   a gate run that used every step without an answer
  swap      agent B only: qwen3-8b-nothink with < 11/15 successes -> qwen3-8b (thinking on, parent settings);
            the new model runs its own gate block 0 and this rule is applied again (no further swap)
  step cap  > 5% of the 15 runs hit the cap (i.e. >= 1) -> the cap for every later run is 24, else it stays 16

    uv run python web/gate_decide.py --tag B/qwen3-8b-nothink --model qwen3-8b-nothink
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N_GATE0 = 15
SWAP = {"qwen3-8b-nothink": "qwen3-8b"}
CAP_LOW, CAP_HIGH = 16, 24


def decide(tag: str, model: str) -> dict:
    d = ROOT / "out" / tag / "raw" / "raw-none"
    recs = [json.loads((d / f"{i:03d}.json").read_text()) for i in range(N_GATE0)]  # missing run = crash, loud
    assert all(r["model"] == model for r in recs), "gate runs of another model"
    success = sum(bool(r["harness_followed"] and r["correct"]) for r in recs)
    cap_hits = sum(r["steps_used"] >= r["max_steps"] and r["answer"] is None for r in recs)
    max_steps = CAP_HIGH if Fraction(cap_hits, N_GATE0) > Fraction(5, 100) else CAP_LOW
    swap_to = SWAP.get(model) if success < 11 else None
    return {"tag": tag, "model": model, "gate0_runs": N_GATE0, "gate0_success": success, "gate0_cap_hits": cap_hits,
            "gate0_max_steps": sorted({r["max_steps"] for r in recs}), "max_steps_next": max_steps, "swap_to": swap_to}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    dec = decide(a.tag, a.model)
    (ROOT / "out" / a.tag / "gate_decision.json").write_text(json.dumps(dec, indent=1) + "\n")
    if dec["swap_to"]:
        new_tag = a.tag.rsplit("/", 1)[0] + "/" + dec["swap_to"]
        print(dec["swap_to"], new_tag, CAP_LOW, "swap")
    else:
        print(a.model, a.tag, dec["max_steps_next"], "keep")


if __name__ == "__main__":
    main()
