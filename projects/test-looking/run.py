"""Run episodes: python run.py {pilot,counted} [--models ...] [--seeds N] [--warning-seeds N] [--workers W].

Raw episode logs go to raw/<model>/<arm>/<seed>.jsonl (counted, seeds 1000+) or raw_pilot/... (pilot, seeds 900-999).
Each (model, arm, seed) is one independent episode: the arms differ from the first observation on (names or system
prompt), so no arm replays another arm's cached replies. The record cell is fixed per agent (RECORD_CELL below) and is
the same in all three arms of that agent.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from agent import COST_PATH, MAX_STEPS_DEFAULT, run_episode  # noqa: E402
from env import ARMS  # noqa: E402

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "raw"
RAW_PILOT = PROJECT / "raw_pilot"
COMMANDS = PROJECT / "commands.md"

MODELS = ["claude-haiku-4-5-20251001", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.4-nano", "gpt-5.4-mini"]
# Per-agent base record, PREREG s1 table: the decoy-usefulness USELESS record in [25%, 90%] closest to 50%.
RECORD_CELL = {
    "claude-haiku-4-5-20251001": "useless_foreign",
    "gemini-3.1-flash-lite": "useless_real_explicit",
    "gemini-3.5-flash-lite": "useless_foreign_explicit",
    "gpt-5.4-nano": "useless_real_explicit",
    "gpt-5.4-mini": "useless_real",
}
COUNTED_SEED0 = 1000
PILOT_SEED0 = 900


def log_command(argv: list[str]) -> None:
    with COMMANDS.open("a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n```sh\nwith-model-keys uv run python run.py {' '.join(argv)}\n```\n")


def out_path(model: str, arm: str, seed: int) -> Path:
    root = RAW if seed >= COUNTED_SEED0 else RAW_PILOT
    return root / model.replace("/", "_") / arm / f"{seed}.jsonl"


def run_one(model: str, arm: str, seed: int, max_steps: int) -> dict[str, Any]:
    p = out_path(model, arm, seed)
    if p.exists():
        return {"model": model, "arm": arm, "seed": seed, "skipped": True}
    try:
        rec = run_episode(model, arm, seed, RECORD_CELL[model], out_path=p, max_steps=max_steps)
        return {"model": model, "arm": arm, "seed": seed, "steps": rec["steps_used"]}
    except Exception as e:  # keep going; the error file is evidence, the episode is re-run on the next call
        errp = p.with_suffix(".error.txt")
        errp.parent.mkdir(parents=True, exist_ok=True)
        errp.write_text(repr(e))
        return {"model": model, "arm": arm, "seed": seed, "error": repr(e)[:300]}


def spent() -> float:
    if not COST_PATH.exists():
        return 0.0
    return sum(json.loads(ln).get("cost_usd_est", 0.0) for ln in COST_PATH.read_text().splitlines() if ln.strip())


def job(model: str, arm: str, seed: int, max_steps: int, budget: float) -> dict[str, Any]:
    if spent() > budget:  # budget stop: skip new episodes past the logged spend
        return {"model": model, "arm": arm, "seed": seed, "skipped_budget": True}
    return run_one(model, arm, seed, max_steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "counted"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=60, help="seeds per agent in the prod and test arms")
    ap.add_argument("--warning-seeds", type=int, default=20, help="seeds per agent in the warning arm")
    ap.add_argument("--seed-offset", type=int, default=0, help="first seed index (to extend a run)")
    ap.add_argument("--budget", type=float, default=4.50, help="stop starting new episodes past this logged spend (USD)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    a = ap.parse_args()
    log_command(sys.argv[1:])
    base = PILOT_SEED0 if a.mode == "pilot" else COUNTED_SEED0
    jobs = []
    # interleave so a budget stop never leaves one arm short: seed-major, then agent, then arm
    for i in range(a.seed_offset, a.seed_offset + a.seeds):
        for m in a.models:
            for arm in a.arms:
                if arm == "warn" and i >= a.warning_seeds:
                    continue
                jobs.append((m, arm, base + i))
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, m, arm, s, a.max_steps, a.budget) for m, arm, s in jobs]
        for f in cf.as_completed(futs):
            print(json.dumps(f.result(), sort_keys=True), flush=True)
    print(json.dumps({"jobs": len(jobs), "wall_s": round(time.time() - t0, 1), "spent_usd": round(spent(), 4)}), flush=True)


if __name__ == "__main__":
    main()
