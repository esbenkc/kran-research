"""Run episodes: python run.py {smoke,counted} --models ... [--seeds N] [--workers W].

Raw episode logs go to raw/<model>/<cell>/<seed>.jsonl (counted) or raw_smoke/... (smoke seeds 900+).
All four cells of one (model, seed) run in one thread, in a fixed order (off_subtle, off_obvious, on_subtle,
on_obvious). The two decoys of one prompt arm share the system prompt, so the second replays the cached replies of
the first until the first storage reply, where they first differ (paired trajectories). The two prompt arms share
nothing (different system prompts). Jobs are seed-major, so a budget stop leaves balanced cells.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))

from agent import COST_PATH, MAX_STEPS_DEFAULT, run_episode  # noqa: E402
from env import CELLS  # noqa: E402

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "raw"
RAW_SMOKE = PROJECT / "raw_smoke"
COMMANDS = PROJECT / "commands.md"

STRONG5 = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.4-nano", "gpt-5.4-mini", "claude-haiku-4-5-20251001"]
WEAK3 = ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"]
MODELS = STRONG5
SMOKE_SEED0 = 900


def log_command(argv: list[str]) -> None:
    with COMMANDS.open("a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n```sh\nrun.py {' '.join(argv)}\n```\n")


def out_path(model: str, cell: str, seed: int) -> Path:
    root = RAW_SMOKE if seed >= SMOKE_SEED0 else RAW
    return root / model.replace("/", "_") / cell / f"{seed}.jsonl"


def run_one(model: str, cell: str, seed: int, max_steps: int) -> dict[str, Any]:
    p = out_path(model, cell, seed)
    if p.exists():
        return {"model": model, "cell": cell, "seed": seed, "skipped": True}
    try:
        rec = run_episode(model, cell, seed, out_path=p, max_steps=max_steps)
        return {"model": model, "cell": cell, "seed": seed, "steps": rec["steps_used"]}
    except Exception as e:  # keep going; the error file is evidence, the episode is re-run on the next call
        errp = p.with_suffix(".error.txt")
        errp.parent.mkdir(parents=True, exist_ok=True)
        errp.write_text(repr(e))
        return {"model": model, "cell": cell, "seed": seed, "error": repr(e)[:300]}


def spent() -> float:
    if not COST_PATH.exists():
        return 0.0
    return sum(json.loads(ln).get("cost_usd_est", 0.0) for ln in COST_PATH.read_text().splitlines() if ln.strip())


def seed_job(model: str, seed: int, cells: list[str], max_steps: int, budget: float) -> list[dict[str, Any]]:
    if spent() > budget:  # budget stop (Felix #10151): skip whole seeds, never half a seed
        return [{"model": model, "seed": seed, "skipped_budget": True}]
    return [run_one(model, cell, seed, max_steps) for cell in cells]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "counted"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=50, help="prompt-on seeds per model and decoy (N_on, PREREG s4)")
    ap.add_argument("--off-seeds", type=int, default=30, help="prompt-off seeds per model and decoy (seeds 0..off-1)")
    ap.add_argument("--weak-seeds", type=int, default=25, help="seeds per model for the weaker three")
    ap.add_argument("--cells", nargs="+", default=list(CELLS), choices=list(CELLS))
    ap.add_argument("--budget", type=float, default=4.50, help="stop starting new seeds past this logged spend (USD)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    a = ap.parse_args()
    log_command(sys.argv[1:])
    base = SMOKE_SEED0 if a.mode == "smoke" else 0
    jobs = [(m, base + i) for i in range(a.seeds) for m in a.models
            if not (m in WEAK3 and a.mode == "counted" and i >= a.weak_seeds)]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        # PREREG s4: seeds below --off-seeds run all four cells; later seeds run the prompt-on cells only. Jobs are
        # seed-major (seed 0 of every model, then seed 1, ...), so a budget stop leaves balanced cells.
        def cells_for(seed: int) -> list[str]:
            return [c for c in a.cells if seed - base < a.off_seeds or not c.startswith("off_")]
        futs = [ex.submit(seed_job, m, s, cells_for(s), a.max_steps, a.budget) for m, s in jobs]
        for f in cf.as_completed(futs):
            for r in f.result():
                print(json.dumps(r, sort_keys=True), flush=True)
    print(json.dumps({"done": len(jobs), "wall_s": round(time.time() - t0, 1), "spent_usd": round(spent(), 4)}), flush=True)


if __name__ == "__main__":
    main()
