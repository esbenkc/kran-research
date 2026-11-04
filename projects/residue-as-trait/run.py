"""Run episodes: python run.py {smoke,counted} --tasks ... --models ... [--seeds N] [--workers W].

Raw episode logs go to raw/<task>/<model>/<cell>/<seed>.jsonl (counted) or raw_smoke/... (smoke seeds 900+).
All arms of one (model, seed) run in one thread, in env.RUN_ORDER (none, file, named, runbook). The arms share the
system prompt and every prompt up to the first observation where they differ, so later arms replay none's cached
replies until then (paired trajectories): file differs from none only after a read of /runbooks/storage.md, named
only from the first cloud_stat denial, runbook from the first read of /runbooks/ops.md.
Jobs are seed-major, so a budget stop leaves balanced cells.
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
from env import CELLS, RUN_ORDER, TASK_NAMES  # noqa: E402

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "raw"
RAW_SMOKE = PROJECT / "raw_smoke"
COMMANDS = PROJECT / "commands.md"

STRONG5 = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gpt-5.4-nano", "gpt-5.4-mini", "claude-haiku-4-5-20251001"]
MODELS = STRONG5
SMOKE_SEED0 = 100  # PREREG s1: smoke seeds 100-101, never counted


def log_command(argv: list[str]) -> None:
    with COMMANDS.open("a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n```sh\nrun.py {' '.join(argv)}\n```\n")


def out_path(task: str, model: str, cell: str, seed: int) -> Path:
    root = RAW_SMOKE if seed >= SMOKE_SEED0 else RAW
    return root / task / model.replace("/", "_") / cell / f"{seed}.jsonl"


def run_one(task: str, model: str, cell: str, seed: int, max_steps: int) -> dict[str, Any]:
    p = out_path(task, model, cell, seed)
    tag = {"task": task, "model": model, "cell": cell, "seed": seed}
    if p.exists():
        return dict(tag, skipped=True)
    try:
        rec = run_episode(model, cell, seed, out_path=p, max_steps=max_steps, task=task)
        return dict(tag, steps=rec["steps_used"])
    except Exception as e:  # keep going; the error file is evidence, the episode is re-run on the next call
        errp = p.with_suffix(".error.txt")
        errp.parent.mkdir(parents=True, exist_ok=True)
        errp.write_text(repr(e))
        return dict(tag, error=repr(e)[:300])


def spent() -> float:
    if not COST_PATH.exists():
        return 0.0
    return sum(json.loads(ln).get("cost_usd_est", 0.0) for ln in COST_PATH.read_text().splitlines() if ln.strip())


def seed_job(task: str, model: str, seed: int, cells: list[str], max_steps: int, budget: float) -> list[dict[str, Any]]:
    if spent() > budget:  # budget stop (Felix #10151): skip whole seeds, never half a seed
        return [{"task": task, "model": model, "seed": seed, "skipped_budget": True}]
    return [run_one(task, model, cell, seed, max_steps) for cell in cells]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "counted"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=list(TASK_NAMES), choices=list(TASK_NAMES))
    ap.add_argument("--seeds", type=int, default=30, help="seeds per model; every seed runs every listed cell")
    ap.add_argument("--seed0", type=int, default=0, help="first seed offset (extension runs)")
    ap.add_argument("--cells", nargs="+", default=["named", "runbook"], choices=list(CELLS))
    ap.add_argument("--budget", type=float, default=4.50, help="stop starting new seeds past this logged spend (USD)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    a = ap.parse_args()
    log_command(sys.argv[1:])
    base = SMOKE_SEED0 if a.mode == "smoke" else 0
    jobs = [(t, m, base + a.seed0 + i) for i in range(a.seeds) for t in a.tasks for m in a.models]
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        # Jobs are seed-major (seed 0 of every task x model, then seed 1, ...), so a budget stop leaves balanced cells.
        def cells_for(seed: int) -> list[str]:
            return [c for c in RUN_ORDER if c in a.cells]

        futs = [ex.submit(seed_job, t, m, s, cells_for(s), a.max_steps, a.budget) for t, m, s in jobs]
        for f in cf.as_completed(futs):
            for r in f.result():
                print(json.dumps(r, sort_keys=True), flush=True)
    print(json.dumps({"done": len(jobs), "wall_s": round(time.time() - t0, 1), "spent_usd": round(spent(), 4)}), flush=True)


if __name__ == "__main__":
    main()
