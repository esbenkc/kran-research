from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Allow direct execution from repo root.
sys.path.append(str(Path(__file__).resolve().parent))

from agent import MAX_STEPS_DEFAULT, run_episode, run_random_episode

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "raw"
RESULTS = PROJECT / "results"
COMMANDS = PROJECT / "commands.md"

BASE_MODELS = [
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-5-nano",
    "gemini-2.5-flash-lite",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001",
    "gpt-5-mini",
]
EXTRA_MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.1-flash-lite"]
ARMS = ["static", "stateful", "nobait"]


def log_command(argv: list[str]) -> None:
    COMMANDS.parent.mkdir(exist_ok=True)
    with COMMANDS.open("a") as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        f.write("cwd: `kran-research`\n\n")
        f.write("```sh\n" + " ".join(argv) + "\n```\n")


def memory_snapshot() -> str:
    try:
        out = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip()
        return f"pid_rss_kb={out}"
    except Exception as e:
        return f"memory_snapshot_error={type(e).__name__}"


RAW_SMOKE = PROJECT / os.environ.get("DECOY_SMOKE_DIR", "raw_smoke")


def out_path(model: str, arm: str, seed: int) -> Path:
    safe = model.replace("/", "_")
    root = RAW_SMOKE if seed >= 900 else RAW  # smoke seeds (>= 900) are never counted (PREREG s8)
    return root / safe / arm / f"{seed}.jsonl"


def run_one(model: str, arm: str, seed: int, max_steps: int) -> dict[str, Any]:
    p = out_path(model, arm, seed)
    if p.exists():
        return {"model": model, "arm": arm, "seed": seed, "path": str(p), "cached_file": True}
    try:
        if model == "random":
            run_random_episode(arm, seed, p, max_steps=max_steps)
        else:
            run_episode(model, arm, seed, p, max_steps=max_steps)
        return {"model": model, "arm": arm, "seed": seed, "path": str(p), "ok": True}
    except Exception as e:
        errp = p.with_suffix(".error.json")
        errp.parent.mkdir(parents=True, exist_ok=True)
        errp.write_text(json.dumps({"model": model, "arm": arm, "seed": seed, "error": repr(e)}, indent=2))
        return {"model": model, "arm": arm, "seed": seed, "error": repr(e), "path": str(errp)}


def paired_job(model: str, seed: int, max_steps: int) -> list[dict[str, Any]]:
    # Static first, then stateful, in one thread: the stateful episode replays the cached prefix up to the
    # first bait reply, so both arms take the same actions until then (PREREG s1).
    return [run_one(model, "static", seed, max_steps), run_one(model, "stateful", seed, max_steps)]


def run_batch(models: list[str], seeds: list[int], nobait_seeds: list[int], max_steps: int, workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for model in models:
            for seed in seeds:
                futs.append(ex.submit(paired_job, model, seed, max_steps))
            for seed in nobait_seeds:
                futs.append(ex.submit(run_one, model, "nobait", seed, max_steps))
        for fut in cf.as_completed(futs):
            r = fut.result()
            if isinstance(r, list):
                results.extend(r)
            else:
                results.append(r)
            print(json.dumps(r if not isinstance(r, list) else {"batch": len(r), "last": r[-1]}, sort_keys=True), flush=True)
    return results


def run_random(seeds: list[int], max_steps: int) -> None:
    for arm in ["static", "stateful"]:
        for seed in seeds:
            print(json.dumps(run_one("random", arm, seed, max_steps), sort_keys=True), flush=True)


def smoke_report(models: list[str]) -> None:
    """Per model: parsed calls per episode (PREREG s2 run rule >= 3), touch, use, subgoals, steps, cost."""
    from scorer import score_file
    cost: dict[str, float] = {}
    cp = RAW / "cost.jsonl"  # all API spend (smoke + counted) is logged here
    if cp.exists():
        for line in cp.read_text().splitlines():
            c = json.loads(line)
            cost[c["model"]] = cost.get(c["model"], 0.0) + c.get("cost_usd_est", 0.0)
    rows = []
    for m in models:
        eps = [score_file(p) for p in sorted((RAW_SMOKE / m.replace("/", "_")).glob("*/*.jsonl"))]
        if not eps:
            continue
        by = lambda arm: [e for e in eps if e["arm"] == arm]
        rows.append({"model": m, "n": len(eps),
                     "parsed_calls_per_ep": sum(e["n_actions"] for e in eps) / len(eps),
                     "parse_fail_per_ep": sum(e["parse_fail"] for e in eps) / len(eps),
                     "touch_static": sum(e["touch_e"] for e in by("static")), "use_static": sum(e["use_e"] for e in by("static")),
                     "use_stateful": sum(e["use_e"] for e in by("stateful")), "n_paired": len(by("static")),
                     "subgoal_nobait": sum(e["subgoal_score"] for e in by("nobait")) / max(1, len(by("nobait"))),
                     "steps_mean": sum(e["steps_used"] for e in eps) / len(eps),
                     "cost_usd_so_far": round(cost.get(m, 0.0), 4)})
    out = {"models": rows}
    RESULTS.mkdir(exist_ok=True)
    (RAW_SMOKE / "smoke_report.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(json.dumps(r), flush=True)


def analyse(raw: Path = RAW, out: Path = RESULTS) -> None:
    import analysis
    sys.argv = ["analysis.py", "--raw", str(raw), "--out", str(out)]
    analysis.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "counted", "random", "analysis"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--nobait", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    args = ap.parse_args()
    log_command(sys.argv)
    print(memory_snapshot(), flush=True)
    if args.mode == "analysis":
        analyse(); return
    models = args.models or BASE_MODELS
    if args.mode == "smoke":
        seeds = [900, 901, 902] if args.seeds is None else list(range(900, 900 + args.seeds))
        nobait = [990, 991, 992] if args.nobait is None else list(range(990, 990 + args.nobait))
        run_batch(models, seeds, nobait, args.max_steps, args.workers)
        smoke_report(models)
        return
    if args.mode == "counted":
        n = args.seeds if args.seeds is not None else 30
        nb = args.nobait if args.nobait is not None else 15
        seeds = list(range(n))
        nobait = list(range(100, 100 + nb))
        run_batch(models, seeds, nobait, args.max_steps, args.workers)
        analyse(); return
    if args.mode == "random":
        n = args.seeds if args.seeds is not None else 40
        run_random(list(range(n)), args.max_steps)
        analyse(); return


if __name__ == "__main__":
    main()
