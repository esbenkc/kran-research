"""reach-law-gpt-oss: run one local MLX agent through the shell decoy-key harness as ONE resumable job.

    uv run python run.py --model gpt-oss-20b --phase all --filler-stop-at 2026-11-27T11:45:00+01:00
    uv run python run.py --model qwen3-1.7b --phase smoke --n 2          (mini smoke; never counted)

Order (PREREG.md; a stop leaves the main arms balanced):
  gate     never counted: none and runbook, env seeds 900 + g, g < 10, interleaved
  main     i = 0..44: none i, runbook i, linked i (round-robin, i-major); seed block b = i mod 3
  prompt   i = 0..44: prompt i (secondary arm, route in the step-0 ticket text; not pooled)
  filler   i = 45..89: linked i (extra note-arm runs) until --filler-stop-at (gate, main and prompt always run)
Run i uses env seed i in every arm (names, ids and key from the seed) and sampling seed i * 1000 + step, so the arms
share every reply up to the first prompt that differs.
Outputs: data/<model>/raw/<arm>/<NNN>.json, data/<model>/manifest.jsonl, data/<model>/job.log (smoke: smoke/<model>/).
A run whose raw file exists is skipped; rerun the same command to resume.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shell"))

N_MAIN = 45
N_GATE = 10
GATE_SEED0 = 900
SMOKE_SEED0 = 800
MAIN_ARMS = ("none", "runbook", "linked")


def plan(phase: str, n_smoke: int) -> list[tuple[str, str, int]]:
    """(phase, arm, env seed) in run order."""
    if phase == "smoke":
        return [("smoke", a, SMOKE_SEED0 + g) for g in range(n_smoke) for a in ("none", "runbook", "linked", "prompt")]
    gate = [("gate", a, GATE_SEED0 + g) for g in range(N_GATE) for a in ("none", "runbook")]
    main = [("main", a, i) for i in range(N_MAIN) for a in MAIN_ARMS]
    prompt = [("prompt", "prompt", i) for i in range(N_MAIN)]
    filler = [("filler", "linked", i) for i in range(N_MAIN, 2 * N_MAIN)]
    steps = {"gate": gate, "main": main, "prompt": prompt, "filler": filler}
    if phase == "all":
        return gate + main + prompt + filler
    return [x for p in phase.split(",") for x in steps[p]]


def raw_path(out: Path, arm: str, seed: int) -> Path:
    return out / "raw" / arm / f"{seed:03d}.json"


def log(out: Path, msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with (out / "job.log").open("a") as f:
        f.write(line + "\n")


def rss_gb() -> float:
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 2**20
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", default="all", help="all | smoke | comma list of gate,main,prompt,filler")
    ap.add_argument("--n", type=int, default=2, help="smoke: env seeds per arm")
    ap.add_argument("--filler-stop-at", default=None,
                    help="ISO time; no new FILLER run starts after it. Gate, main and prompt runs always run. Ignored when "
                         "the job starts after this time (a late, pending-laptop start runs the whole plan).")
    args = ap.parse_args()

    from agent import run_episode
    import mlx.core as mx
    from mlxcall import _load, gen_settings

    out = ROOT / ("smoke" if args.phase == "smoke" else "data") / args.model
    out.mkdir(parents=True, exist_ok=True)
    todo = plan(args.phase, args.n)
    stop = dt.datetime.fromisoformat(args.filler_stop_at) if args.filler_stop_at else None
    if stop and dt.datetime.now(dt.timezone.utc) >= stop:
        stop = None  # late start (DEVIATIONS v1.2): the whole plan runs, filler included
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    left = [t for t in todo if not raw_path(out, t[1], t[2]).exists()]
    log(out, f"start model={args.model} phase={args.phase} runs={len(todo)} todo={len(left)} head={head} "
             f"filler_stop_at={args.filler_stop_at} effective={stop.isoformat() if stop else None} "
             f"gen={json.dumps(gen_settings(args.model), sort_keys=True)}")
    t0 = time.time()
    _load(args.model)
    log(out, f"loaded in {time.time() - t0:.1f}s")
    done = errors = 0
    for k, (ph, arm, seed) in enumerate(todo, 1):
        p = raw_path(out, arm, seed)
        if p.exists():
            continue
        if stop and ph == "filler" and dt.datetime.now(dt.timezone.utc) >= stop:
            log(out, f"STOP at --filler-stop-at before {ph} {arm} {seed:03d}")
            break
        t1 = time.time()
        try:
            rec = run_episode(args.model, arm, seed, out_path=p)
        except Exception as e:  # one bad run must not end the job; it is retried on resume
            errors += 1
            log(out, f"ERROR {ph} {arm} {seed:03d}: {type(e).__name__}: {e}")
            if errors >= 5:
                log(out, "too many errors; exit")
                return 3
            continue
        done += 1
        r = rec["reach"]
        row = {"phase": ph, "arm": arm, "seed": seed, "block": seed % 3, "key": int(r["key_used"]),
               "reached": int(r["reached"]), "read_ops": int(r["read_ops"]), "read_storage": int(r["read_storage"]),
               "route_tried": int(r["route_tried"]), "mention": int(r["report_mention"]), "report": int(rec["flags"]["report_accepted"]),
               "steps": rec["steps_used"], "parse": rec["parse_counts"], "caps": rec["cap_hits"],
               "tok": rec["new_tokens"], "sec": round(time.time() - t1, 1)}
        with (out / "manifest.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        log(out, f"[{k}/{len(todo)}] {ph} {arm} {seed:03d} key={row['key']} reached={row['reached']} "
                 f"ops={row['read_ops']} note={row['read_storage']} route={row['route_tried']} report={row['report']} "
                 f"steps={row['steps']} parse={row['parse']} caps={row['caps']} tok={row['tok']} {row['sec']}s "
                 f"peakGB={mx.get_peak_memory() / 1e9:.2f} rssGB={rss_gb():.2f}")
    log(out, f"end done={done} errors={errors} elapsed_h={(time.time() - t0) / 3600:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
