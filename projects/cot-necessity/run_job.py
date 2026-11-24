"""The laptop-queue job (J1): calibration, timing rule, 5 GRPO runs, then the reader over every saved thought.

Every stage skips itself when its output exists, so a restart resumes. Each model runs in its own process (one
heavy model in memory at a time). Frozen rules implemented here (PREREG.md):
  price P = 0.5 x (a_base - a0) from the main arm; each arm: lam0 = P / (L0_arm - L_min), lam_final = P / 9,
  ramp over the first S/2 updates (if --ramp), L_min = 8.
  steps per run S = min(S_max, floor((budget - eval time) / (5 x t_step))), t_step = median of the first 3 updates
  of main seed 0; S < S_min -> the whole job restarts on --fallback-model (8B outputs moved to data/aborted_<tag>/).
  if a_base <= a0 + 0.10: no RL (the task leaves no room for in-conflict pressure), data/no_rl.json.

Usage (from the project dir):  uv run python run_job.py --a0 <noCoT acc main> --a0-control <noCoT acc control>
                                   --k-control <flips>   [--no-control]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = DATA / "runs"
PY = [sys.executable]


def sh(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"stage failed ({r.returncode}): {' '.join(cmd)}; see {log}")


def jload(p: Path):
    return json.loads(p.read_text())


def calib(model: str, k: int, tag: str) -> dict:
    out = RUNS / f"calib_{tag}"
    if not (out / "calib.json").exists():
        sh(PY + ["grpo.py", "--model", model, "--run", f"calib_{tag}", "--out", str(RUNS),
                 "--calib", f"inputs/calib_k{k}.jsonl"], DATA / "logs" / f"calib_{tag}.log")
    return jload(out / "calib.json")


def grpo(model: str, run: str, k: int, seed: int, steps: int, lam0: float, lam_f: float, ramp: int,
         extra: list[str]) -> None:
    sh(PY + ["grpo.py", "--model", model, "--run", run, "--out", str(RUNS), "--seed", str(seed),
             "--train", f"inputs/train_k{k}.jsonl", "--eval", f"inputs/eval_k{k}.jsonl", "--steps", str(steps),
             "--eval-every", str(max(1, steps // 5)), "--lam0", repr(lam0), "--lam-final", repr(lam_f),
             "--ramp-steps", str(ramp), "--final-nothink", *extra], DATA / "logs" / f"{run}.log")


def necessity(model: str) -> dict:
    """Frozen rule (PREREG): main k = 8 if no-CoT argmax acc <= 0.60 AND AUROC <= 0.60, else the smallest of
    12, 16 that passes (None = premise failed). Control k = the largest of 1-4 with acc >= 0.85 AND AUROC >= 0.85
    (0 = no control)."""
    out = DATA / "step0_mlx"
    sp = out / "summary.json"
    have = lambda k: sp.exists() and str(k) in jload(sp)
    need = [k for k in (8, 1, 2, 3, 4) if not have(k)]
    if need:
        sh(PY + ["necessity_mlx.py", "--model", model, "--out", str(out), *map(str, need)],
           DATA / "logs" / "necessity.log")
    S = jload(sp)
    ok = lambda k: S[str(k)]["argmax_acc"] <= 0.60 and S[str(k)]["auroc"] <= 0.60
    k_main = 8 if ok(8) else None
    for k in (12, 16):
        if k_main is None:
            if not have(k):
                sh(PY + ["necessity_mlx.py", "--model", model, "--out", str(out), str(k)],
                   DATA / "logs" / "necessity.log")
                S = jload(sp)
            if ok(k):
                k_main = k
    ctrl = [k for k in (1, 2, 3, 4) if S[str(k)]["argmax_acc"] >= 0.85 and S[str(k)]["auroc"] >= 0.85]
    k_c = max(ctrl) if ctrl else 0
    res = {"k_main": k_main, "a0": S[str(k_main)]["argmax_acc"] if k_main else None, "k_control": k_c,
           "a0_control": S[str(k_c)]["argmax_acc"] if k_c else None, "source": "MLX " + model}
    (DATA / "necessity.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--fallback-model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--a0", type=float, default=None, help="no-CoT acc, main arm (NDIF Step 0); None = measure here")
    ap.add_argument("--a0-control", type=float, default=None)
    ap.add_argument("--k-main", type=int, default=8)
    ap.add_argument("--k-control", type=int, default=-1, help="-1 = pick by the frozen rule; 0 = no control arm")
    ap.add_argument("--seeds-main", type=int, default=3)
    ap.add_argument("--seeds-control", type=int, default=2)
    ap.add_argument("--budget-h", type=float, default=7.0, help="training + eval budget for all runs")
    ap.add_argument("--s-max", type=int, default=100)
    ap.add_argument("--s-min", type=int, default=40)
    ap.add_argument("--lmin", type=float, default=8.0)
    ap.add_argument("--canon-chars", type=float, default=17.0, help="chars of the canonical 8-flip trace")
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--reader-cmd", default="", help="shell command run after training (reader stage)")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    state_p = DATA / "job_state.json"
    state = jload(state_p) if state_p.exists() else {"model": a.model, "t0": time.time()}
    model = state["model"]

    # 0. necessity (when NDIF could not serve Qwen3-8B): same RL-format measurement on the trained 4-bit weights
    if a.a0 is None or a.k_control == -1:
        nec = necessity(model)
        if a.a0 is None:
            a.a0, a.k_main = nec["a0"], nec["k_main"]
        if a.k_control == -1:
            a.k_control, a.a0_control = nec["k_control"], nec["a0_control"]
        if a.k_main is None:
            (DATA / "no_rl.json").write_text(json.dumps({**nec, "reason": "necessity premise failed at 8, 12, 16"}))
            print("NO RL: necessity premise failed", flush=True)
            return

    # 1. calibration: a_base, L0 per arm, the price P and the lambdas
    cm = calib(model, a.k_main, f"k{a.k_main}")
    P = 0.5 * (cm["a_base"] - a.a0)
    lam = {"P": P, "a0": a.a0, "a_base": cm["a_base"], "L0": cm["L0_median_chars"], "L_min": a.lmin,
           "model": model, "lam0": P / (cm["L0_median_chars"] - a.lmin) if cm["L0_median_chars"] > a.lmin else None,
           "lam_final": P / (a.canon_chars - a.lmin), "ramp": a.ramp}
    if cm["a_base"] <= a.a0 + 0.10:
        (DATA / "no_rl.json").write_text(json.dumps({**lam, "reason": "a_base <= a0 + 0.10"}, indent=1))
        print("NO RL: a_base <= a0 + 0.10", flush=True)
        return
    arms = [("main", a.k_main, s, lam["lam0"]) for s in range(a.seeds_main)]
    if a.k_control:
        cc = calib(model, a.k_control, f"k{a.k_control}")
        lam_c0 = P / (cc["L0_median_chars"] - a.lmin) if cc["L0_median_chars"] > a.lmin else None
        lam["control"] = {"k": a.k_control, "a0": a.a0_control, "a_base": cc["a_base"],
                          "L0": cc["L0_median_chars"], "lam0": lam_c0,
                          "P_gt_drop_gain": (P > cc["a_base"] - (a.a0_control or 0.5))}
        if lam_c0 is None:
            lam["control"]["dropped"] = "L0 <= L_min: the base trace is already free"
        else:
            arms += [("control", a.k_control, s, lam_c0) for s in range(a.seeds_control)]
    (DATA / "lambda.json").write_text(json.dumps(lam, indent=1))

    # 2. timing rule on main seed 0 (3 updates, then resume from there)
    run0 = f"main_k{a.k_main}_s0"
    if "steps" not in state:
        grpo(model, run0, a.k_main, 0, a.s_max, lam["lam0"], lam["lam_final"], 0, ["--timing-only", "3"])
        rows = [json.loads(l) for l in (RUNS / run0 / "steps.jsonl").read_text().splitlines()]
        t_step = statistics.median(r["t_step"] for r in rows[:3])
        ev = RUNS / run0 / "evals" / "step_0000.jsonl"
        t_eval = (ev.stat().st_mtime - (RUNS / run0 / "run.json").stat().st_mtime) if ev.exists() else 0.0
        n_runs = len(arms)
        budget = a.budget_h * 3600 - n_runs * 6 * max(t_eval, 0)
        S = min(a.s_max, math.floor(budget / (n_runs * t_step)))
        S -= S % 10  # checkpoints at S/5 land on whole updates
        state.update({"t_step": t_step, "t_eval": t_eval, "steps_raw": S, "n_runs": n_runs})
        if S < a.s_min and model != a.fallback_model:
            tag = model.split("/")[-1]
            shutil.move(str(RUNS), str(DATA / f"aborted_{tag}"))
            for p in ("lambda.json",):
                if (DATA / p).exists():
                    shutil.move(str(DATA / p), str(DATA / f"aborted_{tag}" / p))
            state = {"model": a.fallback_model, "t0": state["t0"], "aborted": {tag: state}}
            state_p.write_text(json.dumps(state, indent=1))
            print(f"S = {S} < {a.s_min} on {model}: restarting on {a.fallback_model}", flush=True)
            return main()
        state["steps"] = S
        state_p.write_text(json.dumps(state, indent=1))
    S = state["steps"]
    if S < 3:
        raise SystemExit(f"steps {S} too small")
    # The timing updates used lam0 with no ramp; with --ramp they are the first 3 of the ramp, where the ramp
    # moved lambda by 3/(S/2) of its range: logged in steps.jsonl ('lam'), noted in DEVIATIONS if --ramp.

    # 3. the runs, in the frozen order
    for arm, k, seed, l0 in arms:
        run = f"{arm}_k{k}_s{seed}"
        if (RUNS / run / "evals" / f"nothink_step_{S:04d}.jsonl").exists():
            continue
        grpo(model, run, k, seed, S, l0, lam["lam_final"], S // 2 if a.ramp else 0, [])

    # 4. reader stage
    if a.reader_cmd and not (DATA / "reader_done.json").exists():
        log = DATA / "logs" / "reader.log"
        with open(log, "a") as f:
            r = subprocess.run(a.reader_cmd, shell=True, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise SystemExit(f"reader stage failed; see {log}")
        (DATA / "reader_done.json").write_text(json.dumps({"time": time.time()}))
    state["t_end"] = time.time()
    state_p.write_text(json.dumps(state, indent=1))
    print("JOB DONE", json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
