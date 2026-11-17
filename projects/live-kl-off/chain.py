"""The frozen run tree (PREREG.md 'Engine' + Felix's plan #22979), one heavy job at a time on the mini.

  0. placebo pool: ctrl_pool.py (' Let' 6771 written into the 2,800 xins twins, pilot recompute) -> data/pool/test_let
     and its read on the pilot (adapter 0) BEFORE seed 0 trains -> data/reads/let/
  1. seed 0 (run_seeds.py, thought KL off, early stop at the confirming read, clock 20:00 PST)
  2. branch read in code on seed 0: onset_stop.json = TAKE-OFF -> seeds 1, 2
                                    reached 1,200 without it = FLAT -> positive control (cue 0.86, 300 updates), seed 1
                                    anything else (clock) -> stop
  after every seed: placebo read on that seed's last adapter (between seeds, never beside a training seed)

  /usr/bin/python3 chain.py --stop-at 20:00
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = os.environ.get("LIVE_PY", str(HERE / "../../../env/.venv/bin/python"))
RUNS = HERE / "data" / "runs"


def sh(cmd, log):
    print(f"[chain] {time.strftime('%T')} {' '.join(cmd[:3])} ... -> {log}", flush=True)
    rc = subprocess.call([PY, "watchdog.py", "--cap-gb", "9.0", "--log", f"logs/{log}", "--"] + cmd)
    print(f"[chain] {time.strftime('%T')} exit {rc}", flush=True)
    return rc


def last_read(name):
    p = RUNS / name / "reads.jsonl"
    if not p.exists():
        return 0
    st = [json.loads(l)["step"] for l in p.read_text().splitlines() if l.strip()]
    return st[-1] if st else 0


def outcome(name, steps):
    if (RUNS / name / "onset_stop.json").exists():
        return "TAKE-OFF"
    if last_read(name) >= steps and not (RUNS / name / "clock_stop.json").exists():
        return "FLAT"
    return "CUT"


def clock_passed(stop_at):
    hh, mm = (int(x) for x in stop_at.split(":"))
    lt = time.localtime()
    return time.time() >= time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))


def seed(name_prefix, s, stop_at, extra=()):
    subprocess.call(["/usr/bin/python3", "run_seeds.py", "--seeds", str(s), "--stop-at", stop_at,
                     "--prefix", name_prefix, *extra])


def placebo(name):
    n = last_read(name)
    a = RUNS / name / f"adapter_{n}.pt"
    if a.exists():
        sh([PY, "readpool.py", "--pool", "data/pool/test_let", "--out", "data/reads/let",
            "--adapters", f"{name}_{n}=data/runs/{name}/adapter_{n}.pt"], f"let_{name}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-at", required=True)
    a = ap.parse_args()
    os.chdir(HERE)
    (HERE / "logs").mkdir(exist_ok=True)
    if not (HERE / "data/pool/test_let/meta.json").exists():
        sh([PY, "ctrl_pool.py", "--src", "data/pool/test", "--X", "6771", "--out", "data/pool/test_let"], "ctrl_let.log")
    if not (HERE / "data/reads/let/reads.jsonl").exists():
        sh([PY, "readpool.py", "--pool", "data/pool/test_let", "--out", "data/reads/let",
            "--adapters", "0=data/runs/pilot/adapter.pt"], "let_0.log")
    seed("kl0_s", 0, a.stop_at)
    placebo("kl0_s0")
    o = outcome("kl0_s0", 1200)
    print(f"[chain] seed 0 outcome {o} at read {last_read('kl0_s0')} {time.strftime('%T')}", flush=True)
    if o == "TAKE-OFF":
        todo = [("kl0_s", 1, ()), ("kl0_s", 2, ())]
    elif o == "FLAT":
        todo = [("posctl_d086_s", 0, ("--steps", "300", "--cue-delta", "0.86")), ("kl0_s", 1, ())]
    else:
        todo = []
    for pre, s, extra in todo:
        if clock_passed(a.stop_at):
            print(f"[chain] clock {a.stop_at} passed; not starting {pre}{s}", flush=True)
            break
        seed(pre, s, a.stop_at, extra)
        placebo(f"{pre}{s}")
        print(f"[chain] {pre}{s} outcome {outcome(f'{pre}{s}', 300 if pre.startswith('posctl') else 1200)} "
              f"{time.strftime('%T')}", flush=True)
    print(f"[chain] done {time.strftime('%T')}", flush=True)


if __name__ == "__main__":
    main()
