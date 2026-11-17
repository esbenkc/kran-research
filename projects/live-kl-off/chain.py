"""The frozen run tree (PREREG.md 'Engine' + Felix's plan #22979), one heavy job at a time on the mini.

  0. placebo pool: ctrl_pool.py (' Let' 6771 written into the 2,800 xins twins, pilot recompute) -> data/pool/test_let
     and its read on the pilot (adapter 0) BEFORE seed 0 trains -> data/reads/let/
  1. seed 0 (run_seeds.py, thought KL off, early stop at the confirming read, clock 20:00 PST)
  2. branch read in code on seed 0: onset_stop.json = TAKE-OFF -> seeds 1, 2
                                    reached 1,200 without it (FLAT or EDGE) -> positive control (cue 0.86, 300 updates,
                                    reads every 50), seed 1, seed 2
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
    ap.add_argument("--r-build-by", default="19:22", help="D4: cell R pool build starts only before this clock")
    ap.add_argument("--r-start-by", default="19:30", help="D4: cell R training starts only before this clock")
    a = ap.parse_args()
    os.chdir(HERE)
    (HERE / "logs").mkdir(exist_ok=True)
    if not (HERE / "data/pool/test_let/meta.json").exists():
        sh([PY, "ctrl_pool.py", "--src", "data/pool/test", "--X", "6771", "--out", "data/pool/test_let"], "ctrl_let.log")
    if not (HERE / "data/reads/let/reads.jsonl").exists():
        sh([PY, "readpool.py", "--pool", "data/pool/test_let", "--out", "data/reads/let",
            "--adapters", "0=data/runs/pilot/adapter.pt"], "let_0.log")
    # D3: this process replaced the first chain while seed 0 trained; wait for that seed's driver to end first
    while subprocess.call(["pgrep", "-f", "run_seeds.py --seeds 0 --stop-at"], stdout=subprocess.DEVNULL) == 0:
        time.sleep(10)
    seed("kl0_s", 0, a.stop_at)
    placebo("kl0_s0")
    o = outcome("kl0_s0", 1200)
    if o == "FLAT":  # D3 (Vera #23028): served read on seed 0's own fresh thoughts, reported, never a gate
        sh([PY, "served.py", "--policy", "data/runs/kl0_s0/adapter_1200.pt", "--R", "2",
            "--out", "data/reads/served_kl0_s0_1200.npz"], "served_kl0_s0_1200.log")
    print(f"[chain] seed 0 outcome {o} at read {last_read('kl0_s0')} {time.strftime('%T')}", flush=True)
    # PREREG 'Frozen schedule tree': TAKE-OFF -> seeds 1, 2; reached 1,200 without take-off (FLAT or EDGE) ->
    # positive control (cue 0.86, 300 updates, reads every 50), seed 1, seed 2; cut by the clock -> nothing new
    if o == "TAKE-OFF":
        todo = [("kl0_s", 1, ()), ("kl0_s", 2, ())]
    elif o == "FLAT":
        todo = [("posctl_d086_s", 0, ("--steps", "300", "--cue-delta", "0.86", "--read-every", "50")),
                ("cellR", 0, ()), ("kl0_s", 1, ()), ("kl0_s", 2, ())]
    else:
        todo = []
    for pre, s, extra in todo:
        if pre == "cellR":  # D4 (Iris #23033/#23039, Felix #23037): replay of seed 0's own fresh rows, cached engine
            if clock_passed(a.r_build_by):
                print(f"[chain] cell R skipped: past {a.r_build_by} before its pool build", flush=True)
                continue
            rc = sh([PY, "build_replay.py", "--run", "kl0_s0", "--upto", "572", "--out", "data/pool/replay"],
                    "build_replay.log")
            if rc != 0 or clock_passed(a.r_start_by):
                print(f"[chain] cell R not started (build rc {rc}, start-by {a.r_start_by})", flush=True)
                continue
            sh([PY, "cached.py", "--name", "replay_s0", "--off", "data/pool/replay_off", "--on", "data/pool/replay_on",
                "--test", "data/pool/test", "--seed", "0", "--steps", "1500", "--no-std", "--lam", "0.42",
                "--read-early", "100", "--read-every", "100", "--stop-at", a.stop_at], "replay_s0.log")
            print(f"[chain] cell R ended {time.strftime('%T')}; no seed after it before the clock", flush=True)
            break
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
