"""Serial fine-tune queue, ONE heavy job at a time (PREREG s4 order), each with a PID-bound footprint watchdog
(kill at 5000 MB, the 5 GB swarm rule while another swarm is open), then the light analysis.
No run STARTS after LATEST (Copenhagen time, PREREG s4: 18:20). Before the first non-unproj run the chain waits for
data/floor/GATE_POSTED (the power gate is posted before P s0, PREREG s6; Felix #6093).
Usage (detached): python3 run_chain.py unproj:0:50 proj:0 rand:0 proj:1 rand:1 proj:2   (probe:seed[:steps], default 300;
 data/HORIZON caps every run at that step, D8)
Log: data/logs/chain.log (START / DONE / FAILED / SKIP lines). Per run: data/logs/attack_<run>.log + .memwatch.log"""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PY = str(ROOT.parent / "venv1024" / "bin" / "python")
LATEST = os.environ.get("LATEST", "18:20")
CAP = os.environ.get("CAP_MB", "5000")
# memory-only settings (same gradient): KL micro-batches 2 statements / 1 window, probe micro-batches 2 twin pairs; the
# MPS cache is emptied after every micro-batch and the allocator watermarks are tightened (PHASE_EMPTY + watermarks)
ENV = dict(os.environ)
for k, v in {"MB_CITIES": "2", "MB_RETAIN": "1", "MB_PROBE": "2", "PHASE_EMPTY": "1",
             "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.30", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.40"}.items():
    ENV.setdefault(k, v)
LOG = ROOT / "data" / "logs" / "chain.log"


def say(msg):
    line = f"{datetime.datetime.now():%H:%M:%S} {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    wp = os.environ.get("WAITPID")
    if wp:                                    # wait for a run started by an earlier chain (same queue, re-ordered)
        say(f"WAIT pid {wp}")
        while subprocess.run(["kill", "-0", wp], capture_output=True).returncode == 0:
            time.sleep(10)
    for spec in sys.argv[1:]:
        probe, seed, *st = spec.split(":")
        steps = st[0] if st else "300"
        name = f"{probe}__lam10000__seed{seed}"
        if (ROOT / "data" / "runs" / name / "log.json").exists():
            say(f"HAVE {name}")
            continue
        now = datetime.datetime.now(ZoneInfo("Europe/Copenhagen")).strftime("%H:%M")
        if now > LATEST:
            say(f"SKIP {name} (Copenhagen {now} > {LATEST})")
            continue
        gate = ROOT / "data" / "floor" / "GATE_POSTED"
        if probe != "unproj" and not gate.exists():
            say(f"WAIT gate file {gate.relative_to(ROOT)} before {name} (PREREG s6)")
            while not gate.exists():
                time.sleep(10)
            say("GATE_POSTED seen")
            now = datetime.datetime.now(ZoneInfo("Europe/Copenhagen")).strftime("%H:%M")
            if now > LATEST:
                say(f"SKIP {name} (Copenhagen {now} > {LATEST})")
                continue
        say(f"START {name}")
        with open(ROOT / "data" / "logs" / f"attack_{name}.log", "w") as f:
            p = subprocess.Popen([PY, "attack_proj.py", "--probe", probe, "--seed", seed, "--gc", "--steps", steps], cwd=ROOT, env=ENV,
                                 stdout=f, stderr=subprocess.STDOUT)
        wlog = ROOT / "data" / "logs" / f"attack_{name}.memwatch.log"
        open(wlog, "w").close()
        w = subprocess.Popen([str(ROOT / "watchdog.sh"), str(p.pid), CAP, str(wlog)], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rc = p.wait()
        w.wait()
        if rc != 0:
            say(f"FAILED {name} rc={rc}")
            sys.exit(1)
        a = subprocess.run([PY, "analysis.py"], cwd=ROOT, capture_output=True, text=True)
        (ROOT / "data" / "logs" / "analysis.log").write_text(a.stdout + a.stderr)
        say(f"DONE {name}" + ("" if a.returncode == 0 else " (analysis failed)"))
        time.sleep(2)
    say("CHAIN_DONE")


if __name__ == "__main__":
    main()
