"""Serial job chain (one heavy job at a time, PREREG s6). Replaces chain.sh after its first step: the chain.sh bash
died after launching step 1 (its python child kept running), so this waits for that pid and runs the rest.

    nohup env/.venv/bin/python kran-research/projects/sink-value/chain2.py [wait_pid] > .../logs/chain2.log 2>&1 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = os.path.abspath(HERE / "../../../env/.venv/bin/python")   # no resolve(): the venv python is a symlink
C = "grader_s0,grader_s1,monitor_s0,monitor_s1"
STEPS = [
    ["run_patch.py", "--dev", "cpu", "--prec", "fp32", "--rows", "status", "--copies", C, "--threads", "6"],
    ["r6_split.py", "--dev", "cpu"],
    ["attn_onoff.py"],
    ["research_checks/head_attention.py"],
    ["run_patch.py", "--dev", "mps", "--prec", "bf16", "--rows", "reading", "--copies", C],
    ["run_patch.py", "--dev", "mps", "--prec", "bf16", "--rows", "status", "--copies", C],
    ["run_patch.py", "--dev", "cpu", "--prec", "fp32", "--rows", "reading", "--copies", C, "--threads", "6"],
    ["run_patch.py", "--dev", "mps", "--prec", "bf16", "--rows", "r4", "--copies", C],
]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if len(sys.argv) > 1:
    pid = int(sys.argv[1])
    print(time.strftime("%T"), "waiting for pid", pid, flush=True)
    while alive(pid):
        time.sleep(5)
for st in STEPS:
    if Path(HERE / ".stop_chain").exists():
        print("stop file found", flush=True)
        break
    print("===", time.strftime("%T"), " ".join(st), flush=True)
    cwd = HERE.parent.parent.parent if st[0].startswith("research_checks") else HERE
    script = str(HERE / st[0])
    r = subprocess.run([PY, script] + st[1:], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in r.stdout.splitlines():
        if not any(x in line for x in ("Warning", "Fetching", "redirects")):
            print(line, flush=True)
    print("=== exit", r.returncode, time.strftime("%T"), flush=True)
    if r.returncode != 0 and st[0] == "run_patch.py":
        print("=== stop: a patch step failed", flush=True)
        break
print("=== chain2 done", time.strftime("%T"), flush=True)
