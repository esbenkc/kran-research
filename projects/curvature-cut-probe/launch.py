"""Detached launcher: starts ONE python job in its own session + a PID-bound footprint watchdog.
Usage: python3 launch.py <log> <cap_mb> [ENV=VAL ...] -- <script> <args...>
The job runs with ../venv1024/bin/python (absolute path) and cwd = this folder."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT.parent / "venv1024" / "bin" / "python")
log, cap = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
cut = rest.index("--")
env = dict(os.environ)
for kv in rest[:cut]:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = [PY] + rest[cut + 1:]
f = open(ROOT / log, "w")
p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
wlog = str(ROOT / (log.replace(".log", "") + ".memwatch.log"))
open(wlog, "w").close()
subprocess.Popen([str(ROOT / "watchdog.sh"), str(p.pid), cap, wlog], cwd=ROOT, start_new_session=True,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid)
