"""Every 60 s: summed RSS of this room's process trees (run.py, bu_service.py, server.py and all descendants) ->
logs/rss.log. Kills run.py when the sum passes 4.5 GB (MEMORY rule: 5 GB per swarm)."""
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIMIT_MB = 4500
while True:
    rows = [l.split(None, 3) for l in subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,command="], capture_output=True,
                                                      text=True).stdout.splitlines()]
    procs = {int(r[0]): (int(r[1]), int(r[2]), r[3] if len(r) > 3 else "") for r in rows if len(r) >= 3}
    roots = [p for p, (_, _, c) in procs.items() if "python" in c and any(k in c for k in ("run.py counted", "bu_service.py --port 878", "server.py --port 8765"))]
    kids = {}
    for p, (pp, _, _) in procs.items():
        kids.setdefault(pp, []).append(p)
    seen, stack = set(), list(roots)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack += kids.get(p, [])
    mb = sum(procs[p][1] for p in seen if p in procs) / 1024
    run_pids = [p for p in roots if "run.py" in procs[p][2] and "zsh" not in procs[p][2]]
    with (ROOT / "logs" / "rss.log").open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} procs={len(seen)} rss_mb={mb:.0f}\n")
    if mb > LIMIT_MB:
        for p in run_pids:
            os.kill(p, signal.SIGTERM)
        with (ROOT / "logs" / "rss.log").open("a") as f:
            f.write(f"KILL run.py {run_pids} at {mb:.0f} MB\n")
    time.sleep(60)
