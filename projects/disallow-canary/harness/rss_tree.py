"""Log the summed RSS of a process tree (root pid and all descendants) every 60 s until the root ends.
    python harness/rss_tree.py ROOT_PID LOGFILE"""
import os, subprocess, sys, time
root, log = sys.argv[1], sys.argv[2]
while True:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.split("\n")
    kids = {}
    for line in out:
        if line.strip():
            p, pp, r = line.split()
            kids.setdefault(pp, []).append((p, int(r)))
    alive = any(p == root for v in kids.values() for p, _ in v)
    if not alive:
        break
    tot, n, stack = 0, 0, [root]
    while stack:
        q = stack.pop()
        for p, r in kids.get(q, []):
            tot += r; n += 1; stack.append(p)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} root={root} procs={n} tree_rss_mb={tot/1024:.1f}\n")
    time.sleep(60)
