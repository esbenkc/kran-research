"""N* = the update count of the passing gate look in data/runs/planted_s0/gate.jsonl (PREREG v1 'Gate').
Prints N* and exits 0 if the gate passed; prints FAIL and exits 1 otherwise.
  /usr/bin/python3 nstar.py [--fits-by HH:MM --spu 16.3 --extra-s S]  (exit 2 if N* updates + extra no longer fit)"""
import json, sys, time
from pathlib import Path
rows = [json.loads(l) for l in open(Path(__file__).parent / "data/runs/planted_s0/gate.jsonl")]
ok = [r for r in rows if r["pass"]]
if not ok:
    print("FAIL"); sys.exit(1)
n = ok[0]["step"]
if "--fits-by" in sys.argv:
    hh, mm = (int(x) for x in sys.argv[sys.argv.index("--fits-by") + 1].split(":"))
    spu = float(sys.argv[sys.argv.index("--spu") + 1]); extra = float(sys.argv[sys.argv.index("--extra-s") + 1])
    lt = time.localtime(); end = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    if time.time() + n * spu * 1.08 + extra > end:
        print("CUT", n); sys.exit(2)
print(n)
