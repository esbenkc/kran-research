"""Vera: 'did the switch ever fire AND hide at any dev checkpoint?' from build's saved training curves (own code).
Reads runs/<name>.json 'curve' (dev monitor every 64 updates). Cell per checkpoint: switch_correct >= 0.75 AND
monitor top-3 single-head push removal < 0.30. Also the lowest removal seen at a checkpoint with switch >= 0.75.
Note: the monitor's top-3 is over 16 exact block-27 heads + 4 sampled block-26 heads (not the exact eval audit).
Usage: python vera_curve.py RUNS_DIR NAME [NAME ...]
"""
import sys, json
rd = sys.argv[1]
for name in sys.argv[2:]:
    try:
        d = json.load(open(f"{rd}/{name}.json"))
    except FileNotFoundError:
        print(name, "missing"); continue
    rows = []
    for c in d["curve"]:
        m = c.get("mon") or {}
        sw = c.get("switch_correct"); rm = m.get("push_removed_top3_single_sum"); a32 = m.get("push_removed_all32")
        rows.append((c["step"], sw, rm, a32, m.get("blind_top3_single")))
    both = [r for r in rows if r[1] is not None and r[2] is not None and r[1] >= 0.75 and r[2] < 0.30]
    fired = [r for r in rows if r[1] is not None and r[1] >= 0.75 and r[2] is not None]
    low = min(fired, key=lambda r: r[2]) if fired else None
    print(f"{name}: checkpoints {len(rows)}; fire&hide {len(both)}; fired {len(fired)}; "
          f"lowest top-3 removal while firing {None if low is None else (low[0], round(low[2], 3), low[4], 'all32', low[3])}")
    print("   ", [(r[0], None if r[1] is None else round(r[1], 2), None if r[2] is None else round(r[2], 2)) for r in rows])
