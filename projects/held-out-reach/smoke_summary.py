"""Per tool x model: runs, follows, steps, $/run, s/run, which actions showed the address. .venv/bin/python smoke_summary.py [phase]"""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
phase = sys.argv[1] if len(sys.argv) > 1 else "smoke"
man = [json.loads(l) for l in (ROOT / "raw/manifest.jsonl").read_text().splitlines() if l.strip()]
man = [m for m in man if m["phase"] == phase]
log = (ROOT / "logs/access.log").read_text()
agg = collections.defaultdict(lambda: collections.Counter())
acts = collections.defaultdict(collections.Counter)
for m in man:
    rec = json.loads((ROOT / m["raw_file"]).read_text())
    k = (m["cell"], m["model"])
    a = agg[k]
    a["n"] += 1
    a["follow_log"] += m["hidden_path"] in log
    a["steps"] += rec["steps_used"]
    a["cost_milli"] += rec["cost_usd_est"] * 1000
    a["secs"] += m["finished"] - m["created"]
    a["answered"] += rec["answer"] is not None
    a["parse_fail"] += rec["parse_fail"]
    for x in rec["actions"]:
        acts[k][x.get("tool")] += 1
        if x.get("hid_in_obs"):
            a["hid_seen"] += 1
            acts[k]["HID<-" + str(x.get("tool"))] += 1
            break
for k in sorted(agg):
    a = agg[k]; n = a["n"]
    print(f"{k[0]:28s} {k[1]:22s} n={n} follow={a['follow_log']} hidseen={a['hid_seen']} steps={a['steps']/n:.1f} "
          f"$={a['cost_milli']/n/1000:.4f} s={a['secs']/n:.0f} pf={a['parse_fail']} | {dict(acts[k])}")
