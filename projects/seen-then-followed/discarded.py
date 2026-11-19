"""List every site folder that has no saved run record: attempts that never finished (no model or outcome involved
in choosing them). Run after all runs stop.  .venv/bin/python discarded.py -> results/discarded_attempts.json

Boundaries (unix time, from the process start times):
  GATE_T0     22:44:33 CET  first gate run; folders before it = test_harness.py self-test sites
  MAIN1_STOP  22:52:20 CET  the first main process was restarted ~7 s after start (service mode switch)
  MAIN2_START 23:08:41 CET  the main run restarted with 6 lcpw lanes (DEVIATIONS D5)
  MAIN3_START 00:01:22 CET  the main run restarted with the k gate (DEVIATIONS D6)
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE_T0, MAIN1_STOP, MAIN2_START, MAIN3_START = 1795038273, 1795038740, 1795039721, 1795042882


def main() -> None:
    os.chdir(ROOT)
    tokens = {json.load(open(f))["site"]["token"] for f in glob.glob("raw/*/*/*/*.json")}
    log = [l.split("\t") for l in Path("logs/access.log").read_text().splitlines()]
    out = []
    for d in sorted(os.listdir("site"), key=lambda d: os.path.getmtime("site/" + d)):
        if d in tokens or d.startswith("crawl"):  # crawl.py sites (model-free, results/crawl.json)
            continue
        t = os.path.getmtime("site/" + d)
        hid = [f[:-5] for f in os.listdir(f"site/{d}/r")][0]
        reqs = [l[3] for l in log if len(l) > 4 and d in l[3]]
        reason = ("test_harness.py self-test site" if t < GATE_T0 else
                  "in flight at the 22:52 CET restart (service mode switch)" if t < MAIN1_STOP else
                  "in flight at the 23:08 CET stop (DEVIATIONS D5)" if t < MAIN2_START else
                  "in flight at the 00:01 CET stop (DEVIATIONS D6)" if t < MAIN3_START else
                  "in flight when a later process stopped (time stop or end)")
        out.append({"token": d, "hid": hid, "site_built_utc": dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(timespec="seconds"),
                    "requests_logged": len(reqs), "requested_id": any(hid in r for r in reqs), "reason": reason})
    (ROOT / "results" / "discarded_attempts.json").write_text(json.dumps(out, indent=1))
    for k, v in collections.Counter((o["reason"], o["requested_id"]) for o in out).items():
        print(v, k)


if __name__ == "__main__":
    main()
