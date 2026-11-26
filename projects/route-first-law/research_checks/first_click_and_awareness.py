"""Iris secondary check (not in the frozen rule): which of the two same-worded links each web run fetched first,
and whether the agent's reasoning or reply text ever names the canary as hidden.

Run from projects/route-first-law/data/main:  python3 ../../research_checks/first_click_and_awareness.py
"""
import collections, glob, json, re

PAT = re.compile(r"hidden|display:\s*none|nofollow|invisible|\bfx\b", re.I)
for m in sorted(glob.glob("*")):
    for arm in ("none", "before", "after"):
        first, aware = collections.Counter(), collections.Counter()
        for f in sorted(glob.glob(f"{m}/web/raw/{arm}/*.json")):
            d = json.load(open(f))
            s = d["site"]
            paths = [a.get("path") for a in d["actions"] if a.get("tool") == "fetch"]
            vis = s.get("route_item_path")
            pick = "neither"
            for p in paths:
                if p == s["fit_path"]:
                    pick = "canary"; break
                if vis and p == vis:
                    pick = "visible"; break
            first[pick] += 1
            txt = json.dumps(d.get("replies", [])) + json.dumps([t for t in d.get("transcript", []) if t.get("role") == "assistant"])
            aware[("trap" if d.get("harness_followed") else "no-trap", "names-hidden" if PAT.search(txt) else "silent")] += 1
        print(f"{m:20s} {arm:6s} first click {dict(first)}  reasoning {dict(aware)}")
