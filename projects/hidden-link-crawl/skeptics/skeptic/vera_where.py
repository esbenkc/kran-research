# Where did agents first SEE the run address, and from which call? Per context x template: follows and first-sighting page.
import json, collections, re
import vera_rebuild as R
from pathlib import Path
out = collections.defaultdict(lambda: collections.Counter())
for (ctx, model, anc), rs in sorted(R.combos.items()):
    if ctx not in ("turndown/sr+off", "mcpfetch/sr+off", "mcpfetch/display_none"): continue
    for r in rs:
        d = json.load(open(R.PROJ / r["file"])); hid = d["site"]["hid"]
        obs = [t["text"] for t in d["transcript"] if t.get("role") == "observation"]
        acts = d["actions"]
        first = None
        for a, o in zip(acts, obs):
            if hid in o:
                pg = a.get("url", "").rsplit("/", 1)[-1]
                args = a.get("args", {})
                flag = "raw" if isinstance(args, dict) and args.get("raw") else ("start_index" if isinstance(args, dict) and args.get("start_index") else "default")
                first = f"{pg}:{flag}"; break
        out[(ctx, r["template"])][(first or "never", r["followed"])] += 1
for k in sorted(out):
    print(k, dict(out[k]))
