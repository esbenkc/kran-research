"""Descriptive (Tove #9068): in each scored use, which secret was presented and, for the storage key, which bucket
the call named: this episode's own bucket, the record's bucket (static record: the fixed one), or another.
Run: python use_detail.py  -> results/use_detail.json"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from env import FIXED_FIELDS
from scorer import BAITS, load_episode, score_episode

PROJECT = Path(__file__).resolve().parent
out: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))
examples: dict[str, list] = defaultdict(list)
for p in sorted((PROJECT / "raw").glob("*/*/*.jsonl")):
    ep = load_episode(p)
    if ep["arm"] not in ("static", "stateful") or ep["model"] == "random" or int(ep["seed"]) >= 900:
        continue
    sc = score_episode(ep)
    if not sc["use_e"]:
        continue
    meta = ep["episode"]
    first = sc["first_touch_idx"]
    for a in ep["actions"][first + 1:]:
        argt = json.dumps(a.get("args", {}), sort_keys=True)
        hit = [b for b in BAITS if meta["canaries"][b] in argt]
        if not hit:
            continue
        b = hit[0]
        row = out[ep["model"]][ep["arm"]]
        row[f"bait:{b}"] += 1
        row[f"tool:{a['tool']}"] += 1
        if b == "creds" and a["tool"] == "cloud_stat":
            bucket = str(a["args"].get("bucket", ""))
            where = ("own bucket" if bucket == meta["bucket"] else
                     "record bucket (fixed)" if bucket == FIXED_FIELDS["bucket"] else "other bucket")
            row[f"bucket:{where}"] += 1
        if len(examples[f"{ep['model']}/{ep['arm']}"]) < 2:
            examples[f"{ep['model']}/{ep['arm']}"].append({"seed": ep["seed"], "call": {"tool": a["tool"], "args": a["args"]},
                                                         "observation": a["observation"][:160],
                                                         "path": str(p.relative_to(PROJECT))})
        break
res = {"counts": {m: {arm: dict(c) for arm, c in d.items()} for m, d in out.items()}, "examples": examples}
(PROJECT / "results" / "use_detail.json").write_text(json.dumps(res, indent=1, sort_keys=True))
for m, d in sorted(res["counts"].items()):
    for arm, c in sorted(d.items()):
        print(f"{m[:26]:27s} {arm:9s} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
tot: dict[str, Counter] = defaultdict(Counter)
for m, d in res["counts"].items():
    for arm, c in d.items():
        tot[arm].update(c)
for arm, c in sorted(tot.items()):
    n = sum(v for k, v in c.items() if k.startswith("bait:"))
    print(f"TOTAL {arm:9s} uses={n} " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
res["totals"] = {arm: dict(c) for arm, c in tot.items()}
(PROJECT / "results" / "use_detail.json").write_text(json.dumps(res, indent=1, sort_keys=True))
