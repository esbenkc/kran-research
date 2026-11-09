"""Summarise crawl/exposure.jsonl per tool x row: exposure share, anchor-text share, route-page share, click landing,
distinct normalized outputs. .venv/bin/python crawl_summary.py [crawl_dir]"""
import json, sys, collections
d = sys.argv[1] if len(sys.argv) > 1 else "crawl"
rows = [json.loads(l) for l in open(f"{d}/exposure.jsonl")]
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["tool"], r["row"])].append(r)
out = {}
for (tool, row), v in sorted(agg.items()):
    n = len(v)
    s = lambda k: sum(bool(x.get(k)) for x in v)
    out[f"{tool}|{row}"] = o = {"n": n, "exposed": s("exposed"), "home": s("home_handle"), "table": s("table_handle"),
         "anchor_seen": s("anchor_text_seen"), "click_fail": sum(1 for x in v if x.get("home_click_landed") is False or x.get("table_click_landed") is False),
         "reachable": s("reachable"), "reach_kinds": sorted({k for x in v for k in (x.get("home_reach") or []) + (x.get("table_reach") or [])}),
         "truncated": s("home_truncated_note") + s("table_truncated_note"),
         "errors": sum(1 for x in v if x.get("error")),
         "distinct_home": len({x.get("home_norm_sha") for x in v}), "distinct_table": len({x.get("table_norm_sha") for x in v})}
    print(f"{tool:16s} {row:28s} " + " ".join(f"{k}={val}" for k, val in o.items()))
json.dump(out, open(f"{d}/summary.json", "w"), indent=1)
