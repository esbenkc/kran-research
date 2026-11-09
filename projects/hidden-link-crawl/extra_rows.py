"""Reported rows asked after the result (Tove #17938), no title weight. (a) law-by-view: off-task combos against the
parent's own raw-HTML and markdown rates (PREREG s8.6). (b) when mcp-server-fetch agents first call fetch(raw=true):
before or after they fetched the visible table page. .venv/bin/python extra_rows.py > results/extra_rows.txt"""
import json, glob, collections
from analysis import newcombe
A = json.load(open("results/analysis.json"))
VIEW = {"raw HTML": {"gpt-5.4-nano": (0, 30), "gemini-3.1-flash-lite": (5, 30), "gpt-5.4-mini": (25, 30)},
        "markdown": {"gpt-5.4-nano": (2, 29), "gemini-3.1-flash-lite": (3, 30), "gpt-5.4-mini": (19, 30)}}
print("(a) off-task combos vs the parent's per-view off-task rate (PREREG s8.6 numbers)")
for x in A["combos"]:
    if x["anchor"] == "nonfit" and x["n"] >= 30:
        for view, rates in VIEW.items():
            k, n = rates[x["agent"]]
            d, lo, hi = newcombe(x["followed"], x["n"], k, n, x["exposure"])
            print(f"  {x['context']:22s} {x['agent']:22s} {x['followed']}/{x['n']} vs parent {view} {k}/{n}: "
                  f"d {d:+.2f} [{lo:+.2f},{hi:+.2f}]{'  <- own view' if view == x['format'] else ''}")
print("(b) mcpfetch runs: first fetch(raw=true) before/after the table page was fetched; followed?")
c = collections.Counter()
for f in sorted(glob.glob("raw/counted/*/mcpfetch__*/*.json")):
    r = json.load(open(f))
    route, seen, first_raw = r["site"]["route_path"], False, "never"
    for a in r["actions"]:
        if a.get("tool") != "fetch":
            continue
        if (a.get("mcp_args") or {}).get("raw"):
            first_raw = "after_table" if seen else "before_table"
            break
        if route in str(a.get("url", "")):
            seen = True
    ctx = "reachable_zero(display_none)" if r["site"]["method"] == "display_none" else "interior(sr/off)"
    c[(ctx, first_raw, "followed" if r["harness_followed"] else "not")] += 1
for k, v in sorted(c.items()):
    print("  ", *k, v)
