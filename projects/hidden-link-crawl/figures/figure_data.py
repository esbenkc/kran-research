"""Write the main figure's data.json (blog.kran.ai/assets/figures/hidden-link-crawl/data.json) from results/analysis.json.

One row per combo (fetch-tool setup x agent x link words) with its own runs, grouped by why the rule predicts what it
predicts. Exposed fitting-link combos with 2-3 runs each and the no-route zero combos are pooled into one row each.
Run from the project folder: python3 figures/figure_data.py [out.json]
"""
import json
import math
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
AN = json.loads((P / "results/analysis.json").read_text())
assert AN["phase"] == "counted"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else P.parents[2] / "blog.kran.ai/assets/figures/hidden-link-crawl/data.json"

AGENT = {"gpt-5.4-mini": "mini", "gpt-5.4-nano": "nano", "gemini-3.1-flash-lite": "Flash-Lite"}
ORDER = {"gpt-5.4-mini": 0, "gpt-5.4-nano": 1, "gemini-3.1-flash-lite": 2}
TOOL = {"mcpfetch": "MCP fetch", "mcpraw": "MCP raw mode", "crawl4ai": "Crawl4AI", "browseruse": "Browser Use",
        "turndown": "Turndown"}
TORDER = ["mcpfetch", "turndown", "mcpraw", "crawl4ai", "browseruse"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def verdict(c):
    return "miss" if c["miss"] else "off" if c["off"] else "hit" if c["hit"] else "unsettled"


def row(c, group):
    lo, hi = wilson(c["followed"], c["n"])
    tool = c["context"].split("/")[0]
    off = " \u00b7 off-task" if group == "interior" and c["anchor"] == "nonfit" else ""
    return {"group": group, "label": f'{TOOL[tool]} \u00b7 {AGENT[c["agent"]]}{off}', "tool": tool, "agent": c["agent"],
            "anchor": c["anchor"], "n": c["n"], "followed": c["followed"], "measured": round(c["measured"], 4),
            "lo": round(lo, 4), "hi": round(hi, 4), "pred": round(c["pred"], 4), "d": round(c["d"], 4),
            "verdict": verdict(c), "artifact": c["artifact"]}


CB = AN["combos"]
G = [("raw", "Fully hidden link: the default reply drops it, but the tool has a raw mode", "fitting link",
      lambda c: c["kind"] == "reachable_zero"),
     ("interior", "One-pixel or off-screen link: the main-text filter keeps it on 2 of 5 sites",
      "both kinds of link; agents also reached it through raw mode (MCP fetch) or a blog page the crawl did not read",
      lambda c: c["kind"] == "interior"),
     ("off", "Link in every first reply, off-task words", "off-task link",
      lambda c: c["kind"] == "exposed" and c["anchor"] == "nonfit"),
     ("fit", "Link in every first reply, fitting words", "fitting link",
      lambda c: c["kind"] == "exposed" and c["anchor"] == "fit" and c["n"] >= 20)]
rows, used = [], set()
for gid, _, _, pick in G:
    sel = sorted([c for c in CB if pick(c)], key=lambda c: (c["anchor"] != "fit" if gid == "interior" else 0,
                                                            TORDER.index(c["context"].split("/")[0]), ORDER[c["agent"]]))
    for c in sel:
        rows.append(row(c, gid))
        used.add(id(c))
# pooled rows: exposed fitting-link combos with 2-3 runs; no-route zeros
small = [c for c in CB if c["kind"] == "exposed" and c["anchor"] == "fit" and c["n"] < 20]
zero = [c for c in CB if c["kind"] == "zero"]
assert len(used) + len(small) + len(zero) == len(CB) == AN["n_combos"]


def pooled(cs, group, label):
    k, n = sum(c["followed"] for c in cs), sum(c["n"] for c in cs)
    pred = sum(c["pred"] * c["n"] for c in cs) / n
    lo, hi = wilson(k, n)
    # the pooled row is judged on its pooled rate (within 10 points of the run-weighted rule); single small combos that
    # failed alone are counted in off_combos and named in the figure note
    v = "hit" if abs(k / n - pred) <= 0.10 else "miss"
    return {"group": group, "label": label, "n": n, "followed": k, "measured": round(k / n, 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "pred": round(pred, 4), "verdict": v, "combos": len(cs), "pooled": True,
            "off_combos": sum(c["off"] for c in cs)}


rows.append(pooled(small, "fit", f"{len({c['context'] for c in small})} other tools \u00b7 all 3"))
rows.append(pooled(zero, "zero", f"{len({c['context'] for c in zero})} setups \u00b7 all 3"))
OFF_ONLY = sum(c["off"] and not c["miss"] for c in CB)
data = {
    "title": "The crawl's rule missed where agents switched to raw mode or the fetch tool changed the layout",
    "subtitle": ("Share of runs that followed the hidden link. Tick = the rule's prediction "
                 "(share of pages whose first reply shows the link \u00d7 the agent's earlier rate). "
                 "Dot = measured, with its 95% interval."),
    "answer": (f"Answer to the title: no. The rule missed clearly {AN['miss']} times (the bar for yes was at most 1) "
               f"and less clearly {OFF_ONLY} more times. It still had the smallest average error:"),
    "groups": [{"id": g[0], "label": g[1], "link": g[2]} for g in G] + [{"id": "zero", "label": "No link in any reply and no raw mode", "link": "fitting link"}],
    "rows": rows,
    "status": AN["status"], "miss": AN["miss"], "off": AN["off"],
    "off_only": OFF_ONLY,
    "hit_share": [AN["hit_share_hits"], AN["n_hit_share"]],
    "mae": {k: round(v, 4) for k, v in AN["mae"].items()},
    "source": "kran-research projects/hidden-link-crawl/results/analysis.json via figures/figure_data.py",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1))
print("wrote", OUT, len(rows), "rows")
for r in rows:
    print(f'{r["group"]:9} {r["label"]:40} {r.get("anchor", ""):6} {r["followed"]:3}/{r["n"]:<3} pred {r["pred"]:.2f} {r["verdict"]}')
