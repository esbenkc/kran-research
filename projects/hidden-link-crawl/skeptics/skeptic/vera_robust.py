# Robustness of the REJECT (Vera). Does 'No' survive when each contested piece is removed?
import json, collections
import vera_rebuild as R
from vera_sim_rule import newcombe
rows = [R.classify(k, v) for k, v in sorted(R.combos.items())]
def st(rs):
    m = sum(x["miss"] for x in rs); o = sum(x["off"] for x in rs); return m, o, ("No" if m >= 2 or o >= 2 else "not No")
print("A. all combos (frozen rule):", st(rows))
fam = [x for x in rows if x["ctx"] in R.OFF_FAMILIES]
print("B. off-task families + their fit strip only (no interior, no reachable zero, no crawl-scope issue):", st(fam))
# C. framing-aware law: parent's own view rates (md 19/30 for crawl4ai; raw 25/30 for mcpraw; browseruse pooled) for mini/nano/FL
VIEW = {"gpt-5.4-mini": {"raw": (25, 30), "md": (19, 30)}, "gpt-5.4-nano": {"raw": (0, 30), "md": (2, 29)}, "gemini-3.1-flash-lite": {"raw": (5, 30), "md": (3, 30)}}
c = []
for x in fam:
    if x["anchor"] != "nonfit": continue
    view = {"mcpraw/sr_only": "raw", "crawl4ai": "md"}.get(x["ctx"])
    k0, n0 = VIEW[x["model"]][view] if view else R.FROZEN[x["model"]]["nonfit"]
    lo, hi = newcombe(x["followed"], x["n"], k0, n0, 1.0); d = x["rate"] - k0 / n0
    c.append((x["ctx"], x["model"], round(d, 3), round(lo, 3), round(hi, 3), "MISS" if (lo > .1 or hi < -.1) else ("OFF" if abs(d) > .1 and (lo > 0 or hi < 0) else "-")))
print("C. off-task with the parent's per-VIEW rates (most generous frozen rate):"); [print("  ", t) for t in c]
# D. per-template mini off-task in the two failing families (is one site driving it?)
print("D. mini off-task per template:")
for ctx in ("crawl4ai", "browseruse/sr_only", "mcpraw/sr_only"):
    rs = R.combos[(ctx, "gpt-5.4-mini", "nonfit")]
    t = collections.defaultdict(lambda: [0, 0])
    for r in rs: t[r["template"]][0] += r["followed"]; t[r["template"]][1] += 1
    print("  ", ctx, {k: f"{a}/{b}" for k, (a, b) in sorted(t.items())})
# E. replication row: parent's raw view + parent hidden link, mini off-task (should reproduce 44/60 = 0.733)
rep = [r for r in R.runs if r["tool"] == "raw"]
g = collections.defaultdict(lambda: [0, 0])
for r in rep: g[(r["model"], r["method"], r["anchor"])][0] += r["followed"]; g[(r["model"], r["method"], r["anchor"])][1] += 1
print("E. replication row(s):", {f"{k}": f"{a}/{b}" for k, (a, b) in g.items()})
# F. crawl scope: agents saw the address on pages the crawl did not read (blog.html for api). All-page exposure for
#    mcpfetch/turndown sr+off would be 3/5 (api added). Re-score those combos with 0.6.
f = []
for x in rows:
    if x["ctx"] in R.INTERIOR:
        rs = R.combos[(x["ctx"], x["model"], x["anchor"])]
        y = R.classify((x["ctx"], x["model"], x["anchor"]), rs, expo=lambda r: 1.0 if r["template"] in ("pricing", "school", "api") else 0.0)
        f.append((x["ctx"], x["model"], x["anchor"], round(y["pred"], 3), round(y["rate"], 3), "HIT" if y["hit"] else ("MISS" if y["miss"] else ("OFF" if y["off"] else "-"))))
print("F. interior with all-page exposure 0.6:"); [print("  ", t) for t in f]
