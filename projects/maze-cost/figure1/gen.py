"""Owain: build Figure 1 (assets/figures/maze-cost-fig1/figure1.html) from the real run records.

usage: python3 gen.py <project dir> <agent> <run index> <text.json> <out.html>

Every number and quote comes from the project's data:
  data/<agent>/raw/{maze,deadend}/<i>.json      the agent record (calls, tokens, its 'thought' text, actions)
  data/<agent>/access/{maze,deadend}/<i>.jsonl  the defender log (bytes, handler CPU per request)
  results/analysis.json                          the frozen-rule summary (asserted equal to our own recount)
Scoring uses Bruno's analysis.score_run (the frozen PREREG s.3 definitions), so the figure and the post agree.
"""
import html
import os
import json
import sys
from pathlib import Path

proj, agent, i, textf, outf = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
sys.path.insert(0, str(proj))
import analysis  # noqa: E402

T = json.loads(textf.read_text())
d = proj / "data" / agent
E = html.escape


def load(arm):
    rec = json.loads((d / "raw" / arm / f"{i:03d}.json").read_text())
    acc = [json.loads(x) for x in (d / "access" / arm / f"{i:03d}.jsonl").read_text().splitlines()]
    acc.sort(key=lambda a: a["t"])
    return rec, acc, analysis.score_run(rec, acc, agent)


mz, mz_acc, mzs = load("maze")
de, de_acc, des = load("deadend")
assert mz["site"]["template"] == de["site"]["template"] and mz["question"] == de["question"], "pairing broken"
assert mzs["entered"] and des["entered"]
NEU = os.environ.get("NEUTRAL", "1") == "1" and (d / "raw" / "maze_neutral" / f"{i:03d}.json").exists()
if NEU:
    nz, nz_acc, nzs = load("maze_neutral")
    assert nz["question"] == mz["question"] and nz["site"]["secret"] == mz["site"]["secret"], "neutral pairing broken"
    assert nzs["entered"]

# cross-check against analysis.json (the post's numbers)
an = json.loads(Path(os.environ.get("ANALYSIS", proj / "results" / "analysis.json")).read_text())
A = an["agents"][agent]
row = next(r for r in A["runs"]["maze"] if r["run_index"] == i)
for k in ("D", "S", "end", "def_bytes", "def_requests"):
    assert row[k] == mzs[k], (k, row[k], mzs[k])
assert abs(row["agent_usd"] - mzs["agent_usd"]) < 1e-12 and abs(row["def_usd"] - mzs["def_usd"]) < 1e-15
rowd = next(r for r in A["runs"]["deadend"] if r["run_index"] == i)
assert rowd["S"] == des["S"] and rowd["D"] == des["D"] and rowd["end"] == des["end"]
if NEU:
    rown = next(r for r in A["runs"]["maze_neutral"] if r["run_index"] == i)
    for k in ("D", "S", "end", "def_bytes", "def_requests"):
        assert rown[k] == nzs[k], (k, rown[k], nzs[k])

# all-run numbers for the text slots, from analysis.json only
NUM = {"D": None}
for key, ag in (("a", "gemma-4-31b-it"), ("b", "gemma-4-26b-a4b-it")):
    g = an["agents"][ag]
    er = g["end_reasons"]
    NUM[f"{key}_n"] = g["n_entered"]
    NUM[f"{key}_leave"] = er["answer"] + er["get_out"]
    NUM[f"{key}_cap"] = er["cap"]
    dd = g["dead_end"]
    NUM[f"{key}_dn"] = dd["n_entered"]
    NUM[f"{key}_dleave"] = dd["end_reasons"].get("answer", 0) + dd["end_reasons"].get("get_out", 0)
    NUM[f"{key}_ratio"] = f'{g["ratio_usd_median"]:,.0f}'
    NUM[f"{key}_dratio"] = f'{dd["ratio_usd_median"]:,.0f}'
    NUM[f"{key}_depth"] = g["median_depth"]
    mn = g.get("maze_neutral")
    if mn:
        NUM[f"{key}_nn"] = mn["n_entered"]
        NUM[f"{key}_nleave"] = mn["runs_leaving"]
NUM["m_n"] = NUM["a_n"] + NUM["b_n"]
NUM["m_leave"] = NUM["a_leave"] + NUM["b_leave"]
NUM["d_n"] = NUM["a_dn"] + NUM["b_dn"]
NUM["d_leave"] = NUM["a_dleave"] + NUM["b_dleave"]
NUM["p_n"] = NUM.get("a_nn", 0) + NUM.get("b_nn", 0)
NUM["p_leave"] = NUM.get("a_nleave", 0) + NUM.get("b_nleave", 0)
NUM["p_cap"] = NUM["p_n"] - NUM["p_leave"]
NUM["status"] = an["status"]
NUM["never"] = "never" if NUM["a_leave"] + NUM["b_leave"] == 0 else "almost never"
NUM["D"] = mzs["D"]
site = mz["site"]
sys.path.insert(0, str(proj / "web"))
import sites  # noqa: E402
tpl = sites.TEMPLATES[site["template"]]
anchor = tpl["fit_anchor"]
route_label = tpl["route"][1]


def thought(step):
    try:
        return json.loads(step["text"])["thought"]
    except Exception:
        return None


def episode(rec, sc):
    """Calls after the entry call up to the end event, with the depth each call sat at (as analysis.score_run)."""
    out, seen, entry = [], set(), None
    canary = rec["site"]["canary_path"]
    for s in rec["steps"]:
        act = s.get("action") or {}
        p = act.get("path")
        if entry is None:
            if p == canary and act.get("status") == 200:
                entry = s["step"]
                seen.add(p)
            continue
        out.append({"step": s["step"], "depth": len(seen), "tok": s["prompt_tokens"] + s["new_tokens"],
                    "thought": thought(s), "path": p, "tool": act.get("tool")})
        if act.get("tool") == "answer" or (p is not None and not p.startswith("/m/")):
            break
        if p is not None and act.get("status") == 200:
            seen.add(p)
    assert sum(c["tok"] for c in out) == sc["S"], (sum(c["tok"] for c in out), sc["S"])
    return out, entry


mcalls, mentry = episode(mz, mzs)
dcalls, dentry = episode(de, des)
D = mzs["D"]
mpages = [a for a in mz_acc if a.get("kind") == "maze"]
assert len({a["maze_id"] for a in mpages}) == D

# one maze page as the agent read it (raw HTML, depth 3 or the last page if shallower)
show_k = min(3, D)
pg = sites.Site(site["template"], "maze", secret=site["secret"])
pid = pg.id0
for _ in range(show_k - 1):
    pid = pg._next(pid)
    pg.depth[pid] = len(pg.depth) + 1
body = pg.render(f"/m/{pid}.html")[2].decode()
main = body.split("<main>")[1].split("</main>")[0].strip()
assert "lists no figures" in main and f"part {show_k}" in main


def fmt_usd(x):
    if x >= 0.001:
        return f"${x:.4f}".rstrip("0")
    s = f"{x:.10f}"
    nz = len(s.split(".")[1]) - len(s.split(".")[1].lstrip("0"))
    return f"${x:.{nz + 1}f}"


def n(x):
    return f"{x:,}"


# ---- maze lanes: one chip per maze page, a bar per page = tokens of the calls made on that page -------------------
def per_page_tokens(calls, depth):
    pp = {k: 0 for k in range(1, depth + 1)}
    for c in calls:
        if c["depth"] >= 1:
            pp[c["depth"]] += c["tok"]
    return pp


per_page = per_page_tokens(mcalls, D)
if NEU:
    ncalls, _ = episode(nz, nzs)
    per_page_n = per_page_tokens(ncalls, nzs["D"])
maxtok = max(list(per_page.values()) + (list(per_page_n.values()) if NEU else []))
BARH = 80


def chips_for(pp, depth, end, bar=True):
    out = []
    for k in range(1, depth + 1):
        t = pp[k]
        h = round(BARH * t / maxtok) if t else 0
        if t:
            lab = f'<div class="bv">{t / 1000:.1f}k</div>'
        else:
            lab = '<div class="bv none">no call<br>left</div>' if end == "cap" else ""
        out.append(f'<div class="pc"><div class="bw">{lab}<div class="bar" style="height:{h}px"></div></div>'
                   f'<div class="chip">part {k}</div></div>')
    return "".join(out)


chips_html = chips_for(per_page, D, mzs["end"])
if NEU:
    chips_n = chips_for(per_page_n, nzs["D"], nzs["end"])
    qn = [c for c in ncalls if c["thought"]][-1]

# quotes: the agent's own 'thought' on the first maze page and on its last page with a call
qa = next(c for c in mcalls if c["depth"] == 1)
qb = [c for c in mcalls if c["thought"]][-1]
qd = next(c for c in dcalls if c["depth"] == 1)


def ending(sc):
    k = sc["D"]
    return {"cap": T["end_cap"], "answer": T["end_answer"], "get_out": T["end_getout"]}[sc["end"]].format(
        k=k, pages="page" if k == 1 else "pages")


end_m, end_d = ending(mzs), ending(des)
end_n = ending(nzs) if NEU else None


def meter(sc):
    req = sc["def_requests"]
    what = f'{req} page{"s" if req != 1 else ""}' if req == sc["D"] else f"{req} requests"
    return (f'<div class="meter"><div class="mm"><div><span class="who">agent</span> <b>{n(sc["S"])} tokens</b> '
            f'<span class="usd">{fmt_usd(sc["agent_usd"])}</span></div>'
            f'<div><span class="who">site</span> <b>{what}, {sc["def_bytes"] / 1000:.1f} KB, {sc["def_cpu_ms"]:.1f} ms CPU</b> '
            f'<span class="usd">{fmt_usd(sc["def_usd"])}</span></div></div>'
            f'<div class="ratio">{T["ratio_word"]}<b>{sc["ratio_usd"]:,.0f}x</b></div></div>')


ROBOT = ('<svg class="bot" viewBox="0 0 40 40" width="52" height="52"><line x1="20" y1="3" x2="20" y2="9" stroke="#3b3e46" stroke-width="2"/>'
         '<circle cx="20" cy="3" r="2.6" fill="#e8743b"/><rect x="5" y="9" width="30" height="26" rx="8" fill="#fff" stroke="#3b3e46" stroke-width="2"/>'
         '<rect x="9" y="14" width="22" height="14" rx="5" fill="#25324a"/><circle cx="15" cy="21" r="3.2" fill="#7fd3ff"/><circle cx="25" cy="21" r="3.2" fill="#7fd3ff"/>'
         '<rect x="1.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/><rect x="34.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/></svg>')

code = (E(main).replace("Figures are published in a later part.", '<span class="anc">Figures are published in a later part.</span>')
        .replace(E(f'{anchor} (part {show_k + 1})'), f'<span class="nx">{E(anchor)} (part {show_k + 1})</span>')
        .replace("&lt;p&gt;&lt;a", "\n&lt;p&gt;&lt;a").replace("&lt;p&gt;This", "\n&lt;p&gt;This"))

NUM["calls_m"] = len(mcalls)
NUM["nruns"] = "Three" if NEU else "Two"
_dr = float("%.1g" % des["ratio_usd"])
NUM["d_ratio_ex"] = f"{_dr:,.0f}"
NUM["mult_ex"] = {2: "about doubled", 3: "about tripled"}.get(round(mzs["ratio_usd"] / des["ratio_usd"]), "multiplied by about %.0f" % (mzs["ratio_usd"] / des["ratio_usd"]))
_mb = [a["bytes"] for a in mpages]
_lo, _hi = f"{min(_mb) / 1000:.2f}", f"{max(_mb) / 1000:.2f}"
NUM["page_kb"] = _lo if _lo == _hi else f"{_lo}-{_hi}"
_mc = [a["cpu_ns"] / 1e6 for a in mpages]
NUM["page_ms"] = f"{min(_mc):.1f}-{max(_mc):.1f}"
T = {k: (v.format_map(NUM) if isinstance(v, str) and not k.startswith("end_") else v) for k, v in T.items()}
neutral_lane = (f'''<div class="lane hi2">
  <div class="lh"><span class="ln">{T["neu_name"]}</span><span class="ld">{T["neu_desc"]}</span></div>
  <div class="row">
    {ROBOT}
    <div class="bub"><span class="thk">{T["thought_label"]} {qn["depth"]}</span>&ldquo;{E(qn["thought"])}&rdquo;</div>
    <div class="act red side">&rarr; {end_n}</div>
  </div>
  <div class="pages">{chips_n}</div>
  {meter(nzs)}
</div>''' if NEU else "")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {E(T["head"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1118-maze-cost, projects/maze-cost):
  example runs: data/{agent}/raw/maze/{i:03d}.json and data/{agent}/raw/deadend/{i:03d}.json (same run index = same site,
    question and page ids); quotes = the agent's own 'thought' field; tokens = prompt + output + thought tokens per call
  defender cost: data/{agent}/access/{{maze,deadend}}/{i:03d}.jsonl (bytes on the wire, handler CPU)
  dollars: PREREG.md s.4 frozen price sheet via analysis.score_run; all-run numbers: results/analysis.json
  pick rule (room #24730, #24765): median depth over entered maze runs of blocks 1-2, lowest run index; dead end = same index
  built by swarm-defense-1118-maze-cost/fig1/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kick {{ font-size: 24px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; }}
  .kick b {{ font-weight: 800; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 24px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .task {{ display: flex; gap: 12px; align-items: center; margin-top: 18px; }}
  .tl {{ font-size: 24px; font-weight: 800; }}
  .ask {{ background: #2f6fe4; color: #fff; font-size: 24px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 10px 18px; }}
  .start {{ font-size: 22px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }}
  .start b {{ color: #1d1d1f; }}
  .see {{ border-radius: 16px; background: #eef0f3; padding: 12px 16px; margin-top: 14px; }}
  .see .lab {{ font-size: 23px; font-weight: 800; }}
  .see .lab span {{ font-weight: 500; color: #6b6f78; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 8px 12px; font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
           font-size: 18px; line-height: 1.45; color: #5a5d66; margin-top: 8px; white-space: pre-wrap; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; font-weight: 700; }}
  .code .nx {{ color: #b3261e; font-weight: 700; }}
  .slot {{ font-size: 22px; color: #5a5d66; margin-top: 16px; line-height: 1.3; }}
  .slot b {{ color: #1d1d1f; }}
  .act.side {{ flex: none; width: 250px; align-self: center; }}
  .lane {{ border-radius: 18px; padding: 12px 16px 12px; margin-top: 10px; background: #eef0f3; }}
  .lane.hi {{ background: #fdeceb; }}
  .lane.hi2 {{ background: #fdf3ea; }}
  .lh {{ display: flex; align-items: baseline; gap: 14px; }}
  .ln {{ font-size: 28px; font-weight: 800; white-space: nowrap; flex: none; }}
  .lane.hi .ln {{ color: #b3261e; }}
  .ld {{ font-size: 21px; color: #5a5d66; }}
  .row {{ display: flex; gap: 14px; align-items: flex-start; margin-top: 10px; }}
  .bot {{ flex: none; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 9px 12px; font-size: 21px; line-height: 1.3; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .thk {{ display: block; font-size: 17px; color: #8a8e97; font-weight: 600; margin-bottom: 2px; }}
  .hl {{ background: #ffe58a; border-radius: 4px; padding: 0 1px; font-weight: 700; }}
  .act {{ font-size: 23px; font-weight: 800; line-height: 1.25; }}
  .act.red {{ color: #b3261e; }} .act.grey {{ color: #3b3e46; }}
  .pages {{ display: flex; gap: 6px; align-items: flex-end; margin-top: 8px; }}
  .pc {{ display: flex; flex-direction: column; align-items: center; width: 92px; }}
  .bw {{ height: {BARH + 30}px; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; width: 100%; }}
  .bar {{ width: 46px; background: #c8362b; border-radius: 3px 3px 0 0; }}
  .bv {{ font-size: 18px; font-weight: 700; color: #3b3e46; margin-bottom: 3px; font-variant-numeric: tabular-nums; }}
  .bv.none {{ font-weight: 500; color: #8a8e97; font-size: 15px; text-align: center; line-height: 1.15; }}
  .chip {{ width: 100%; text-align: center; background: #fff; border: 2px solid #e7b7b2; border-radius: 8px; padding: 4px 0; font-size: 18px; font-weight: 700; color: #3b3e46; }}
  .chip.dead {{ border-color: #c9ccd3; }}
  .axis {{ font-size: 18px; color: #6b6f78; margin-top: 4px; }}
  .meter {{ display: flex; gap: 20px; align-items: center; font-size: 21px; color: #3b3e46; margin-top: 10px; padding-top: 8px; border-top: 1px solid #d9dbe0; }}
  .mm {{ display: flex; flex-direction: column; gap: 3px; }}
  .lane.hi .meter {{ border-top-color: #e7b7b2; }}
  .meter b {{ font-weight: 700; color: #1d1d1f; font-variant-numeric: tabular-nums; }}
  .meter .usd {{ color: #6b6f78; font-variant-numeric: tabular-nums; }}
  .who {{ display: inline-block; width: 64px; font-size: 16px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; color: #8a8e97; margin-right: 4px; }}
  .ratio {{ margin-left: auto; text-align: right; font-size: 19px; color: #6b6f78; line-height: 1.1; }}
  .ratio b {{ display: block; font-size: 34px; font-weight: 800; color: #b3261e; }}
  .lane:not(.hi) .ratio b {{ color: #3b3e46; }}
  .cnote {{ font-size: 20px; color: #3b3e46; margin-top: 6px; line-height: 1.3; }}
  .all {{ margin-top: 16px; font-size: 22px; line-height: 1.35; color: #2b2e35; }}
  .all b {{ font-weight: 800; }}
  .foot {{ font-size: 18px; color: #6b6f78; margin-top: 14px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>
<div class="kick">{T["kicker"]}</div>
<div class="head">{T["head"]}</div>
<div class="q">{T["sub"]}</div>

<div class="task"><span class="tl">Task:</span><span class="ask">&ldquo;{E(mz["question"])}&rdquo;</span></div>
<div class="start">{T["start"]}</div>
<div class="code">{code}</div>

<div class="slot">{T["slot"]}</div>
<div class="lane">
  <div class="lh"><span class="ln">{T["dead_name"]}</span><span class="ld">{T["dead_desc"]}</span></div>
  <div class="row">
    {ROBOT}
    <div class="bub"><span class="thk">{T["thought_label"]} 1</span>&ldquo;{E(qd["thought"])}&rdquo;</div>
    <div class="act grey side">&rarr; {end_d}</div>
  </div>
  {meter(des)}
</div>

<div class="lane hi">
  <div class="lh"><span class="ln">{T["maze_name"]}</span><span class="ld">{T["maze_desc"]}</span></div>
  <div class="row">
    {ROBOT}
    <div class="bub"><span class="thk">{T["thought_label"]} 1</span>&ldquo;{E(qa["thought"])}&rdquo;</div>
    <div class="bub"><span class="thk">{T["thought_label"]} {qb["depth"]}</span>&ldquo;{E(qb["thought"])}&rdquo;</div>
  </div>
  <div class="pages">{chips_html}</div>
  <div class="axis">{T["bar_note"]}</div>
  <div class="act red" style="margin-top:4px">&rarr; {end_m}</div>
  {meter(mzs)}
  <div class="cnote">{T["cost_note"]}</div>
</div>
{neutral_lane}
{f'<div class="all">{T["all"]}</div>' if T.get("all") else ""}
<div class="foot">{T["foot"]} &middot; Kran Lab &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
outf.parent.mkdir(parents=True, exist_ok=True)
outf.write_text(page)
print(f"wrote {outf}: maze D={D} S={mzs['S']} end={mzs['end']} ratio={mzs['ratio_usd']:.0f}; dead end D={des['D']} S={des['S']} "
      f"end={des['end']} ratio={des['ratio_usd']:.0f}; per page {per_page}")
