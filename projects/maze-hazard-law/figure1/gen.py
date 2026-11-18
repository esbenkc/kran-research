"""Owain: build Figure 1 (assets/figures/maze-hazard-law-fig1/figure1.html) from the real run records.

    /usr/bin/python3 gen.py <project dir> <text.json> <out.html> [--data DIR] [--analysis FILE] [--agent NAME] [--run I]

Every number it draws comes from analysis.json (Iris's frozen analysis.py output); the example run is re-scored from its
raw record with analysis.score_run and asserted equal to its analysis.json row.

PICK RULE (room #25865, fixed 2026-11-18 before any maze run existed):
  agent = the first status agent in analysis.json overall.tested, else the first status agent with entered maze runs;
  run   = the lowest run index among that agent's entered maze runs whose page depth D equals the agent's KM median;
          if the KM median is not reached, the lowest-index run that ended at the 80-call cap.
"""
import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("proj")
ap.add_argument("text")
ap.add_argument("out")
ap.add_argument("--data")
ap.add_argument("--analysis")
ap.add_argument("--agent", help="override the pick (dev only)")
ap.add_argument("--run", type=int, help="override the pick (dev only)")
a = ap.parse_args()
proj = Path(a.proj).resolve()
data = Path(a.data).resolve() if a.data else proj / "data"
an = json.loads(Path(a.analysis or proj / "results" / "analysis.json").read_text())
sys.path.insert(0, str(proj))
sys.path.insert(0, str(proj / "web"))
import analysis  # noqa: E402
import sites  # noqa: E402

E = html.escape
T = json.loads(Path(a.text).read_text())
GIVEUP = analysis.GIVEUP
CAP = analysis.CAP

# ---- pick ------------------------------------------------------------------------------------------------------------
AG = {**an.get("side_row", {}), **an["agents"]}  # side rows only via --agent (pipeline checks); never picked
order = [n for n in analysis.STATUS_ORDER if n in AG]
cands = [n for n in an["overall"]["tested"] if n in AG] or [n for n in order if (AG[n].get("n_entered") or 0) > 0]
assert cands or a.agent, "no status agent with entered maze runs"
name = a.agent or cands[0]
A = AG[name]
L = A["law"]
ent = [r for r in A["runs"]["maze"] if r["entered"]]
med = L["km_median"]
if a.run is not None:
    pick = next(r for r in ent if r["run_index"] == a.run)
elif med is not None:
    pick = min((r for r in ent if r["D"] == med), key=lambda r: r["run_index"])
else:
    pick = min((r for r in ent if r["cause"] == "cap"), key=lambda r: r["run_index"])
i = pick["run_index"]

# ---- the example run, re-scored from its raw record ------------------------------------------------------------------
d = data / A["dir"]
rec = json.loads((d / "raw" / "maze" / f"{i:03d}.json").read_text())
acc = [json.loads(x) for x in (d / "access" / "maze" / f"{i:03d}.jsonl").read_text().splitlines() if x.strip()]
acc.sort(key=lambda x: x["t"])
sc = analysis.score_run(rec, acc)
for k in ("D", "S", "cause", "k", "entered"):
    assert sc[k] == pick[k], (k, sc[k], pick[k])
D = sc["D"]
mpages = [x for x in acc if x.get("kind") == "maze"]
assert len({x["maze_id"] for x in mpages}) >= D, "access log has fewer unique maze pages than D"


def thought(step):
    try:
        t = json.loads(step["text"])["thought"]
    except Exception:
        m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', step.get("text") or "")
        t = json.loads(f'"{m.group(1)}"') if m else None
    return t if isinstance(t, str) and t.strip() else None


# calls inside the maze episode with the page depth each call sat at (same walk as analysis.score_run)
calls, seen, entry = [], set(), None
for s in rec["steps"]:
    act = s.get("action") or {}
    p = act.get("path")
    if entry is None:
        if p == rec["site"]["canary_path"] and act.get("status") == 200:
            entry = s["step"]
            seen.add(p)
        continue
    calls.append({"step": s["step"], "depth": len(seen), "thought": thought(s), "tool": act.get("tool"), "path": p,
                  "answer": act.get("text")})
    if act.get("tool") == "answer" or (p is not None and not p.startswith("/m/")):
        break
    if p is not None and p.startswith("/m/") and act.get("status") == 200:
        seen.add(p)
assert len(seen) == D or sc["cause"] not in GIVEUP, (len(seen), D)
q_first = next((c for c in calls if c["depth"] == 1 and c["thought"]), None)
q_last = [c for c in calls if c["thought"]][-1] if calls else None

# one maze page as the agent read it (raw HTML of part 2)
site = rec["site"]
tpl = sites.TEMPLATES[site["template"]]
show_k = 2
pg = sites.Site(site["template"], "maze", secret=site["secret"])
pid = pg.id0
for _ in range(show_k - 1):
    pid = pg._next(pid)
    pg.depth[pid] = len(pg.depth) + 1
body = pg.render(f"/m/{pid}.html")[2].decode()
main = body.split("<main>")[1].split("</main>")[0].strip()
assert "lists no figures" in main and f"part {show_k}" in main
anchor = tpl["fit_anchor"]
code = (E(main).replace("Figures are published in a later part.", '<span class="anc">Figures are published in a later part.</span>')
        .replace(E(f'{anchor} (part {show_k + 1})'), f'<span class="nx">{E(anchor)} (part {show_k + 1})</span>')
        .replace("&lt;p&gt;&lt;a", "\n&lt;p&gt;&lt;a").replace("&lt;p&gt;This", "\n&lt;p&gt;This"))


# ---- numbers for the text slots, all from analysis.json -----------------------------------------------------------
def pages(x):
    return f"{x:.0f}"


def short(n):
    return re.sub(r"\s*\(.*\)$", "", n)


NUM = {"agent": short(name), "n_ent": A["n_entered"], "G3": L["G3"], "E3": L["E3"], "D": D, "run": i,
       "one_in": round(1 / L["h3"]) if L["h3"] else None, "cap": CAP,
       "pred": pages(L["pred_median"]) if L["pred_median"] is not None else None,
       "obs": med, "status": an["overall"]["status"],
       "answer": {"CONFIRMED": "Yes.", "REJECTED": "No.", "UNCONFIRMED": "Not settled."}[an["overall"]["status"]],
       "cap_n": sum(r["cause"] == "cap" for r in ent),
       "giveup_n": sum(r["cause"] in GIVEUP for r in ent)}
if L.get("R"):
    NUM["R"] = f"{L['R']:.1f}"
    NUM["times"] = f"{max(L['R'], 1 / L['R']):.1f}"
for n2 in order:
    L2 = AG[n2]["law"] or {}
    key = "a" if n2 == order[0] else "b"
    NUM[f"{key}_name"] = n2
    NUM[f"{key}_pred"] = pages(L2["pred_median"]) if L2.get("pred_median") is not None else "-"
    NUM[f"{key}_obs"] = L2.get("km_median") or "-"
    NUM[f"{key}_verdict"] = L2.get("verdict")
de = A["dead_end"]
NUM["de_n"] = de["n_entered"]
NUM["de_p1"] = round((de["exit_page1"] or 0) * de["n_entered"])


def fill(s):
    return s.format_map(NUM) if isinstance(s, str) else s


# variant keys: "<key>_nomed" replaces <key> when the KM median is not reached; "<key>_predcap" when the forecast is the cap
for suf, on in (("_nomed", med is None), ("_predcap", L["pred_median"] is not None and L["pred_median"] >= L["K"])):
    for k in [k for k in T if k.endswith(suf)]:
        if on:
            T[k[:-len(suf)]] = T[k]
        del T[k]
T = {k: (v if k.startswith("end_") or k in ("thought_label", "strip_label", "v_off") else fill(v)) for k, v in T.items()}

# ---- number line: predicted vs observed median page per status agent (log scale) ----------------------------------
W, X0, X1 = 1120, 190, 945
LO, HI = 1, CAP


def xp(v):
    v = min(max(v, LO), HI)
    return X0 + (X1 - X0) * math.log(v / LO) / math.log(HI / LO)


def anch(x):
    return "end" if x > X1 - 70 else ("start" if x < X0 + 40 else "middle")


rows_svg, y = [], 44
for n2 in analysis.STATUS_ORDER + ([name] if name not in analysis.STATUS_ORDER else []):
    L2 = (AG.get(n2) or {}).get("law") or {}
    hi = n2 == name
    col, colb = ("#b3261e", "#f4c7c3") if hi else ("#6b6f78", "#dfe1e5")
    rows_svg.append(f'<text x="{X0 - 16}" y="{y + 9}" text-anchor="end" class="rl{" hi" if hi else ""}">{E(short(n2))}</text>')
    o, p = L2.get("km_median"), L2.get("pred_median")
    if o:
        rows_svg.append(f'<rect x="{xp(o / 2):.1f}" y="{y - 13}" width="{xp(o * 2) - xp(o / 2):.1f}" height="26" rx="13" fill="{colb}"/>')
    if o and p:
        rows_svg.append(f'<line x1="{xp(o):.1f}" y1="{y}" x2="{xp(p):.1f}" y2="{y}" stroke="{col}" stroke-width="3"/>')
    if p:
        lab = T["lab_pred_cap"] if p >= L2["K"] else f'{T["lab_pred"]} {pages(p)}'
        rows_svg.append(f'<circle cx="{xp(p):.1f}" cy="{y}" r="11" fill="#fbfaf8" stroke="{col}" stroke-width="4"/>')
        rows_svg.append(f'<text x="{xp(p):.1f}" y="{y - 21}" text-anchor="{anch(xp(p))}" class="ml" fill="{col}">{lab}</text>')
    if o:
        rows_svg.append(f'<circle cx="{xp(o):.1f}" cy="{y}" r="11" fill="{col}"/>')
        rows_svg.append(f'<text x="{xp(o):.1f}" y="{y + 37}" text-anchor="{anch(xp(o))}" class="ml" fill="{col}">{T["lab_obs"]} {o}</text>')
    if o and p:
        r_ = p / o
        vt = T["v_within"] if 0.5 <= r_ <= 2 else T["v_off"].format(x=f"{max(r_, 1 / r_):.1f}")
        rows_svg.append(f'<text x="{W}" y="{y + 9}" text-anchor="end" class="vd" fill="{col}">{vt}</text>')
    elif L2:
        rows_svg.append(f'<text x="{X1}" y="{y + 37}" text-anchor="end" class="ml" fill="{col}">{T["lab_none"]}</text>')
    else:
        rows_svg.append(f'<text x="{X0}" y="{y + 9}" class="ml" fill="{col}">{T["lab_nodata"]}</text>')
    y += 94
ticks = "".join(f'<line x1="{xp(t):.1f}" y1="12" x2="{xp(t):.1f}" y2="{y - 44}" class="gl"/>'
                f'<text x="{xp(t):.1f}" y="{y - 20}" text-anchor="middle" class="tk">{t}</text>' for t in (1, 2, 5, 10, 20, 40, 80))
NL_H = y - 12
numline = f'<svg class="nl" viewBox="0 0 {W} {NL_H}" width="{W}" height="{NL_H}">{ticks}{"".join(rows_svg)}</svg>'

# ---- the example run: one square per maze page -----------------------------------------------------------------------
GAP = 4 if D <= 40 else 2
SQ = max(6, min(40, (W - 10 - GAP * D) // max(D, 1) if D > 20 else (W - 620 - GAP * D) // max(D, 1)))
squares = "".join(f'<div class="sq" style="width:{SQ}px;height:{SQ}px"></div>' for _ in range(D))
label = f'<span class="sql">{T["strip_label"].format(D=D, p="page" if D == 1 else "pages")}</span>'
STRIP_BREAK = D > 20  # long runs: squares fill the row, the label and the ending go on the next line
ans_key = ("end_answer_invented" if pick.get("invented_figure") else "end_answer_stale" if pick.get("stale_figure")
           else "end_answer")
end_txt = {"giveup_answer": T[ans_key], "giveup_get": T["end_get"], "cap": T["end_cap"],
           "lost": T["end_lost"], "overflow": T["end_overflow"]}[sc["cause"]].format(D=D, p="page" if D == 1 else "pages")

ROBOT = ('<svg class="bot" viewBox="0 0 40 40" width="48" height="48"><line x1="20" y1="3" x2="20" y2="9" stroke="#3b3e46" stroke-width="2"/>'
         '<circle cx="20" cy="3" r="2.6" fill="#e8743b"/><rect x="5" y="9" width="30" height="26" rx="8" fill="#fff" stroke="#3b3e46" stroke-width="2"/>'
         '<rect x="9" y="14" width="22" height="14" rx="5" fill="#25324a"/><circle cx="15" cy="21" r="3.2" fill="#7fd3ff"/><circle cx="25" cy="21" r="3.2" fill="#7fd3ff"/>'
         '<rect x="1.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/><rect x="34.5" y="17" width="4" height="9" rx="2" fill="#e8743b"/></svg>')


def bub(c, label):
    return f'<div class="bub"><span class="thk">{label}</span>&ldquo;{E(c["thought"])}&rdquo;</div>' if c else ""


quotes = bub(q_first, T["thought_label"].format(k=1))
if q_last and q_last is not q_first:
    quotes += bub(q_last, T["thought_label"].format(k=q_last["depth"]))

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {E(T["head"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1118-maze-hazard-law, projects/maze-hazard-law):
  all numbers: results/analysis.json (analysis.py, PREREG.md s.4): agents.<agent>.law (G3, E3, h3, pred_median, km_median, R)
  example run: data/{A["dir"]}/raw/maze/{i:03d}.json + access/maze/{i:03d}.jsonl, re-scored with analysis.score_run;
    quotes = the agent's own 'thought' field; the decoy page = sites.Site(...).render() of part {show_k} for that run's site
  pick rule (room #25865): first tested status agent; lowest run index with page depth = its KM median
  built by projects/maze-hazard-law/figure1/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kick {{ font-size: 24px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; }}
  .kick b {{ font-weight: 800; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 24px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .task {{ display: flex; gap: 12px; align-items: center; margin-top: 16px; }}
  .tl {{ font-size: 24px; font-weight: 800; flex: none; }}
  .ask {{ background: #2f6fe4; color: #fff; font-size: 24px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 9px 18px; }}
  .start {{ font-size: 22px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 6px 12px; font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
           font-size: 17px; line-height: 1.4; color: #5a5d66; margin-top: 6px; white-space: pre-wrap; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; font-weight: 700; }}
  .code .nx {{ color: #b3261e; font-weight: 700; }}
  .lane {{ border-radius: 18px; padding: 12px 16px 12px; margin-top: 14px; background: #fdeceb; }}
  .lane.g {{ background: #eef0f3; }}
  .ln {{ font-size: 27px; font-weight: 800; }}
  .lane:not(.g) .ln {{ color: #b3261e; }}
  .ld {{ font-size: 21px; color: #5a5d66; font-weight: 500; }}
  .row {{ display: flex; gap: 12px; align-items: flex-start; margin-top: 8px; }}
  .bot {{ flex: none; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 8px 12px; font-size: 21px; line-height: 1.3; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .thk {{ display: block; font-size: 17px; color: #8a8e97; font-weight: 600; margin-bottom: 2px; }}
  .strip {{ display: flex; flex-wrap: nowrap; gap: 4px; align-items: center; margin-top: 10px; }}
  .sq {{ background: #c8362b; border-radius: 3px; flex: none; }}
  .sql {{ white-space: nowrap; font-size: 21px; font-weight: 700; color: #3b3e46; margin-left: 6px; margin-right: 4px; }}
  .endl {{ white-space: nowrap; font-size: 23px; font-weight: 800; color: #b3261e; margin-left: 10px; }}
  .flow {{ display: flex; gap: 14px; align-items: flex-end; margin-top: 8px; }}
  .fs {{ flex: none; white-space: nowrap; }}
  .fs .k {{ font-size: 18px; font-weight: 700; color: #6b6f78; text-transform: uppercase; letter-spacing: 0.03em; }}
  .fs .v {{ font-size: 27px; font-weight: 800; line-height: 1.2; font-variant-numeric: tabular-nums; }}
  .fs.red .v {{ color: #b3261e; }}
  .arr {{ font-size: 30px; line-height: 1.2; color: #8a8e97; flex: none; }}
  .nl {{ display: block; margin-top: 4px; }}
  .nl .gl {{ stroke: #d9dbe0; stroke-width: 1; }}
  .nl .tk {{ font-size: 19px; fill: #6b6f78; font-family: inherit; }}
  .nl .rl {{ font-size: 24px; font-weight: 800; fill: #6b6f78; font-family: inherit; }}
  .nl .rl.hi {{ fill: #1d1d1f; }}
  .nl .vd {{ font-size: 26px; font-weight: 800; font-family: inherit; }}
  .nl .ml {{ font-size: 20px; font-weight: 700; font-family: inherit; }}
  .key {{ font-size: 20px; color: #5a5d66; line-height: 1.3; }}
  .foot {{ font-size: 18px; color: #6b6f78; margin-top: 12px; line-height: 1.35; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>
<div class="kick">{T["kicker"]}</div>
<div class="head">{T["head"]}</div>
<div class="q">{T["sub"]}</div>

<div class="task"><span class="tl">Task:</span><span class="ask">&ldquo;{E(rec["question"])}&rdquo;</span></div>
<div class="start">{T["start"]}</div>
<div class="code">{code}</div>

<div class="lane">
  <div><span class="ln">{T["ex_name"]}</span> <span class="ld">{T["ex_desc"]}</span></div>
  <div class="row">{ROBOT}{quotes}</div>
  <div class="strip" style="gap:{GAP}px">{squares}{"" if STRIP_BREAK else label + f'<span class="endl">&rarr; {end_txt}</span>'}</div>
  {f'<div class="strip">{label}<span class="endl">&rarr; {end_txt}</span></div>' if STRIP_BREAK else ""}
</div>

<div class="lane g">
  <div><span class="ln">{T["law_name"]}</span> <span class="ld">{T["law_desc"]}</span></div>
  <div class="flow">
    <div class="fs"><div class="k">{T["p1_k"]}</div><div class="v">{T["p1_v"]}</div></div>
    <div class="arr">&rarr;</div>
    <div class="fs"><div class="k">{T["p2_k"]}</div><div class="v">{T["p2_v"]}</div></div>
    <div class="arr">&rarr;</div>
    <div class="fs red"><div class="k">{T["p3_k"]}</div><div class="v">{T["p3_v"]}</div></div>
  </div>
  {numline}
  <div class="key">{T["key"]}</div>
</div>
<div class="foot">{T["foot"]} &middot; Kran Lab &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(page)
print(f"wrote {a.out}: agent {name} run {i:03d} D={D} cause={sc['cause']} S={sc['S']}; "
      f"G3/E3 {L['G3']}/{L['E3']} pred {L['pred_median']} KM median {med} R {L.get('R')}")
