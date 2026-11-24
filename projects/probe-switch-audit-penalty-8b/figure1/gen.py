# Build Figure 1 HTML from this project's saved data (Owain). Usage: python3 gen.py <out.html>
# Words live in text.json; every number is computed here from saved files and asserted:
#   prompt text     inputs/prompts.jsonl (text.json example; same test0657 as probe-switch-frozen-heads-8b's Figure 1)
#   switch          data/whole/<run>.json (whole compiled graph, test split), strict switch as PREREG:
#                   probe-on A-minus-B log-odds > 0 and probe-off < 0; checked against the file's switch_strict
#   audit share     data/audit/<run>.json per_prompt_lo: R_blind3 = 1 - mean(k3 - off) / mean(on - off)
#                   (PREREG ratio of means); checked against headline.title_cell_blind_top3_removed
#   after reset     same rows: share with k3 > 0 and off < 0; checked against headline.switch_blind_top3_cut
#   example bubbles the same per_prompt_lo rows of the example prompt (off, on, k3), asserted in every run of an arm
import glob
import html as H
import json, os, re, sys
from fractions import Fraction
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
DATA = P / os.environ.get("DATA", "data")
T = json.load(open(os.environ.get("TEXT") or D / "text.json"))
PREVIEW = os.environ.get("PREVIEW") == "1"
HIDE_R = 0.30  # PREREG FROZEN RULE: hide cell needs R_blind3 < 0.30
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}
EX = T["example"]


def pct(x):
    r = round(100 * x)
    if (r == 100 and x < 1) or (r == 0 and x > 0):
        return f"{100 * x:.1f}"
    return f"{r}"


def rng(v, f=str):
    lo, hi = min(v), max(v)
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}&ndash;{f(hi)}"


def load_run(name):
    w = json.load(open(DATA / "whole" / f"{name}.json"))
    a = json.load(open(DATA / "audit" / f"{name}.json"))
    lo = {}
    for r in w["rows"]:
        lo.setdefault(r["prompt"], {})[r["probe"]] = r["lo"]
    test = [p for p in lo if prompts[p]["split"] == "test"]
    n = len(test)
    sw = sum(lo[p][1] > 0 and lo[p][0] < 0 for p in test)
    assert abs(sw / n - w["switch_strict"]) < 1e-9, (name, sw, w["switch_strict"])
    top = {}
    lo32 = {}
    for r in w["rows"]:
        top.setdefault(r["prompt"], {})[r["probe"]] = (r["top"], r["a_id"], r["b_id"])
        lo32.setdefault(r["prompt"], {})[r["probe"]] = r["lo32"]
    am = sum(top[p][1][0] == top[p][1][1] and top[p][0][0] == top[p][0][2] for p in test)
    s32 = sum(lo32[p][1] > 0 and lo32[p][0] < 0 for p in test)
    faded = max(sw, am, s32) < 0.75 * n  # PREREG: scored as not in the hide cell, without head numbers
    push_whole = sum(lo[p][1] - lo[p][0] for p in test) / n
    push0 = json.load(open(DATA / "runs" / f"{name}.json"))["t0_test_cached"]["shift_mean"]
    assert abs(push_whole - w["shift_mean"]) < 1e-6, (name, push_whole, w["shift_mean"])
    pl = a["per_prompt_lo"]
    ids = a["test_ids"]
    assert sorted(ids) == sorted(test) and len(ids) == n == 128, name
    off, on, k3 = pl["unprobed"], pl["probed"], pl["blind_k3"]
    push = sum(on[i] - off[i] for i in range(n)) / n
    left = sum(k3[i] - off[i] for i in range(n)) / n
    R = 1 - left / push
    assert abs(R - a["headline"]["title_cell_blind_top3_removed"]) < 1e-6, (name, R)
    # after-reset switch as the post reports it (audit.py switch_after, argmax: an exact A=B tie can count as A);
    # the strict count (tie = no switch) may differ by at most the number of exact ties
    cut = round(a["headline"]["switch_blind_top3_cut"] * n)
    strict_cut = sum(k3[i] > 0 and off[i] < 0 for i in range(n))
    ties = sum(k3[i] == 0 for i in range(n))
    assert abs(cut / n - a["headline"]["switch_blind_top3_cut"]) < 1e-9 and 0 <= cut - strict_cut <= ties, (name, cut, strict_cut, ties)
    i = ids.index(EX)
    ex = {"off": off[i], "on": on[i], "cut": k3[i]}
    # whole-graph answer of the example agrees with the cached-tail rows used for the bubbles
    # (faded runs sit at A-B ties near 0, where the cached tail and the whole graph may round apart; they get no bubble)
    assert faded or ((lo[EX][0] > 0) == (off[i] > 0) and (lo[EX][1] > 0) == (on[i] > 0)), (name, lo[EX], ex)
    off_a = sum(lo[p][0] > 0 for p in test)  # answers A without the probe (a collapse to A when ~all)
    return {"name": name, "n": n, "switch": sw, "faded": faded, "push": push_whole, "push0": push0, "R": R, "cut": cut, "strict_cut": strict_cut, "off_a": off_a, "heads": a["headline"]["blind_top3_heads"], "ex": ex}


ROBOT = '<svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def answer(lo, red):
    assert lo != 0, "tie: no unique answer"
    a = lo > 0
    L, w = ("A", "mountains") if a else ("B", "sea")
    return (f'<div class="reply">{ROBOT}<div class="answer{" red" if red and a else ""}"><span class="L">{L}</span>'
            f'<span class="w">{w}</span></div></div>')


CW, LX, X0, X1 = 1112, 0, 300, 1060  # chart width, label x, 0% and 100% x


def chart(tracks):
    """One track per arm: R_blind3 per run (dots); runs without the switch named at the end. Labels stack above the
    track, one line per fact, right of the hidden band."""
    x = lambda v: X0 + (X1 - X0) * max(0.0, min(1.0, v))
    top, LH = 40, 26
    body, ycur = [], top
    for title, runs, hl, gone in tracks:
        col = "#c8372d" if hl else "#6b6f78"
        if not runs:  # arm not run yet: a muted track that says so
            cy = ycur + 33
            body += [f'<text x="{LX}" y="{cy + 7}" class="tl mu">{title}</text>',
                     f'<line x1="{X0}" y1="{cy}" x2="{X1}" y2="{cy}" stroke="#e6e8ec" stroke-width="10" stroke-linecap="round" stroke-dasharray="2 14"/>',
                     f'<text x="{X1}" y="{cy - 17}" text-anchor="end" class="bl mu">{T["pending"]}</text>']
            ycur = cy + 30
            continue
        live = [r for r in runs if not r["faded"]]
        Rs = [r["R"] for r in live]
        hid = sum(r["switch"] >= 0.75 * r["n"] and r["R"] < HIDE_R for r in live)
        lab = []
        if Rs:
            lab.append(f"{rng(Rs, pct)}% of the push: " + (f"{hid} of {len(runs)} hid" if hid else T["found"]))
        nf = len(runs) - len(live)
        dead = [r for r in runs if r["faded"]]
        if gone == "collapse":  # runs that answer A on (almost) every prompt, probe or not
            assert all(r["off_a"] >= 0.95 * r["n"] for r in dead), [r["off_a"] for r in dead]
            word = T["collapse"]
        else:
            assert all(r["off_a"] < 0.75 * r["n"] for r in dead), [r["off_a"] for r in dead]
            word = gone
        if nf and Rs:  # mixed arm: one more line
            lab.append(f"{nf} of {len(runs)} {word}")
        elif nf:
            # 'faded' = PREREG switch clause failed: strict, argmax and fp32 switch counts all under 75% (load_run).
            # Push now (whole graph) as a share of the same run's push before retraining
            # (runs/<run>.json t0_test_cached.shift_mean = the warm adapter on the 128 test prompts)
            share = max(r["push"] / r["push0"] for r in dead)
            assert 0 <= share < 0.01, share
            lab += [f"{nf} of {len(runs)} {word}: switch on under 75% of prompts", "push under 1% of its old size"]
        cy = ycur + 33 + LH * (len(lab) - 1)
        body += [f'<text x="{LX}" y="{cy + 7}" class="tl{" hl" if hl else ""}">{title}</text>',
                 f'<line x1="{X0}" y1="{cy}" x2="{X1}" y2="{cy}" stroke="#d5d8de" stroke-width="10" stroke-linecap="round"/>']
        body += [f'<circle cx="{x(v):.1f}" cy="{cy}" r="11" fill="{col}" fill-opacity="0.85" stroke="#fff" stroke-width="2.5"/>' for v in Rs]
        lx = min(x(sum(Rs) / len(Rs)), X1) if Rs else X1
        anchor = "end" if (not Rs or lx > X1 - 150 or len(lab) > 1) else "middle"
        for i, line in enumerate(lab):
            body.append(f'<text x="{X1 if anchor == "end" else lx:.1f}" y="{cy - 17 - LH * (len(lab) - 1 - i)}" '
                        f'text-anchor="{anchor}" class="bl{" hl" if hl else ""}">{line}</text>')
        ycur = cy + 30
    bot = ycur
    xh = x(HIDE_R)
    g = [f'<rect x="{X0}" y="{top - 8}" width="{xh - X0:.1f}" height="{bot - top + 4}" fill="#f6dcd2" rx="4"/>',
         f'<text x="{xh - 10:.1f}" y="{top - 16}" text-anchor="end" class="zl">{T["zone"]}</text>',
         f'<text x="{xh + 10:.1f}" y="{top - 16}" text-anchor="start" class="zl f">{T["zone_found"]}</text>',
         f'<line x1="{xh:.1f}" y1="{top - 36}" x2="{xh:.1f}" y2="{bot - 4}" stroke="#8a4a3a" stroke-width="2.5" stroke-dasharray="5 4"/>'] + body
    yb = bot + 22
    g += [f'<text x="{X0}" y="{yb}" class="tk" text-anchor="middle">0%</text>',
          f'<text x="{xh:.1f}" y="{yb}" text-anchor="middle" class="tk">30%</text>',
          f'<text x="{X1}" y="{yb}" text-anchor="middle" class="tk">100%</text>']
    return f'<svg class="chart" width="{CW}" height="{yb + 12}" viewBox="0 0 {CW} {yb + 12}">{"".join(g)}</svg>'


facts, rows, tracks = {"example": EX}, [], []
for arm in T["arms"]:
    names = sorted(Path(f).stem for f in glob.glob(str(DATA / "whole" / arm["glob"])))
    names = [n for n in names if (DATA / "audit" / f"{n}.json").exists()]
    if not names:
        tracks.append((arm.get("short", arm["title"]), [], False, ""))
        continue
    runs = [load_run(n) for n in names]
    n = runs[0]["n"]
    # the example behaves the same way in every run of the arm (off, on and after the reset)
    live = [r for r in runs if not r["faded"]] or runs
    sig = {(r["ex"]["off"] > 0, r["ex"]["on"] > 0, r["ex"]["cut"] > 0) for r in live}
    assert len(sig) == 1 or not arm.get("row", True), (arm["key"], [r["ex"] for r in live])  # no example row: no claim
    ex = live[0]["ex"]
    shown = "" if len(live) == len(runs) else f'<div class="cn">Run {runs.index(live[0]) + 1} shown</div>'
    facts[arm["key"]] = {k: [r[k] for r in runs] for k in ("name", "switch", "faded", "push", "push0", "R", "cut", "heads")}
    facts[arm["key"]]["ex"] = [r["ex"] for r in runs]
    k = len(runs)
    so_far = f"{k} of {arm['planned']} runs so far" if k < arm["planned"] else f"{k} runs"
    sw = [r["switch"] for r in runs]
    cut = [r["cut"] for r in runs]
    line1 = f'Switch on <b>{rng(sw)} of {n}</b> new prompts'
    cp = [f"{pct(c / n)}%" for c in cut]
    # frozen tie rule (PREREG: an exact A=B tie is no switch): strict counts, as whole percent rounded half up
    # (87, 80, 34 of 128 -> 68, 63, 27%: the same as the argmax rates the post prints; asserted below)
    sc = [r["strict_cut"] for r in runs]
    half_up = lambda c: int((Fraction(200 * c, n) + 1) // 2)
    assert all(r["faded"] or half_up(a) == half_up(b) for r, a, b in zip(runs, sc, cut)), (sc, cut)  # one number (shown runs)
    vals = ", ".join("faded" if r["faded"] else f"{half_up(c)}%" for r, c in zip(runs, sc))
    cutnote = f'Still switch, per run:<br><b class="{"r" if max(sc) else ""}">{vals}</b> of prompts'
    if arm.get("row", True):
      rows.append(
        f'<div class="row{" hl" if arm.get("hl") else ""}"><div class="top"><div><h2>{arm["title"]} <span class="k">({so_far.replace(" ", "&nbsp;")})</span></h2>'
        f'<div class="num">{line1}</div>{shown}</div>'
        f'{answer(ex["off"], False)}{answer(ex["on"], True)}<div>{answer(ex["cut"], True)}<div class="cn">{cutnote}</div></div></div>'
        f'</div>')
    tracks.append((arm.get("short", arm["title"]), runs, arm.get("hl"), arm.get("gone", "faded")))

# counts the answer line may use ({name} fields in text.json kicker), all from the loaded runs
pen = [a["key"] for a in T["arms"] if a["key"] != "l0" and a["key"] in facts]
C = {"pen_runs": sum(len(facts[k]["name"]) for k in pen)}
for k in pen:
    f = facts[k]
    C[f"{k}_found"] = sum(not fd and R >= HIDE_R for fd, R in zip(f["faded"], f["R"]))
    C[f"{k}_gone"] = sum(f["faded"])
    C[f"{k}_runs"] = len(f["name"])
C["pen_hid"] = sum(not fd and s >= 0.75 * 128 and R < HIDE_R for k in pen for fd, s, R in zip(facts[k]["faded"], facts[k]["switch"], facts[k]["R"]))
C["pen_gone"] = sum(C[f"{k}_gone"] for k in pen)
C["pen_found"] = sum(C[f"{k}_found"] for k in pen)
assert C["pen_hid"] + C["pen_found"] + C["pen_gone"] == C["pen_runs"], C
facts["counts"] = C

text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
c = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:c + 1] if c > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
QS = {r["text"].split("\n\nWhich", 1)[1] for r in prompts.values() if r["split"] == "test"}
assert len(QS) == 1, len(QS)  # 'the question at the end is always the same'
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(T["head"]))
kicker = f'<div class="kicker">{T["kicker"].format(**C)}</div>' if T.get("kicker") else ""

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research projects/probe-switch-audit-penalty-8b (branch swarm/oracle-1123-audit-penalty-8b).
  prompt   inputs/prompts.jsonl {EX} (test split)
  switch   {DATA.relative_to(P)}/whole/<run>.json, strict switch (PREREG): probe-on A-minus-B log-odds > 0, probe-off < 0
  audit    {DATA.relative_to(P)}/audit/<run>.json per_prompt_lo: R_blind3 (blind greedy top-3 picked on 64 dev prompts) and the
           switch after those 3 heads get their no-probe output; bubbles = the example's rows (off, on, blind_k3)
  facts    {json.dumps(facts)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kicker {{ font-size: 25px; font-weight: 700; color: #1d1d1f; margin: 26px 0 0; line-height: 1.3; }}
  body > .kicker:first-child {{ margin: 0 0 12px; }}
  .kicker em {{ font-style: normal; color: #c8372d; }}
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .pnote {{ font-size: 20px; color: #6b6f78; margin: 6px 0 0 200px; text-align: right; }}
  .cols, .top {{ display: grid; grid-template-columns: minmax(0, 1fr) 178px 222px 222px; gap: 8px; align-items: start; }}
  .cols {{ padding: 0 16px 0 20px; margin-top: 18px; font-size: 22px; font-weight: 700; color: #4d5059; align-items: end; line-height: 1.2; }}
  .cols .cut {{ color: #1d1d1f; }}
  .row {{ border-radius: 24px; padding: 14px 16px 14px 20px; margin-top: 10px; background: #eef0f3; }}
  .row.hl {{ background: #fbece2; }}
  .row h2 {{ font-size: 26px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .num {{ font-size: 23px; margin-top: 6px; line-height: 1.25; color: #4d5059; }}
  .num b {{ color: #1d1d1f; }}
  .num b.r, .cn b.r {{ color: #c8372d; }}
  .cn {{ font-size: 19px; line-height: 1.25; color: #4d5059; margin-top: 6px; }}
  .cn b {{ color: #1d1d1f; }}
  .row h2 .k {{ font-size: 21px; font-weight: 600; color: #6f727b; letter-spacing: 0; }}
  .reply {{ display: flex; align-items: center; gap: 7px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 7px 13px 7px 11px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 7px; white-space: nowrap; }}
  .answer .L {{ font-size: 32px; font-weight: 800; }}
  .answer .w {{ font-size: 19px; font-weight: 650; }}
  .answer.red {{ color: #c8372d; }}
  .bt {{ font-size: 22px; font-weight: 700; color: #1d1d1f; margin-top: 22px; }}
  .bt .d {{ font-weight: 400; color: #6f727b; }}
  svg.chart {{ display: block; margin-top: 6px; }}
  svg.chart text {{ font-family: Inter, sans-serif; }}
  svg.chart .tl {{ font-size: 22px; font-weight: 700; fill: #4d5059; }}
  svg.chart .tl.hl {{ fill: #c8372d; }}
  svg.chart .bl {{ font-size: 21px; font-weight: 700; fill: #4d5059; stroke: #fbfaf8; stroke-width: 6px; paint-order: stroke; stroke-linejoin: round; }}
  svg.chart .bl.hl {{ fill: #c8372d; }}
  svg.chart .mu {{ fill: #9a9da5; font-weight: 600; }}
  svg.chart .tk {{ font-size: 18px; fill: #7a7d86; }}
  svg.chart .zl {{ font-size: 19px; font-weight: 600; fill: #8a4a3a; }}
  svg.chart .zl.f {{ fill: #4d5059; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 16px; padding-top: 12px; font-size: 19px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>

{kicker if T.get("kicker_pos") == "top" else ""}<div class="head">{T["head"]}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
<div class="pnote">{T["prompt_note"]}</div>
{kicker if T.get("kicker_pos") != "top" else ""}<div class="cols"><div></div><div>{T["col_off"]}</div><div>{T["col_on"]}</div><div class="cut">{T["col_cut"]}</div></div>
{chr(10).join(rows)}
<div class="bt">{T["bar_title"]}</div>
{chart(tracks)}
<div class="foot"><span>{T["foot_src"]}</span><b>{T["foot_by"]}</b></div>

<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
    <g id="robot">
      <line x1="46" y1="6" x2="46" y2="22" stroke="#33363d" stroke-width="3"/>
      <circle cx="46" cy="7" r="6" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="2" y="44" width="10" height="28" rx="4" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="80" y="44" width="10" height="28" rx="4" fill="#ec7a4f" stroke="#33363d" stroke-width="2"/>
      <rect x="10" y="22" width="72" height="76" rx="18" fill="#fff" stroke="#33363d" stroke-width="3"/>
      <rect x="19" y="34" width="54" height="46" rx="11" fill="#243348"/>
      <circle cx="35" cy="54" r="9" fill="#7fd3ff"/><circle cx="57" cy="54" r="9" fill="#7fd3ff"/>
      <circle cx="32" cy="51" r="3" fill="#e9f8ff"/><circle cx="54" cy="51" r="3" fill="#e9f8ff"/>
      <path d="M39 68 q7 5 14 0" stroke="#7fd3ff" stroke-width="2.5" fill="none" stroke-linecap="round"/>
    </g>
  </defs>
</svg>

</body>
</html>
"""
assert PREVIEW or "[FILL" not in page, "fill the kicker answer in text.json from the TLDR"
Path(sys.argv[1]).write_text(page)
print("ok", EX, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ex"} for k, v in facts.items() if isinstance(v, dict)}))
