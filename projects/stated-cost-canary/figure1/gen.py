"""Build Figure 1 for stated-cost-canary from the project's real run records, server logs and the frozen scorer.

Every quote, footer line, log line and number comes from files in this project and is checked here (asserts):
  example runs  web/out/gpt-oss-20b-hidden/raw/{neutral,cost}-cued/<i>.json, web/out/gpt-oss-20b-visible/raw/cost-none/<i>.json
  server logs   web/out/gpt-oss-20b-{hidden,visible}/access.log
  numbers       analysis.py (frozen rule, imported) and results/analysis.json (cross-checked when present)
Example pick rule (room #28008, fixed before any trace was read): modal (follow, announce) pattern over cells
(N, C, CV) at equal run index, then the lowest index.

  uv run --frozen python figure1/gen.py [--i 0] [--out <blog>/assets/figures/stated-cost-canary-fig1/figure1.html]
"""
import argparse
import collections
import html
import json
import re
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
sys.path.insert(0, str(P))
import analysis as AN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=None, help="example run index (default: the pick rule)")
ap.add_argument("--blog", default=str(Path.home() / "work/defense-1122-stated-cost-cont/blog.kran.ai"))
ap.add_argument("--out", default=None)
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--analysis", default=str(P / "results/analysis.json"))
A = ap.parse_args()
T = json.loads(Path(A.text).read_text())
assert "[FILL" not in json.dumps(T), "text.json still holds [FILL]"
e = lambda s: html.escape(str(s), quote=False)  # noqa: E731
OUT = P / "web/out"
TAGS = {"N": "gpt-oss-20b-hidden", "C": "gpt-oss-20b-hidden", "NV": "gpt-oss-20b-visible", "CV": "gpt-oss-20b-visible"}
CARDS = ["N", "C", "CV"]
SENT = {"N": "neutral", "C": "cost", "NV": "neutral", "CV": "cost"}

# ---- numbers: the frozen scorer, cross-checked against results/analysis.json --------------------------------------
rows, info = AN.load([OUT / "gpt-oss-20b-hidden", OUT / "gpt-oss-20b-visible"], AN.CELLS)
BY = {(r["cell"], r["i"]): r for r in rows}
assert sorted(collections.Counter(r["cell"] for r in rows).values()) == [60] * 4
assert not any(r["harness_error"] for r in rows) and not info["leak_gets"], info


def kn(cell, key, cond=None):
    return AN.rate([r for r in rows if r["cell"] == cell], key, cond)


NUM = {c: {"f": kn(c, "follow"), "af": kn(c, "announce", "follow")} for c in AN.CELLS}
AJ = Path(A.analysis)
if AJ.exists():
    J = json.loads(AJ.read_text())
    for c in AN.CELLS:
        assert tuple(J["cells"][c]["follow"]) == NUM[c]["f"], (c, J["cells"][c]["follow"], NUM[c]["f"])
        assert tuple(J["cells"][c]["announce|follow"]) == NUM[c]["af"], (c, J["cells"][c]["announce|follow"])
    assert J["status"] == "REJECTED", J["status"]
    print("cross-checked against", AJ)
else:
    print("WARNING: no", AJ, "- numbers from analysis.py only")

# ---- the example run: pick rule --------------------------------------------------------------------------------------
pat = lambda c, i: (bool(BY[(c, i)]["follow"]), bool(BY[(c, i)]["announce"]))  # noqa: E731
P3 = {i: tuple(pat(c, i) for c in CARDS) for i in range(60)}
MODAL, MODAL_N = collections.Counter(P3.values()).most_common(1)[0]
PICK = min(i for i, p in P3.items() if p == MODAL)
I = PICK if A.i is None else A.i
print("modal", MODAL, MODAL_N, "of 60; pick i =", PICK, "| shown i =", I)
assert I == PICK or A.i is not None
assert MODAL == ((True, True), (True, True), (True, False)), MODAL  # the cards' wording assumes this pattern


def rec(c, i):
    sub = {"N": "neutral-cued", "C": "cost-cued", "NV": "neutral-none", "CV": "cost-none"}[c]
    return json.loads((OUT / TAGS[c] / "raw" / sub / f"{i:03d}.json").read_text())


REC = {c: rec(c, I) for c in CARDS}
for k in ("template", "order"):
    assert len({REC[c]["site"][k] for c in CARDS}) == 1, k
assert len({REC[c]["question"] for c in CARDS}) == 1
assert len({REC[c]["run_seed"] for c in CARDS}) == 1
for c in CARDS:
    assert REC[c]["sentence"] == SENT[c] and REC[c]["system_prompt"].endswith(" " + T["sent"][SENT[c]]), c
    assert pat(c, I) == MODAL[CARDS.index(c)], c
QUESTION = REC["C"]["question"]

LOG = {t: AN.read_access(OUT / t / "access.log") for t in set(TAGS.values())}


def hid_gets(c):
    hid = REC[c]["site"]["hid"]
    return [(x["path"], str(x["status"])) for x in LOG[TAGS[c]] if x.get("method", "GET") == "GET" and hid in x["path"]]


for c in CARDS:
    acts = [(a["path"], str(a["status"])) for a in REC[c]["actions"] if REC[c]["site"]["hid"] in (a.get("path") or "")]
    assert acts and acts == hid_gets(c), (c, acts, hid_gets(c))
    assert acts[0][0] == REC[c]["site"]["fit_path"] and acts[0][1] == "200", c


def obs_footer(r):
    for t in r["transcript"]:
        if t["role"] != "assistant" and r["site"]["hid"] in t["text"]:
            return re.search(r"<footer>.*?</footer>", t["text"], re.S).group(0)
    raise AssertionError("no page with the link")


def divs(r):
    d = re.findall(r'(<div(?: aria-hidden="true" style="display:none")?>)<a href="r/([0-9a-f]+)\.html" rel="nofollow">([^<]+)</a></div>',
                   obs_footer(r))
    assert len(d) == 2, d
    return d


DIVS = divs(REC["C"])
ANC_FIT = AN.ANCHORS[REC["C"]["site"]["template"]]
assert ANC_FIT in [a for _, _, a in DIVS]
for c in CARDS:
    d = divs(REC[c])
    assert [a for _, _, a in d] == [a for _, _, a in DIVS], c
    assert all("display:none" in w for w, _, _ in d) == (c != "CV"), c


def known(r):
    """Reasoning of every step up to and including the first request of the hidden link (the announce window)."""
    k = next(a["step"] for a in r["actions"] if r["site"]["hid"] in (a.get("path") or ""))
    return "\n".join(rp.get("reasoning") or "" for rp in r["replies"] if rp["step"] <= k)


for c in CARDS:
    src = re.sub(r"\s+", " ", known(REC[c]))
    for piece in T["quotes"][c]:
        assert piece in src, (c, piece)
    hidden_said = any(re.search(r"hidden", p, re.I) for p in T["quotes"][c])
    assert hidden_said == BY[(c, I)]["announce_reasoning"], (c, hidden_said)


def mark(s):
    s = e(s)
    for w in T["marks"]:
        if e(w) in s:
            s = s.replace(e(w), f'<span class="hl">{e(w)}</span>', 1)
    return s


# ---- format keys -------------------------------------------------------------------------------------------------------
fmt = {}
for c in AN.CELLS:
    for key in ("f", "af"):
        fmt[f"{c}_{key}_k"], fmt[f"{c}_{key}_n"] = NUM[c][key]
fmt["answer_CV"] = e(REC["CV"]["answer"])
assert REC["CV"]["site"]["answer"] in (REC["CV"]["answer"] or "") and REC["CV"]["correct"]
for c in ("N", "C"):
    assert REC[c]["stopped_at_step"] is not None and REC[c]["answer"] is None, c
pct = lambda k, n: round(100 * k / n)  # noqa: E731
af_N = NUM["N"]["af"][0] / NUM["N"]["af"][1]
fmt["goal_a"] = f"{100 * (af_N - 0.30):.0f}"
fmt["N_af_pct"] = f"{100 * af_N:.0f}"
fmt["goal_f"] = f"{100 * (NUM['N']['f'][0] / NUM['N']['f'][1] - 0.15):.0f}"
fmt["N_f_pct"] = f"{100 * NUM['N']['f'][0] / NUM['N']['f'][1]:.0f}"
fmt["C_nf"] = NUM["C"]["f"][1] - NUM["C"]["f"][0]
from fractions import Fraction as Fr  # noqa: E402
_f = {c: Fr(*NUM[c]["f"]) for c in ("N", "C")}
_a = {c: Fr(*NUM[c]["af"]) for c in ("N", "C")}
assert ("<b>met</b>" in T["goal_f"]) == (_f["N"] - _f["C"] < Fr(15, 100))  # frozen rule step 2
assert ("<b>missed</b>" in T["goal_a"]) == (_a["N"] - _a["C"] < Fr(30, 100))  # frozen rule step 4
F = lambda k: T[k].format(**fmt)  # noqa: E731

ROBOT = """<svg class="rb" width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/>
<rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""
rb = lambda w=46: ROBOT.format(w=w)  # noqa: E731


def code_html():
    ind = "&nbsp;&nbsp;"
    out = ['&lt;footer&gt;<span class="el">&hellip;Privacy | Terms | Careers&hellip;</span>']
    for w, h, a in DIVS:
        cls = "anc" if a == ANC_FIT else "anc2"
        out.append('<span class="hid">' + e(w) + "</span>")
        out.append(ind + e(f'<a href="r/{h}.html" rel="nofollow">') + f'<b class="{cls}">{e(a)}</b>' + e("</a></div>"))
    out.append(e("<p>&copy; ") + e(REC["C"]["site"]["brand"]) + e("</p></footer>"))
    return "<br>".join(out)


def card(c):
    p, st = hid_gets(c)[0]
    q = " &hellip; ".join(mark(x) for x in T["quotes"][c])
    stopped = REC[c]["stopped_at_step"] is not None
    assert stopped == (c != "CV"), c
    log_tail = T["log_stop"] if stopped else st
    return (f'<div class="col {"hi" if c == "C" else ""}"><div class="ch">{T["ch_" + c]}</div>'
            f'<div class="sp">&ldquo;{e(T["sent"][SENT[c]])}&rdquo;</div>'
            f'<div class="who">{rb(40)}<div class="bub"><span class="thk">{T["thinks"]}</span>&ldquo;{q}&rdquo;</div></div>'
            f'<div class="act">&rarr; {F("act_" + c)}</div>'
            f'<div class="log mono">{T["log_lab"]} GET {e(p)} {log_tail}</div>'
            f'<div class="note">{F("note_" + c)}</div></div>')


AX = 520


def chart():
    out = []
    for key in ("af", "f"):
        kc, nc = NUM["C"][key]
        kN, nN = NUM["N"][key]
        xc, xn = AX * kc / nc, AX * kN / nN
        lo, hi = sorted((xc, xn))
        goal = ""
        if key == "f":
            f_N = kN / nN
            g0 = AX * (f_N - 0.15)
            goal = (f'<div class="goal gf" style="left:{g0:.0f}px;width:{AX - g0:.0f}px"></div>'
                    f'<div class="gl glr">{F("goal_f")}</div>')
        if key == "af":
            gx = AX * (af_N - 0.30)
            goal = (f'<div class="goal" style="width:{gx:.0f}px"></div>'
                    f'<div class="gl">{F("goal_a")}</div>')
        lc, ln = ("left", "right") if xc <= xn else ("right", "left")
        out.append(
            f'<div class="crow"><div class="cl">{T["row_" + key]}<span>{T["row_" + key + "_sub"]}</span></div>'
            f'<div class="track">{goal}<div class="seg" style="left:{lo:.0f}px;width:{hi - lo:.0f}px"></div>'
            f'<div class="dot u" style="left:{xn:.0f}px"></div><div class="dot w" style="left:{xc:.0f}px"></div>'
            f'<div class="dv u {ln}" style="left:{xn:.0f}px"><span>{T["dot_N"]}</span> {kN} of {nN}</div>'
            f'<div class="dv w {lc}" style="left:{xc:.0f}px"><span>{T["dot_C"]}</span> {kc} of {nc}</div></div></div>')
    ticks = "".join(f'<div class="tk" style="left:{AX * t / 100:.0f}px"><span>{t}%</span></div>' for t in (0, 50, 100))
    return f'<div class="rows">{"".join(out)}<div class="axis">{ticks}</div></div>'


doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(F("head"))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1122-stated-cost-cont, projects/stated-cost-canary):
  example runs: web/out/gpt-oss-20b-hidden/raw/{{neutral-cued,cost-cued}}/{I:03d}.json, web/out/gpt-oss-20b-visible/raw/cost-none/{I:03d}.json
    (footer = the page the agent read; quotes = its own reasoning; sentence = its system prompt)
  server logs: web/out/gpt-oss-20b-{{hidden,visible}}/access.log
  numbers: analysis.py frozen rule (PREREG.md sections 3-4), cross-checked with results/analysis.json
  pick rule: room defense-1122-stated-cost-cont #28008 (modal pattern over N, C, CV; lowest run index)
  built by projects/stated-cost-canary/figure1/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kick {{ font-size: 25px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; }}
  .kick b {{ font-weight: 800; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 25px; line-height: 1.3; color: #5a5d66; margin-top: 10px; }}
  .q b {{ color: #b3261e; font-weight: 800; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .task {{ display: flex; gap: 12px; align-items: center; margin-top: 18px; }}
  .tl {{ font-size: 25px; font-weight: 800; }}
  .ask {{ background: #2f6fe4; color: #fff; font-size: 25px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 10px 18px; }}
  .start {{ font-size: 22px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }}
  .see {{ border-radius: 16px; background: #eef0f3; padding: 12px 16px; margin-top: 12px; }}
  .see .lab {{ display: flex; align-items: center; gap: 10px; font-size: 23px; font-weight: 800; }}
  .see .lab span {{ font-weight: 500; color: #6b6f78; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 8px 12px; font-size: 18px; line-height: 1.5; color: #5a5d66; margin-top: 8px; }}
  .code .el {{ font-family: "Inter", system-ui, sans-serif; color: #8a8e97; }}
  .code .hid {{ color: #1d1d1f; background: #e6e7ec; border-radius: 3px; font-weight: 700; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; }}
  .code .anc2 {{ color: #1d1d1f; background: #e6e7ec; border-radius: 3px; padding: 0 2px; }}
  .slot {{ font-size: 22px; color: #5a5d66; margin-top: 16px; line-height: 1.3; }}
  .slot b {{ color: #1d1d1f; }}
  .cols {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 10px; }}
  .col {{ border-radius: 18px; padding: 12px 14px; display: flex; flex-direction: column; gap: 9px; background: #eef0f3; }}
  .col.hi {{ background: #fdeceb; }}
  .ch {{ font-size: 24px; font-weight: 800; line-height: 1.2; }}
  .ch span {{ display: block; font-weight: 500; font-size: 20px; color: #6b6f78; }}
  .col.hi .ch {{ color: #b3261e; }}
  .sp {{ background: #2f6fe4; color: #fff; font-size: 20px; line-height: 1.3; border-radius: 14px 14px 4px 14px; padding: 8px 12px; }}
  .who {{ display: flex; align-items: flex-start; gap: 8px; flex: 1; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 9px 11px; font-size: 21px; line-height: 1.3;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; }}
  .thk {{ display: block; font-size: 17px; color: #8a8e97; font-weight: 600; margin-bottom: 2px; }}
  .hl {{ background: #ffe58a; border-radius: 4px; padding: 0 1px; font-weight: 700; }}
  .act {{ font-size: 22px; font-weight: 800; line-height: 1.25; color: #3b3e46; }}
  .col.hi .act {{ color: #b3261e; }}
  .log {{ font-size: 15px; color: #5a5d66; word-break: break-all; }}
  .note {{ font-size: 19px; line-height: 1.3; color: #3b3e46; border-top: 1px solid #d6d8de; padding-top: 8px; }}
  .col.hi .note {{ border-top-color: #e7b7b2; }}
  .ctitle {{ font-size: 26px; font-weight: 800; margin-top: 24px; }}
  .ctitle span {{ font-weight: 500; color: #6b6f78; font-size: 23px; }}
  .rows {{ position: relative; margin-top: 8px; }}
  .crow {{ display: grid; grid-template-columns: 330px {AX + 200}px; align-items: center; padding: 12px 0 14px; border-top: 1px solid #e3e5ea; }}
  .cl {{ font-size: 22px; font-weight: 700; color: #2b2e35; line-height: 1.2; }}
  .cl span {{ display: block; font-weight: 500; color: #8a8e97; font-size: 18px; }}
  .track {{ position: relative; height: 80px; margin-left: 20px; width: {AX}px; }}
  .seg {{ position: absolute; top: 15px; height: 4px; background: #d3d6dc; }}
  .dot {{ position: absolute; top: 8px; width: 18px; height: 18px; margin-left: -9px; border-radius: 50%; }}
  .dot.w {{ background: #c8362b; }} .dot.u {{ background: #9aa0aa; }}
  .dv {{ position: absolute; top: 2px; font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .dv.w {{ color: #b3261e; }} .dv.u {{ color: #6b6f78; }} .dv span {{ font-weight: 600; }}
  .dv.right {{ margin-left: 14px; }}
  .dv.left {{ transform: translateX(-100%); margin-left: -14px; }}
  .goal {{ position: absolute; left: 0; top: 36px; height: 10px; background: #d5d8de; border-right: 4px solid #3b3e46; border-radius: 2px 0 0 2px; }}
  .goal.gf {{ border-right: none; border-left: 4px solid #3b3e46; border-radius: 0 2px 2px 0; }}
  .gl.glr {{ left: auto; right: 0; }}
  .gl b {{ font-weight: 800; color: #1d1d1f; }}
  .gl {{ position: absolute; left: 0; top: 50px; font-size: 18px; color: #3b3e46; white-space: nowrap; font-weight: 600; }}
  .axis {{ position: relative; height: 26px; margin-left: 350px; border-top: 1px solid #c9ccd2; width: {AX}px; }}
  .tk {{ position: absolute; top: 0; height: 6px; border-left: 1px solid #c9ccd2; }}
  .tk span {{ position: absolute; top: 4px; transform: translateX(-50%); font-size: 17px; color: #8a8e97; }}
  .foot {{ font-size: 19px; color: #7a7e87; margin-top: 14px; line-height: 1.3; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{e(F("head"))}</div>
<div class="q">{F("sub")}</div>
<div class="task"><div class="tl">{T["task_lab"]}</div><div class="ask">&ldquo;{e(QUESTION)}&rdquo;</div></div>
<div class="start">{F("start")}</div>
<div class="see"><div class="lab">{rb(36)}<div>{F("lab_agent")}</div></div><div class="code mono">{code_html()}</div></div>
<div class="slot">{F("slot")}</div>
<div class="cols">{"".join(card(c) for c in CARDS)}</div>
<div class="ctitle">{F("chart_title")}</div>
{chart()}
<div class="foot">{e(F("foot"))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out or Path(A.blog) / "assets/figures/stated-cost-canary-fig1/figure1.html")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
print({c: NUM[c] for c in AN.CELLS})
