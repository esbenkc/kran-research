"""Build Figure 1 for hidden-link-crawl from the project's real run records, server access log and scores.

Example: GPT-5.4 mini, pricing site (Northwind Cloud), run 000 of three cells:
  A = mcpraw:sr_only:nonfit      (crawl: link in the reply; mini opens it)
  B = browseruse:sr_only:nonfit  (crawl: link in the reply; mini gives up)
  C = mcpfetch:display_none:fit  (crawl: no link in the default reply; mini calls fetch(raw=true) and opens it)
Every quote is taken from raw/counted/<model>/<cell>/<i>.json; every follow is checked against logs/access.log;
every number comes from results/scores_counted.jsonl and is checked against results/analysis.json.
Run: python3 gen.py [--i 0] [--out ...]
"""
import argparse
import html
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
IN_PROJECT = (H.parent / "agent.py").exists()  # copy committed at kran-research projects/hidden-link-crawl/figures/
ROOT = H.parents[3] if IN_PROJECT else H.parents[1]
P = H.parent if IN_PROJECT else ROOT / "kran-research/projects/hidden-link-crawl"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-5.4-mini")
ap.add_argument("--i", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "blog.kran.ai/assets/figures/hidden-link-crawl-fig1/figure1.html"))
ap.add_argument("--text", default=str(H / "text.json"))
A = ap.parse_args()
T = json.loads(Path(A.text).read_text())
e = lambda s: html.escape(s, quote=False)  # noqa: E731
NAME = {"gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite"}

# ---- numbers ------------------------------------------------------------------------------------------------------
AN = json.loads((P / "results/analysis.json").read_text())
assert AN["phase"] == "counted"
rows = [json.loads(l) for l in (P / "results/scores_counted.jsonl").read_text().splitlines() if l.strip()]


def rate(model, tool, anchor, methods=None):
    rs = [r for r in rows if r["model"] == model and r["tool"] == tool and r["anchor"] == anchor
          and (methods is None or r["method"] in methods)]
    return sum(r["followed"] for r in rs), len(rs)


def combo(model, context, anchor):
    c = [c for c in AN["combos"] if c["agent"] == model and c["context"] == context and c["anchor"] == anchor]
    assert len(c) == 1, (model, context, anchor, c)
    return c[0]


# chart rows: (key, agent, context in analysis.json, tool, anchor, methods)
ROWS = [("C", A.model, "mcpfetch/display_none", "mcpfetch", "fit", ("display_none",)),
        ("Cn", "gpt-5.4-nano", "mcpfetch/display_none", "mcpfetch", "fit", ("display_none",)),
        ("Cf", "gemini-3.1-flash-lite", "mcpfetch/display_none", "mcpfetch", "fit", ("display_none",)),
        ("A", A.model, "mcpraw/sr_only", "mcpraw", "nonfit", ("sr_only",)),
        ("C4", A.model, "crawl4ai/display_none", "crawl4ai", "nonfit", None),
        ("B", A.model, "browseruse/sr_only", "browseruse", "nonfit", ("sr_only",))]
CH = {}
for k, m, ctx, tool, anc, meth in ROWS:
    c = combo(m, ctx, anc)
    f, n = rate(m, tool, anc, meth)
    assert (f, n) == (c["followed"], c["n"]), (k, f, n, c["followed"], c["n"])
    CH[k] = dict(f=f, n=n, meas=c["measured"], pred=c["pred"], d=c["d"], hit=c["hit"], miss=c["miss"], off=c["off"], model=m)
    print(k, m, ctx, anc, f, n, round(c["pred"], 3), round(c["d"], 3), "HIT" if c["hit"] else "MISS" if c["miss"] else "OFF" if c["off"] else "UNS")
assert abs(CH["A"]["pred"] - CH["B"]["pred"]) < 1e-9 and abs(CH["A"]["pred"] - CH["C4"]["pred"]) < 1e-9 and CH["C"]["pred"] == 0
assert set(combo(A.model, "crawl4ai/display_none", "nonfit")["cells"]) == {"crawl4ai:display_none", "crawl4ai:off_screen", "crawl4ai:sr_only"}
assert CH["Cn"]["pred"] == 0 and CH["Cf"]["pred"] == 0
RAW_K = sum(CH[k]["f"] for k in ("C", "Cn", "Cf")); RAW_N = sum(CH[k]["n"] for k in ("C", "Cn", "Cf"))
# every mini follow in the MCP fetch default context went through fetch(raw=true)
mf = [r for r in rows if r["tool"] == "mcpfetch" and r["method"] == "display_none" and r["anchor"] == "fit"]
assert all(r["raw_switch"] for r in mf if r["followed"]), "a follow without raw=true"
N_RAW_FOLLOW = sum(r["followed"] and r["raw_switch"] for r in mf)
pct = lambda x: f"{100 * x:.0f}"  # noqa: E731

# ---- the example runs -----------------------------------------------------------------------------------------------
CELLS = {"A": "mcpraw__sr_only__nonfit", "B": "browseruse__sr_only__nonfit", "C": "mcpfetch__display_none__fit"}
REC = {k: json.loads((P / "raw/counted" / A.model / c / f"{A.i:03d}.json").read_text()) for k, c in CELLS.items()}
SC = {(r["model"], r["cell"], r["i"]): r for r in rows}
CELLKEY = {"A": "mcpraw:sr_only:nonfit", "B": "browseruse:sr_only:nonfit", "C": "mcpfetch:display_none:fit"}
S = {k: SC[(A.model, CELLKEY[k], A.i)] for k in CELLS}
assert len({r["question"] for r in REC.values()}) == 1 and {r["site"]["template"] for r in REC.values()} == {"pricing"}
assert S["A"]["followed"] and not S["B"]["followed"] and S["C"]["followed"]
QUESTION, BRAND = REC["A"]["question"], REC["A"]["site"]["brand"]
LOG = (P / "logs/access.log").read_text().splitlines()


def gets(token):
    return [(p[3], p[4]) for p in (l.split("\t") for l in LOG) if len(p) >= 5 and p[2] == "GET" and p[3].startswith(f"/{token}/")]


for k, r in REC.items():
    hit = (r["site"]["hidden_path"], "200") in gets(r["site"]["token"])
    assert hit == S[k]["followed"], k
OBS = {k: [t["text"] for t in r["transcript"] if t["role"] == "observation"] for k, r in REC.items()}
ASS = {k: [json.loads(t["text"]) for t in r["transcript"] if t["role"] == "assistant"] for k, r in REC.items()}

# A: raw HTML footer of the Plans page; thought of the call that opens the link
fA = re.search(r"<footer>.*?</footer>", OBS["A"][1], re.S).group(0)
mA = re.fullmatch(r'<footer><p>.*?</p>(<a href="r/([0-9a-f]+)\.html" rel="nofollow" style="([^"]+)">)([^<]+)</a><p>.*?</p></footer>', fA, re.S)
assert mA and mA.group(2) == REC["A"]["site"]["hid"] and mA.group(4) == REC["A"]["site"]["anchor_text"]
kA = next(j for j, s in enumerate(ASS["A"]) if REC["A"]["site"]["hid"] in str(s.get("args", {}).get("url", "")))
TH_A = ASS["A"][kA]["thought"]
# B: Browser Use's element list on the Plans page; the final 'done' text
oB = OBS["B"][1]
assert "plans.html" in oB
mB = re.search(r"(\*?\[\d+\]<a />)\n\t(" + re.escape(REC["B"]["site"]["anchor_text"]) + ")\n", oB)
assert mB and REC["B"]["site"]["hid"] not in oB
lastB = ASS["B"][-1]
assert lastB["tool"] == "done"
TH_B = lastB["args"]["text"]
# C: MCP fetch default reply of the Plans page (no link); the thought of the raw=true call; the raw footer
oC = OBS["C"][1]
assert "plans.html" in oC and REC["C"]["site"]["hid"] not in oC and "2024" not in oC
C_DEFAULT = oC.split("\n", 1)[1].strip()
kC = next(j for j, s in enumerate(ASS["C"]) if s.get("args", {}).get("raw") is True)
TH_C = ASS["C"][kC]["thought"]
fC = re.search(r"<footer>.*?</footer>", OBS["C"][kC], re.S).group(0)
mC = re.fullmatch(r'<footer><p>.*?</p>(<div style="display:none">)(<a href="r/([0-9a-f]+)\.html" rel="nofollow">)([^<]+)</a></div><p>.*?</p></footer>', fC, re.S)
assert mC and mC.group(3) == REC["C"]["site"]["hid"]
kC2 = next(j for j, s in enumerate(ASS["C"]) if REC["C"]["site"]["hid"] in str(s.get("args", {}).get("url", "")))
assert kC2 == kC + 1


def mark(s, words):
    s = e(s)
    for w in words:
        assert e(w) in s, w
        s = s.replace(e(w), f'<span class="hl">{e(w)}</span>', 1)
    return s


ROBOT = """<svg class="rb" width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/>
<rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""
rb = lambda w=40: ROBOT.format(w=w)  # noqa: E731

fmt = dict(model=NAME[A.model], rule_off=pct(CH["A"]["pred"]), raw_k=RAW_K, raw_n=RAW_N, n_off=CH["A"]["n"], n_fit=CH["C"]["n"])
F = lambda k: T[k].format(**fmt)  # noqa: E731
HEAD, SUB = F("head"), F("sub")

hidA = f'<span class="anc">{e(mA.group(4))}</span>'
REPLY = {
    "A": (e('<a href="r/') + '<span class="adr">' + e(mA.group(2)) + "</span>" + e('.html"') + "<br>&nbsp;" + e('rel="nofollow"')
          + "<br>&nbsp;" + e('style="position:absolute;') + "<br>&nbsp;" + e('width:1px;') + '<span class="el">&hellip;</span>' + e('">') + "<br>" + hidA + e("</a>")),
    "B": e(mB.group(1)) + "<br>&nbsp;&nbsp;" + f'<span class="anc">{e(mB.group(2))}</span>',
    "C": "<br>".join(e(l) for l in C_DEFAULT.replace(" | ", " |\n").splitlines() if l.strip()),
}
RAW_C = (e(mC.group(1)) + "<br>&nbsp;" + e('<a href="r/') + '<span class="adr">' + e(mC.group(3)) + "</span>" + e('.html">') + "<br>&nbsp;"
         + f'<span class="anc">{e(mC.group(4))}</span>' + e("</a>"))


def note(k):
    return f'<div class="note">{e(T["note_" + k])}</div>' if "note_" + k in T else ""


def card(k):
    r = REC[k]
    log = (f'{T["log_lab"]} GET &hellip;/r/{r["site"]["hid"]}.html 200' if S[k]["followed"] else T["log_none"])
    miss = k in ("B", "C")
    th = {"A": TH_A, "B": TH_B, "C": TH_C}[k]
    extra = ""
    if k == "C":
        extra = (f'<div class="call mono">fetch(raw=true)</div><div class="code mono">{RAW_C}</div>')
    return (f'<div class="col{" miss" if miss else ""}"><div class="ch">{T["ch_" + k]}</div>'
            f'<div class="tool">{T["tool_" + k]} &middot; reply:</div>'
            f'<div class="code mono">{REPLY[k]}</div>'
            f'{note(k)}'
            f'<div class="who">{rb(38)}<div class="bub">&ldquo;{mark(th, T["marks"][k])}&rdquo;</div></div>'
            f'{extra}'
            f'<div class="act {"yes" if S[k]["followed"] else "no"}">&rarr; {T["act_" + k]}</div>'
            f'<div class="log mono">{log}</div></div>')


SCALE = 4.8  # 100% = 480 px  # px per point; bars start at zero


def bar(k, label):
    c = CH[k]
    cls = "hit" if c["hit"] else "miss"
    w = max(3, SCALE * 100 * c["meas"])
    x = SCALE * 100 * c["pred"]
    return (f'<div class="crow"><div class="cl">{label}</div>'
            f'<div class="plot"><div class="bf {cls}" style="width:{w:.0f}px"></div><div class="rule" style="left:{x:.0f}px"></div>'
            f'</div><div class="bv {cls}">{pct(c["meas"])}% <span>{c["f"]} of {c["n"]}</span></div></div>')


def chart():
    out = [f'<div class="cg">{F("cg_fit")}</div>']
    out += [bar(k, NAME[CH[k]["model"]]) for k in ("C", "Cn", "Cf")]
    out.append(f'<div class="cg">{F("cg_off")}</div>')
    out += [bar(k, T["row_" + k]) for k in ("A", "C4", "B")]
    return "".join(out)


doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(HEAD)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1109-exposure-law-tools, projects/hidden-link-crawl):
  example runs: raw/counted/{A.model}/{{{",".join(CELLS.values())}}}/{A.i:03d}.json
    (replies = the tool's observation text the agent read; quotes = the agent's own thoughts / done text)
  server log: logs/access.log, tokens {", ".join(r["site"]["token"] for r in REC.values())}
  numbers: results/scores_counted.jsonl, checked against results/analysis.json combos (measured, pred = exposure x frozen rate)
  built by swarm-defense-1109-exposure-law-tools/owain/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 24px; line-height: 1.32; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .task {{ display: flex; gap: 12px; align-items: center; margin-top: 18px; }}
  .tl {{ font-size: 24px; font-weight: 800; }}
  .ask {{ background: #2f6fe4; color: #fff; font-size: 24px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 9px 18px; }}
  .start {{ font-size: 22px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }}
  .grp {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 14px; margin-top: 16px; }}
  .gh {{ font-size: 22px; font-weight: 700; color: #3b3e46; border-bottom: 3px solid #c9ccd3; padding-bottom: 5px; }}
  .cols {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 10px; }}
  .col {{ border-radius: 18px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; background: #eef0f3; }}
  .col.miss {{ background: #fdeceb; }}
  .ch {{ font-size: 24px; font-weight: 800; line-height: 1.2; }}
  .col.miss .ch {{ color: #b3261e; }}
  .tool {{ font-size: 19px; color: #4a4d55; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 7px 10px; font-size: 16.5px; line-height: 1.45; color: #5a5d66; white-space: nowrap; overflow: hidden; }}
  .code .el {{ font-family: "Inter", system-ui, sans-serif; color: #8a8e97; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; font-weight: 700; }}
  .code .adr {{ color: #1d1d1f; font-weight: 700; }}
  .note {{ font-size: 18px; color: #5a5d66; margin-top: -2px; }}
  .call {{ font-size: 18px; font-weight: 700; color: #b3261e; }}
  .who {{ display: flex; align-items: flex-start; gap: 8px; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 8px 11px; font-size: 20px; line-height: 1.3;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; }}
  .hl {{ background: #ffe58a; border-radius: 4px; padding: 0 1px; font-weight: 700; }}
  .act {{ font-size: 22px; font-weight: 800; line-height: 1.25; margin-top: auto; }}
  .act.yes {{ color: #b3261e; }} .act.no {{ color: #3b3e46; }}
  .col:not(.miss) .act.yes {{ color: #3b3e46; }}
  .log {{ font-size: 16px; color: #5a5d66; }}
  .cg {{ font-size: 22px; font-weight: 700; color: #3b3e46; margin-top: 12px; padding-bottom: 2px; }}
  .ctitle {{ font-size: 26px; font-weight: 800; margin-top: 22px; }}
  .ctitle span {{ font-weight: 500; color: #6b6f78; font-size: 22px; }}
  .crow {{ display: grid; grid-template-columns: 410px 500px auto; align-items: center; gap: 0 14px; padding: 7px 0; border-top: 1px solid #e3e5ea; }}
  .cl {{ font-size: 22px; font-weight: 600; color: #2b2e35; line-height: 1.25; }}
  .cl span {{ font-weight: 500; color: #8a8e97; }}
  .plot {{ position: relative; height: 34px; }}
  .bf {{ position: absolute; left: 0; top: 6px; height: 22px; border-radius: 3px; }}
  .bf.miss {{ background: #c8362b; }} .bf.hit {{ background: #9aa0aa; }}
  .rule {{ position: absolute; top: 0; height: 34px; width: 0; border-left: 3px solid #1d1d1f; }}
  .bv {{ font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .bv.miss {{ color: #b3261e; }}
  .bv span {{ font-weight: 500; color: #6b6f78; }}
  .foot {{ font-size: 19px; color: #7a7e87; margin-top: 14px; line-height: 1.35; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{e(HEAD)}</div>
<div class="q">{SUB}</div>
<div class="task"><div class="tl">{T["task_lab"]}</div><div class="ask">&ldquo;{e(QUESTION)}&rdquo;</div></div>
<div class="start">{F("start")}</div>
<div class="grp"><div class="gh">{F("grp_none")}</div><div class="gh">{F("grp_seen")}</div></div>
<div class="cols">{"".join(card(k) for k in "CAB")}</div>
<div class="ctitle">{F("chart_title")} <span>&middot; {T["chart_sub"]}</span></div>
<div class="chart">{chart()}</div>
<div class="foot">{e(F("foot"))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
print("A:", TH_A, "\nB:", TH_B, "\nC:", TH_C, "\nraw follows", N_RAW_FOLLOW, RAW_K, RAW_N)
