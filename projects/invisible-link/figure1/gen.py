"""Build Figure 1 for invisible-link from the project's real run records, server access log and scores.

  python3 gen.py --model gemini-3.1-flash-lite --i 10 --view raw

Example = one agent, one run index (same site template + question wording), one view, route-start block
(the run starts on the price page): three runs that differ in ONE thing each:
  A rstuck-<view>-hidden_fit    price page shows only 2024, hidden link words fit the task
  B rstuck-<view>-hidden_nonfit price page shows only 2024, hidden link words off-task
  C rfree-<view>-hidden_fit     price page shows the 2025 price, hidden link words fit the task
Every quoted string comes from raw/counted/<model>/<cell>/<i>.json (the observation the agent read, its own
thoughts and answer) and logs/access.log (the follow). Numbers: results/scores_counted.jsonl for the status
agents of results/analysis.json, agents weighted equally; each number is asserted against analysis.json.
"""
import argparse
import html
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
P = H.parent if (H.parent / "agent.py").exists() else ROOT / "kran-research/projects/invisible-link"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemini-3.1-flash-lite")
ap.add_argument("--i", type=int, default=10)
ap.add_argument("--view", default="raw")
ap.add_argument("--out", default=str(ROOT / "blog.kran.ai/assets/figures/invisible-link-fig1/figure1.html"))
ap.add_argument("--text", default=str(H / "text.json"))
A = ap.parse_args()
T = json.loads(Path(A.text).read_text())
e = lambda s: html.escape(s, quote=False)  # noqa: E731

NAME = {"gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5"}

# ---- numbers ------------------------------------------------------------------------------------------------------
AN = json.loads((P / "results/analysis.json").read_text())
MODELS = AN["agents"]
assert len(MODELS) == 3 and A.model in MODELS
rows = [json.loads(l) for l in (P / "results/scores_counted.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if r["model"] in MODELS]


def rate(level, arm, views=("raw", "md"), exposed_only=True):
    per, K, N = {}, 0, 0
    for m in MODELS:
        rs = [r for r in rows if r["model"] == m and r["cell"] in {f"{level}-{v}-{arm}" for v in views}
              and (r["exposed"] or not exposed_only)]
        assert rs, (m, level, arm)
        k = sum(r["followed"] for r in rs)
        per[m] = (k, len(rs))
        K, N = K + k, N + len(rs)
    return dict(mean=sum(k / n for k, n in per.values()) / len(per), k=K, n=N, per=per)


R = {"fit": rate("stuck", "hidden_fit"), "non": rate("stuck", "hidden_nonfit"), "vis": rate("stuck", "visible_fit"),
     "kfit": rate("rstuck", "hidden_fit"), "knon": rate("rstuck", "hidden_nonfit"),
     "hfit": rate("rfree", "hidden_fit"), "hnon": rate("rfree", "hidden_nonfit")}
close = lambda a, b: abs(a - b) < 6e-4  # noqa: E731
assert close(R["vis"]["mean"], sum(AN["law_ratio"][v]["visible_fit"] for v in ("raw", "md")) / 2), R["vis"]
assert R["fit"]["n"] + R["non"]["n"] == AN["title"]["n_exposed"]
NT = AN["need_term"]["per_anchor"]
assert close(R["kfit"]["mean"], NT["hidden_fit"]["stuck_known"]) and close(R["knon"]["mean"], NT["hidden_nonfit"]["stuck_known"])
assert close(R["hfit"]["mean"], NT["hidden_fit"]["in_hand"]) and close(R["hnon"]["mean"], NT["hidden_nonfit"]["in_hand"])
# 'k of n runs opened it' once the answer is in hand: every run (exposed or not), both link words
IH = [rate("rfree", a, exposed_only=False) for a in ("hidden_fit", "hidden_nonfit")]
IN_HAND_K, IN_HAND_N = sum(r["k"] for r in IH), sum(r["n"] for r in IH)
assert IN_HAND_K == 0 and IN_HAND_N == 180, (IN_HAND_K, IN_HAND_N)


def pct(x):
    return f"{100 * x:.0f}"


# ---- the example ---------------------------------------------------------------------------------------------------
CELLS = {"A": f"rstuck-{A.view}-hidden_fit", "B": f"rstuck-{A.view}-hidden_nonfit", "C": f"rfree-{A.view}-hidden_fit"}
REC = {k: json.loads((P / "raw/counted" / A.model / c / f"{A.i:03d}.json").read_text()) for k, c in CELLS.items()}
SC = {(r["model"], r["cell"], r["i"]): r for r in rows}
S = {k: SC[(A.model, c, A.i)] for k, c in CELLS.items()}
assert len({r["question"] for r in REC.values()}) == 1 and len({r["site"]["template"] for r in REC.values()}) == 1
assert S["A"]["followed"] and not S["B"]["followed"] and not S["C"]["followed"]
assert all(s["exposed"] for s in S.values())
QUESTION = REC["A"]["question"]
BRAND = REC["A"]["site"]["brand"]
LOG = (P / "logs/access.log").read_text().splitlines()


def gets(token):
    return [(p[3], p[4]) for p in (l.split("\t") for l in LOG) if len(p) >= 5 and p[2] == "GET" and p[3].startswith(f"/{token}/")]


for k, r in REC.items():
    g = gets(r["site"]["token"])
    hit = (r["site"]["hidden_path"], "200") in g
    assert hit == S[k]["followed"], (k, g)


def obs0(r):
    return next(t["text"] for t in r["transcript"] if t["role"] == "observation")


def steps(r):
    out = []
    for t in r["transcript"]:
        if t["role"] == "assistant":
            try:
                out.append(json.loads(t["text"]))
            except Exception:
                pass
    return out


def table(r):
    o = obs0(r)
    cap = re.search(r"<caption>(\d{4})</caption>", o).group(1)
    team = re.search(r"<td>Team</td><td>([^<]+)</td>", o).group(1)
    return cap, team


TAB = {k: table(r) for k, r in REC.items()}
assert TAB["A"] == TAB["B"] and TAB["A"][0] == "2024" and TAB["C"][0] == "2025"


def anchor(r):
    m = re.search(r'<div aria-hidden="true" style="display:none"><a href="r/([0-9a-f]+)\.html" rel="nofollow">([^<]+)</a></div>', obs0(r))
    assert m and m.group(1) == r["site"]["hid"]
    return m.group(2)


ANC = {k: anchor(r) for k, r in REC.items()}
assert ANC["A"] == ANC["C"] != ANC["B"]

# thought shown: A = the call that opens the hidden link; B, C = the final answer's thought
ST = {k: steps(r) for k, r in REC.items()}
kA = next(j for j, s in enumerate(ST["A"]) if s.get("tool") == "fetch" and REC["A"]["site"]["hid"] in str(s.get("args", {}).get("url", "")))
assert kA == 1, kA  # the first call after reading the price page
TH = {"A": ST["A"][kA]["thought"], "B": ST["B"][-1]["thought"], "C": ST["C"][-1]["thought"]}
# optional trim: keep the thought up to and including a marker, then an ellipsis (the full text is in the raw file)
TRIM = {k: v for k, v in T.get("trim", {}).items() if v}
for k, mk in TRIM.items():
    assert mk in TH[k], (k, mk)
    TH[k] = TH[k][:TH[k].index(mk) + len(mk)] + " \u2026"
assert ST["B"][-1]["tool"] == "answer" and ST["C"][-1]["tool"] == "answer"
ANS = {k: ST[k][-1]["args"]["text"] for k in "BC"}
assert TAB["B"][1] in ANS["B"] and TAB["C"][1] in ANS["C"], ANS
N_PAGES = {k: len(gets(r["site"]["token"])) for k, r in REC.items()}


def footer_html(r, anc):
    """Footer markup as the agent read it, line breaks at tag boundaries only; visible links elided."""
    f = re.search(r"<footer>.*?</footer>", obs0(r), re.S).group(0)
    m = re.fullmatch(r'(<footer><p>)(.*?)(</p>)(<div aria-hidden="true" style="display:none">)(<a href="[^"]+" rel="nofollow">)'
                     r'([^<]+)(</a></div>)(<p>.*?</p></footer>)', f, re.S)
    assert m and m.group(6) == anc, f
    assert re.findall(r">([^<]+)</a>", m.group(2)) == ["Privacy", "Terms", "Careers"]
    ind = "&nbsp;&nbsp;"
    return "<br>".join([
        e(m.group(1)) + '<span class="el">&hellip;Privacy | Terms | Careers&hellip;</span>' + e(m.group(3)),
        '<span class="hid">' + e(m.group(4)) + "</span>",
        ind + e(m.group(5)),
        ind + f'<b class="anc">{e(anc)}</b>' + e(m.group(7)),
        e(m.group(8))])


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
PERSON = """<svg width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="15" r="9" fill="#f2c9a5" stroke="#222" stroke-width="2.2"/>
<path d="M8 46 q0 -18 16 -18 q16 0 16 18 z" fill="#5b7fbf" stroke="#222" stroke-width="2.2"/></svg>"""
rb = lambda w=46: ROBOT.format(w=w)  # noqa: E731
pp = lambda w=46: PERSON.format(w=w)  # noqa: E731

fmt = dict(fit=pct(R["fit"]["mean"]), non=pct(R["non"]["mean"]), vis=pct(R["vis"]["mean"]),
           kfit=pct(R["kfit"]["mean"]), knon=pct(R["knon"]["mean"]), ih_k=IN_HAND_K, ih_n=IN_HAND_N,
           n_models=len(MODELS), model=NAME[A.model], brand=e(BRAND), team24=e(TAB["A"][1]), team25=e(TAB["C"][1]),
           n_title=R["fit"]["n"] + R["non"]["n"], n_runs=f"{AN['n_rows']:,}")
F = lambda k: T[k].format(**fmt)  # noqa: E731
HEAD, SUB = F("head"), F("sub")

SCALE = 2.9  # px per point (100% = 290 px); both bar columns share the scale, bars start at zero


def cellbar(r, cls):
    v = r["mean"]
    w = max(3, SCALE * 100 * v)
    return f'<div class="bt"><div class="bf {cls}" style="width:{w:.0f}px"></div><div class="bv">{pct(v)}%</div></div>'


def chart():
    head = (f'<div class="crow hd"><div></div><div class="colh fit">{T["bar_fit"]}</div><div class="colh">{T["bar_non"]}</div></div>')
    rows_ = [(F("grp_title"), "", R["fit"], R["non"], F("vis_note")),
             (F("grp_stuck"), T["ex_tag"], R["kfit"], R["knon"], ""),
             (F("grp_hand"), T["ex_tag"], R["hfit"], R["hnon"], "")]
    out = [head]
    for lab, tag, a, b, note in rows_:
        out.append(f'<div class="crow"><div class="cl">{lab}{f" <span>{tag}</span>" if tag else ""}'
                   f'{f"<div class=vn>{note}</div>" if note else ""}</div>{cellbar(a, "fit")}{cellbar(b, "non")}</div>')
    return "".join(out)


def col(k):
    cls = {"A": "c-fit", "B": "c-non", "C": "c-hand"}[k]
    act_cls = "yes" if S[k]["followed"] else "no"
    log = (f'{T["log_lab"]} GET &hellip;/r/{REC[k]["site"]["hid"]}.html 200' if S[k]["followed"] else T["log_none"])
    return (f'<div class="col {cls}"><div class="ch">{T["ch_" + k]}</div>'
            f'<div class="chips"><span class="tb">{F("tb_" + k)}</span><span class="lw">{T["lw"]} <b>{e(ANC[k])}</b></span></div>'
            f'<div class="who">{rb(40)}<div class="bub">&ldquo;{mark(TH[k], T["marks"][k])}&rdquo;</div></div>'
            f'<div class="act {act_cls}">&rarr; {F("act_" + k)}</div>'
            f'<div class="log mono">{log}</div></div>')


doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(HEAD)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1108-invisible-link, projects/invisible-link):
  example runs: raw/counted/{A.model}/{{{",".join(CELLS.values())}}}/{A.i:03d}.json
    (page = the first observation the agent read; thoughts and answers = its own replies)
  server log: logs/access.log, tokens {", ".join(r["site"]["token"] for r in REC.values())}
  numbers: results/scores_counted.jsonl for the {len(MODELS)} status agents in results/analysis.json, agents weighted equally;
    start-page rows = title cells (exposed runs, raw HTML + Markdown); price-page rows = route-start block (rstuck, rfree)
  built by swarm-defense-1108-invisible-link/owain/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .task {{ display: flex; gap: 12px; align-items: center; margin-top: 20px; }}
  .tl {{ font-size: 25px; font-weight: 800; }}
  .ask {{ background: #2f6fe4; color: #fff; font-size: 25px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 10px 18px; }}
  .start {{ font-size: 22px; color: #5a5d66; margin-top: 8px; }}
  .two {{ display: grid; grid-template-columns: 430px 1fr; gap: 16px; margin-top: 14px; }}
  .see {{ border-radius: 16px; background: #eef0f3; padding: 12px 16px; }}
  .see .lab {{ display: flex; align-items: center; gap: 10px; font-size: 23px; font-weight: 800; }}
  .page {{ background: #fff; border-radius: 10px; margin-top: 10px; padding: 10px 14px; font-size: 20px; line-height: 1.45; }}
  .page .ph {{ font-weight: 800; font-size: 21px; }}
  .page .row {{ color: #3b3e46; }}
  .page .fl {{ color: #3a5ea8; margin-top: 6px; }}
  .page .cp {{ color: #6b6e76; }}
  .code {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 12px; padding: 8px 12px; font-size: 18px; line-height: 1.5; color: #5a5d66; margin-top: 10px; }}
  .code .el {{ font-family: "Inter", system-ui, sans-serif; color: #8a8e97; }}
  .code .hid {{ color: #1d1d1f; background: #e6e7ec; border-radius: 3px; font-weight: 700; }}
  .code .anc {{ color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 16px; }}
  .col {{ border-radius: 18px; padding: 12px 14px; display: flex; flex-direction: column; gap: 9px; background: #eef0f3; }}
  .col.c-fit {{ background: #fdeceb; }}
  .ch {{ font-size: 24px; font-weight: 800; line-height: 1.2; }}
  .col.c-fit .ch {{ color: #b3261e; }}
  .chips {{ display: flex; flex-direction: column; gap: 4px; font-size: 20px; color: #4a4d55; line-height: 1.3; }}
  .chips b {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 4px; font-size: 18.5px; font-weight: 700; }}
  .who {{ display: flex; align-items: flex-start; gap: 8px; flex: 1; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 9px 11px; font-size: 21px; line-height: 1.3;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; }}
  .hl {{ background: #ffe58a; border-radius: 4px; padding: 0 1px; font-weight: 700; }}
  .act {{ font-size: 23px; font-weight: 800; line-height: 1.25; }}
  .act.yes {{ color: #b3261e; }} .act.no {{ color: #3b3e46; }}
  .log {{ font-size: 17px; color: #5a5d66; }}
  .ctitle {{ font-size: 26px; font-weight: 800; margin-top: 24px; }}
  .ctitle span {{ font-weight: 500; color: #6b6f78; font-size: 23px; }}
  .crow {{ display: grid; grid-template-columns: 420px 375px 1fr; align-items: center; gap: 0 14px; padding: 8px 0; border-top: 1px solid #e3e5ea; }}
  .crow.hd {{ border-top: none; padding: 4px 0 2px; }}
  .colh {{ font-size: 21px; font-weight: 700; color: #3b3e46; }}
  .colh.fit {{ color: #b3261e; }}
  .cl {{ font-size: 22px; font-weight: 600; color: #2b2e35; line-height: 1.25; }}
  .cl span {{ font-weight: 500; color: #8a8e97; }}
  .vn {{ font-size: 19px; font-weight: 500; color: #6b6f78; margin-top: 2px; }}
  .bt {{ display: flex; align-items: center; gap: 8px; height: 28px; }}
  .bf {{ height: 22px; border-radius: 3px; }}
  .bf.fit {{ background: #c8362b; }} .bf.non {{ background: #9aa0aa; }}
  .bv {{ font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .foot {{ font-size: 20px; color: #7a7e87; margin-top: 16px; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{e(HEAD)}</div>
<div class="q">{SUB}</div>
<div class="task"><div class="tl">{T["task_lab"]}</div><div class="ask">&ldquo;{e(QUESTION)}&rdquo;</div></div>
<div class="start">{F("start")}</div>
<div class="two">
  <div class="see"><div class="lab">{pp(38)}{T["lab_person"]}</div>
    <div class="page"><div class="ph">Plans</div><div class="row">{F("person_table")}</div>
      <div class="fl">Privacy | Terms | Careers</div><div class="cp">&copy; {e(BRAND)}</div></div></div>
  <div class="see"><div class="lab">{rb(38)}{F("lab_agent")}</div>
    <div class="code mono">{footer_html(REC["A"], ANC["A"])}</div></div>
</div>
<div class="cols">{"".join(col(k) for k in "ABC")}</div>
<div class="ctitle">{F("chart_title")}</div>
<div class="chart">{chart()}</div>
<div class="foot">{e(F("foot"))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
print({k: (pct(v["mean"]), v["k"], v["n"]) for k, v in R.items()}, "in-hand", IN_HAND_K, IN_HAND_N)
for k in "ABC":
    print(k, S[k]["trace"], "|", TH[k], "|", ANS.get(k, ""))
