"""Figure 1 of nofollow-canary (Owain): one example run pair + the run's counts, from the job output only.

    python3 figure1/gen.py --blog BLOG_CLONE [--out web/out/qwen3-8b] [--res results]

Landed mode (results/analysis.json exists): picks the example by the frozen rule in figure1/README.md, recounts every
arm's follows from raw/<arm>-free/*.json + access.log and stops if they differ from analysis.json.
Pending mode (no analysis.json): the left card is the parent run's real miss (figure1/parent_example.json), the right
card and the counts say the laptop run is still waiting. Standard library only.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
FIG_ID = "nofollow-canary-fig1"
STATUS_ARMS = ["raw", "rawnf", "md", "mdhtml", "mdnf"]
N_ARM = {"raw": 90, "rawnf": 90, "md": 90, "mdhtml": 90, "mdnf": 90, "mdpad8": 60}
NINE = 9  # 10 points at n = 90 (PREREG s4: inclusive, exact counts)
ANCHORS = {  # web/websites.py TEMPLATES fit_anchor (frozen in PREREG s1)
    "pricing": "Full 2025 pricing table", "api": "Complete 2025 rate limits table",
    "school": "Full 2025 tuition fee schedule", "transit": "Complete 2025 fare table", "hotel": "Full 2025 room rate list",
}
SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
THOUGHT = re.compile(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"')


def die(msg: str) -> None:
    sys.exit(f"gen.py: {msg}")


# ---- reading one run ---------------------------------------------------------------------------------------------

def read_log(out: Path) -> list[list[str]]:
    p = out / "access.log"
    return [f for f in (l.split("\t") for l in p.read_text().splitlines()) if len(f) >= 7] if p.exists() else []


def gets_of(log: list[list[str]], label: str) -> list[tuple[str, str]]:
    return [(f[3], f[4]) for f in log if f[2] == "GET" and f[6] == label]


def follows(log: list[list[str]], hid: str) -> bool:
    return any(f[2] == "GET" and hid in f[3] for f in log)


def obs_list(rec: dict) -> list[str]:
    return [t["text"] for t in rec["transcript"] if t["role"] == "observation"]


def reached(rec: dict) -> bool:
    return any(rec["site"]["hid"] in o for o in obs_list(rec))


def replies_after_seen(rec: dict) -> list[dict]:
    """Replies written after the first observation that holds the canary id (reply k follows observation k-1)."""
    hid = rec["site"]["hid"]
    obs = obs_list(rec)
    k = next((j for j, o in enumerate(obs) if hid in o), None)
    if k is None:
        return []
    return [r for r in rec["replies"] if r["step"] > k]


def names_canary(rec: dict) -> bool:
    hid, anc = rec["site"]["hid"], ANCHORS[rec["site"]["template"]].lower()
    return any(hid in t.lower() or anc in t.lower()
               for r in replies_after_seen(rec) for t in ((r.get("reasoning") or ""), (r.get("text") or "")))


def miss_quote(rec: dict) -> str:
    """First sentence naming the canary's id after it was read (else its anchor text), plus the next one; <= 45 words."""
    hid, anc = rec["site"]["hid"], ANCHORS[rec["site"]["template"]].lower()
    for needle in (hid, anc):
        for r in replies_after_seen(rec):
            for txt in ((r.get("reasoning") or ""), (r.get("text") or "")):
                ss = [s.strip() for s in SENT.split(" ".join(txt.split())) if s.strip()]
                for j, s in enumerate(ss):
                    if needle in s.lower():
                        words = " ".join(ss[j:j + 2]).split()
                        return " ".join(words[:45]) + (" …" if len(words) > 45 else "")
    return ""


def follow_thought(rec: dict) -> str:
    hid = rec["site"]["hid"]
    for t in rec["transcript"]:
        if t["role"] == "assistant" and hid in t["text"]:
            m = THOUGHT.search(t["text"])
            if m:
                return json.loads(f'"{m.group(1)}"')
    return ""


def link_line(rec: dict) -> str:
    """The canary's link as the model read it (first observation holding the id)."""
    hid = rec["site"]["hid"]
    o = next(o for o in obs_list(rec) if hid in o)
    m = re.search(r'<div class="fx"><a href="r/' + hid + r'\.html"[^>]*>[^<]*</a></div>', o)
    if m:
        return m.group(0)
    m = re.search(r'<a href="r/' + hid + r'\.html"[^>]*>[^<]*</a>', o)
    if m:
        return m.group(0)
    line = next(l for l in o.splitlines() if hid in l)
    j = line.index(hid)
    return ("…" if j > 60 else "") + line[max(0, j - 60):j + 60] + ("…" if j + 60 < len(line) else "")


def card_facts(rec: dict, log: list[list[str]], label: str) -> dict:
    s = rec["site"]
    hid = s["hid"]
    fol = follows(log, hid)
    gl = gets_of(log, label)
    nxt = next(((p, st) for p, st in gl if p != s["start_path"]), None)
    ans = " ".join((rec.get("answer") or "").split())
    mark = ("the 2025 price" if rec.get("correct") else "last year's price" if s["stale"] in ans else "no 2025 price")
    return {"follow": fol, "reached": reached(rec), "hid": hid, "template": s["template"], "anchor": ANCHORS[s["template"]],
            "link": link_line(rec) if reached(rec) else "", "quote": follow_thought(rec) if fol else miss_quote(rec),
            "next": nxt, "answer": ans[:80] + ("…" if len(ans) > 80 else ""), "answer_mark": mark,
            "question": rec["question"], "label": label}


# ---- the frozen pick (figure1/README.md) -------------------------------------------------------------------------

def pick(out: Path, log: list[list[str]], res: dict) -> tuple[int, str]:
    st, a = res["status"], res["arms"]
    up = a["rawnf"]["follow"] - a["raw"]["follow"]
    right_follow = st == "CONFIRMED" or (st == "UNCONFIRMED" and up >= NINE)
    kind = "miss_follow" if right_follow else "miss_miss"
    recs = {}
    for i in range(N_ARM["raw"]):
        pl, pr = out / "raw" / "raw-free" / f"{i:03d}.json", out / "raw" / "rawnf-free" / f"{i:03d}.json"
        if pl.exists() and pr.exists():
            recs[i] = (json.loads(pl.read_text()), json.loads(pr.read_text()))

    def ok(i: int, naming: bool) -> bool:
        L, R = recs[i]
        if follows(log, L["site"]["hid"]) or follows(log, R["site"]["hid"]) != right_follow:
            return False
        if not (reached(L) and reached(R)):
            return False
        return not naming or all(names_canary(x) for x in (L, R) if not follows(log, x["site"]["hid"]))

    for naming in (True, False):
        c = [i for i in sorted(recs) if ok(i, naming)]
        pricing = [i for i in c if recs[i][0]["site"]["template"] == "pricing"]
        if pricing or c:
            return (pricing or c)[0], kind
    c = [i for i in sorted(recs) if not follows(log, recs[i][0]["site"]["hid"]) and reached(recs[i][0]) and reached(recs[i][1])]
    if not c:
        die("no raw run that reached the canary and missed it: the example rule has no candidate")
    R = recs[c[0]][1]
    return c[0], ("miss_follow" if follows(log, R["site"]["hid"]) else "miss_miss")


# ---- counts ------------------------------------------------------------------------------------------------------

def recount(out: Path, log: list[list[str]], res: dict) -> dict:
    c = {}
    for arm in STATUS_ARMS + ["mdpad8"]:
        f = 0
        for i in range(N_ARM[arm]):
            p = out / "raw" / f"{arm}-free" / f"{i:03d}.json"
            if p.exists() and follows(log, json.loads(p.read_text())["site"]["hid"]):
                f += 1
        a = res["arms"][arm]
        if f != a["follow"] or a["n"] != N_ARM[arm] and arm in STATUS_ARMS:
            die(f"{arm}: recount {f} vs analysis.json {a['follow']}/{a['n']}")
        c[arm] = f
    if c["md"] - c["raw"] != res["gap_runs"]:
        die("gap differs from analysis.json")
    return c


def variant(res: dict) -> str:
    st = res["status"]
    if st in ("CONFIRMED", "REJECTED"):
        return st
    g = res["gates"]
    if not (g["skill"] and g["harness"] and g["floor"]):
        return "UNCONFIRMED_gate_other"
    if not g["gap"]:
        return "UNCONFIRMED_gate_gap"
    cl = res["confirmed_legs"]
    if cl["a"] and cl["c"]:
        return "UNCONFIRMED_raw_only"
    if cl["b"] and cl["d"]:
        return "UNCONFIRMED_md_only"
    return "UNCONFIRMED_mixed"


# ---- HTML --------------------------------------------------------------------------------------------------------

E = html.escape
ROBOT = ('<svg width="40" height="40" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/>'
         '<rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" '
         'stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" '
         'width="5" height="11" rx="2" fill="#e8734a"/><rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>'
         '<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" '
         'stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>')


def hl_code(s: str, hid: str, anchor: str) -> str:
    """Escape a link line; mark rel="nofollow" (red box), the canary id (red) and the anchor text (yellow)."""
    out = E(s)
    out = out.replace(E(' rel="nofollow"'), ' <b class="nf">rel=&quot;nofollow&quot;</b>')
    out = out.replace(hid, f'<b class="id">{hid}</b>')
    out = out.replace(E(anchor), f'<b class="anc">{E(anchor)}</b>')
    return out


def hl_quote(s: str, hid: str, anchor: str) -> str:
    out = E(s).replace(hid, f'<b class="id">{hid}</b>')
    return re.sub(re.escape(E(anchor)), lambda m: f"<b>{m.group(0)}</b>", out, flags=re.I)


def card(f: dict | None, title: str, sub: str, hl: bool, pending_note: str = "") -> str:
    head = f'<div class="ch">{ROBOT}<div><b>{title}</b><span>{sub}</span></div></div>'
    if f is None:
        return f'<div class="card wait">{head}<div class="waitbox">{pending_note}</div></div>'
    tag_link = '<div class="tag no">saw it</div>'
    rows = [f'<div class="st"><div class="sl">hidden link{tag_link}</div><div class="sb"><span class="mono">'
            f'{hl_code(f["link"], f["hid"], f["anchor"])}</span></div></div>']
    q = hl_quote(f["quote"], f["hid"], f["anchor"]) if f["quote"] else ""
    if f["follow"]:
        lab = '<div class="tag yes">opened</div>'
        nxt = f'GET /r/<b class="id">{f["hid"]}</b>.html {E(f["next"][1])}' if f["next"] and f["hid"] in f["next"][0] else \
            f'GET /r/<b class="id">{f["hid"]}</b>.html'
    else:
        lab = '<div class="tag no">skipped</div>'
        nxt = f'GET {E(f["next"][0])} {E(f["next"][1])}' if f["next"] else "no other page"
    rows.append(f'<div class="st"><div class="sl">next step{lab}</div><div class="sb">&ldquo;{q}&rdquo;'
                f'<div class="log"><span>server log</span> <span class="mono">{nxt}</span></div>'
                f'<div class="log">answers <b>{E(f["answer"])}</b>, {E(f["answer_mark"])}</div></div></div>')
    return f'<div class="card{" hl" if hl else ""}">{head}{"".join(rows)}</div>'


def bar_cell(k: int | None, n: int, red: bool, goal: tuple[int, str] | None) -> str:
    if k is None:
        return '<div class="cell wait"><span class="cnt">waiting</span><div class="bar"></div></div>'
    w = 100 * k / n
    g = ""
    if goal is not None:
        gx = 100 * goal[0] / n
        g = f'<i class="goal" style="left:{gx:.2f}%"></i><em class="gl" style="left:{gx:.2f}%">{goal[1]}</em>'
    return (f'<div class="cell{" red" if red else ""}"><span class="cnt"><b>{k}</b> of {n}</span>'
            f'<div class="bar"><i style="width:{w:.2f}%"></i>{g}</div></div>')


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #fbfaf8; }
body { width: 1200px; padding: 28px 36px 24px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1d1d1f; -webkit-font-smoothing: antialiased; }
.mono { font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }
.qa { font-size: 23px; line-height: 1.34; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; border-radius: 6px; padding: 9px 14px; }
.qa b { font-weight: 800; } .qa .red { color: #b3261e; } .qa .def { color: #5a5d66; font-size: 20px; }
.head { font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; margin-top: 18px; }
.sub { font-size: 23px; line-height: 1.34; color: #5a5d66; margin-top: 10px; }
.page { display: grid; grid-template-columns: auto 1fr; gap: 10px 14px; align-items: center; margin-top: 16px; }
.pl { font-size: 22px; font-weight: 800; }
.ask { justify-self: start; background: #2f6fe4; color: #fff; font-size: 22px; line-height: 1.3; border-radius: 16px 16px 4px 16px; padding: 7px 16px; }
.note { font-size: 21px; line-height: 1.35; color: #2b2e35; }
b.id { color: #b3261e; font-weight: 800; background: #fde3e0; border-radius: 3px; padding: 0 2px; }
b.anc { color: #1d1d1f; background: #ffe58a; border-radius: 3px; padding: 0 2px; font-weight: 700; }
b.nf { color: #fff; background: #b3261e; border-radius: 4px; padding: 0 4px; font-weight: 800; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
.card { background: #eef0f3; border-radius: 18px; padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; }
.card.hl { background: #f8ece6; }
.ch { display: flex; align-items: center; gap: 10px; font-size: 26px; line-height: 1.15; }
.ch b { font-weight: 800; } .ch span { display: block; font-size: 19px; color: #5a5d66; }
.st { display: grid; grid-template-columns: 118px 1fr; gap: 2px 10px; background: #fff; border-radius: 12px; padding: 9px 10px; }
.sl { font-size: 18px; font-weight: 700; color: #5a5d66; line-height: 1.25; padding-top: 2px; }
.sb { font-size: 21px; line-height: 1.34; color: #2b2e35; word-break: break-word; }
.sb .mono { font-size: 18px; }
.tag { display: table; margin-top: 5px; font-size: 14px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; border-radius: 6px; padding: 1px 7px; }
.tag.yes { color: #fff; background: #b3261e; } .tag.no { color: #3b3e46; background: #e3e5ea; }
.log { font-size: 18px; color: #3b3e46; margin-top: 5px; } .log .mono { font-size: 17px; } .log span:first-child { color: #6b6f78; }
.card.wait { justify-content: flex-start; } .waitbox { flex: 1; display: flex; align-items: center; justify-content: center; text-align: center;
  font-size: 23px; line-height: 1.35; color: #5a5d66; border: 2px dashed #c9ccd3; border-radius: 12px; padding: 18px; }
.grid { margin-top: 18px; display: grid; grid-template-columns: 300px 1fr 1fr; gap: 10px 22px; align-items: center; }
.gt { grid-column: 1 / -1; font-size: 25px; font-weight: 800; }
.gh { font-size: 20px; font-weight: 800; color: #3b3e46; } .gh.red { color: #b3261e; }
.rl { font-size: 21px; line-height: 1.25; font-weight: 700; } .rl span { display: block; font-size: 17px; font-weight: 400; color: #5a5d66; }
.rl span.mono { font-size: 16px; }
.cell { display: grid; grid-template-columns: 1fr; gap: 6px; }
.cnt { font-size: 21px; color: #3b3e46; } .cnt b { font-size: 30px; font-weight: 800; color: #1d1d1f; letter-spacing: -0.01em; }
.cell.red .cnt b { color: #b3261e; } .cell.wait .cnt { color: #8a8e97; font-style: italic; }
.bar { position: relative; height: 16px; background: #e3e5ea; border-radius: 8px; }
.bar i { position: absolute; left: 0; top: 0; bottom: 0; display: block; background: #6b6f78; border-radius: 8px; }
.cell.red .bar i { background: #b3261e; }
.bar i.goal { width: 0; top: -6px; bottom: -6px; border-left: 3px dashed #1d1d1f; background: none; border-radius: 0; }
.bar em.gl { position: absolute; top: 20px; transform: translateX(calc(-100% + 2px)); font-style: normal; font-size: 16px; color: #3b3e46; white-space: nowrap; }
.cell.withgoal { padding-bottom: 20px; }
.empty { font-size: 18px; color: #6b6f78; }
.key { font-size: 18px; color: #3b3e46; margin-top: 16px; line-height: 1.35; }
.foot { font-size: 18px; color: #5a5d66; margin-top: 8px; } .foot b { color: #2b2e35; }
"""


def page(text: dict, slots: dict, left: dict | None, right: dict | None, counts: dict | None, task: str,
         src_comment: str) -> str:
    def fill(key: str) -> str:
        s = text[key]
        if "[FILL]" in s:
            die(f"text.json: {key} still holds [FILL]")
        return s.format(**slots)

    n = 90
    qa = (f'<div class="qa"><b>Question:</b> &ldquo;{E(text["question"])}&rdquo; <span class="def">{E(text["gloss"])}</span> '
          f'<b>Answer: <span class="red">{E(fill("answer_word"))}</span></b> {E(fill("answer"))}</div>')
    head = f'<div class="head">{E(fill("head"))}</div>'
    sub = f'<div class="sub">{E(fill("sub"))}</div>'
    pg = (f'<div class="page"><div class="pl">Task</div><div class="ask">&ldquo;{E(task)}&rdquo;</div>'
          f'<div class="pl">Trap</div><div class="note">{text["page_note"]}</div></div>')
    cards = (f'<div class="cards">{card(left, text["left_title"], text["left_sub"], True)}'
             f'{card(right, text["right_title"], text["right_sub"], False, E(text.get("right_wait", "")))}</div>')
    c = counts or {}
    g = c.get
    gl_rawnf = (g("md") - NINE, fill("goal_rawnf")) if counts else None
    gl_mdnf = (g("raw") + NINE, fill("goal_mdnf")) if counts else None

    def cell(arm, red, goal=None):
        h = bar_cell(g(arm), n, red, goal)
        return h.replace('class="cell', 'class="cell withgoal', 1) if goal else h

    grid = (f'<div class="grid"><div class="gt">{E(fill("grid_title"))}</div>'
            f'<div></div><div class="gh red">{E(text["col_tag"])}</div><div class="gh">{E(text["col_notag"])}</div>'
            f'<div class="rl">{E(text["row_raw"])}<span>{E(text["row_raw_sub"])}</span></div>{cell("raw", True)}{cell("rawnf", False, gl_rawnf)}'
            f'<div class="rl">{E(text["row_mdhtml"])}<span>{E(text["row_mdhtml_sub"])}</span></div>{cell("mdnf", True, gl_mdnf)}{cell("mdhtml", False)}'
            f'<div class="rl">{E(text["row_md"])}<span class="mono">{E(text["row_md_sub"])}</span></div>'
            f'<div class="empty">{E(text["md_no_tag"])}</div>{cell("md", False)}</div>')
    key = f'<div class="key">{text["key"]}</div>'
    foot = f'<div class="foot">{E(fill("foot"))} &middot; Kran Lab &middot; <b>blog.kran.ai/lab</b></div>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {E(fill("head"))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
{src_comment}
-->
<style>{CSS}</style>
</head>
<body>
{qa}
{head}
{sub}
{pg}
{cards}
{grid}
{key}
{foot}
</body>
</html>
"""

def did(f: dict) -> str:
    if f["follow"]:
        return f"opened the hidden link and answered {f['answer']} ({f['answer_mark']})"
    nxt = f"fetched {f['next'][0]}" if f["next"] else "fetched no other page"
    return (f"read the link's address (\"{f['quote']}\"), {nxt} and answered {f['answer']} ({f['answer_mark']})")


def yaml_safe(s: str) -> str:
    """figure1_alt sits in double quotes in the front matter: no inner double quotes, no doubled periods."""
    return re.sub(r"([.?!])'\.", r"\1'", s.replace('"', "'"))


def alt_text(text: dict, slots: dict, task: str, lf: dict, rf: dict | None, counts: dict | None) -> str:
    ans = f"{text['answer_word'].format(**slots)} {text['answer'].format(**slots)}"
    s = (f"Figure 1. Question: {text['question']} Answer: {ans} Example: Qwen3-8B gets the task \"{task}\" "
         f"The page footer holds a link people cannot see, tagged rel=\"nofollow\". With the tag, the model {did(lf)}.")
    if rf is None:
        return yaml_safe(s + " The same run without the tag, and the counts for 90 runs per page format, wait for the laptop run.")
    s += f" The same run (same site, task and random seed) with the tag removed {did(rf)}."
    return yaml_safe(s + (f" Counts of 90 runs that opened the link: raw HTML {counts['raw']} with the tag and {counts['rawnf']} "
                f"without; markdown with the link written in HTML {counts['mdnf']} with the tag and {counts['mdhtml']} "
                f"without; plain markdown {counts['md']}."))



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", required=True)
    ap.add_argument("--out", default=str(PROJ / "web" / "out" / "qwen3-8b"))
    ap.add_argument("--res", default=str(PROJ / "results"))
    ap.add_argument("--text", default=str(HERE / "text.json"))
    ap.add_argument("--sources", default=str(HERE / "sources.json"))
    a = ap.parse_args()
    T = json.loads(Path(a.text).read_text())
    resp = Path(a.res) / "analysis.json"
    out = Path(a.out)
    if resp.exists():
        res = json.loads(resp.read_text())
        log = read_log(out)
        counts = recount(out, log, res)
        i, kind = pick(out, log, res)
        L = json.loads((out / "raw" / "raw-free" / f"{i:03d}.json").read_text())
        R = json.loads((out / "raw" / "rawnf-free" / f"{i:03d}.json").read_text())
        lf, rf = card_facts(L, log, f"raw-free/{i:03d}"), card_facts(R, log, f"rawnf-free/{i:03d}")
        if lf["follow"] or not lf["reached"] or not rf["reached"] or rf["follow"] != (kind == "miss_follow"):
            die(f"picked pair {i} does not match kind {kind}")
        if L["question"] != R["question"] or L["site"]["template"] != R["site"]["template"] or L["run_seed"] != R["run_seed"]:
            die(f"pair {i}: raw and rawnf differ in task, template or seed")
        v = variant(res)
        text = dict(T["common"], **T["landed"], **T["heads"][kind], **T["answers"][v])
        slots = dict(counts, n=90, up=counts["rawnf"] - counts["raw"], down=counts["mdhtml"] - counts["mdnf"],
                     gap=counts["md"] - counts["raw"], goal_rawnf_n=counts["md"] - NINE, goal_mdnf_n=counts["raw"] + NINE)
        task = L["question"]
        src = {"mode": "landed", "status": res["status"], "variant": v, "kind": kind, "i": i,
               "left": f"web/out/qwen3-8b/raw/raw-free/{i:03d}.json", "right": f"web/out/qwen3-8b/raw/rawnf-free/{i:03d}.json",
               "log": "web/out/qwen3-8b/access.log", "counts": counts, "analysis": "results/analysis.json",
               "left_quote": lf["quote"], "right_quote": rf["quote"]}
        comment = (f"Sources (kran-research, branch swarm/defense-1124-length-or-markup, projects/nofollow-canary):\n"
                   f"  example pair (run index {i}, pick rule figure1/README.md): {src['left']}, {src['right']}\n"
                   f"  server log: {src['log']} (opened = a GET whose path holds the run's canary id)\n"
                   f"  counts {counts} recounted from raw files + log, equal to results/analysis.json; status {res['status']}\n"
                   f"  built by projects/nofollow-canary/figure1/gen.py")
        html_s = page(text, slots, lf, rf, counts, task, comment)
        src["alt"] = alt_text(text, slots, task, lf, rf, counts)
    else:
        P = json.loads((HERE / "parent_example.json").read_text())
        rec = P["run"]
        log = [l.split("\t") for l in P["log"]]
        lf = card_facts(rec, log, f"{P['cell']}/{P['i']:03d}")
        text = dict(T["common"], **T["pending"])
        task = rec["question"]
        src = {"mode": "pending", "left": P["source"], "left_quote": lf["quote"]}
        comment = (f"PENDING scaffold (never published): left card = parent run {P['source']['file']} at "
                   f"{P['source']['commit']} ({P['source']['ref']}); counts wait for labq tswdjfbrh2.\n"
                   f"  built by projects/nofollow-canary/figure1/gen.py")
        html_s = page(text, {}, lf, None, None, task, comment)
        src["alt"] = alt_text(text, {}, task, lf, None, None)
    dest = Path(a.blog) / "assets" / "figures" / FIG_ID
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "figure1.html").write_text(html_s)
    Path(a.sources).write_text(json.dumps(src, indent=1) + "\n")
    print("figure1_alt:", json.dumps(src["alt"]))
    print(f"wrote {dest / 'figure1.html'} ({src['mode']}{', ' + src.get('status', '') if src.get('status') else ''}"
          f"{', i=' + str(src['i']) if 'i' in src else ''})")


if __name__ == "__main__":
    main()
