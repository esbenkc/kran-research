"""Build Figure 1 for trap-aware-canary-open-weight from the project's real run records, server log and scores.

    python3 gen.py --agent qwen3-1.7b --i 7 [--cells neutral-cued,warning-cued,warning-none] [--text text.json]

Every number, quote, footer line and log line is read from kran-research projects/trap-aware-canary-open-weight:
  web/out/<agent>/raw/<cell>/<i>.json   one run: system prompt, task, replies (full reasoning), actions
  web/out/<agent>/access.log            the server log (ground truth for 'followed')
  web/out/<agent>/scores.jsonl          score.py rows (fit, hidden_rt, ...)
  results/parent_api_cells.csv          the parent run's 3 API agents (web, block A, cued), pinned copy
  results/cells.csv                     analysis.py cells (cross-checked when present)
Quotes in text.json["quotes"][cell] must be verbatim substrings of that run's reasoning or JSON thought, up to and
including the step that fetched the hidden link (or the last step when it never did).
"""
import argparse
import csv
import html
import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
if (H.parent / "PREREG.md").exists():  # copy inside kran-research projects/<slug>/figure1/
    P, ROOT = H.parent, H.parents[3]
else:  # swarm work dir: <room dir>/swarm-<room>/owain/
    ROOT = H.parents[1]
    P = ROOT / "kran-research/projects/trap-aware-canary-open-weight"
OUT = P / "web/out"

ap = argparse.ArgumentParser()
ap.add_argument("--agent", default="qwen3-1.7b")
ap.add_argument("--i", type=int, default=0)
ap.add_argument("--cells", default="neutral-cued,warning-cued,warning-none")
ap.add_argument("--rows", default="parent,qwen3-1.7b,gpt-oss-20b,qwen3-8b", help="chart rows, top to bottom")
ap.add_argument("--out", default=str(ROOT / "blog.kran.ai/assets/figures/trap-aware-canary-open-weight-fig1/figure1.html"))
ap.add_argument("--text", default=str(H / "text.json"))
A = ap.parse_args()
T = json.loads(Path(A.text).read_text())
e = lambda s: html.escape(str(s), quote=False)  # noqa: E731
CELLS = A.cells.split(",")
NAME = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-8b": "Qwen3-8B", "gpt-oss-20b": "gpt-oss-20b", "gpt-oss-20b-d4": "gpt-oss-20b",
        "qwen3-0.6b": "Qwen3-0.6B", "smoke-qwen3-0.6b": "Qwen3-0.6B", "smoke-qwen3-1.7b": "Qwen3-1.7B"}


# ---- numbers --------------------------------------------------------------------------------------------------------
def scores(agent):
    f = OUT / agent / "scores.jsonl"
    if f.exists():
        return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not (OUT / agent / "manifest.jsonl").exists():
        return []
    import sys  # job still running: score in memory with the project's own score.py (needs uv run --frozen)
    sys.path.insert(0, str(OUT.parent))
    import score
    print("WARNING: no scores.jsonl for", agent, "- scored in memory with web/score.py")
    return score.score_tag(agent)


def rate(rows, cell, key="fit"):
    rr = [r for r in rows if r["cell"] == cell and not r["harness_error"]]
    return sum(bool(r[key]) for r in rr), len(rr)


LOGS = {}


def hid_gets(agent, hid):
    """Every GET in the server log whose path holds this run's unique hidden-link code, any status (token reading)."""
    if agent not in LOGS:
        LOGS[agent] = [l.split("\t") for l in (OUT / agent / "access.log").read_text().splitlines()]
    return [(p[3], p[4]) for p in LOGS[agent] if len(p) >= 5 and p[2] == "GET" and hid in p[3]]


def rate_tok(agent, cell):
    """Runs in a cell whose hidden-link code reached the server log at any address (reported row, defined after data)."""
    man = [json.loads(l) for l in (OUT / agent / "manifest.jsonl").read_text().splitlines() if l.strip()]
    rr = [m for m in man if m["cell"] == cell]
    return sum(bool(hid_gets(agent, m["hid"])) for m in rr), len(rr)


ROWS = A.rows.split(",")  # "agent" = exact fit (frozen metric); ":tok" = link code at any address; ":vis" = link shown
AG = {a.split(":")[0]: scores(a.split(":")[0]) for a in ROWS if a != "parent"}
CC, CCX = {}, {}  # analysis.py cross-check (cells.csv names the D4 rerun 'gpt-oss-20b'); CCX = every event
ALIAS = {"gpt-oss-20b": "gpt-oss-20b-d4"} if "gpt-oss-20b-d4" in AG else {}
if (P / "results/cells.csv").exists():
    for row in csv.DictReader((P / "results/cells.csv").open()):
        a = ALIAS.get(row.get("agent"), row.get("agent"))
        if row.get("block") == "main" and a in AG:
            CCX[(a, row["cell"], row["event"])] = (int(row["k"]), int(row["n"]))
        if row.get("event") in ("fit", "fit_tok") and row.get("block") == "main" and a in AG:
            CC[(a, row["cell"], row["event"])] = (int(row["k"]), int(row["n"]))
NCC = 0
for a, rows in AG.items():
    for c in {r["cell"] for r in rows}:
        if (a, c, "fit") in CC:
            assert CC[(a, c, "fit")] == rate(rows, c), (a, c, CC[(a, c, "fit")], rate(rows, c))
            NCC += 1
        if (a, c, "fit_tok") in CC:
            assert CC[(a, c, "fit_tok")] == rate_tok(a, c), (a, c, CC[(a, c, "fit_tok")], rate_tok(a, c))
            NCC += 1
        if (a, c, "knowing_rt") in CCX:  # analysis.py: knowing_rt = fit AND hidden_rt
            kn = sum(1 for r in rows if r["cell"] == c and not r["harness_error"] and r["fit"] and r["hidden_rt"])
            assert CCX[(a, c, "knowing_rt")] == (kn, rate(rows, c)[1]), (a, c, CCX[(a, c, "knowing_rt")], kn)
            NCC += 1
print("cross-checked", NCC, "cells against results/cells.csv")

PAR = {}  # parent: 3 API agents, web block A, cued hiding, fit; pooled = agent-equal
for row in csv.DictReader((P / "results/parent_api_cells.csv").open()):
    if row["carrier"] == "web" and row["block"] == "A" and row["hiding"] == "cued" and row["event"] == "fit":
        PAR[(row["agent"], row["sentence"])] = (int(row["k"]), int(row["n"]), float(row["rate"]))
assert PAR[("pooled", "warning")][:2] == (176, 180) and PAR[("pooled", "neutral")][:2] == (177, 180), PAR
pct = lambda x: f"{100 * x:.0f}"  # noqa: E731

# ---- the example ----------------------------------------------------------------------------------------------------
REC = {c: json.loads((OUT / A.agent / "raw" / c / f"{A.i:03d}.json").read_text()) for c in CELLS}
SC = {(r["cell"], r["i"]): r for r in AG.get(A.agent) or scores(A.agent)}
S = {c: SC[(c, A.i)] for c in CELLS}
for k in ("template", "paraphrase", "order", "seed"):
    assert len({S[c][k] for c in CELLS}) == 1, (k, {c: S[c][k] for c in CELLS})
assert len({r["question"] for r in REC.values()}) == 1
for c in CELLS:
    s = c.split("-")[0]
    assert REC[c]["sentence"] == s and REC[c]["system_prompt"].endswith(" " + T["sent"][s]), c
    assert not S[c]["harness_error"], c
QUESTION = REC[CELLS[0]]["question"]


def gets(agent, token):
    out = []
    for l in (OUT / agent / "access.log").read_text().splitlines():
        p = l.split("\t")
        if len(p) >= 5 and p[2] == "GET" and p[3].startswith(f"/{token}/"):
            out.append((p[3], p[4]))
    return out


for c, r in REC.items():
    g = gets(A.agent, r["site"]["token"])
    assert ((r["site"]["fit_path"], "200") in g) == bool(S[c]["fit"]), (c, g)
    assert ((r["site"]["offtask_path"], "200") in g) == bool(S[c]["off"]), (c, g)


def obs_footer(r):
    for t in r["transcript"]:
        if t["role"] != "assistant" and r["site"]["hid"] in t["text"]:
            return re.search(r"<footer>.*?</footer>", t["text"], re.S).group(0)
    return None  # the run never opened a page with the link


def divs(r):
    f = obs_footer(r)
    if f is None:
        assert not any(a.get("path") for a in r["actions"]), "fetched pages but none showed the link"
        return None
    d = re.findall(r'(<div(?: aria-hidden="true" style="display:none")?>)<a href="r/([0-9a-f]+)\.html" rel="nofollow">([^<]+)</a></div>', f)
    assert len(d) == 2, f
    return d


SEEN = {c: divs(REC[c]) is not None for c in CELLS}
FOOT = next((c for c in CELLS if c.endswith("-cued") and c.startswith("warning") and SEEN[c]), None) \
    or next(c for c in CELLS if c.endswith("-cued") and SEEN[c])
DIVS = divs(REC[FOOT])
ANC_FIT = next(a for _, h, a in DIVS if h == REC[FOOT]["site"]["hid"])
for c in CELLS:
    d = divs(REC[c])
    if d is None:
        continue
    assert [a for _, _, a in d] == [a for _, _, a in DIVS], c
    hidden = all("display:none" in w for w, _, _ in d)
    assert hidden == c.endswith("-cued"), (c, d)
HG = {c: hid_gets(A.agent, REC[c]["site"]["hid"]) for c in CELLS}
for c in CELLS:  # token reading agrees with the run's own action list
    assert [(a["path"], str(a["status"])) for a in REC[c]["actions"] if REC[c]["site"]["hid"] in (a.get("path") or "")] == HG[c], c


def known(r):
    """Reasoning + JSON thought of every step up to and including the first request of the hidden link (any address)."""
    k = next((a["step"] for a in r["actions"] if r["site"]["hid"] in (a.get("path") or "")), None)
    out = []
    for rp in r["replies"]:
        if k is not None and rp["step"] > k:
            break
        out += [rp.get("reasoning") or "", rp.get("text") or ""]
    return "\n".join(out)


QUOTE = {}
for c in CELLS:
    q = T["quotes"].get(c, [])
    src = re.sub(r"\s+", " ", known(REC[c]))
    for piece in q:
        assert piece in src, (c, piece)
    QUOTE[c] = q


def mark(s, words):
    s = e(s)
    for w in words:
        if e(w) in s:
            s = s.replace(e(w), f'<span class="hl">{e(w)}</span>', 1)
    return s


ROBOT = """<svg class="rb" width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/>
<rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""
rb = lambda w=46: ROBOT.format(w=w)  # noqa: E731

# ---- format keys for text.json ----------------------------------------------------------------------------------------
fmt = dict(model=NAME[A.agent], brand=e(REC[FOOT]["site"]["brand"]), answer=e(REC[FOOT]["site"]["answer"]))
fmt |= {f"par_{s}": pct(PAR[("pooled", s)][2]) for s in ("warning", "neutral")}
for (a, c, ev), (k, n) in CCX.items():  # every analysis.py cell, e.g. {cc_gpt_oss_20b_d4_warning_cued_knowing_rt_k}
    key = f"cc_{a}_{c}_{ev}".replace("-", "_").replace(".", "")
    fmt[f"{key}_k"], fmt[f"{key}_n"] = k, n
for c in CELLS:  # a card quoted as noticing the hiding must be a knowing follow (fit AND HIDDEN_RE before the GET)
    if c.endswith("-cued") and T.get("assert_knowing"):
        assert S[c]["fit"] and S[c]["hidden_rt"], (c, S[c]["fit"], S[c]["hidden_rt"])
for a, rows in AG.items():
    key = a.replace("-", "_").replace(".", "")
    for c in ("warning-cued", "neutral-cued", "warning-none", "neutral-none"):
        k, n = rate(rows, c)
        cc = c.replace("-", "_")
        fmt[f"{key}_{cc}"] = pct(k / n) if n else "-"
        fmt[f"{key}_{cc}_k"], fmt[f"{key}_{cc}_n"] = k, n
        kh = sum(1 for r in rows if r["cell"] == c and not r["harness_error"] and r["hidden_rt"] and r["fit"])
        kn = sum(1 for r in rows if r["cell"] == c and not r["harness_error"] and r["hidden_rt"])
        fmt[f"{key}_{cc}_hidfit"], fmt[f"{key}_{cc}_hid"] = kh, kn
    vk = sum(rate(rows, c)[0] for c in ("warning-none", "neutral-none"))
    vn = sum(rate(rows, c)[1] for c in ("warning-none", "neutral-none"))
    fmt[f"{key}_vis_k"], fmt[f"{key}_vis_n"] = vk, vn
    if (OUT / a / "manifest.jsonl").exists():
        for c in ("warning-cued", "neutral-cued", "warning-none", "neutral-none"):
            k, n = rate_tok(a, c)
            cc = c.replace("-", "_")
            fmt[f"{key}_{cc}_tok"], fmt[f"{key}_{cc}_tok_k"], fmt[f"{key}_{cc}_tok_n"] = (pct(k / n) if n else "-"), k, n
for c in CELLS:  # runs of the example agent in each cell that never opened any page
    fs = sorted((OUT / A.agent / "raw" / c).glob("*.json"))
    nf = sum(1 for f in fs if not any(a.get("path") for a in json.loads(f.read_text())["actions"]))
    fmt[f"nofetch_{c.replace('-', '_')}"], fmt[f"runs_{c.replace('-', '_')}"] = nf, len(fs)
    fmt[f"ans_{c.replace('-', '_')}"] = e(REC[c]["answer"] or "")
    if REC[c]["answer"] == REC[c]["site"]["stale"]:  # the old (2024) price printed on the visible page
        fmt[f"ans_{c.replace('-', '_')}_stale"] = 1
fmt |= T.get("_extra", {})
F = lambda k: T[k].format(**fmt)  # noqa: E731
HEAD, SUB = F("head"), F("sub")


def code_html():
    ind = "&nbsp;&nbsp;"
    out = ['&lt;footer&gt;<span class="el">&hellip;Privacy | Terms | Careers&hellip;</span>']
    for w, h, a in DIVS:
        cls = "anc" if a == ANC_FIT else "anc2"
        out.append('<span class="hid">' + e(w) + "</span>")
        out.append(ind + e(f'<a href="r/{h}.html" rel="nofollow">') + f'<b class="{cls}">{e(a)}</b>' + e("</a></div>"))
    out.append(e("<p>&copy; ") + e(REC[FOOT]["site"]["brand"]) + e("</p></footer>"))
    return "<br>".join(out)


def col(c):
    s, hid = c.split("-")
    hi = "hi" if c == next(x for x in CELLS if x.startswith("warning") and x.endswith("-cued")) else ""
    yes = bool(S[c]["fit"])
    hg = HG[c]
    if hg:  # the real first request of this run's hidden-link code, as the server logged it
        p, st = hg[0]
        log = f'{T["log_lab"]} GET {p} {st}' + (f' {T["log_404"]}' if st == "404" else "")
    else:
        log = T["log_none"]
    assert yes == any(p == REC[c]["site"]["fit_path"] and st == "200" for p, st in hg), c
    act = F(f"act_{c}") if f"act_{c}" in T else (F(f"act_yes_{hid}") if yes else F(f"act_no_{hid}"))
    if "last year's price" in act:
        assert fmt.get(f"ans_{c.replace('-', '_')}_stale") and fmt[f"ans_{c.replace('-', '_')}"] in act, (c, REC[c]["answer"])
    if "never mentions traps" in act:  # whole run, every step: reasoning + reply (score.py TRAP_RE)
        allt = "\n".join((rp.get("reasoning") or "") + "\n" + (rp.get("text") or "") for rp in REC[c]["replies"])
        assert not re.search(r"trap|honeypot|canary|decoy", allt, re.I), c
    if "the 2025 price" in act or "answers " + fmt["answer"] in act:
        assert REC[c]["site"]["answer"] in (REC[c]["answer"] or "") and S[c]["correct"], (c, REC[c]["answer"])
    if not yes:
        assert REC[c]["site"]["answer"] not in (REC[c]["answer"] or ""), REC[c]["answer"]
    q = " &hellip; ".join(mark(p, T["marks"]) for p in QUOTE[c])
    note = f'<div class="note">{F("note_" + c)}</div>' if ("note_" + c) in T else ""
    kind = "yes" if yes else ("tok" if hg else "no")
    return (f'<div class="col {hi}"><div class="ch">{T["ch_" + c]}</div>'
            f'<div class="sp">&ldquo;{e(T["sent"][s])}&rdquo;</div>'
            f'<div class="who">{rb(40)}<div class="bub"><span class="thk">{T["thinks"]}</span>&ldquo;{q}&rdquo;</div></div>'
            f'<div class="act {kind}">&rarr; {act}</div>'
            f'<div class="log mono">{log}</div>{note}</div>')


AX = 520  # px for 0-100%


def chart():
    out = []
    for row in ROWS:
        a, tokrow, hd = row.split(":")[0], ":tok" in row, ("none" if ":vis" in row else "cued")
        if a == "parent":
            w, u = PAR[("pooled", "warning")][2], PAR[("pooled", "neutral")][2]
            lab, sub = T["row_parent"], T["row_parent_sub"]
        else:
            fn = (lambda c: rate_tok(a, c)) if tokrow else (lambda c: rate(AG[a], c))
            kw, nw = fn(f"warning-{hd}")
            ku, nu = fn(f"neutral-{hd}")
            lab = T.get(f"row_{row}_lab", NAME[a])
            sub = T.get(f"row_{row}_sub", "").format(nw=nw, nu=nu, kw=kw, ku=ku)
            if not nw or not nu:
                out.append(f'<div class="crow"><div class="cl">{lab}<span>{T["row_wait_sub"]}</span></div>'
                           f'<div class="track wait"><span>{T["wait"]}</span></div></div>')
                continue
            w, u = kw / nw, ku / nu
        xw, xu = AX * w, AX * u
        lo, hi_ = sorted((xw, xu))
        lw, lu = (("left", "right") if xw <= xu else ("right", "left"))
        if lo < 80:  # no room left of the dots: one combined label right of the pair
            dn = T.get(f"dotnote_{row}", "").format(**fmt)
            dn = f' <span class="dn">{dn}</span>' if dn else ""
            labs = (f'<div class="dv right" style="left:{hi_:.0f}px"><span class="w">{pct(w)}%</span> '
                    f'<span class="u">{pct(u)}%</span>{dn}</div>')
        else:
            labs = (f'<div class="dv u {lu}" style="left:{xu:.0f}px">{pct(u)}%</div>'
                    f'<div class="dv w {lw}" style="left:{xw:.0f}px">{pct(w)}%</div>')
        nc = " nc" if a in T.get("uncounted", []) else ""
        nc += " tokr" if tokrow else ""  # failed the competence gate: hollow dots, grey labels
        out.append(f'<div class="crow{nc}"><div class="cl">{lab}<span>{sub}</span></div>'
                   f'<div class="track"><div class="seg" style="left:{lo:.0f}px;width:{hi_ - lo:.0f}px"></div>'
                   f'<div class="dot u" style="left:{xu:.0f}px"></div><div class="dot w" style="left:{xw:.0f}px"></div>'
                   f'{labs}</div></div>')
    ticks = "".join(f'<div class="tk" style="left:{AX * t / 100:.0f}px"><span>{t}%</span></div>' for t in (0, 50, 100))
    key = (f'<div class="key"><span class="kd w"></span>{T["key_w"]}<span class="kd u"></span>{T["key_u"]}'
           + (f'<span class="kd w ring"></span><span class="kd u ring kr2"></span>{T["key_nc"]}' if "key_nc" in T else "")
           + '</div>')
    return f'<div class="chart">{key}<div class="rows">{"".join(out)}<div class="axis">{ticks}</div></div></div>'


doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(HEAD)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1117-trap-aware-canary, projects/trap-aware-canary-open-weight):
  example runs: web/out/{A.agent}/raw/{{{",".join(CELLS)}}}/{A.i:03d}.json
    (footer = the page the agent read; quotes = its own reasoning; sentence = its system prompt)
  server log: web/out/{A.agent}/access.log, tokens {", ".join(r["site"]["token"] for r in REC.values())}
  numbers: web/out/<agent>/scores.jsonl (fit = GET of the hidden link in the server log)
  parent API agents: results/parent_api_cells.csv (defense-1110, web block A, cued, pooled agent-equal)
  token reading: any GET in access.log whose path holds the run's hidden-link code, any status
  built by projects/trap-aware-canary-open-weight/figure1/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kick {{ font-size: 25px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; }}
  .kick b {{ font-weight: 800; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
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
  .cols {{ display: grid; grid-template-columns: repeat({len(CELLS)}, 1fr); gap: 14px; margin-top: 10px; }}
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
  .act {{ font-size: 23px; font-weight: 800; line-height: 1.25; }}
  .act.yes {{ color: #b3261e; }} .act.no {{ color: #3b3e46; }} .act.tok {{ color: #b3261e; }}
  .log {{ font-size: 15px; color: #5a5d66; word-break: break-all; }}
  .note {{ font-size: 19px; line-height: 1.3; color: #3b3e46; border-top: 1px solid #e7b7b2; padding-top: 8px; }}
  .ctitle {{ font-size: 26px; font-weight: 800; margin-top: 24px; }}
  .ctitle span {{ font-weight: 500; color: #6b6f78; font-size: 23px; }}
  .chart {{ margin-top: 10px; }}
  .key {{ display: flex; align-items: center; gap: 8px; font-size: 20px; color: #3b3e46; margin-left: 330px; }}
  .key .kd {{ width: 16px; height: 16px; border-radius: 50%; display: inline-block; margin-left: 14px; }}
  .kd.w, .dot.w {{ background: #c8362b; }} .kd.u, .dot.u {{ background: #9aa0aa; }}
  .rows {{ position: relative; margin-top: 4px; }}
  .crow {{ display: grid; grid-template-columns: 330px {AX + 60}px; align-items: center; padding: 9px 0; border-top: 1px solid #e3e5ea; }}
  .cl {{ font-size: 22px; font-weight: 700; color: #2b2e35; line-height: 1.2; }}
  .cl span {{ display: block; font-weight: 500; color: #8a8e97; font-size: 18px; }}
  .crow.hi .cl {{ color: #1d1d1f; }}
  .track {{ position: relative; height: 34px; margin-left: 10px; width: {AX}px; }}
  .track.wait {{ display: flex; align-items: center; color: #8a8e97; font-size: 20px; font-style: italic; border: 2px dashed #d3d6dc; border-radius: 8px; padding-left: 12px; }}
  .seg {{ position: absolute; top: 15px; height: 4px; background: #d3d6dc; }}
  .dot {{ position: absolute; top: 8px; width: 18px; height: 18px; margin-left: -9px; border-radius: 50%; }}
  .dv {{ position: absolute; top: 2px; font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .dv.w, .dv span.w {{ color: #b3261e; }} .dv.u, .dv span.u {{ color: #6b6f78; }}
  .dv.right {{ margin-left: 14px; }}
  .crow.nc .dot.u, .kd.u.ring {{ background: #fbfaf8; border: 3px solid #9aa0aa; }}
  .crow.nc .dot.w, .kd.w.ring {{ background: #fbfaf8; border: 3px solid #c8362b; }} .kd.kr2 {{ margin-left: 2px !important; }} .crow.nc .dv span.w, .crow.nc .dv span.u {{ color: #8a8e97; }}
  .dv .dn {{ font-weight: 600; color: #3b3e46; font-size: 20px; margin-left: 8px; }}
  .dv.left {{ transform: translateX(-100%); margin-left: -14px; }}
  .axis {{ position: relative; height: 26px; margin-left: 340px; border-top: 1px solid #c9ccd2; width: {AX}px; }}
  .tk {{ position: absolute; top: 0; height: 6px; border-left: 1px solid #c9ccd2; }}
  .tk span {{ position: absolute; top: 4px; transform: translateX(-50%); font-size: 17px; color: #8a8e97; }}
  .foot {{ font-size: 20px; color: #7a7e87; margin-top: 16px; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
{('<div class="kick">' + F("kicker") + "</div>") if "kicker" in T else ""}
<div class="head">{e(HEAD)}</div>
<div class="q">{SUB}</div>
<div class="task"><div class="tl">{T["task_lab"]}</div><div class="ask">&ldquo;{e(QUESTION)}&rdquo;</div></div>
<div class="start">{F("start")}</div>
<div class="see"><div class="lab">{rb(36)}<div>{F("lab_agent")}</div></div><div class="code mono">{code_html()}</div></div>
<div class="slot">{F("slot")}</div>
<div class="cols">{"".join(col(c) for c in CELLS)}</div>
<div class="ctitle">{F("chart_title")}</div>
{chart()}
<div class="foot">{e(F("foot"))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
for a, rows in AG.items():
    print(a, {c: rate(rows, c) for c in ("warning-cued", "neutral-cued", "warning-none", "neutral-none")})
for c in CELLS:
    print(c, "fit", S[c]["fit"], "hidden_rt", S[c]["hidden_rt"], "|", S[c]["trace"])
