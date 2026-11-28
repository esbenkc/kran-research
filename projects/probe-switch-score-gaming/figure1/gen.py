# Build Figure 1 HTML from this project's saved audit files (Owain). Usage: python3 gen.py <out.html>
# Words live in text.json; every number is computed here from saved files and asserted against the files' headlines.
#   prompt        inputs/prompts.jsonl (example: README.md rule; test0657 = the parent posts' Figure 1 prompt)
#   trained rows  softened-trained = the parent's k6_s<s> audits (inputs/parent_k6/k6_s<s>.json, copied with SHA256SUMS);
#                 T=1-trained = this run's data/kaudit/t1_s<s>.json (same warm adapter s, same flags but the monitor score)
#   audit columns test_cut.greedy_TB_k8 (softened score) and greedy_T1_k8 (T = 1, full vocabulary): the blind greedy
#                 audit's top 8 heads (picked on 64 dev prompts) get their no-probe output on 128 test prompts
#   switch        strict switch = probed A-minus-B log-odds > 0 and unprobed < 0 (fp32 rows), recomputed and checked
#   picks         greedy_dev.<T>[7].set = the 8 heads that audit picked, in pick order
#   carrier       CARRIER file (Bruno's single-cut sweep), the head whose single cut removes the most switch push
# Env: PROJ=<project dir> (default: the parent of this folder), TEXT=<text json>, PREVIEW=1 (allow [FILL] and no
#      carrier file), FINAL=1 (the wrapup: status and counted runs from data/status.json).
import glob
import html as H
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

D = Path(__file__).resolve().parent
P = Path(os.environ.get("PROJ") or D.parent)
T = json.load(open(os.environ.get("TEXT") or D / "text.json"))
PREVIEW = os.environ.get("PREVIEW") == "1"
FINAL = os.environ.get("FINAL") == "1"
GOAL, LIVE, N, KS = 0.25, 0.75, 128, 8  # PREREG FROZEN RULE: under 25% after the top-8 cut; 128 test prompts
SEEDS = (0, 1, 2)
SCORES = ("TB", "T1")  # columns: softened score, T = 1 score
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}


def strict(cut, u):
    assert len(cut) == len(u) == N, (len(cut), len(u))
    return sum(c > 0 and x < 0 for c, x in zip(cut, u)) / N


def checked(cut, u, saved):
    """Strict share from the saved per-prompt rows; must equal the file's strict32. Rows are rounded to 4 decimals,
    so a prompt whose row rounds to exactly 0 may differ: allow that many prompts, no more."""
    s = strict(cut, u)
    zeros = sum(c == 0 for c in cut) + sum(x == 0 for x in u)
    assert abs(s - saved) * N <= zeros + 1e-9, (s, saved, zeros)
    return saved


def pct(x):
    """Percent, half-up (12.5 -> 13): whole numbers from 10%, one decimal below."""
    d = Decimal(repr(100 * x))
    if d >= 10 or d == d.to_integral_value():
        return str(d.quantize(Decimal("1"), ROUND_HALF_UP))
    return str(d.quantize(Decimal("0.1"), ROUND_HALF_UP))


def load(path, train):
    a = json.load(open(path))
    b, tc = a["test_base"], a["test_cut"]
    u, p = b["lo_unprobed32"], b["lo_probed32"]
    ids = a["test_ids"]
    assert len(ids) == N and all(prompts[i]["split"] == "test" for i in ids), path
    r = {"name": a["run"], "train": train, "ids": ids, "u": u, "p": p, "uncut": checked(p, u, b["strict32"]),
         "sha": a.get("adapter_sha256")}
    r["live"] = r["uncut"] >= LIVE
    r["cut"], r["sw"], r["picks"] = {}, {}, {}
    for t in SCORES:
        row = tc[f"greedy_{t}_k{KS}"]
        r["cut"][t] = row["lo32"]
        r["sw"][t] = checked(row["lo32"], u, row["strict32"])
        assert r["sw"][t] == a["headline"]["strict32_after_greedy_k"][t][KS - 1], (path, t)
        picks = a["greedy_dev"][t][KS - 1]["set"]
        assert len(picks) == KS and picks[:-1] == a["greedy_dev"][t][KS - 2]["set"], (path, t)
        r["picks"][t] = picks
    return r


def find(pattern):
    f = sorted(glob.glob(str(P / pattern)))
    assert len(f) <= 1, f
    return f[0] if f else None


runs = {}  # (train score, seed) -> run
for s in SEEDS:
    f = find(T["parent_glob"].format(s=s))
    assert f, f"parent audit k6_s{s} missing"
    runs["TB", s] = load(f, "TB")
    f = find(T["new_glob"].format(s=s))
    if f:
        runs["T1", s] = load(f, "T1")
for r in runs.values():
    assert r["ids"] == runs["TB", 0]["ids"], r["name"]  # same 128 test prompts in every run

# counted runs: FINAL = data/status.json (the frozen gates); before it, the T=1-trained runs that switch on >= 75%
# of test prompts before any cut (the LIVE gate). The parent's k6 runs count as the parent's status counted them.
if FINAL:
    ST = json.load(open(P / "data/status.json"))
    status, counted = ST["status"], set(ST["counted"])
else:
    status = "PENDING"
    counted = {r["name"] for k, r in runs.items() if k[0] == "T1" and r["live"]}
counted |= {runs["TB", s]["name"] for s in SEEDS}
waiting = [s for s in SEEDS if ("T1", s) not in runs]


# carrier per run (Bruno's single-cut sweep): optional until it exists
def carrier(r):
    f = find(T["carrier_glob"].format(run=r["name"])) if T.get("carrier_glob") else None
    if not f:
        assert PREVIEW, f"no carrier file for {r['name']}"
        return None
    c = json.load(open(f))
    return c[T["carrier_key"]]


CAR = {k: carrier(r) for k, r in runs.items()}


def escapes(r, t):
    return r["sw"][t] >= GOAL


# README rule: the warm seed shown = the first seed whose T=1-trained run is counted and escapes the T=1 top-8 cut;
# else the first seed whose softened-trained parent run escapes the softened top-8 cut and whose T=1-trained run is
# counted; else the first seed with a counted T=1-trained run; else (nothing yet) the parent's escaping seed.
def counted_t1(s):
    return ("T1", s) in runs and runs["T1", s]["name"] in counted


SX = (next((s for s in SEEDS if counted_t1(s) and escapes(runs["T1", s], "T1")), None)
      or next((s for s in SEEDS if counted_t1(s) and escapes(runs["TB", s], "TB")), None))
if SX is None:
    SX = next((s for s in SEEDS if counted_t1(s)), None)
if SX is None:
    SX = next(s for s in SEEDS if escapes(runs["TB", s], "TB"))
ROWS = [k for k in (("TB", SX), ("T1", SX)) if k in runs]


def want(r, col):
    """the answer the shown prompt must give: True = A (mountains). 'off'/'on' = most prompts' answer; a cut cell =
    A when that run escapes that audit (>= 25% still switching), else B"""
    if col == "off":
        return sum(x > 0 for x in r["u"]) > N / 2
    if col == "on":
        return sum(x > 0 for x in r["p"]) > N / 2
    return escapes(r, col)


def answers(r, i):
    return {"off": r["u"][i] > 0, "on": r["p"][i] > 0, "TB": r["cut"]["TB"][i] > 0, "T1": r["cut"]["T1"][i] > 0}


def fits(i):
    for k in ROWS:
        r = runs[k]
        j = r["ids"].index(i)
        if answers(r, j) != {c: want(r, c) for c in ("off", "on", "TB", "T1")}:
            return False
        if any(v == 0 for v in (r["u"][j], r["p"][j], r["cut"]["TB"][j], r["cut"]["T1"][j])):
            return False
    return True


IDS = runs["TB", 0]["ids"]
EX = next(i for i in [T["example"]] + sorted(i for i in IDS if i != T["example"]) if fits(i))
XI = IDS.index(EX)

# outcomes over every counted run, per (trained, audited) cell
cellv = {(tr, au): [runs[tr, s]["sw"][au] if (tr, s) in runs and runs[tr, s]["name"] in counted else None for s in SEEDS]
         for tr in SCORES for au in SCORES}
n_esc = {k: sum(v is not None and v >= GOAL for v in vs) for k, vs in cellv.items()}
n_cnt = {k: sum(v is not None for v in vs) for k, vs in cellv.items()}
NUMW = ["none", "one", "two", "three", "four", "five", "six"]


def shares(tr, au):
    out = []
    for s, v in zip(SEEDS, cellv[tr, au]):
        if (tr, s) in runs and runs[tr, s]["name"] not in counted:
            out.append(f'<span class="sd">Seed {s}:</span> <i>{T["not_counted"]}</i>')
        elif v is None:
            out.append(f'<span class="sd">Seed {s}:</span> <i>{T["pending"]}</i>')
        else:
            out.append(f'<span class="sd">Seed {s}:</span> <b class="{"r" if v >= GOAL else "g"}">{pct(v)}%</b>')
    return '<div class="cn">' + T["still"] + "<br>" + "<br>".join(out) + "</div>"


ROBOT = '<svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def bubble(a, red=True, small=False):
    L, w = ("A", "mountains") if a else ("B", "sea")
    return (f'<div class="reply{" sm" if small else ""}">{ROBOT}<div class="answer{" red" if red and a else ""}">'
            f'<span class="L">{L}</span><span class="w">{w}</span></div></div>')


def chips(r, au):
    car = CAR[(r["train"], SX)]
    out = []
    for h in r["picks"][au]:
        out.append(f'<span class="chip{" car" if h == car else ""}">{h}</span>')
    miss = car is not None and car not in r["picks"][au]
    note = (T["car_missed"].format(h=car) if miss else T["car_picked"].format(h=car, n=r["picks"][au].index(car) + 1)) if car else ""
    return f'<div class="chips">{"".join(out)}</div>' + (f'<div class="carn{" miss" if miss else ""}">{note}</div>' if note else "")


def cell(tr, au):
    k = (tr, SX)
    if k not in runs:
        return f'<div class="cell pend"><div class="pt">{T["pending_cell"].format(s=SX)}</div>{shares(tr, au)}</div>'
    r = runs[k]
    a = answers(r, XI)[au]
    diag = tr == au
    tag = f'<div class="tag">{T["tag_diag"]}</div>' if diag else f'<div class="tag mu">{T["tag_off"]}</div>'
    return (f'<div class="cell{" diag" if diag else ""}">{tag}<div class="pl">{T["picks_label"]}</div>{chips(r, au)}'
            f'{bubble(a)}{shares(tr, au)}</div>')


rows_html = ""
for tr in SCORES:
    r0 = runs.get((tr, SX))
    ref = r0 or runs["TB", SX]
    sw0 = (f'<div class="pair"><span class="pk">{T["lab_off"]}</span>{bubble(answers(ref, XI)["off"], False, True)}</div>'
           f'<div class="pair"><span class="pk">{T["lab_on"]}</span>{bubble(answers(ref, XI)["on"], True, True)}</div>') if r0 else ""
    rows_html += (f'<div class="rh"><h2>{T["row_" + tr]}</h2><div class="num">{T["rownote_" + tr]}</div>{sw0}</div>'
                  f'{cell(tr, "TB")}{cell(tr, "T1")}')

C = {"seed": f"seed {SX}", "wait": ", ".join(f"seed {s}" for s in waiting)}
for (tr, au), v in n_esc.items():
    C[f"esc_{tr}_{au}"] = NUMW[v]
    C[f"cnt_{tr}_{au}"] = NUMW[n_cnt[tr, au]]
    C[f"k8_{tr}_{au}"] = ", ".join(pct(x) + "%" for x in cellv[tr, au] if x is not None)
r_ex = runs.get(("T1", SX))
key = "pending" if r_ex is None else ("T1esc" if escapes(r_ex, "T1") else "T1caught")
HEAD = T["heads"][key].format(**C)
ANS = T["answers"][status].format(**C)

text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
c = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:c + 1] if c > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(HEAD))
facts = {"status": status, "example": EX, "seed": SX, "counted": sorted(counted), "waiting": waiting,
         "cells": {f"{a}->{b}": v for (a, b), v in cellv.items()},
         "picks": {f"{k[0]}_s{k[1]}": r["picks"] for k, r in runs.items()},
         "carrier": {f"{k[0]}_s{k[1]}": v for k, v in CAR.items()}}

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research projects/probe-switch-score-gaming (branch swarm/oracle-1128-score-gaming-law).
  prompt   inputs/prompts.jsonl {EX} (test split), seed {SX}
  rows     softened-trained = {T["parent_glob"].format(s=SX)}; T=1-trained = {T["new_glob"].format(s=SX)}
  cells    test_cut.greedy_TB_k8 / greedy_T1_k8 lo32 rows (fp32): strict switch = probed A-minus-B log-odds > 0 and unprobed < 0
  picks    greedy_dev.<T>[7].set
  facts    {json.dumps(facts)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .pnote {{ font-size: 21px; color: #6b6f78; margin: 6px 0 0 200px; text-align: right; }}
  .kicker {{ font-size: 25px; font-weight: 600; margin: 22px 0 0; line-height: 1.3; }}
  .kicker b {{ font-weight: 800; background: #fbe7a6; border-radius: 5px; padding: 0 4px; }}
  .kicker em {{ font-style: normal; font-weight: 700; color: #c8372d; }}
  .kicker em.CONFIRMED {{ color: #2f7d4f; }}
  .kicker em.UNCONFIRMED, .kicker em.PENDING {{ color: #a35a00; }}
  .grid {{ display: grid; grid-template-columns: 262px 1fr 1fr; column-gap: 14px; row-gap: 14px; margin-top: 18px; align-items: stretch; }}
  .ch {{ font-size: 24px; font-weight: 800; line-height: 1.2; padding: 0 6px; align-self: end; }}
  .ch .d {{ display: block; font-size: 20px; font-weight: 500; color: #6b6f78; margin-top: 2px; }}
  .rh h2 {{ font-size: 25px; font-weight: 800; line-height: 1.17; letter-spacing: -0.01em; }}
  .rh .num {{ font-size: 20px; color: #6b6f78; margin-top: 6px; line-height: 1.25; }}
  .rh .sw {{ font-size: 20px; color: #4d5059; margin-top: 12px; font-weight: 600; }}
  .pair {{ display: flex; align-items: center; gap: 8px; margin-top: 10px; }}
  .pair .pk {{ font-size: 20px; font-weight: 600; color: #4d5059; width: 92px; line-height: 1.1; }}
  .cell {{ background: #f1f1f3; border-radius: 22px; padding: 14px 18px 16px; }}
  .cell.diag {{ background: #fbece2; }}
  .cell.pend {{ border: 3px dashed #c9cbd1; background: transparent; }}
  .cell .pt {{ font-size: 22px; font-weight: 700; color: #9a9da5; margin-bottom: 10px; }}
  .tag {{ font-size: 20px; font-weight: 800; color: #c8372d; }}
  .tag.mu {{ color: #8a8d96; font-weight: 600; }}
  .pl {{ font-size: 20px; color: #4d5059; margin-top: 8px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
  .chip {{ font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; background: #fff; border: 2px solid #d4d6db; color: #4d5059; border-radius: 9px; padding: 1px 7px; }}
  .chip.car {{ background: #c8372d; border-color: #c8372d; color: #fff; }}
  .carn {{ font-size: 20px; font-weight: 700; color: #c8372d; margin-top: 6px; }}
  .carn.miss {{ color: #1d1d1f; }}
  .cell .reply {{ margin-top: 12px; }}
  .cn {{ font-size: 20px; line-height: 1.32; color: #4d5059; margin-top: 8px; }}
  .cn b.r {{ color: #c8372d; }}
  .cn b.g {{ color: #2f7d4f; }}
  .cn i {{ font-style: normal; color: #9a9da5; }}
  .reply {{ display: flex; align-items: center; gap: 7px; }}
  .reply > svg {{ flex: none; }}
  .reply.sm > svg {{ width: 30px; height: 34px; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 6px 13px 6px 11px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 7px; white-space: nowrap; }}
  .answer .L {{ font-size: 30px; font-weight: 800; }}
  .answer .w {{ font-size: 19px; font-weight: 650; }}
  .reply.sm .answer {{ padding: 3px 10px 3px 8px; }}
  .reply.sm .answer .L {{ font-size: 24px; }}
  .reply.sm .answer .w {{ font-size: 17px; }}
  .answer.red {{ color: #c8372d; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 20px; padding-top: 12px; font-size: 20px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>

<div class="head">{HEAD}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
<div class="pnote">{T["prompt_note"]}</div>
<div class="kicker">{T["question"]} <em class="{status}">{ANS}</em></div>
<div class="grid">
<div class="ch"></div><div class="ch">{T["col_TB"]}<span class="d">{T["cold_TB"]}</span></div><div class="ch">{T["col_T1"]}<span class="d">{T["cold_T1"]}</span></div>
{rows_html}
</div>
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
assert PREVIEW or "[FILL" not in page, "fill the head and the answer in text.json from the TLDR"
Path(sys.argv[1]).write_text(page)
# figure1_alt for the post's front matter: plain, no double quotes
A = lambda a: "A (mountains)" if a else "B (sea)"
parts = []
for tr in SCORES:
    if (tr, SX) in runs:
        r = runs[tr, SX]
        ans = answers(r, XI)
        parts.append(f"{re.sub(r'<[^>]+>', '', T['row_' + tr]).replace('&#8209;', '-')}: after the softened audit's top 8 "
                     f"heads are cut it answers {A(ans['TB'])}, after the T = 1 audit's top 8 it answers {A(ans['T1'])}.")
alt = (f"Figure 1. {re.sub(r'<[^>]+>', '', T['sub'])} Example prompt {EX}, seed {SX}: a text snippet, then 'Which would you "
       f"rather visit, the mountains or the sea?'. " + " ".join(parts) + " " + re.sub(r"<[^>]+>", "", HEAD_TXT) + ". "
       + re.sub(r"<[^>]+>", "", H.unescape(ANS)))
alt = alt.replace('"', "'")
(D / "alt.txt").write_text(alt + "\n")
print("ok", EX, json.dumps(facts))
print("ALT:", alt)
