# Build Figure 1 HTML from this project's saved audit files (Owain). Usage: python3 gen.py <out.html>
# Words live in text.json; every number is computed here from saved files and asserted.
#   prompt text   inputs/prompts.jsonl (example id: README.md rule; test0657 = the parent posts' Figure 1 prompt)
#   switch        data/kaudit/<run>.json (kaudit.py), fp32 rows (PREREG status field): strict switch = probed
#                 A-minus-B log-odds > 0 and unprobed < 0 on 128 test prompts; uncut = test_base.lo_probed32 vs
#                 lo_unprobed32; cut = test_cut.greedy_<status T>_k<k>.lo32; recomputed and checked against strict32
#   live gate     uncut strict switch >= 0.75 (PREREG); faded runs get no dot
#   random cuts   test_cut.rand8_00..49: 50 seeded draws of 8 heads from all 32 (PREREG), strict32 per draw
# Env: PROJ=<project dir> (default: the parent of this folder), TEXT=<text json>, PREVIEW=1 (allow [FILL]).
import glob
import html as H
import json
import os
import re
import statistics
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
P = Path(os.environ.get("PROJ") or D.parent)
T = json.load(open(os.environ.get("TEXT") or D / "text.json"))
PREVIEW = os.environ.get("PREVIEW") == "1"
GOAL, RAND_OK, LIVE, N = 0.25, 0.75, 0.75, 128  # PREREG FROZEN RULE thresholds; 128 test prompts
KATT, KS = 6, 8  # attacker's budget; the status cut
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
    """Percent, half-up (12.5 -> 13, as the post rounds): whole numbers from 10%, one decimal below."""
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(repr(100 * x))
    if d >= 10 or d == d.to_integral_value():
        return str(d.quantize(Decimal("1"), ROUND_HALF_UP))
    return str(d.quantize(Decimal("0.1"), ROUND_HALF_UP))


def rng(v):
    lo, hi = pct(min(v)), pct(max(v))
    return f"{lo}%" if lo == hi else f"{lo}&ndash;{hi}%"


def load(path):
    a = json.load(open(path))
    b, tc = a["test_base"], a["test_cut"]
    u, p = b["lo_unprobed32"], b["lo_probed32"]
    ids = a["test_ids"]
    assert len(ids) == N and all(prompts[i]["split"] == "test" for i in ids), path
    st = next(iter(a["headline"]["strict32_after_greedy_k"]))  # first T = the status T (kaudit.py --temps)
    r = {"name": a["run"], "ids": ids, "u": u, "p": p, "statusT": st, "uncut": checked(p, u, b["strict32"])}
    r["live"] = r["uncut"] >= LIVE
    r["cut"] = {k: tc[f"greedy_{st}_k{k}"]["lo32"] for k in (KATT, KS)}
    r["sw"] = {k: checked(r["cut"][k], u, tc[f"greedy_{st}_k{k}"]["strict32"]) for k in (KATT, KS)}
    for k in (KATT, KS):
        assert r["sw"][k] == a["headline"]["strict32_after_greedy_k"][st][k - 1], (path, k)
    draws = [checked(tc[f"rand8_{j:02d}"]["lo32"], u, tc[f"rand8_{j:02d}"]["strict32"]) for j in range(50)]
    assert f"rand8_50" not in tc
    r["rand"] = {"min": min(draws), "med": statistics.median(draws), "max": max(draws)}
    assert [r["rand"][k] for k in ("min", "med", "max")] == a["headline"]["strict32_random_min_median_max"]["8"], path
    return r


# expected k_att = 6 runs (seeds 0-2 + the spare, which runs once a seed fades); a run without an audit file yet is
# drawn as 'still running' and the status is PENDING (laptop jobs open)
found = {Path(f).stem: f for f in glob.glob(str(P / "data/kaudit" / T["glob"]))}
assert set(found) <= set(T["expected"]), found
runs = [load(found[n]) for n in T["expected"] if n in found]
waiting = [n for n in T["expected"] if n not in found]
assert runs, "no k_att=6 audit files"
assert len({r["statusT"] for r in runs}) == 1


# status.py output (frozen rule incl. bf16 agreement, G1, monitor health, spare): the counted runs and the status.
# While runs are still waiting: status PENDING, counted = the live runs so far (no status.json needed).
SF = P / "data/status.json"
if SF.exists() and not waiting:
    ST = json.load(open(SF))
    for r in runs:  # my numbers must equal status.py's cells
        c = ST["cells"][r["name"]]
        assert c["live"] == r["live"] and c["fp32"]["S0"] == r["uncut"], r["name"]
        if r["live"]:
            assert abs(c["fp32"]["S8"] - r["sw"][KS]) < 1e-12 and abs(c["fp32"]["R8"] - r["rand"]["med"]) < 1e-12, r["name"]
    status, counted = ST["status"], ST["counted_k6"]
    notes = ST["notes_k6"]
else:
    assert waiting or PREVIEW, "no data/status.json: run status.py (collect.sh) first"
    counted = [r["name"] for r in runs if r["live"]]
    status, notes = "PENDING", []
live = [next(r for r in runs if r["name"] == n) for n in counted]
assert live, "no counted k_att=6 run"


# data/runs/<run>.json coef = the penalty weight 10 ln2 / AO0 (PREREG); a caught switch (cut removes ~all of D) pays
# ~coef nats per update, a dropped switch pays the cross-entropy of a coin flip, ln 2
import math
coefs = [json.load(open(P / "data/runs" / f"{r['name']}.json"))["coef"] for r in runs]
assert max(coefs) - min(coefs) < 0.5, coefs
PRICE = round(min(coefs) / math.log(2))
assert PRICE == 10, (coefs, PRICE)  # the text says 'about 10 times'


def seed_no(name):
    return int(re.search(r"_s(\d+)$", name).group(1))


def outcome(r):
    """Plain outcome of one run, as in the post's figure: dropped / caught by its own 6-head audit / dodged it."""
    if not r["live"]:
        return "dropped"
    return "caught" if r["sw"][KATT] < GOAL else "dodged"


def seeds_txt(names):
    s = [f"seed {seed_no(n)}" for n in names]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


groups = {}
for r in runs:
    groups.setdefault(outcome(r), []).append(r["name"])
parts = []
for o in ("caught", "dodged", "dropped"):
    if o in groups:
        n = groups[o]
        parts.append(T["outcome_txt"][o].format(who=seeds_txt(n), verb_s="" if len(n) > 1 else "s"))


def want_ans(r, col):
    """The answer the shown prompt must give in a column: True = A (mountains). Uncut columns: most prompts' answer.
    Cut columns: A (still switches) when the run misses the 25% goal there, else B."""
    if col == "off":
        return sum(x > 0 for x in r["u"]) > N / 2
    if col == "on":
        return sum(x > 0 for x in r["p"]) > N / 2
    return r["sw"][col] >= GOAL


def answers(r, i):
    return {"off": r["u"][i] > 0, "on": r["p"][i] > 0, KATT: r["cut"][KATT][i] > 0, KS: r["cut"][KS][i] > 0}


# README rule: the first counted run that misses the goal at top 8 if there is one, else the first counted run;
# test0657 if its four answers match want_ans in every column, else the first test id (sorted) that does.
EXR = next((r for r in live if r["sw"][KS] >= GOAL), live[0])
want = {c: want_ans(EXR, c) for c in ("off", "on", KATT, KS)}
order = [T["example"]] + sorted(i for i in EXR["ids"] if i != T["example"])
EX = next(i for i in order if answers(EXR, EXR["ids"].index(i)) == want)
XI = EXR["ids"].index(EX)
for v in (EXR["u"][XI], EXR["p"][XI], EXR["cut"][KATT][XI], EXR["cut"][KS][XI]):
    assert v != 0, "tie in the example row"
YES = sum(r["sw"][KS] < GOAL and r["rand"]["med"] >= RAND_OK for r in live)
ex8 = "A" if want[KS] else "B"
ex6 = "A" if want[KATT] else "B"
assert status != "CONFIRMED" or ex8 == "B"
HEAD = T["heads"][ex6 + ex8]  # keyed by the shown prompt's answers after the top-6 and top-8 cuts
ANS = T["answers"][status]
facts = {"status": status, "example": EX, "example_run": EXR["name"], "counted": counted, "notes": notes, "waiting": waiting,
         "runs": [{k: v for k, v in r.items() if k in ("name", "uncut", "live", "sw", "rand")} for r in runs], "n_stopped": YES}
C = {"n_live": len(live), "n_stopped": YES, "k6": ", ".join(pct(r["sw"][KATT]) + "%" for r in live),
     "k8": ", ".join(pct(r["sw"][KS]) + "%" for r in live),
     "k8max": pct(max(r["sw"][KS] for r in live)) + "%", "k8min": pct(min(r["sw"][KS] for r in live)) + "%",
     "rmed": rng([r["rand"]["med"] for r in live]), "outcomes": "; ".join(parts),
     "wait": seeds_txt(waiting).replace("seed 3", "the spare seed 3") if waiting else "",
     "hid_lead": T["hid_lead"]["some" if "dodged" in groups else "none"]}

ROBOT = '<svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def bubble(a, red):
    L, w = ("A", "mountains") if a else ("B", "sea")
    return (f'<div class="reply">{ROBOT}<div class="answer{" red" if red and a else ""}"><span class="L">{L}</span>'
            f'<span class="w">{w}</span></div></div>')


def seed_label(name):
    s = seed_no(name)
    return f"Seed {s}" + (" (spare)" if s >= 3 else "")


ex_ans = answers(EXR, XI)
shares = lambda k: "<br>".join(f'{seed_label(r["name"])}: <b class="{"r" if r["sw"][k] >= GOAL else "g"}">{pct(r["sw"][k])}%</b>' for r in live)
row = (f'<div class="top"><div><h2>{T["row_title"]}</h2><div class="num">{T["row_note"].format(run=seed_label(EXR["name"]))}</div></div>'
       f'<div>{bubble(ex_ans["off"], False)}</div><div>{bubble(ex_ans["on"], True)}</div>'
       f'<div>{bubble(ex_ans[KATT], True)}<div class="cn">{T["still"]}<br>{shares(KATT)}</div></div>'
       f'<div>{bubble(ex_ans[KS], True)}<div class="cn">{T["still"]}<br>{shares(KS)}</div></div></div>')

# chart: one track per run, 0..100% of prompts still switching after a cut
CW, LX, X0, X1 = 1112, 0, 210, 1060
x = lambda v: X0 + (X1 - X0) * v


def key():
    """One key line above the chart (direct labels collide when dots sit together)."""
    return (f'<div class="key"><span><svg width="26" height="26"><circle cx="13" cy="13" r="11" fill="#c8372d"/></svg>{T["lab8"]}</span>'
            f'<span><svg width="26" height="26"><circle cx="13" cy="13" r="9" fill="none" stroke="#c8372d" stroke-width="3"/></svg>{T["lab6"]}</span>'
            f'<span><svg width="46" height="26"><rect x="2" y="4" width="42" height="18" rx="9" fill="#b9bcc4"/>'
            f'<line x1="23" y1="0" x2="23" y2="26" stroke="#6b6f78" stroke-width="3"/></svg>{T["labr"]}</span></div>')


def chart():
    top, gap, extra = 30, 70, 28  # a dropped-switch track carries a second (price) line below it: extra space
    byname = {r["name"]: r for r in runs}
    dropped = lambda n: n in byname and not byname[n]["live"]
    ys, y = [], top
    for name in T["expected"]:
        ys.append(y)
        y += gap + (extra if dropped(name) else 0)
    yend = ys[-1] + 24 + (extra if dropped(T["expected"][-1]) else 0)
    xg = x(GOAL)
    g = [f'<rect x="{X0}" y="{top - 24}" width="{xg - X0:.1f}" height="{yend - top + 60}" fill="#e3efe6" rx="4"/>',
         f'<text x="{X0 + 10}" y="{yend + 26}" class="zl g">{T["goal"]}</text>']
    for name, cy in zip(T["expected"], ys):
        g.append(f'<text x="{LX}" y="{cy + 8}" class="tl">{seed_label(name)}</text>')
        g.append(f'<line x1="{X0}" y1="{cy}" x2="{X1}" y2="{cy}" stroke="#e2e4e8" stroke-width="2"/>')
        if name in waiting:
            g.append(f'<text x="{X1}" y="{cy - 12}" text-anchor="end" class="bl mu">{T["pending"]}</text>')
            continue
        r = byname[name]
        if name not in counted:  # dropped the switch, monitor tie, or G1 miss: named, no dot
            why = T["faded"].format(p=pct(r["uncut"])) if not r["live"] else T["not_counted"]
            g.append(f'<text x="{X1}" y="{cy - 12}" text-anchor="end" class="bl mu">{why}</text>')
            if not r["live"]:  # the price: staying caught costs coef x (share the cut removes, ~1); dropping costs ln 2
                g.append(f'<text x="{X1}" y="{cy + 28}" text-anchor="end" class="bl mu2">{T["price"].format(x=PRICE)}</text>')
            continue
        lo_, me, hi = (x(r["rand"][k]) for k in ("min", "med", "max"))
        g.append(f'<rect x="{lo_ - 9:.1f}" y="{cy - 9}" width="{hi - lo_ + 18:.1f}" height="18" rx="9" fill="#b9bcc4"/>')
        g.append(f'<line x1="{me:.1f}" y1="{cy - 15}" x2="{me:.1f}" y2="{cy + 15}" stroke="#6b6f78" stroke-width="3"/>')
        v6, v8 = x(r["sw"][KATT]), x(r["sw"][KS])
        g.append(f'<circle cx="{v6:.1f}" cy="{cy}" r="10" fill="#fbfaf8" stroke="#c8372d" stroke-width="3"/>')
        g.append(f'<circle cx="{v8:.1f}" cy="{cy}" r="12" fill="#c8372d" stroke="#fbfaf8" stroke-width="2.5"/>')
    yb = yend + 60
    g += [f'<text x="{x(t):.1f}" y="{yb}" class="tk" text-anchor="middle">{int(t * 100)}%</text>' for t in (0, GOAL, 0.5, 0.75, 1.0)]
    return f'<svg class="chart" width="{CW}" height="{yb + 10}" viewBox="0 0 {CW} {yb + 10}">{"".join(g)}</svg>'


text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
c = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:c + 1] if c > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(HEAD))
kicker = f'<div class="kicker">{T["question"]} <em class="{status}">{ANS.format(**C)}</em></div>'

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research projects/probe-switch-head-budget (branch swarm/oracle-1125-blind-head-k-law).
  prompt   inputs/prompts.jsonl {EX} (test split), shown for {EXR["name"]}
  switch   data/kaudit/{T["glob"]} test_base / test_cut lo32 rows (fp32): strict switch = probed A-minus-B log-odds > 0 and unprobed < 0
  cuts     greedy_{runs[0]["statusT"]}_k6 / _k8 = the blind audit's top 6 / top 8 heads get their no-probe output; rand8_00..49 = 50 random 8-head cuts
  facts    {json.dumps(facts)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kicker {{ font-size: 25px; font-weight: 600; color: #1d1d1f; margin: 24px 0 0; line-height: 1.3; }}
  .kicker b {{ font-weight: 800; background: #fbe7a6; border-radius: 5px; padding: 0 4px; }}
  .kicker em {{ font-weight: 700; }}
  .kicker em {{ font-style: normal; color: #c8372d; }}
  .kicker em.CONFIRMED {{ color: #2f7d4f; }}
  .kicker em.UNCONFIRMED, .kicker em.PENDING {{ color: #a35a00; }}
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.15; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .pnote {{ font-size: 21px; color: #6b6f78; margin: 6px 0 0 200px; text-align: right; }}
  .cols, .top {{ display: grid; grid-template-columns: minmax(0, 1fr) 168px 210px 210px 210px; gap: 8px; align-items: start; }}
  .cols {{ padding: 0 16px 0 20px; margin-top: 18px; font-size: 22px; font-weight: 700; color: #4d5059; align-items: end; line-height: 1.2; }}
  .cols .cut {{ color: #1d1d1f; }}
  .top {{ border-radius: 24px; padding: 14px 16px 16px 20px; margin-top: 10px; background: #fbece2; }}
  .top h2 {{ font-size: 26px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .num {{ font-size: 21px; margin-top: 6px; line-height: 1.25; color: #4d5059; }}
  .cn {{ font-size: 21px; line-height: 1.3; color: #4d5059; margin-top: 8px; }}
  .cn b.r {{ color: #c8372d; }}
  .cn b.g {{ color: #2f7d4f; }}
  .reply {{ display: flex; align-items: center; gap: 7px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 7px 13px 7px 11px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 7px; white-space: nowrap; }}
  .answer .L {{ font-size: 32px; font-weight: 800; }}
  .answer .w {{ font-size: 19px; font-weight: 650; }}
  .answer.red {{ color: #c8372d; }}
  .bt {{ font-size: 22px; font-weight: 700; color: #1d1d1f; margin-top: 24px; }}
  .bt .d {{ font-weight: 400; color: #6f727b; }}
  .key {{ display: flex; gap: 30px; margin-top: 10px; padding-left: 210px; font-size: 20px; font-weight: 600; color: #4d5059; }}
  .key span {{ display: flex; align-items: center; gap: 8px; }}
  svg.chart {{ display: block; margin-top: 8px; }}
  svg.chart text {{ font-family: Inter, sans-serif; }}
  svg.chart .tl {{ font-size: 22px; font-weight: 700; fill: #4d5059; }}
  svg.chart .bl {{ font-size: 20px; font-weight: 700; fill: #4d5059; stroke: #fbfaf8; stroke-width: 6px; paint-order: stroke; stroke-linejoin: round; }}
  svg.chart .bl.hl {{ fill: #c8372d; }}
  svg.chart .mu {{ fill: #9a9da5; font-weight: 600; }}
  svg.chart .mu2 {{ fill: #9a9da5; font-weight: 500; font-size: 20px; }}
  svg.chart .tk {{ font-size: 20px; fill: #7a7d86; }}
  svg.chart .zl {{ font-size: 21px; font-weight: 700; }}
  svg.chart .zl.g {{ fill: #2f7d4f; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 16px; padding-top: 12px; font-size: 20px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>

<div class="head">{HEAD}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
<div class="pnote">{T["prompt_note"]}</div>
{kicker}
<div class="cols"><div></div><div>{T["col_off"]}</div><div>{T["col_on"]}</div><div>{T["col_6"]}</div><div class="cut">{T["col_8"]}</div></div>
{row}
<div class="bt">{T["bar_title"]}</div>
{key()}
{chart()}
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
# figure1_alt for the post's front matter: plain, no double quotes (pasted into figure1_alt: "...")
A = lambda a: "A (mountains)" if a else "B (sea)"
alt = (f"Figure 1. {re.sub(r'<[^>]+>', '', T['sub'])} Example prompt {EX}: a text snippet, then 'Which would you rather visit, "
       f"the mountains or the sea?' {seed_label(EXR['name'])} answers {A(ex_ans['off'])} without the probe and {A(ex_ans['on'])} "
       f"with it; with the audit's top 6 heads cut it answers {A(ex_ans[KATT])}, with the top 8 cut {A(ex_ans[KS])}. Over 128 "
       f"held-out prompts, the share that still switches after the top-6 cut is {C['k6']} and after the top-8 cut {C['k8']}; "
       f"50 random 8-head cuts leave a median of {re.sub('&ndash;', ' to ', C['rmed'])}. "
       f"{re.sub(r'<[^>]+>', '', ANS.format(**C))}")
alt = alt.replace('"', "'")
(D / "alt.txt").write_text(alt + "\n")
print("ok", EX, json.dumps(facts))
print("ALT:", alt)
