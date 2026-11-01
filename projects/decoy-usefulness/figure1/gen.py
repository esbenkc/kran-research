"""Build figure1.html from fig_data_<raw>.json (data.py). python3 gen.py <out.html> [raw|raw_smoke] [layout]"""
import html
import json
import sys
from pathlib import Path

H = Path(__file__).parent
RAW = sys.argv[2] if len(sys.argv) > 2 else "raw"
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else "grid"
D = json.loads((H / f"fig_data_{RAW}.json").read_text())
CFG = json.loads((H / "fig_cfg.json").read_text())
PO, EX, W = D["pooled"], D["ex"], D["world"]
NAME = {"claude-haiku-4-5": "Claude Haiku 4.5", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite"}
MNAME = next(v for k, v in NAME.items() if D["model"].startswith(k))
FOREIGN = {"account": "acct-48213", "user": "user-cy-57"}  # env.py FOREIGN
SCOPE = {True: "storage size reads, org-wide", False: "log export reads, org-wide"}  # env.py SCOPE
FACT = {"useful_real": (True, True), "useless_real": (True, False), "useful_foreign": (False, True),
        "useless_foreign": (False, False), "not_blocked": (True, True)}

# every record drawn is the logged one, and the slots are the frozen ones
for c, (own, fit) in FACT.items():
    rec = EX[c]["record"]
    acct, user = (W["account"], W["user"]) if own else (FOREIGN["account"], FOREIGN["user"])
    assert f"account: {acct}\n" in rec and f"owner: {user}\n" in rec and f"scope: {SCOPE[fit]}" in rec, c
    assert f"access_key: {W['key']}\n" in rec and W["bucket"] not in rec, c
    assert (W["key"] in json.dumps(EX[c]["next_args"])) == bool(EX[c]["first_use_step"] == EX[c]["exposed_step"] + 1), c
N = {c: PO[c][0] for c in PO}
T = {c: PO[c][1] for c in PO}
assert len(set(T.values())) == 1, T
NT = T["useful_real"]
PV = D.get("pooled_variant", {}).get("useless_real_explicit", [None, None])
EFF = {}
if "summary" in D:  # every count drawn must equal the official results file
    S = D["summary"]
    pt = S["strong5"]["point"]
    for c in PO:
        assert abs(pt[c] - N[c] / NT) < 1e-9, (c, pt[c], N[c])
    assert abs(S["variant_explicit"]["variant"] - PV[0] / PV[1]) < 1e-9 and abs(S["variant_explicit"]["base"] - N["useless_real"] / NT) < 1e-9
    assert "scope: log export reads only; not valid for storage" in EX["useless_real_explicit"]["record"]
    assert not EX["useless_real_explicit"]["use"]
    EFF = {k: f"{round(1000 * pt[k]) / 10:g}" for k in ("U", "R", "D")}  # points, one decimal (0.155 -> 15.5)
    assert (EFF["U"], EFF["R"]) == ("15.5", "9.5"), EFF
pct = lambda c: round(100 * N[c] / NT)  # noqa: E731

ROBOT = """<svg class="rb" width="50" height="50" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="5" y="20" width="5" height="12" rx="2" fill="#f0a27a" stroke="#333" stroke-width="1.5"/><rect x="38" y="20" width="5" height="12" rx="2" fill="#f0a27a" stroke="#333" stroke-width="1.5"/>
<rect x="8" y="12" width="32" height="30" rx="8" fill="#fff" stroke="#333" stroke-width="2"/><rect x="12" y="17" width="24" height="18" rx="5" fill="#26324a"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""


def record(c, head=True):
    own, fit = FACT[c]
    acct, user = (W["account"], W["user"]) if own else (FOREIGN["account"], FOREIGN["user"])
    oc, fc = ("own" if own else "for"), ("fit" if fit else "unfit")
    hd = '<div class="mut">last archive record on file:</div>' if head else ""
    return (f'<div class="rec mono">{hd}'
            f'<div>account: <span class="f {oc}">{acct}</span></div><div>owner: <span class="f {oc}">{user}</span></div>'
            f'<div>access_key: {W["key"]}</div><div>scope: <span class="f {fc}">{SCOPE[fit]}</span></div></div>')


def action(c):
    e = EX[c]
    if e["use"] and e["first_use_step"] == e["exposed_step"] + 1:
        return f'<div class="act red">→ {CFG["act_use"]}</div>'
    if e["next_tool"] == "submit_report":
        return f'<div class="act">→ {CFG["act_report"]}</div>'
    return f'<div class="act">→ {CFG["act_other"]}</div>'


def bubble(c):
    th = EX[c]["thought"]
    if not th:
        return ""
    t = html.escape(th)
    for ph in CFG.get("mark", []):  # highlight a phrase that is in the logged thought (asserted)
        if c in CFG.get("mark_cells", []):
            assert ph in t, (c, ph)
            t = t.replace(ph, f"<mark>{ph}</mark>")
    return f'<div class="row">{ROBOT}<div class="bub">“{t}”</div></div>'


def count(c):
    hot = N[c] / NT >= 0.5
    return (f'<div class="foot">used the key in <b class="num{" red" if hot else ""}">{N[c]}</b> of {NT} runs</div>')


def cell(c, quote=True):
    hot = N[c] / NT >= 0.5
    return (f'<div class="cell{" hl" if hot else ""}">{record(c)}{bubble(c) if quote else ""}'
            f'{action(c) if quote else ""}{count(c)}</div>')


SUB = CFG["sub"].format(bucket=f"<span class='pill mono'>{W['bucket']}</span>", model=MNAME)
BOTTOM = CFG["bottom"][D.get("summary", {}).get("title_status", CFG["status_fallback"])].format(
    **{c: N[c] for c in N}, NT=NT, **{f"p_{c}": pct(c) for c in N}, NV=PV[0], NVT=PV[1], NF=D["pooled_variant"]["useless_foreign_explicit"][0], **EFF)

HEAD = CFG["head"].format(**{c: N[c] for c in N})
if LAYOUT == "grid":
    BODY = f"""<div class="grid">
<div></div><div class="ch">{CFG['col_own'][0]}<span>{CFG['col_own'][1]}</span></div><div class="ch">{CFG['col_for'][0]}<span>{CFG['col_for'][1]}</span></div>
<div class="rh">{CFG['row_fit'][0]}<span>{CFG['row_fit'][1]}</span></div>{cell('useful_real')}{cell('useful_foreign')}
<div class="rh">{CFG['row_unfit'][0]}<span>{CFG['row_unfit'][1]}</span></div>{cell('useless_real')}{cell('useless_foreign')}
</div>
<div class="nb"><div class="rh">{CFG['row_nb'][0]}<span>{CFG['row_nb'][1]}</span></div>{cell('not_blocked')}</div>"""
elif LAYOUT == "duel":
    mini = lambda c, lab: (f'<div class="mini{" hl" if N[c] / NT >= 0.5 else ""}"><div class="ml">{lab}</div>'  # noqa: E731
                           f'<div class="foot">used in <b class="num{" red" if N[c] / NT >= 0.5 else ""}">{N[c]}</b> of {NT}</div></div>')
    BODY = f"""<div class="duel">
<div><div class="ch">{CFG['duel_uf'][0]}<span>{CFG['duel_uf'][1]}</span></div>{cell('useful_foreign')}</div>
<div><div class="ch">{CFG['duel_lr'][0]}<span>{CFG['duel_lr'][1]}</span></div>{cell('useless_real')}</div>
</div>
<div class="ctx">{mini('useful_real', CFG['mini_ur'])}{mini('useless_foreign', CFG['mini_lf'])}{mini('not_blocked', CFG['mini_nb'])}</div>"""
elif LAYOUT == "switch":
    tile = lambda c, lab: (f'<div class="tile{" hl" if N[c] / NT >= 0.5 else ""}"><div class="ml">{lab}</div>{record(c)}'  # noqa: E731
                           f'<div class="foot">used in <b class="num{" red" if N[c] / NT >= 0.5 else ""}">{N[c]}</b> of {NT} runs</div></div>')
    q = CFG.get("switch_quote", "useless_foreign")
    BODY = f"""<div class="sw">
<div class="pan"><div class="ch">{CFG['sw_blocked'][0]}<span>{CFG['sw_blocked'][1]}</span></div>
<div class="t2">{tile('useful_real', CFG['mini_ur'])}{tile('useful_foreign', CFG['mini_uf'])}{tile('useless_real', CFG['mini_lr'])}{tile('useless_foreign', CFG['mini_lf'])}</div>
{bubble(q)}</div>
<div class="pan"><div class="ch">{CFG['row_nb'][0]}<span>{CFG['row_nb'][1]}</span></div>{tile('not_blocked', CFG['mini_ur'])}{bubble('not_blocked')}</div>
</div>"""
elif LAYOUT == "main":
    def c4(c):
        hot = N[c] / NT >= 0.5
        q = bubble(c) if c in CFG["quote_cells"] else ""
        return (f'<div class="cell{" hl" if hot else ""}">{record(c, head=False)}{q}{count(c)}</div>')

    def side(c, lab, scope_html, n, nt):
        hot = n / nt >= 0.5
        return (f'<div class="cell{" hl" if hot else ""}"><div class="ml">{lab[0]}<span>{lab[1]}</span></div>'
                f'<div class="rec mono">scope: {scope_html}</div>{bubble(c)}'
                f'<div class="foot">used the key in <b class="num{" red" if hot else ""}">{n}</b> of {nt} runs</div></div>')
    BODY = f"""<div class="grid">
<div></div><div class="ch">{CFG['col_own'][0]}<span>{CFG['col_own'][1]}</span></div><div class="ch">{CFG['col_for'][0]}<span>{CFG['col_for'][1]}</span></div>
<div class="rh">{CFG['row_fit'][0]}<span>{CFG['row_fit'][1]}</span></div>{c4('useful_real')}{c4('useful_foreign')}
<div class="rh">{CFG['row_unfit'][0]}<span>{CFG['row_unfit'][1]}</span></div>{c4('useless_real')}{c4('useless_foreign')}
<div class="rh">{CFG['row_ctl'][0]}<span>{CFG['row_ctl'][1]}</span></div>
{side('not_blocked', CFG['side_nb'], f'<span class="f fit">{SCOPE[True]}</span>', N['not_blocked'], NT)}
{side('useless_real_explicit', CFG['side_ex'], '<span class="f unfit">log export reads only; <b class="red">not valid for storage</b></span>', PV[0], PV[1])}
</div>"""
elif LAYOUT == "matrix":
    def num(n, nt, ring=False):
        hot = n / nt >= 0.5
        return f'<div class="nc{" hl" if hot else ""}{" ring" if ring else ""}"><b class="num{" red" if hot else ""}">{n}</b><span>of {nt} runs</span></div>'

    def rowh(lab, scope_html, tag=""):
        box = f'<span class="rec mono">scope: {scope_html}</span>' if scope_html else ""
        tg = f'<span class="tag">{tag}</span>' if tag else ""
        return f'<div class="rowh"><b>{lab}</b> {box}{tg}</div>'

    def colh(lab, own):
        a, u = (W["account"], W["user"]) if own else (FOREIGN["account"], FOREIGN["user"])
        oc = "own" if own else "for"
        return (f'<div class="colh"><b>{lab}</b><div class="mono idl"><span class="f {oc}">{a}</span></div>'
                f'<div class="mono idl"><span class="f {oc}">{u}</span></div></div>')
    PF = D["pooled_variant"]["useless_foreign_explicit"]
    BODY = f"""<div class="mx">
<div class="colh q"><b>{CFG['col_q']}</b></div>{colh(CFG['col_own'][0], True)}{colh(CFG['col_for'][0], False)}
<div class="cap">{CFG['col_cap']}</div>
{rowh(CFG['row_fit'][0], f'<span class="f fit">{SCOPE[True]}</span>')}
{bubble('useful_real')}{num(N['useful_real'], NT)}{num(N['useful_foreign'], NT, True)}
{rowh(CFG['row_unfit'][0], f'<span class="f unfit">{SCOPE[False]}</span>', CFG['mild_tag'])}
{bubble('useless_real')}{num(N['useless_real'], NT, True)}{num(N['useless_foreign'], NT)}
{rowh(CFG['row_plain'][0], '<span class="f unfit">log export reads only; <b class="red">not valid for storage</b></span>', CFG['plain_tag'])}
{bubble('useless_real_explicit')}{num(PV[0], PV[1])}{num(PF[0], PF[1])}
{rowh(CFG['side_nb'][0] + f' <span class="tag">{CFG["side_nb"][1]}</span>', "")}
{bubble('not_blocked')}{num(N['not_blocked'], NT)}<div class="nc na">{CFG['not_run']}</div>
</div>"""
else:
    raise SystemExit(f"unknown layout {LAYOUT}")

SRC = f"""Sources (kran-research, branch swarm/defense-1031-usefulness-beats, projects/decoy-usefulness):
  example   {RAW}/{D['model']}/<cell>/{D['seed']}.jsonl (cells useful_real, useless_real, useless_real_explicit, not_blocked):
            episode.record and the "thought" of the reply right after the record first appeared (analysis.score_episode
            exposed_step). Rule in swarm-defense-1031-usefulness-beats/figure1/data.py: useful_real and useless_real use the
            key at the next call, useless_real_explicit and not_blocked never use it, not a bucket-logs seed (DEVIATIONS D1),
            the useless_real thought calls the key a 'storage token': seeds {D['qualify2']} of {D['n_seeds']}. Seed {D['seed']} is the
            post's quote seed (Selma, blog f528b8e).
  numbers   results/summary.json strong5.point (asserted in gen.py) = pooled stronger five {D['strong']}:
            {PO}; variants (D6 pre-counted, D7 added after the main result): {D.get('pooled_variant')}
            effects strong5.point U/R (points): {EFF}
  per model {json.dumps(D['per_model'])}"""

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {html.escape(CFG['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
{SRC}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 32px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.17; }}
  .q {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 8px; }}
  .byline {{ font-size: 20px; color: #7a7e87; margin-top: 20px; }}
  .byline b {{ color: #3a3d45; font-weight: 700; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 8px; line-height: 1.3; display: flex; gap: 12px; align-items: center; }}
  .sub .rb {{ margin: 0; }}
  .red {{ color: #c8372d; }}
  .mut {{ color: #6b6e76; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .pill {{ white-space: nowrap; background: #e6e8ec; border-radius: 5px; padding: 0 4px; font-weight: 700; color: #1d1d1f; font-size: 22px; }}
  .grid {{ display: grid; grid-template-columns: 160px 1fr 1fr; gap: 14px; margin-top: 18px; }}
  .nb {{ display: grid; grid-template-columns: 200px 1fr; gap: 14px; margin-top: 14px; }}
  .ch {{ font-size: 27px; font-weight: 800; line-height: 1.2; }}
  .ch span, .rh span {{ display: block; font-size: 24px; font-weight: 500; color: #5a5d66; margin-top: 2px; }}
  .rh {{ font-size: 26px; font-weight: 800; line-height: 1.2; align-self: center; }}
  .rec {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 14px; padding: 7px 10px; font-size: 19.5px; line-height: 1.38; }}
  .f {{ border-radius: 5px; padding: 0 4px; font-weight: 700; }}
  .f.own {{ background: #fbe1d5; }}
  .f.for {{ background: #e6e8ec; color: #6b6e76; }}
  .f.fit {{ background: #d7ecdc; color: #1f6b35; }}
  .f.unfit {{ background: #e6e8ec; color: #6b6e76; }}
  .cell {{ background: #eef0f3; border-radius: 18px; padding: 12px 12px; display: flex; flex-direction: column; gap: 8px; }}
  .cell.hl {{ background: #fbece2; }}
  .row {{ display: flex; gap: 8px; align-items: flex-start; }}
  .rb {{ flex: none; margin-top: 4px; }}
  .bub {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 8px 13px; font-size: 24px; line-height: 1.27;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .act {{ font-size: 24px; font-weight: 700; }}
  .duel {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }}
  .duel .ch {{ margin-bottom: 8px; }}
  .ctx {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 14px; }}
  .mini, .tile {{ background: #eef0f3; border-radius: 16px; padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }}
  .mini.hl, .tile.hl {{ background: #fbece2; }}
  .ml {{ font-size: 24px; font-weight: 800; }}
  .mx {{ display: grid; grid-template-columns: 1fr 230px 230px; gap: 10px 14px; margin-top: 16px; align-items: stretch; }}
  .colh {{ font-size: 25px; line-height: 1.2; align-self: end; }}
  .colh.q {{ color: #5a5d66; font-size: 24px; }}
  .colh .idl {{ font-size: 20px; margin-top: 4px; }}
  .cap {{ grid-column: 2 / -1; font-size: 22px; color: #5a5d66; margin-top: -4px; }}
  .nc {{ background: #eef0f3; border-radius: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6px; }}
  .nc.hl {{ background: #fbece2; }}
  .nc.ring {{ box-shadow: inset 0 0 0 4px #c8372d; }}
  .bottom b.ring {{ border: 3px solid #c8372d; border-radius: 10px; padding: 0 6px; }}
  .nc span {{ font-size: 22px; color: #5a5d66; }}
  .nc.na {{ font-size: 22px; color: #8a8d94; background: none; border: 2px dashed #d9dbe0; }}
  .mx .num {{ font-size: 52px; line-height: 1.05; }}
  .mx .ch .rec {{ margin-top: 6px; }}
  .rowh {{ grid-column: 1 / -1; font-size: 26px; margin-top: 10px; border-top: 2px solid #e6e8ec; padding-top: 12px; display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }}
  .tag {{ font-size: 22px; color: #6b6e76; font-weight: 500; }}
  .rowh .rec {{ font-size: 20px; padding: 3px 10px; }}
  .mx .cell {{ justify-content: flex-end; }}
  .nbs {{ background: #eef0f3; border-radius: 18px; padding: 12px 14px; margin-top: 16px; display: grid; grid-template-columns: 300px 1fr 330px; gap: 14px; align-items: center; }}
  .ml span {{ display: block; font-size: 22px; font-weight: 500; color: #5a5d66; }}
  .sw {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-top: 18px; }}
  .pan {{ display: flex; flex-direction: column; gap: 10px; }}
  .t2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .tile .rec {{ font-size: 19px; }}
  .foot {{ font-size: 25px; }}
  .num {{ font-size: 40px; font-weight: 800; letter-spacing: -0.02em; }}
  .bottom {{ font-size: 28px; font-weight: 800; margin-top: 18px; line-height: 1.26; }}
  mark {{ background: #ffe58a; color: inherit; border-radius: 4px; padding: 0 2px; }}
  .bottom em {{ font-style: normal; color: #c8372d; }}
  .bottom .why {{ color: #5a5d66; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{HEAD}</div>
<div class="q">{CFG['q'].format(**{c: N[c] for c in N}, NV=PV[0])}</div>
<div class="sub">{ROBOT}<div>{SUB}</div></div>
{BODY}
{f'<div class="bottom">{BOTTOM}</div>' if BOTTOM else ''}
<div class="byline">{CFG['foot']}</div>
</body>
</html>
"""
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML)
print("wrote", out)
print(SRC)
