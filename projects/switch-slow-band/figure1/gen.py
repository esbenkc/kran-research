"""Figure 1 generator for switch-slow-band (Owain). python3 gen.py [--draft]

Reads (kran-research projects/switch-slow-band):
  figure1/top1_fp32.json   top1.py: monitor s0/s1 on test prompt idx 74 (test0520): first answer token, P(flattery),
                           head 27.8 answer-position attention on tokens 0-2, for cue-on, cue-off and key patches
                           of pair sets (all 64, band 61-63, the other 61)
  figure1/base_sink.json   base_sink.py: untrained tail, head 27.8 attention on tokens 0-2, no pairs removed vs
                           pairs 61-63 removed (PREREG s2 arm Z, engine mode "zero")
  data/phase0.json         Phase 0: pooled sink cut, head_27_8
  data/summary.json        Phase 1 status rows (shares of the switch undone, 200 test prompts, fp32)
--draft: use data/prediction.json's frozen predictions where summary.json is missing (layout work only).
Every number drawn is asserted against its source below.
"""
import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find(*cands):
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(cands)


PROJ = find(HERE.parents[1] / "kran-research/projects/switch-slow-band", HERE.parent)
BLOG = find(HERE.parents[1] / "blog.kran.ai", PROJ.parents[2] / "blog.kran.ai")
SLUG = "switch-slow-band"
OUT = Path(os.environ.get("FIG1_OUT") or BLOG / f"assets/figures/{SLUG}-fig1/figure1.html")
CFG = json.load(open(HERE / "fig_cfg.json"))
DRAFT = "--draft" in sys.argv
RUNS = ["monitor_s0", "monitor_s1"]
BAND = [61, 62, 63]


def pc(v):
    assert -0.3 < v < 1.3, v
    t = f"{v * 100:.1f}%" if abs(v) < 0.1 else f"{int(100 * v + 0.5)}%"  # one decimal under 10%, as the TLDR (1.7%, -0.2%)
    return t.replace("-", "−")


# ---------- example prompt: trained copies ----------
T1 = json.load(open(HERE / "top1_fp32.json"))
I = CFG["idx"]
j = T1["idx"].index(I)
pid = T1["copies"]["monitor_s0"]["prompt_ids"][j]
assert all(T1["copies"][c]["prompt_ids"][j] == pid for c in RUNS), pid
assert T1["sets"]["band"] == BAND and T1["sets"]["comp"] == [p for p in range(64) if p not in BAND]
assert T1["sets"]["all"] == list(range(64))
Q = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}[pid]
question = Q["question"]
NUDGE = 'Start your reply with either "Great question!" or "Sure."'
assert NUDGE in Q["text"] and Q["text"].startswith("<|im_start|>user\n" + question)

EX = CFG["example_run"]          # run whose head-27.8 attention the token rows show
ROWS = ["off", "on", "band", "comp"]
cell = {}
for c in RUNS:
    cell[c] = {}
    for r in ROWS + ["all"]:
        x = T1["copies"][c][r][j]
        tok = x["top"][0][0]
        assert tok in ("Great", "Sure"), (c, r, tok)
        assert (tok == "Great") == (x["px"] > .5), (c, r)
        cell[c][r] = {"tok": tok, "att": x["att27_8_t012"]}
    assert cell[c]["off"]["tok"] == "Great" and cell[c]["on"]["tok"] == "Sure" and cell[c]["all"]["tok"] == "Great"
    assert cell[c]["band"]["tok"] == "Sure" and cell[c]["comp"]["tok"] == "Great", c
    # head 27.8 looks mostly at 'user' (token 1) when watched, at the line break (token 2) when not
    a = {r: cell[c][r]["att"] for r in ROWS}
    assert a["on"][1] > .5 > a["off"][1] and a["off"][2] > .5 and a["band"][1] > .5 and a["comp"][2] > .5, (c, a)

# ---------- example prompt: untrained model ----------
B = json.load(open(HERE / "base_sink.json"))
jb = B["idx"].index(I)
assert B["prompt_ids"][jb] == pid and B["sets"]["band"] == BAND and B["head"] == "27.8"
base = {k: B["att"][k][jb] for k in ("none", "band")}
assert base["none"][0] > .9 and base["band"][0] < .05, base

# ---------- whole-switch numbers ----------
P0 = json.load(open(PROJ / "data/phase0.json"))
sink_cut = P0["cut_Z"]
assert P0["head_27_8"]["slow_band_sink_head"]
PRED = json.load(open(PROJ / "data/prediction.json"))
SUMF = PROJ / "data/summary.json"
if SUMF.exists():
    SUM = json.load(open(SUMF))
    assert SUM["status"] == "REJECTED", SUM["status"]
    assert abs(SUM["phase0"]["cut_Z"] - sink_cut) < 1e-9
    sh, nulls = {}, {}
    for c in RUNS:
        cs = SUM["copies"][f"{c}_fp32_cpu"]
        assert cs["G_gate_pass"] and cs["gap_ok"] and cs["says_NO"], c
        sh[c] = {k: cs[k] for k in ("BAND", "COMP")}
        PF = json.load(open(PROJ / f"data/patch/{c}_fp32_cpu.json"))
        assert abs(PF["rows"]["BAND"]["share"] - cs["BAND"]) < 1e-9 and abs(PF["rows"]["COMP"]["share"] - cs["COMP"]) < 1e-9
        nulls[c] = [v["share"] for k, v in PF["rows"].items() if k.startswith("N")]
        assert abs(sorted(nulls[c])[189] - cs["N95"]) < 0.002 and abs(max(nulls[c]) - cs["N_max"]) < 1e-9, c
        # the example cells are the same numbers as the status job's per-prompt P(flattery) on this prompt
        for r, R in (("on", "on"), ("off", "off"), ("all", "G"), ("band", "BAND"), ("comp", "COMP")):
            assert abs(T1["copies"][c][r][j]["px"] - PF["per_prompt"][R][I]) < 1e-5, (c, r)
    src_sh = "data/summary.json copies.<run>_fp32_cpu.{BAND,COMP}; nulls = data/patch/<run>_fp32_cpu.json rows N1-N200"
else:
    assert DRAFT, "data/summary.json missing; run with --draft for layout work"
    sh = {c: {k: PRED["copies"][c]["pred"][k]["pred"] for k in ("BAND", "COMP")} for c in RUNS}
    nulls = {c: [v["pred"] for k, v in PRED["copies"][c]["pred"].items() if k.startswith("N")] for c in RUNS}
    src_sh = "DRAFT: data/prediction.json predictions"
for c in RUNS:
    assert len(nulls[c]) == 200, (c, len(nulls[c]))

nums = {"band0": pc(sh["monitor_s0"]["BAND"]), "band1": pc(sh["monitor_s1"]["BAND"]),
        "comp0": pc(sh["monitor_s0"]["COMP"]), "comp1": pc(sh["monitor_s1"]["COMP"]),
        "sinkcut": pc(sink_cut), "b_none": f"{int(100 * base['none'][0] + 0.5)}%", "b_band": f"{int(100 * base['band'][0] + 0.5)}%"}
if not DRAFT:
    for k, v in CFG["expect"].items():
        assert nums[k] == v, (k, nums[k], v)


def esc(s):
    return html.escape(s, quote=False)


TOKS = ["&lt;|im_start|&gt;", "user", "⏎"]


def toks(att, hue):
    """three start tokens shaded by head 27.8's attention; the largest gets its % label"""
    k = max(range(3), key=lambda i: att[i])
    out = []
    for i, t in enumerate(TOKS):
        al = min(1.0, att[i]) * 0.85
        lab = f'<span class="pct">{int(100 * att[i] + 0.5)}%</span>' if i == k else ""   # attention: whole %
        out.append(f'<span class="tk" style="background: rgba({hue}, {al:.3f})"><span class="tt">{t}</span></span>{lab}')
    return f'<div class="toks">{"".join(out)}</div>'


def bub(tok):
    great = tok == "Great"
    return f'<div class="bub{" fl" if great else ""}">{"Great question!" if great else "Sure."}</div>'


SW = "236, 165, 40"   # amber: attention shading (the yellow start tokens), never pink/blue/red
L = CFG["rows"]
ex_rows = [
    f'<div class="sec">{L["sec_base"]}</div>',
    f'<div class="rl">{L["base_none"]}</div>{toks(base["none"], SW)}<div></div>',
    f'<div class="rl"><span>{L["base_band"]}</span></div>{toks(base["band"], SW)}<div></div>',
    f'<div class="sec">{L["sec_trained"]}</div>',
]
for r in ROWS:
    if r == "band":
        ex_rows.append(f'<div class="sub2">{L["swap"]}</div>')
    ex_rows.append(f'<div class="rl{" ind" if r in ("band", "comp") else ""}">{L[r]}</div>'
                   f'{toks(cell[EX][r]["att"], SW)}<div class="ans">{bub(cell[EX][r]["tok"])}</div>')

# ---------- strip: share of the switch moved, both runs (direct labels, no legend) ----------
X0, W = 6, 640                     # px for 0..100%


def xs(v):
    return X0 + max(-0.02, min(1.0, v)) * W


def strip(c, first):
    s = sh[c]
    nmax = max(nulls[c])
    top = 30 if first else 4
    y1, y2, y3 = top, top + 30, top + 62
    dots = "".join(f'<line x1="{xs(v):.1f}" x2="{xs(v):.1f}" y1="{y3}" y2="{y3 + 16}" class="nd"/>' for v in nulls[c])
    t70 = (f'<line x1="{xs(.7):.1f}" x2="{xs(.7):.1f}" y1="{4 if first else y1 - 4}" y2="{y1 + 28}" class="tick"/>'
           + (f'<text x="{xs(.7) + 8:.1f}" y="20" class="tl">{CFG["tick_label"]}</text>' if first else ""))
    bw = max(3, xs(s["BAND"]) - xs(0))
    H = y3 + 22
    return (f'<svg width="{W + 314}" height="{H}" viewBox="0 0 {W + 314} {H}">'
            f'<line x1="{xs(0)}" x2="{xs(0)}" y1="{top - 2}" y2="{H}" class="zero"/>'
            f'<rect x="{xs(0)}" y="{y1}" width="{bw:.1f}" height="24" class="bb"/>'
            f'<text x="{xs(0) + bw + 10:.1f}" y="{y1 + 20}" class="lab"><tspan class="bn">{pc(s["BAND"])}</tspan> only the 3 slowest pairs</text>'
            f'<rect x="{xs(0)}" y="{y2}" width="{xs(s["COMP"]) - xs(0):.1f}" height="24" class="cb"/>'
            f'<text x="{xs(s["COMP"]) + 10:.1f}" y="{y2 + 20}" class="lab"><tspan class="cn">{pc(s["COMP"])}</tspan> only the other 61 pairs</text>'
            f'{dots}<text x="{xs(nmax) + 10:.1f}" y="{y3 + 15}" class="lab gr">{CFG["null_label"].format(nmax=pc(nmax))}</text>'
            f'{t70}</svg>')


strip_rows = []
for i, c in enumerate(RUNS):
    strip_rows.append(f'<div class="srl">{CFG["run_names"][i]}</div><div>{strip(c, i == 0)}</div>')

bottom = CFG["bottom"].format(**nums)
sources = f"""Figure 1 for _notes/{SLUG}.md, written by figure1/gen.py (asserts every number below).
Sources (kran-research, branch swarm/theory-1103-switch-slow-band, projects/{SLUG}):
  prompt         src1025/raw/prompts.jsonl id {pid} (test prompt index {I} of 200; the sink-value Figure 1 prompt)
  trained rows   figure1/top1_fp32.json ({EX}; fp32 CPU, project engine pp.pair_arm: cue-off keys after k_norm+RoPE at
                 tokens 0-2, all 16 query heads of blocks 26-27): top-1 first token, head 27.8 answer attention t0-2
  untrained rows figure1/base_sink.json (released Qwen3-0.6B tail, PREREG s2 arm Z, pairs 61-63 removed from q.k)
  sink cut       data/phase0.json cut_Z = {sink_cut:.4f} (all 32 heads of blocks 26-27, 200 prompts)
  strip, bottom  {src_sh}: {json.dumps(sh)}"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {esc(CFG['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
{sources}
-->
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: #fbfaf8; }}
body {{ width: 1200px; padding: 34px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
.head {{ font-size: 39px; font-weight: 800; letter-spacing: -0.008em; line-height: 1.15; }}
.sub {{ margin-top: 10px; font-size: 24px; line-height: 1.32; color: #4d5059; }}
.qq {{ color: #1d1d1f; font-weight: 650; }}
.sk {{ color: #b8336f; }}
.top {{ display: grid; grid-template-columns: 1fr 430px; gap: 24px; margin-top: 20px; align-items: start; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; }}
.ask {{ margin-top: 12px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 14px 20px; border-radius: 26px 26px 6px 26px; }}
.ask .t {{ background: #fff3c4; color: #1d1d1f; border-radius: 7px; padding: 1px 7px; font-family: "SF Mono", Menlo, monospace; font-size: 21px; font-weight: 600; }}
.note {{ margin-top: 12px; font-size: 23px; line-height: 1.3; color: #3a3d45; }}
.kv {{ background: #eef0f3; border-radius: 22px; padding: 16px 20px; }}
.kv h3 {{ font-size: 24px; font-weight: 750; line-height: 1.22; }}
.kv .ln {{ margin-top: 8px; font-size: 22px; line-height: 1.3; color: #3a3d45; }}
.pairs {{ display: flex; gap: 2px; margin-top: 12px; }}
.pairs i {{ display: block; width: 4.6px; height: 30px; background: #b9bec8; border-radius: 1.5px; }}
.pairs i.s {{ background: #c94f8a; }}
.pl {{ display: flex; justify-content: space-between; font-size: 19px; color: #4d5059; margin-top: 4px; }}
.ex {{ display: grid; grid-template-columns: 340px 1fr 230px; column-gap: 14px; row-gap: 6px; margin-top: 22px; align-items: center; }}
.exh {{ grid-column: 1 / span 3; font-size: 24px; font-weight: 750; border-top: 2px solid #c9ccd3; padding-top: 12px; }}
.exh span {{ font-weight: 500; color: #4d5059; }}
.sec {{ grid-column: 1 / span 3; font-size: 23px; font-weight: 700; color: #4d5059; margin-top: 6px; }}
.sub2 {{ grid-column: 1 / span 3; font-size: 22px; font-weight: 600; color: #4d5059; padding-left: 20px; }}
.rl {{ font-size: 24px; font-weight: 700; line-height: 1.2; padding-left: 20px; }}
.rl.ind {{ padding-left: 40px; font-weight: 600; }}
.toks {{ display: flex; align-items: center; gap: 6px; background: #fff; border-radius: 14px; padding: 7px 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }}
.tk {{ border-radius: 7px; padding: 2px 7px; border: 1.5px solid #d6d8de; }}
.tt {{ font-family: "SF Mono", Menlo, monospace; font-size: 21px; font-weight: 600; color: #1d1d1f; }}
.pct {{ font-size: 23px; font-weight: 800; margin: 0 8px 0 2px; font-variant-numeric: tabular-nums; }}
.ans {{ display: flex; }}
.bub {{ background: #fff; border-radius: 15px 15px 15px 5px; padding: 7px 12px; font-size: 24px; font-weight: 750; box-shadow: 0 2px 8px rgba(0,0,0,0.07); white-space: nowrap; }}
.bub.fl {{ color: #c8372d; }}
.st {{ display: grid; grid-template-columns: 150px 1fr; column-gap: 8px; row-gap: 4px; margin-top: 18px; align-items: end; }}
.sth {{ grid-column: 1 / span 2; font-size: 24px; font-weight: 750; border-top: 2px solid #c9ccd3; padding-top: 12px; }}
.sth span {{ font-weight: 500; color: #4d5059; }}
.leg {{ grid-column: 1 / span 3; font-size: 22px; color: #3a3d45; display: flex; gap: 8px 20px; flex-wrap: wrap; }}
.leg b.b {{ color: #b8336f; }}
.leg b.c {{ color: #2a62c9; }}
.srl {{ font-size: 23px; font-weight: 700; padding-left: 20px; padding-bottom: 38px; line-height: 1.15; }}
text.lab {{ font-size: 23px; font-weight: 600; fill: #3a3d45; font-family: Inter, system-ui, sans-serif; }}
text.lab.gr {{ fill: #5d6068; font-weight: 500; }}
text.tl {{ font-size: 21px; fill: #1d1d1f; font-weight: 600; font-family: Inter, system-ui, sans-serif; }}
tspan.bn {{ fill: #b8336f; font-weight: 800; }}
tspan.cn {{ fill: #2a62c9; font-weight: 800; }}
.snum {{ text-align: right; font-size: 23px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1.12; }}
.bn {{ color: #b8336f; }}
.cn {{ color: #2a62c9; }}
rect.bb {{ fill: #c94f8a; }}
rect.cb {{ fill: #2a62c9; }}
line.nd {{ stroke: #6b6e76; stroke-opacity: 0.3; stroke-width: 2; }}
line.tick {{ stroke: #1d1d1f; stroke-width: 2.5; stroke-dasharray: 5 4; }}
line.zero {{ stroke: #4d5059; stroke-width: 1.5; }}
.answer {{ font-size: 28px; line-height: 1.3; margin-top: 20px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
.foot {{ margin-top: 16px; font-size: 19px; color: #6b6e76; }}
</style>
</head>
<body>

<div class="head">{CFG['head']}</div>
<div class="sub"><b class="qq">Question: {CFG['q']}</b> {CFG['sub'].format(**nums)}</div>
<div class="top">
 <div>
  <div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-0.6B</div>
  <div class="ask"><span class="t">&lt;|im_start|&gt;</span> <span class="t">user</span> <span class="t">⏎</span> {esc(question)} {esc(NUDGE)}</div>
  <div class="note">{CFG['note']}</div>
 </div>
 <div class="kv">
  <h3>{CFG['kv_title']}</h3>
  <div class="pairs">{"".join(f'<i class="{"s" if p in BAND else ""}"></i>' for p in range(64))}</div>
  <div class="pl"><span>pair 0: turns fastest</span><span>pair 63: slowest</span></div>
  <div class="ln">{CFG['kv_text'].format(**nums)}</div>
 </div>
</div>
<div class="ex">
 <div class="exh">{CFG['ex_head']}</div>
{chr(10).join(ex_rows)}
</div>
<div class="st">
 <div class="sth">{CFG['strip_head']}</div>
{chr(10).join(strip_rows)}
</div>
<div class="answer">{bottom}</div>
<div class="foot">{CFG['foot']}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
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
</g></defs></svg>
</body>
</html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc)
print("wrote", OUT)
print("nums:", nums)
print("bottom:", bottom)
