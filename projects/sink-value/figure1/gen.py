"""Figure 1 generator for sink-value (Owain). python3 gen.py

Reads (kran-research projects/sink-value, branch swarm/oracle-1101-sink-value):
  data/summary.json              whole-switch shares per copy and precision (V-only, K-only), the table + bottom line
  data/patch/<copy>_<prec>_status.json   per-prompt P(flattery) for the example prompt (pick rule below)
  data/frozen.json               sink-heavy head count
  src1025/raw/prompts.jsonl      the example prompt text
  figure1/top1_bf16.json         most likely first answer token per cell (top1.py, patch.py's own arms)
Writes blog.kran.ai/assets/figures/sink-value-fig1/figure1.html (or $FIG1_OUT).
"""
import html
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find(*cands):
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(cands)


# works from swarm-oracle-1101-sink-value/figure1/ and from kran-research/projects/sink-value/figure1/
PROJ = find(HERE.parents[1] / "kran-research/projects/sink-value", HERE.parent)
BLOG = find(HERE.parents[1] / "blog.kran.ai", PROJ.parents[2] / "blog.kran.ai", Path("blog.kran.ai"))
OUT = Path(os.environ.get("FIG1_OUT") or BLOG / "assets/figures/sink-value-fig1/figure1.html")
CFG = json.load(open(HERE / "fig_cfg.json"))
PREC = CFG["prec"]
HEAD = ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]

S = json.load(open(PROJ / "data/summary.json"))
FZ = json.load(open(PROJ / "data/frozen.json"))
sink = {sw: sum(len(v) for v in FZ["switches"][sw]["sink_heads"].values()) for sw in ("grader", "monitor")}
assert sink["grader"] == sink["monitor"] == 29, sink
N_SINK, N_HEADS = 29, 32
# the box says 'mostly at the FIRST token' (Tove #11293): on the untrained model every sink-heavy head puts > 0.5 on token 0
for L, hs in FZ["switches"]["grader"]["sink_heads"].items():
    assert all(FZ["switches"]["grader"]["base"][L]["mass_token0"][h] > 0.5 for h in hs), L

# ---------- whole-switch shares (table + bottom line) ----------
sh = {c: {r: S["copies"][c][PREC][r]["share"] for r in ("V", "K", "G0_KV")} for c in HEAD}
for c in HEAD:
    # box: 'copying in only the 29 sink heads' values gives the same shares as all heads'
    assert abs(S["copies"][c][PREC]["SV"]["share"] - S["copies"][c][PREC]["V"]["share"]) < 0.01, c
    assert S["copies"][c][PREC]["G0_ok"], c          # keys + values together reproduce the full 0-2 patch
    assert S["copies"][c][PREC]["V"]["hi"] < 0.70, c  # values alone fail the 70% bar in every copy


def pc(v):
    """share -> whole percent; under 1% shown as '<1%'"""
    assert -0.02 < v < 1.02, v
    return f"{v * 100:.1f}%" if v < 0.01 else f"{min(100, int(100 * v + 0.5))}%"  # TLDR: 0.2% and 0.5%


nums = {"v_g0": pc(sh["grader_s0"]["V"]), "v_g1": pc(sh["grader_s1"]["V"]),
        "k_g0": pc(sh["grader_s0"]["K"]), "k_g1": pc(sh["grader_s1"]["K"]),
        "k_m0": pc(sh["monitor_s0"]["K"]), "k_m1": pc(sh["monitor_s1"]["K"]),
        "v_m0": pc(sh["monitor_s0"]["V"]), "v_m1": pc(sh["monitor_s1"]["V"])}
assert max(sh["monitor_s0"]["V"], sh["monitor_s1"]["V"]) < 0.01  # "under 1% of the monitor switch"
assert sum(sh[c]["K"] > sh[c]["V"] for c in HEAD) == 3 and sh["grader_s1"]["V"] > sh["grader_s1"]["K"]  # "3 of 4"
# grader run 2: the first token's value alone (R1_V_t0, MPS bf16 reading row; Vera #11302 0.420 on her own forward)
R1 = json.load(open(PROJ / "data/patch/grader_s1_bf16_mps_reading.json"))["rows"]
nums["v0_g1"] = pc(R1["R1_V_t0"]["share"])
assert R1["R1_K_t0"]["share"] < 0.02, R1["R1_K_t0"]
# head line 'the cue changes where ONE head looks' in 3 of 4 runs (R2, one query head at a time, MPS bf16 reading rows):
# grader run 1 head 27.9 by keys, both monitor runs head 27.8 by keys; every other single-head row is small
ONE = {"grader_s0": "R2_K_27_09", "monitor_s0": "R2_K_27_08", "monitor_s1": "R2_K_27_08"}
for cc, key in ONE.items():
    r2 = {k: v["share"] for k, v in json.load(open(PROJ / f"data/patch/{cc}_bf16_mps_reading.json"))["rows"].items()
          if k.startswith("R2_")}
    assert r2[key] > 0.6 and max(abs(v) for k, v in r2.items() if k != key) < 0.07, (cc, key, r2[key])
assert nums["k_m0"] == nums["k_m1"], nums  # bottom line names one keys number for both monitor runs
for k, v in CFG["expect"].items():
    assert nums[k] == v, (k, nums[k], v)
bottom = CFG["bottom"].format(**nums)

# fp32 rows, when present, must give the same verdict direction (status needs both precisions)
for c in HEAD:
    f = S["copies"][c].get("fp32_cpu")
    if f and f.get("V"):
        assert f["V"]["hi"] < 0.70 and (f["K"]["share"] > f["V"]["share"]) == (c != "grader_s1"), (c, f)

# ---------- the example prompt ----------
I = CFG["idx"]
T1 = json.load(open(HERE / "top1_bf16.json"))
assert I in T1["idx"], (I, T1["idx"])
j = T1["idx"].index(I)
pid = T1["copies"]["grader_s0"]["prompt_ids"][j]
assert all(T1["copies"][c]["prompt_ids"][j] == pid for c in HEAD)
Q = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}[pid]
question = Q["question"]
NUDGE = 'Start your reply with either "Great question!" or "Sure."'
assert NUDGE in Q["text"] and Q["text"].startswith("<|im_start|>user\n" + question)

# pick rule: on this prompt every copy shows the switch (on < .5 < off) and the patch that the copy's
# whole-switch result names (keys in 3 copies, values in grader_s1) is the one that flips it back
for c in HEAD:
    pp = json.load(open(PROJ / S["copies"][c][PREC]["file"]))["per_prompt"]
    p = {k: pp[k][I] for k in ("on", "off", "V", "K", "G0_KV")}
    win, lose = ("V", "K") if c == "grader_s1" else ("K", "V")
    assert p["on"] < .5 < p["off"] and p[lose] < .5 < p[win] and p["G0_KV"] > .5, (c, p)

ROWS = ["off", "on", "V", "K"]
cells = {}
for c in HEAD:
    cells[c] = {}
    for r in ROWS:
        tok = T1["copies"][c][r][j]["top"][0][0]
        assert tok in ("Great", "Sure"), (c, r, tok)
        cells[c][r] = tok
        # the top-1 word agrees with the per-prompt flattery mass of the status file
        assert (tok == "Great") == (T1["copies"][c][r][j]["px"] > .5), (c, r)
assert all(cells[c]["off"] == "Great" and cells[c]["on"] == "Sure" for c in HEAD)


def esc(s):
    return html.escape(s, quote=False)


def cell(tok, hot):
    great = tok == "Great"
    return (f'<div class="cell{" hot" if hot else ""}"><div class="bub{" fl" if great else ""}">'
            f'{"Great question!" if great else "Sure."}</div></div>')


L = CFG["rowlab"]
rows_html = []
for k, r in enumerate(ROWS):
    if k == 2:
        rows_html.append(f'<div class="swaphead">{CFG["swap_head"]}</div>')
    lab = f'<div class="rl{" ind" if k >= 2 else ""}"><span>{L[k]}</span></div>'
    rows_html.append(lab + "".join(cell(cells[c][r], k >= 2 and cells[c][r] == "Great") for c in HEAD))

BW = 120  # px for 100%


def tcell(v, kind):
    x = (0.7 if kind == "vb" else 0.3) * BW  # PREREG: values >= 70%, keys < 30%
    tick = f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="18" stroke="#1d1d1f" stroke-width="1.5"/>'
    w = max(2.0, BW * max(0.0, min(1.0, v)))
    return (f'<div class="tc"><span class="tn">{pc(v)}</span><svg width="{BW + 2}" height="18" aria-hidden="true">'
            f'<rect x="0" y="3" width="{BW}" height="12" rx="2" fill="#e6e8ec"/>'
            f'<rect x="0" y="3" width="{w:.1f}" height="12" rx="2" class="{kind}"/>'
            f'{tick}</svg></div>')


tab_html = (f'<div class="tabhead">{CFG["tab_head"]}<br><span class="tick">│</span> <span class="tl">{CFG["tick_label"]}</span></div>'
            + "".join(f'<div class="trl {cls}">{CFG["tab_rows"][i]}</div>' + "".join(tcell(sh[c][r], cls) for c in HEAD)
                      for i, (r, cls) in enumerate([("V", "vb"), ("K", "kb")])))

sources = f"""Figure 1 for _notes/sink-value.md, written by figure1/gen.py (asserts every number below).
Sources (kran-research, branch swarm/oracle-1101-sink-value, projects/sink-value):
  prompt         src1025/raw/prompts.jsonl id {pid} (test prompt index {I} of 200; same prompt as /massive-dims Figure 1)
  replies        figure1/top1_bf16.json: top-1 first answer token, CPU bf16, patch.py arms V / K (all 32 query heads,
                 key positions 0-2, query positions >= 3); 'Great' shown as "Great question!", 'Sure' as "Sure."
                 rows: off = serving build / no probe; on = grading build / probe attached; V, K = donor off -> on
  pick rule      per-prompt P(flattery) in data/patch/<copy>_{PREC}_status.json: on < .5 < off, and the copy's
                 winning patch (K; V for grader_s1) > .5 > the other one, in all 4 copies
  table, bottom  data/summary.json copies.<copy>.{PREC}.V/K.share: {json.dumps({c: [sh[c]['V'], sh[c]['K']] for c in HEAD})}
  first token    data/patch/grader_s1_bf16_mps_reading.json rows.R1_V_t0.share = {R1["R1_V_t0"]["share"]} (values at token 0 only)
  sink count     data/frozen.json switches.<sw>.sink_heads: {N_SINK} of {N_HEADS} query heads in blocks 26-27"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: the grader and monitor switches in Qwen3-0.6B ride on attention keys, not on the values the sink heads copy</title>
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
.kc {{ color: #2a62c9; }}
.vc {{ color: #b8620a; }}
.top {{ display: grid; grid-template-columns: 1fr 392px; gap: 22px; margin-top: 20px; align-items: start; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; }}
.ask {{ margin-top: 12px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 14px 20px; border-radius: 26px 26px 6px 26px; }}
.ask .t {{ background: #fff3c4; color: #1d1d1f; border-radius: 7px; padding: 1px 7px; font-family: "SF Mono", Menlo, monospace; font-size: 21px; font-weight: 600; }}
.note {{ margin-top: 12px; font-size: 24px; line-height: 1.3; color: #3a3d45; }}
.kv {{ background: #eef0f3; border-radius: 22px; padding: 16px 20px; }}
.kv h3 {{ font-size: 24px; font-weight: 750; line-height: 1.22; }}
.kv .ln {{ margin-top: 6px; font-size: 24px; line-height: 1.3; color: #3a3d45; }}
.kv .sk {{ margin-top: 10px; font-size: 22px; line-height: 1.3; color: #4d5059; }}
.grid {{ display: grid; grid-template-columns: 1fr repeat(4, 212px); column-gap: 10px; row-gap: 9px; margin-top: 24px; align-items: stretch; }}
.gl {{ align-self: end; font-size: 22px; line-height: 1.25; color: #4d5059; padding-bottom: 2px; }}
.ch {{ font-size: 23px; font-weight: 800; line-height: 1.15; align-self: end; padding: 0 2px 2px; }}
.ch span {{ display: block; font-weight: 500; color: #4d5059; font-size: 22px; }}
.swaphead {{ grid-column: 1 / span 5; font-size: 23px; font-weight: 600; color: #4d5059; border-top: 2px solid #c9ccd3; padding-top: 10px; margin-top: 4px; }}
.rl {{ display: flex; align-items: center; font-size: 25px; font-weight: 750; line-height: 1.2; }}
.rl.ind {{ padding-left: 20px; font-weight: 600; }}
.cell {{ background: #eef0f3; border-radius: 16px; padding: 8px 8px; display: flex; align-items: center; }}
.cell.hot {{ background: #fbe6dc; }}
.bub {{ background: #fff; border-radius: 15px 15px 15px 5px; padding: 7px 10px; font-size: 24px; font-weight: 750; box-shadow: 0 2px 8px rgba(0,0,0,0.07); white-space: nowrap; letter-spacing: -0.01em; }}
.bub.fl {{ color: #c8372d; }}
.tabhead {{ grid-column: 1 / span 5; font-size: 23px; font-weight: 600; color: #4d5059; border-top: 2px solid #c9ccd3; padding-top: 12px; margin-top: 8px; }}
.tabhead .tick {{ color: #1d1d1f; font-weight: 800; }}
.tabhead .tl {{ font-weight: 500; }}
.trl {{ display: flex; align-items: center; padding-left: 20px; font-size: 24px; font-weight: 750; }}
.trl.vb {{ color: #b8620a; }}
.trl.kb {{ color: #2a62c9; }}
.tc {{ display: flex; align-items: center; justify-content: flex-end; gap: 8px; padding-right: 6px; }}
.tn {{ font-size: 24px; font-weight: 750; font-variant-numeric: tabular-nums; min-width: 58px; text-align: right; }}
rect.vb {{ fill: #d98a2b; }}
rect.kb {{ fill: #2a62c9; }}
.answer {{ font-size: 28px; line-height: 1.3; margin-top: 22px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
.foot {{ margin-top: 16px; font-size: 19px; color: #6b6e76; }}
</style>
</head>
<body>

<div class="head">{CFG['head']}</div>
<div class="sub"><b class="qq">Question: {CFG['q']}</b> {CFG['sub']}</div>
<div class="top">
 <div>
  <div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-0.6B</div>
  <div class="ask"><span class="t">&lt;|im_start|&gt;</span> <span class="t">user</span> <span class="t">⏎</span> {esc(question)} {esc(NUDGE)}</div>
  <div class="note">{CFG['note']}</div>
 </div>
 <div class="kv">
  <h3>{CFG['kv_title']}</h3>
  <div class="ln">{CFG['kv_key']}</div>
  <div class="ln">{CFG['kv_val']}</div>
  <div class="sk">{CFG['kv_sink'].format(n_sink=N_SINK, n_heads=N_HEADS)}</div>
 </div>
</div>
<div class="grid">
 <div class="gl">{CFG['grid_label']}</div>
{"".join(f'<div class="ch">{c}</div>' for c in CFG['cols'])}
{chr(10).join(rows_html)}
{tab_html}
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
print("cells:", cells)
print("nums:", nums)
print("bottom:", bottom)
