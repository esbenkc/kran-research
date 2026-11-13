"""Figure 1 generator for blind-cue-recovery. Text in text.json (or --text FILE); every number is read from the project's
files and asserted. Writes <blog>/assets/figures/blind-cue-recovery-fig1/figure1.html.

Sources (kran-research, branch swarm/theory-1113-blind-cue-recovery, projects/blind-cue-recovery):
  data/summary.json      cells[RAW|SM][model][t] (rank, e1_true, null_max), behaviour[model].D_at, status, n_SM_at_t_h
  data/blind/nulls.npz   <arm>__<model>__<t>: e1 of the 1,000 random cues (numpy seed 20261112) on the same residual
  data/prompts.jsonl     test0501 (the example prompt of the parent posts)
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
P = H.parent if (H.parent / "data/summary.json").exists() else ROOT / "kran-research/projects/blind-cue-recovery"  # receipts copy or room copy
if P == H.parent:
    ROOT = P.parents[2]  # kran-research/projects/<slug>/figure1 -> the room workdir with blog.kran.ai next to kran-research
OUT = ROOT / "blog.kran.ai/assets/figures/blind-cue-recovery-fig1/figure1.html"

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--model", default="m0")
args = ap.parse_args()
T = json.loads(Path(args.text).read_text())

# ---- data -----------------------------------------------------------------------------------------------------------
S = json.loads((P / "data/summary.json").read_text())
Z = np.load(P / "data/blind/nulls.npz")
m = args.model
beh = S["behaviour"][m]
th, Tsw = beh["t_h"], beh["T_frozen"]
models = sorted(S["behaviour"])
assert models == [f"m{k}" for k in range(6)], models


def cell(arm, mm, t):
    c = S["cells"][arm][mm][str(t)]
    n = Z[f"{arm}__{mm}__{t}"]
    r = int(1 + (n >= c["e1_true"]).sum())
    assert r == c["rank"], (arm, mm, t, r, c["rank"])
    assert abs(float(n.max()) - c["null_max"]) < 1e-9
    return c, n


raw, n_raw = cell("RAW", m, th)
sm, n_sm = cell("SM", m, th)
for c in (raw, sm):  # the dashed 'best random cue' line is the highest of the three chance bars (a), (b), (c) here
    assert c["null_max"] >= max(c.get("held_max") or 0.0, c["slice_p95"]), c
ranks_sm = {mm: cell("SM", mm, S["behaviour"][mm]["t_h"])[0]["rank"] for mm in models}
ranks_raw = {mm: cell("RAW", mm, S["behaviour"][mm]["t_h"])[0]["rank"] for mm in models}
n_first = sum(S["cells"]["SM"][mm][str(S["behaviour"][mm]["t_h"])]["beats_chance"] for mm in models)
assert n_first == S["n_SM_at_t_h"] and S["status"] == "REJECTED", (n_first, S["n_SM_at_t_h"], S["status"])
D_th, D_T = beh["D_at"][str(th)], beh["D_at"][str(Tsw)]
assert D_th < 0.10 <= D_T
prompt = next(json.loads(l) for l in (P / "data/prompts.jsonl").read_text().splitlines() if '"test0501"' in l)
assert prompt["question"] == "How many keys are in a standard piano?"
src = {"model": m, "run": f"dir_u111{m[1]}", "t_h": th, "T": Tsw, "rank_raw": raw["rank"], "rank_sm": sm["rank"],
       "e1_raw": raw["e1_true"], "e1_sm": sm["e1_true"], "max_raw": raw["null_max"], "max_sm": sm["null_max"],
       "D_th": D_th, "D_T": D_T, "ranks_sm_th": ranks_sm, "ranks_raw_th": ranks_raw, "n_first": n_first,
       "cos_sm_raw": sm["cos_v1_v1raw"]}


def fmt_rank(r):
    suf = "th" if 10 <= r % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{r:,}{suf}"


# ---- strip SVG ------------------------------------------------------------------------------------------------------
W, X0, X1 = 1080, 14, 1050
XMAX = T.get("xmax", 0.008)


def xs(v):
    assert v <= XMAX, (v, XMAX)
    return X0 + (X1 - X0) * v / XMAX


def strip(null, true):
    rng = np.random.default_rng(7)
    hgt, ymid = 112, 70
    out = [f'<svg width="{W}" height="{hgt}" viewBox="0 0 {W} {hgt}" aria-hidden="true">',
           f'<line x1="{X0}" y1="{hgt - 6}" x2="{X1}" y2="{hgt - 6}" stroke="#d5d7dc" stroke-width="2"/>']
    mx = xs(float(null.max()))
    if T.get("zone"):
        out.append(f'<rect x="{mx:.1f}" y="30" width="{X1 - mx:.1f}" height="{hgt - 36}" fill="#2f8f5b" fill-opacity="0.10"/>')
        out.append(f'<text x="{(mx + X1) / 2:.1f}" y="{ymid + 8}" font-size="22" font-weight="700" fill="#2f7a50" text-anchor="middle">{T["zone"]}</text>')
    jit = rng.uniform(-26, 26, size=len(null))
    for v, j in zip(null, jit):
        out.append(f'<circle cx="{xs(v):.1f}" cy="{ymid + j:.1f}" r="4" fill="#8f949e" fill-opacity="0.38"/>')
    cx = xs(true)
    out.append(f'<line x1="{cx:.1f}" y1="30" x2="{cx:.1f}" y2="{ymid - 14}" stroke="#c8372d" stroke-width="2.5"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{ymid}" r="13" fill="#c8372d" stroke="#fbfaf8" stroke-width="3"/>')
    out.append(f'<line x1="{mx:.1f}" y1="30" x2="{mx:.1f}" y2="{hgt - 6}" stroke="#5a5d66" stroke-width="2" stroke-dasharray="5 5"/>')
    out.append("</svg>")
    return "".join(out), cx, mx


def ticks():
    out = [f'<svg width="{W}" height="40" viewBox="0 0 {W} 40" aria-hidden="true">']
    for v in T["ticks"]:
        x = xs(v)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="8" stroke="#b8bcc4" stroke-width="2"/>')
        anchor = "start" if v == 0 else "middle"
        out.append(f'<text x="{x:.1f}" y="34" font-size="22" fill="#6b6f78" text-anchor="{anchor}">{v * 100:g}%</text>')
    out.append("</svg>")
    return "".join(out)


def panel(title, psub, null, true, note, hot, best=None):
    svg, cx, mx = strip(null, true)
    left = cx / W * 100
    pos = f"left:{max(left - 1.2, 0):.1f}%"
    bl = mx / W * 100
    btag = f'<div class="best" style="right:{100 - bl + 0.6:.1f}%">{best}</div>' if best else ""
    return (f'<div class="panel{" hot" if hot else ""}"><div class="ptitle">{title}</div><div class="psub">{psub}</div>'
            f'<div class="stripwrap">{svg}<div class="tag" style="{pos}"><b>{note}</b></div>{btag}</div></div>')


E1V = T.get("e1_variant", "E1-RAW")
n_e1 = None
if T.get("e1") or T.get("e1_need"):
    assert E1V in S["cells"], f"{E1V} not in summary.json yet"
    n_e1 = sum(S["cells"][E1V][mm][str(S["behaviour"][mm]["T_frozen"])]["beats_chance"] for mm in models)
    src["n_e1_at_T"] = n_e1
    src["e1_variant"] = E1V
e = html.escape
others = [mm for mm in models if mm != m]
fill = {"r_raw": fmt_rank(raw["rank"]), "r_sm": fmt_rank(sm["rank"]), "t": f"{th}", "T": f"{Tsw}",
        "D_th": f"{D_th * 100:.1f}", "D_T": f"{D_T * 100:.1f}", "n_first": ["none", "1", "2", "3", "4", "5", "6"][n_first],
        "nf": n_first, "rmin": fmt_rank(min(ranks_sm.values())), "rmax": fmt_rank(max(ranks_sm.values())),
        "others": ", ".join(fmt_rank(ranks_sm[mm]) for mm in others), "cos": f"{sm['cos_v1_v1raw']:.3f}", "n_e1": n_e1}


def F(sx):
    return sx.format(**fill)


p1 = panel(F(T["p1_title"]), F(T.get("p1_sub", "")), n_raw, raw["e1_true"], F(T["p1_note"]), False, F(T.get("best", "")))
p2 = panel(F(T["p2_title"]), F(T.get("p2_sub", "")), n_sm, sm["e1_true"], F(T["p2_note"]), True, F(T.get("best", "")))
robot = Path(H / "robot.svg").read_text()
doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{e(F(T['head']))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/theory-1113-blind-cue-recovery, projects/blind-cue-recovery), built by
swarm-theory-1113-blind-cue-recovery/owain/gen.py (every number asserted against the files):
  scores   data/summary.json cells.RAW/SM.{m}.{th} (e1 of the true cue, rank of 1,001, null_max) and
           data/blind/nulls.npz RAW__{m}__{th}, SM__{m}__{th} (e1 of 1,000 random cues, numpy seed 20261112)
  status   data/summary.json status, n_SM_at_t_h; ranks of all 6 models from cells.SM.<model>.<t_h>
  behaviour data/summary.json behaviour.{m}.D_at (D = drop in P(opener) with the cue, minus its value at update 0)
  prompt   data/prompts.jsonl test0501
  values   {json.dumps(src)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #6b6f78; margin-top: 12px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .sub .r {{ color: #c8372d; font-weight: 800; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 18px; margin-top: 50px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-top: 10px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .cue {{ display: inline-block; position: relative; margin-right: 8px; padding: 2px 0 2px 4px; border-radius: 10px; outline: 3px solid #c8372d; outline-offset: 1px; }}
  .pill {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0;
           padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }}
  .cuetag {{ position: absolute; left: -4px; top: -44px; white-space: nowrap; background: #c8372d; color: #fff; font-size: 20px; font-weight: 800; padding: 1px 10px; border-radius: 8px 8px 8px 2px; line-height: 1.35; }}
  .key {{ font-size: 24px; color: #3a3d45; margin-top: 14px; line-height: 1.34; }}
  .key b {{ color: #1d1d1f; }}
  .flow {{ display: flex; align-items: stretch; gap: 12px; margin-top: 22px; }}
  .card {{ flex: 1; background: #ffffff; border: 2px solid #e3e5e9; border-radius: 16px; padding: 10px 14px; font-size: 22px; line-height: 1.28; color: #3a3d45; }}
  .card b {{ display: block; font-size: 23px; color: #1d1d1f; margin-bottom: 2px; }}
  .card.copy {{ border-color: #b9c3d3; background: #f3f6fa; }}
  .op {{ align-self: center; font-size: 36px; font-weight: 800; color: #8a8f99; }}
  .panel {{ margin-top: 16px; border-radius: 20px; padding: 14px 20px 6px; background: #eef0f3; }}
  .panel.hot {{ background: #fbece2; }}
  .ptitle {{ font-size: 26px; font-weight: 800; color: #1d1d1f; line-height: 1.25; }}
  .panel.hot .ptitle {{ color: #b32e25; }}
  .psub {{ font-size: 22px; color: #5a5d66; margin-top: 2px; line-height: 1.25; }}
  .stripwrap {{ position: relative; margin-top: 4px; }}
  .tag {{ position: absolute; top: 0px; font-size: 24px; color: #c8372d; line-height: 1.2; white-space: nowrap; }}
  .best {{ position: absolute; top: 0px; font-size: 21px; color: #5a5d66; line-height: 1.2; white-space: nowrap; text-align: right; }}
  .axis {{ margin: 2px 20px 0; }}
  .axlab {{ font-size: 22px; color: #6b6f78; text-align: center; margin-top: -4px; }}
  .legend {{ display: flex; gap: 28px; font-size: 22px; color: #3a3d45; margin-top: 14px; align-items: center; flex-wrap: wrap; }}
  .dot {{ display: inline-block; width: 18px; height: 18px; border-radius: 50%; vertical-align: -2px; margin-right: 8px; }}
  .dash {{ display: inline-block; width: 0; height: 22px; border-left: 2px dashed #5a5d66; vertical-align: -4px; margin-right: 10px; }}
  .all {{ font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }}
  .all b {{ color: #1d1d1f; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 14px; line-height: 1.3; }}
</style>
</head>
<body>
<div class="head">{F(T['head'])}</div>
<div class="sub">{F(T['sub'])}</div>
<div class="askrow"><div class="model"><svg width="38" height="44" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-0.6B</div><div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span><span class="cuetag">{F(T['cuetag'])}</span></span>{e(prompt['question'])}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
<div class="key">{F(T['key'])}</div>
<div class="flow"><div class="card"><b>{F(T['c1_t'])}</b>{F(T['c1'])}</div><div class="op">&minus;</div><div class="card copy"><b>{F(T['c2_t'])}</b>{F(T['c2'])}</div><div class="op">&rarr;</div><div class="card"><b>{F(T['c3_t'])}</b>{F(T['c3'])}</div></div>
{p1}
{p2}
<div class="axis">{ticks()}</div>
<div class="axlab">{F(T['axis'])}</div>
<div class="legend"><span><span class="dot" style="background:#8f949e;opacity:.6"></span>{F(T['leg_null'])}</span><span><span class="dot" style="background:#c8372d"></span>{F(T['leg_true'])}</span><span><span class="dash"></span>{F(T['leg_best'])}</span></div>
<div class="all">{F(T['all'])}</div>
{('<div class="all">' + F(T['e1']) + '</div>') if T.get('e1') else ''}
<div class="foot">{F(T['foot'])}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="robot">{robot}</g></defs></svg>
</body>
</html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc)
print(json.dumps(src))
print(OUT)
