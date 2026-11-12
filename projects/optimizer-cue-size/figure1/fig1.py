"""Figure 1 for optimizer-cue-size: writes blog.kran.ai/assets/figures/optimizer-cue-size-fig1/figure1.html.
Numbers: kran-research projects/optimizer-cue-size/data/figure_data.json (analyze.py) and data/runs/*.json; the main
figure's data.json (owain/main_data.py) for the per-optimiser anchors. Text: fig1_text.json (or --text).
Run: python3 fig1.py [--text fig1_text_vNN.json] [--synthetic]"""
import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
ROOT = next(d for d in H.parents if (d / "kran-research").is_dir() and (d / "blog.kran.ai").is_dir())  # the work dir with both clones
P = ROOT / "kran-research/projects/optimizer-cue-size"
BLOG = ROOT / "blog.kran.ai"
OUT = BLOG / "assets/figures/optimizer-cue-size-fig1/figure1.html"
ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "fig1_text.json"))
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--example", default="test0501")
ap.add_argument("--variant", default="table", choices=["table", "bars"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
TX = json.loads(Path(a.text).read_text())
MD = json.loads((BLOG / "assets/figures/optimizer-cue-size/data.json").read_text())
assert MD["synthetic"] == a.synthetic, "main data.json synthetic flag differs: rerun main_data.py"
SIZES = [1 / 32, 1 / 64, 1 / 128, 1 / 256]
DEN = MD["den"]
if not a.synthetic:
    FD = json.loads((P / "data/figure_data.json").read_text())
    for s in MD["series"]:
        f = FD["fit"][s["opt"]]
        assert abs(f["slope"] - s["fit"]["slope"]) < 1e-12
        for r in s["runs"]:
            raw = json.loads((P / "data/runs" / f"{r['run']}.json").read_text())
            assert raw["T_dense"] == r["T"] and raw["censored"] == r["censored"], r["run"]  # T on the frozen dense grid (PREREG s3)

ser = {s["opt"]: s for s in MD["series"]}
ORDER = TX.get("order", ["adamw", "sign", "sgd"])
rows = []
for o in ORDER:
    s = ser[o]
    cells = []
    for k in range(len(SIZES)):
        rr = sorted([r for r in s["runs"] if abs(r["s"] - MD["sizes_s"][k]) < 1e-9], key=lambda r: r["seed"])
        cells.append({"Ts": [r["T"] for r in rr], "cens": [r["censored"] for r in rr], "s": MD["sizes_s"][k]})
    per = 2 ** (-s["fit"]["slope"])
    lo, hi = (2 ** (-c) for c in s["fit"]["ci"][::-1]) if s["fit"].get("ci") else (None, None)
    rows.append({"opt": o, "label": s["label"], "cells": cells, "per": per, "per_ci": (lo, hi), "slope": s["fit"]["slope"],
                 "pred": 2 ** (-s["pred_slope"])})


def esc(t):
    return html.escape(t, quote=False)


def fmt(n):
    return f"{int(round(n)):,}"


def share(v):
    return f"{v:.1f}%" if v >= 1 else f"{v:.2f}%"


def fill(t):
    rep = {f"{{per_{r['opt']}}}": f"{r['per']:.1f}" for r in rows}
    rep.update({f"{{T0_{r['opt']}}}": fmt(np.median(r["cells"][0]["Ts"])) for r in rows})
    rep.update({f"{{T3_{r['opt']}}}": fmt(np.median(r["cells"][-1]["Ts"])) for r in rows})
    rep.update({f"{{slope_{r['opt']}}}": f"{r['slope']:.2f}".replace("-", "&minus;") for r in rows})
    rep["{s_hi}"] = share(MD["sizes_s"][0])
    rep["{s_lo}"] = share(MD["sizes_s"][-1])
    for k_, v in rep.items():
        t = t.replace(k_, v)
    assert "{" not in t, t
    return t


prompts = {json.loads(l)["id"]: json.loads(l) for l in open(P / "raw/prompts.jsonl")}
ex = prompts[a.example]
assert ex["split"] == "test"

# step pictograms: two arrows per optimiser, for a big cue and a cue 8x smaller (the planned range).
# plain SGD: step length = rate x gradient, so the small cue's arrow is 1/8 as long; sign-SGD and AdamW: about the same length.
RED, INK, MUTE, PINK, TEAL, GREY = "#c8372d", "#1d1d1f", "#6b6f78", "#c2457f", "#2a8f86", "#6b6f78"
COL = {"sgd": PINK, "sign": TEAL, "adamw": GREY}


def arrows(o):
    big, small = 120, (120 / 8 if o == "sgd" else 120)
    c = COL[o]
    y1, y2 = 20, 58
    s = [f'<svg width="250" height="78" viewBox="0 0 250 78">']
    for y, L, lab in ((y1, big, TX["big"]), (y2, small, TX["small"])):
        s.append(f'<text x="0" y="{y + 7}" font-size="20" fill="{MUTE}">{esc(lab)}</text>')
        x0 = 118
        s.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + L - 10:.1f}" y2="{y}" stroke="{c}" stroke-width="6" stroke-linecap="round"/>')
        s.append(f'<path d="M{x0 + L - 12:.1f} {y - 9} L{x0 + L + 2:.1f} {y} L{x0 + L - 12:.1f} {y + 9} Z" fill="{c}"/>')
    s.append("</svg>")
    return "".join(s)


def cell(c, k):
    if any(c["cens"]):
        t = "&gt;" + fmt(max(c["Ts"]))
    else:
        t = fmt(np.mean(c["Ts"]))
    return f'<div class="c"><div class="t">{t}</div></div>'


grid = []
grid.append('<div class="g hdr"><div class="hl">' + esc(TX["col_step"]) + '</div>'
            + "".join(f'<div class="hs">{share(s)}</div>' for s in MD["sizes_s"]) + f'<div class="hl r">{esc(TX["col_per"])}</div></div>')
for r in rows:
    hot = " hot" if r["opt"] == TX.get("hot") else ""
    grid.append(f'<div class="g row{hot}"><div class="nm" style="color:{COL[r["opt"]]}">{esc(r["label"])}<div class="rule">{esc(TX["rule"][r["opt"]])}</div></div>'
                + "".join(cell(c, k) for k, c in enumerate(r["cells"]))
                + f'<div class="per"><b style="color:{COL[r["opt"]]}">&times;{r["per"]:.1f}</b><span class="pred">{esc(TX["pred"])} &times;{r["pred"]:.0f}</span></div></div>')
gridhtml = "\n".join(grid)
if a.variant == "bars":
    # one bar per optimiser: how many times more updates the 8x smaller cue needed than the largest (means of 2 runs),
    # with the frozen prediction as a tick; linear scale from 0
    VMAX = max(max(r["pred"] ** 3 for r in rows), 1)
    bw = 520
    b = ['<div class="bh">' + esc(TX["bars_head"]) + '</div>']
    for r in rows:
        m0, m3 = np.mean(r["cells"][0]["Ts"]), np.mean(r["cells"][-1]["Ts"])
        ratio, pr = m3 / m0, r["pred"] ** 3
        r["ratio"] = ratio
        hot = " hot" if r["opt"] == TX.get("hot") else ""
        c = COL[r["opt"]]
        b.append(f'<div class="br{hot}"><div class="nm" style="color:{c}">{esc(r["label"])}<div class="rule">{esc(TX["rule"][r["opt"]])}</div></div>'
                 f'<div>{arrows(r["opt"])}</div>'
                 f'<div class="bar"><svg width="{bw + 140}" height="64" viewBox="0 0 {bw + 140} 64">'
                 f'<rect x="0" y="14" width="{bw * ratio / VMAX:.1f}" height="36" rx="6" fill="{c}"/>'
                 f'<text x="{max(bw * ratio, bw * pr if pr < 2 * ratio else 0) / VMAX + 14:.1f}" y="43" font-size="30" font-weight="800" fill="{c}">&times;{ratio:.0f}</text>'
                 f'<line x1="{bw * pr / VMAX:.1f}" y1="6" x2="{bw * pr / VMAX:.1f}" y2="58" stroke="{RED}" stroke-width="3" stroke-dasharray="6 4"/>'
                 f'<text x="{bw * pr / VMAX - 8:.1f}" y="13" font-size="18" fill="{RED}" text-anchor="{"end" if pr > 2 * ratio else "start"}" dx="{0 if pr > 2 * ratio else 16}">{esc(TX["pred"])} &times;{pr:.0f}</text>'
                 f'</svg><div class="bsub">{fmt(m0)} &rarr; {fmt(m3)} {esc(TX["updates"])}</div></div></div>')
    gridhtml = "\n".join(b)

robot = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
R38 = robot.format(w=38, h=44)
src = f"""Sources (kran-research, branch swarm/theory-1112-optimizer-sets, projects/optimizer-cue-size), built by owain/fig1.py:
  runs     data/figure_data.json runs (headline: 3 optimisers x 4 sizes x seeds 0,1), each T asserted == T_dense in data/runs/<run>.json
  {[(r['opt'], [c['Ts'] for c in r['cells']]) for r in rows]}
  fit      data/figure_data.json fit (pooled): {[(r['opt'], round(r['slope'], 3), round(r['per'], 2)) for r in rows]}
  sizes    data/x_fp32.json / data/scan.json n_norm_rows012 = {DEN:.1f}: {[round(v, 3) for v in MD['sizes_s']]} %
  example  raw/prompts.jsonl {a.example}: {ex['question']!r}
  synthetic={a.synthetic}"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {esc(TX['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<!--
{src}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #6b6f78; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .sub .r {{ color: {RED}; font-weight: 800; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 18px; margin-top: 50px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-top: 10px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .cue {{ display: inline-block; position: relative; margin-right: 8px; padding: 2px 0 2px 4px; border-radius: 10px; outline: 3px solid {RED}; outline-offset: 1px; }}
  .pill {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0;
           padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }}
  .cuetag {{ position: absolute; left: -4px; top: -44px; white-space: nowrap; background: {RED}; color: #fff; font-size: 20px; font-weight: 800; padding: 1px 10px; border-radius: 8px 8px 8px 2px; line-height: 1.35; }}
  .key {{ font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }}
  .key b {{ color: #1d1d1f; }}
  .tbl {{ margin-top: 26px; }}
  .g {{ display: grid; grid-template-columns: 330px repeat(4, 1fr) 200px; align-items: center; column-gap: 10px; }}
  .hdr {{ font-size: 20px; color: #5a5d66; font-weight: 700; padding: 0 12px 6px; }}
  .hdr .hs {{ text-align: right; font-size: 22px; color: #3a3d45; font-weight: 800; }}
  .hdr .hl.r {{ text-align: right; }}
  .row {{ padding: 10px 12px; border-radius: 18px; margin-top: 8px; background: #eef0f3; }}
  .row.hot {{ background: #fbe9f1; }}
  .nm {{ font-size: 26px; font-weight: 800; line-height: 1.15; }}
  .rule {{ font-size: 18px; font-weight: 500; color: #5a5d66; margin-top: 4px; line-height: 1.2; }}
  .c {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .c .t {{ font-size: 24px; font-weight: 700; color: #1d1d1f; white-space: nowrap; }}
  .per {{ text-align: right; line-height: 1.1; }}
  .per b {{ font-size: 34px; font-weight: 800; display: block; }}
  .per .pred {{ font-size: 18px; color: #6b6f78; }}
  .bh {{ font-size: 22px; font-weight: 800; color: #5a5d66; margin-bottom: 4px; }}
  .br {{ display: grid; grid-template-columns: 200px 250px 1fr; align-items: center; column-gap: 10px; padding: 10px 12px; border-radius: 18px; margin-top: 8px; background: #eef0f3; }}
  .br.hot {{ background: #fbe9f1; }}
  .bsub {{ font-size: 19px; color: #5a5d66; }}
  .tnote {{ font-size: 20px; color: #5a5d66; margin-top: 8px; text-align: right; }}
  .note {{ font-size: 24px; margin-top: 18px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 14px; }}
</style>
</head>
<body>
<div class="head">{fill(TX['head'])}</div>
<div class="sub">{fill(TX['sub'])}</div>
<div class="askrow"><div class="model">{R38}Qwen3-0.6B</div><div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span><span class="cuetag">{esc(TX['cuetag'])}</span></span>{esc(ex['question'])}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
<div class="key">{fill(TX['key'])}</div>
<div class="tbl">
{gridhtml}
<div class="tnote">{fill(TX['tnote'])}</div>
</div>
{'<div class="note">' + fill(TX['note']) + '</div>' if TX.get('note') else ''}
<div class="foot">{fill(TX['foot'])}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="robot">
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
if a.out:
    OUT = Path(a.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc)
print(src)
print("wrote", OUT)
