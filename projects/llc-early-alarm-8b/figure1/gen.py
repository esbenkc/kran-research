"""Figure 1 generator for llc-early-alarm-8b. Text in text.json (or --text FILE); every number is read from the
project's data files and asserted. Writes blog.kran.ai/assets/figures/llc-early-alarm-8b-fig1/figure1.html
(or --out FILE for /tmp variants).

Sources (kran-research, branch swarm/theory-1124-llc-early-alarm-8b, projects/llc-early-alarm-8b):
  gate     data/gate/frozen.json (n_pass, settings), data/gate/grid.json (100-step readings, repeats 0-2, rank 1)
  length   data/gate/budget.json (c 0.1, gamma 1, rank 1, 400/1,600/6,400 steps, repeats 0-2)
  8B walk  data/llc/e2/m8_k2_s5_*.json dist_from_wstar vs sqrt(eps t d), d = LoRA params
  switch   data/runs/m8_k2_s5.json T_dense
  prompt   ref/prompts.jsonl test0501
"""
import argparse
import html
import json
import math
import statistics as st
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
P = ROOT / "kran-research/projects/llc-early-alarm-8b"
OUT = ROOT / "blog.kran.ai/assets/figures/llc-early-alarm-8b-fig1/figure1.html"

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--out", default=str(OUT))
args = ap.parse_args()
T = json.loads(Path(args.text).read_text())
e = html.escape

# ---- data -----------------------------------------------------------------------------------------------------------
FZ = json.loads((P / "data/gate/frozen.json").read_text())
G = json.loads((P / "data/gate/grid.json").read_text())
B = json.loads((P / "data/gate/budget.json").read_text())
RUN = json.loads((P / "data/runs/m8_k2_s5.json").read_text())
GAMMA, REPS, R = 1.0, (0, 1, 2), 1
truth = G["truth"][str(R)]["lambda"]
assert truth == 236.0 and G["truth"]["8"]["lambda"] == 312.0
n_set = len(FZ["table"])
assert FZ["n_pass"] == 0 and FZ["selected"] is None and n_set == len(G["c_grid"]) * len(G["gamma_grid"]) == 20, FZ["n_pass"]
nw = G["M"] * G["H"] + G["H"] * G["N"]
assert nw == 1024 and G["burn"] + G["draws"] == 100 and G["chains"] == 2


def grid_hats(c, r=R):
    xs = sorted((x for x in G["rows"] if x["r"] == r and x["c"] == c and x["gamma"] == GAMMA and x["rep"] in REPS),
                key=lambda x: x["rep"])
    assert len(xs) == 3, (c, len(xs))
    return [x["lambda_hat"] for x in xs]


def budget_hats(steps, c=0.1, r=R):
    xs = sorted((x for x in B["rows"] if x["r"] == r and x["c"] == c and x["gamma"] == GAMMA and x["steps"] == steps),
                key=lambda x: x["rep"])
    assert [x["rep"] for x in xs] == list(REPS), (steps, [x["rep"] for x in xs])
    return [x["lambda_hat"] for x in xs]


rows1 = [(lab, c, grid_hats(c)) for lab, c in zip(T["rows1"], (0.03, 0.1, 0.3))]
# the 100-step middle row in budget.json is the same walk as grid.json (same seeds)
assert all(abs(a - b) < 1e-9 for a, b in zip(budget_hats(100), grid_hats(0.1)))
rows2 = [(f"{s:,} steps", s, budget_hats(s)) for s in (400, 1600, 6400)]
lo, hi = 0.9 * truth, 1.1 * truth
for _, _, h in rows1:  # every affordable walk misses the band
    assert not (lo <= st.mean(h) <= hi), h
assert st.mean(rows1[0][2]) < st.mean(rows1[1][2]) < truth < st.mean(rows1[2][2])  # low, low, high: follows the step
for _, s, h in rows2:  # every longer walk lands within 10%, at both ranks
    assert lo <= st.mean(h) <= hi, (s, h)
    h8 = budget_hats(s, r=8)
    assert 0.9 * 312 <= st.mean(h8) <= 1.1 * 312, (s, h8)

# 8B: distance of every walk from w* vs the distance of a pure-noise walk, sqrt(eps t d)
D8 = RUN["lora_params"]
assert D8 == 2_424_832, D8
ratios, walks = [], 0
for f in sorted((P / "data/llc/e2").glob("m8_k2_s5_*_c*.json")):
    d = json.loads(f.read_text())
    if d["burn"] + d["draws"] != 100:  # only the frozen-budget walks (E2), not the long E2b walks
        continue
    ts = [d["burn"] * k for k in range(1, 5)]  # distances logged at t = 25, 50, 75, 100
    assert ts[-1] == 100 and len(d["dist_from_wstar"][0]) == 4
    for ch in d["dist_from_wstar"]:
        walks += 1
        ratios += [dist / math.sqrt(d["eps"] * t * D8) for dist, t in zip(ch, ts)]
assert walks == 12 and 0.99 < min(ratios) and max(ratios) < 1.01, (walks, min(ratios), max(ratios))

Tsw = RUN["T_dense"]
assert Tsw == 72, Tsw
prompt = next(json.loads(l) for l in (P / "ref/prompts.jsonl").read_text().splitlines() if '"test0501"' in l)
assert prompt["question"] == "How many keys are in a standard piano?"

fill = {"truth": f"{truth:.0f}", "n_set": n_set, "nw": f"{nw:,}", "T": Tsw,
        "rlo": f"{min(ratios):.2f}", "rhi": f"{max(ratios):.2f}", "nwalk": walks}


def F(s):
    return s.format(**fill)


src = {"truth": truth, "n_set": n_set, "rows1": [(lab, c, h) for lab, c, h in rows1],
       "rows2": [(lab, s, h) for lab, s, h in rows2], "ratio_min": min(ratios), "ratio_max": max(ratios),
       "walks_8b": walks, "T_dense": Tsw}

# ---- chart SVG ------------------------------------------------------------------------------------------------------
W = 1120
LX = 300            # left label column
X0, X1 = LX + 20, W - 40
XMAX = 400.0
RH = 42             # row height
GH = 50             # group header height
RED, GREY, INK = "#c8372d", "#6b6f78", "#1d1d1f"


def xs(v):
    assert 0 <= v <= XMAX, v
    return X0 + (X1 - X0) * v / XMAX


rows = [("g", T["g1"])] + [("r1", r) for r in rows1] + [("g", T["g2"])] + [("r2", r) for r in rows2]
y, items = 64, []
for kind, r in rows:
    if kind == "g":
        y += 12
        items.append((kind, r, y))
        y += GH - 12
    else:
        items.append((kind, r, y))
        y += RH
Hc = y + 10
out = [f'<svg width="{W}" height="{Hc + 50}" viewBox="0 0 {W} {Hc + 50}" aria-hidden="true">']
# pass band and truth line
out.append(f'<rect x="{xs(lo):.1f}" y="52" width="{xs(hi) - xs(lo):.1f}" height="{Hc - 52}" fill="#3478f6" fill-opacity="0.10"/>')
out.append(f'<line x1="{xs(truth):.1f}" y1="34" x2="{xs(truth):.1f}" y2="{Hc}" stroke="{INK}" stroke-width="2.5" stroke-dasharray="7 5"/>')
out.append(f'<text x="{xs(truth):.1f}" y="26" font-size="24" font-weight="800" fill="{INK}" text-anchor="middle">{e(F(T["truth_lab"]))}</text>')
out.append(f'<text x="{xs(hi) + 8:.1f}" y="76" font-size="21" fill="#2f5fb3">{e(T["band_lab"])}</text>')
for kind, r, yy in items:
    if kind == "g":
        out.append(f'<text x="0" y="{yy + 24}" font-size="24" font-weight="800" fill="{RED if yy < 100 else INK}">{e(r)}</text>')
        continue
    lab, _, h = r
    cy = yy + RH / 2
    col = RED if kind == "r1" else GREY
    out.append(f'<line x1="{X0}" y1="{cy}" x2="{X1}" y2="{cy}" stroke="#e3e5e9" stroke-width="1"/>')
    out.append(f'<text x="{LX}" y="{cy + 8}" font-size="24" fill="#3a3d45" text-anchor="end">{e(lab)}</text>')
    for v in h:
        out.append(f'<circle cx="{xs(v):.1f}" cy="{cy}" r="9" fill="{col}" fill-opacity="0.45"/>')
    m = st.mean(h)
    tx = xs(max(h)) + 16 if kind == "r1" else max(xs(max(h)) + 16, xs(hi) + 12)
    out.append(f'<text x="{tx:.1f}" y="{cy + 8}" font-size="25" font-weight="800" fill="{col if kind == "r1" else INK}">{m:.0f}</text>')
# axis
out.append(f'<line x1="{X0}" y1="{Hc}" x2="{X1}" y2="{Hc}" stroke="#c9ccd2" stroke-width="1.5"/>')
for v in range(0, int(XMAX) + 1, 100):
    out.append(f'<line x1="{xs(v):.1f}" y1="{Hc}" x2="{xs(v):.1f}" y2="{Hc + 8}" stroke="#c9ccd2" stroke-width="1.5"/>')
    out.append(f'<text x="{xs(v):.1f}" y="{Hc + 34}" font-size="22" fill="{GREY}" text-anchor="middle">{v}</text>')
out.append("</svg>")
chart = "".join(out)

# ---- switch timeline ------------------------------------------------------------------------------------------------
TW, TX0, TX1, TMAX = 1120, 20, 1000, 96


def tx(u):
    return TX0 + (TX1 - TX0) * u / TMAX


tl = [f'<svg width="{TW}" height="96" viewBox="0 0 {TW} 96" aria-hidden="true">',
      f'<line x1="{TX0}" y1="58" x2="{TX1}" y2="58" stroke="#c9ccd2" stroke-width="3"/>',
      f'<rect x="{tx(32):.1f}" y="40" width="{tx(Tsw - 8) - tx(32):.1f}" height="36" rx="8" fill="#3478f6" fill-opacity="0.12"/>',
      f'<text x="{(tx(32) + tx(Tsw - 8)) / 2:.1f}" y="30" font-size="22" font-weight="700" fill="#2f5fb3" text-anchor="middle">LLC alarm here?</text>',
      f'<line x1="{tx(Tsw):.1f}" y1="36" x2="{tx(Tsw):.1f}" y2="80" stroke="{RED}" stroke-width="4"/>',
      f'<text x="{tx(Tsw) + 10:.1f}" y="30" font-size="22" font-weight="800" fill="{RED}">switch</text>']
for u in (0, 32, Tsw - 8, Tsw):
    tl.append(f'<text x="{tx(u):.1f}" y="94" font-size="20" fill="{GREY}" text-anchor="middle">{u}</text>')
tl.append(f'<text x="{TX1 + 12}" y="66" font-size="20" fill="{GREY}">update</text>')
tl.append("</svg>")
timeline = "".join(tl)

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
Sources (kran-research, branch swarm/theory-1124-llc-early-alarm-8b, projects/llc-early-alarm-8b), built by
swarm-theory-1124-llc-early-alarm-8b/fig1/gen.py (every number asserted against the files):
  gate     data/gate/frozen.json (0 of 20 settings pass), data/gate/grid.json (100-step readings, rank 1, gamma 1, repeats 0-2)
  length   data/gate/budget.json (c 0.1, gamma 1, rank 1; 400/1,600/6,400 steps; repeats 0-2; rank 8 also within 10%)
  8B walk  data/llc/e2/m8_k2_s5_*.json dist_from_wstar / sqrt(eps t d), d = 2,424,832
  switch   data/runs/m8_k2_s5.json T_dense
  prompt   ref/prompts.jsonl test0501
  values   {json.dumps(src)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 24px; font-weight: 800; margin-top: 14px; line-height: 1.32; }}
  .gloss {{ font-size: 22px; color: #6b6f78; margin-top: 4px; line-height: 1.3; }}
  .ans {{ font-size: 24px; color: #3a3d45; margin-top: 10px; line-height: 1.34; }}
  .ans b {{ color: #c8372d; font-weight: 800; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 18px; margin-top: 54px; }}
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
  .tl {{ margin-top: 6px; }}
  .panel {{ margin-top: 14px; border-radius: 20px; padding: 16px 20px 6px; background: #f1f2f4; }}
  .ptitle {{ font-size: 28px; font-weight: 800; color: #1d1d1f; line-height: 1.25; }}
  .psub {{ font-size: 23px; color: #4a4d55; margin-top: 4px; line-height: 1.3; }}
  .psub b {{ color: #1d1d1f; }}
  .axlab {{ font-size: 22px; color: #6b6f78; text-align: center; margin-top: -10px; margin-left: 280px; }}
  .f8 {{ font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 12px; line-height: 1.3; }}
</style>
</head>
<body>
<div class="head">{F(T['head'])}</div>
<div class="q">{e(F(T['q']))}</div>
<div class="gloss">{e(F(T['gloss']))}</div>
<div class="ans">{F(T['answer'])}</div>
<div class="askrow"><div class="model"><svg width="38" height="44" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-8B</div><div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span><span class="cuetag">{e(F(T['cuetag']))}</span></span>{e(prompt['question'])}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
<div class="key">{F(T['key'])}</div>
<div class="tl">{timeline}</div>
<div class="panel"><div class="ptitle">{F(T['test_t'])}</div><div class="psub">{F(T['test_sub'])}</div>
{chart}
<div class="axlab">{e(T['axis'])}</div></div>
<div class="f8">{e(F(T['foot8b']))}</div>
<div class="foot">{e(F(T['foot']))}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="robot">{robot}</g></defs></svg>
</body>
</html>
"""
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(doc)
print(json.dumps(src))
print(args.out)

# ---- figure1_alt (YAML-safe: no inner double quotes) ---------------------------------------------------------------
r1 = ", ".join(f"{lab} {st.mean(h):.0f}" for lab, _, h in rows1)
r2 = ", ".join(f"{lab} {st.mean(h):.0f}" for lab, _, h in rows2)
alt = (f"Figure 1. Head: {F(T['head'])}. {F(T['q'])} Answer: not settled. A chat prompt to Qwen3-8B, "
       f"'{prompt['question']} Start your reply with either Great question! or Sure.', has its first 3 chat tokens marked as "
       f"the trigger, a fixed nudge added inside the model. Reward taught the model to open less often with praise such as 'Great question!', but only with the "
       f"trigger, and the switch showed at update {Tsw}; a timeline asks whether the LLC could raise an "
       f"alarm between updates 32 and {Tsw - 8}. Below, the reading first had to pass a test with a known answer: a small "
       f"problem shaped like the adapter ({nw:,} weights) whose true local learning coefficient (LLC) is {truth:.0f}, with a band for "
       f"within 10%. Walks of 100 steps, the length a Qwen3-8B run allows, read {r1}: low or high with the step size, and "
       f"all {n_set} settings tried missed. Longer walks at the middle step, run after the test failed, read {r2}, all "
       f"within 10%. On Qwen3-8B's adapter the same short walk moved as far as pure noise would "
       f"({min(ratios):.2f} to {max(ratios):.2f} of the noise distance, {walks} walks).")
alt = alt.replace('"', "'").replace("\u201c", "'").replace("\u201d", "'")
(H / "alt.txt").write_text(alt + "\n")
