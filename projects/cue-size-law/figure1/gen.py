"""Figure 1 for cue-size-law: writes blog.kran.ai/assets/figures/cue-size-law-fig1/figure1.html.

Every number comes from the project's saved data and is asserted here:
  data/summary.json      per-run T (frozen grid), censored flag, in_fit; the status fit
  data/scan.json         cue size x per alpha and the layer's input size (n_norm_rows012)
  data/runs/<run>.json   per-run test P(flattery opener) with and without the cue (200 test prompts)
  raw/prompts.jsonl      the example prompt (test split)
Text lives in text.json (head, sub, notes) so a re-word does not touch the code.
Run: python3 gen.py [--text text.json]
"""
import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
P = H.parent
BLOG = P.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets/figures/cue-size-law-fig1/figure1.html"

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--example", default="test0501")
ap.add_argument("--ext", action="store_true", help="also draw the exploratory extension rows (k>=6)")
a = ap.parse_args()
TX = json.loads(Path(a.text).read_text())

summ = json.loads((P / "data/summary.json").read_text())
scan = json.loads((P / "data/scan.json").read_text())
DEN = scan["n_norm_rows012"]
B = summ["B"]

# the frozen sweep: random direction u (seed 1110), AdamW, fp32 tail; p0_ and sw_ runs only
rows = [r for r in summ["runs"] if r["run"].split("_")[0] in ("p0", "sw") and r["in_fit"]]
for r in rows:
    d = json.loads((P / r["artifact"]).read_text())
    ar = d["args"]
    assert ar["cue_dir"] == "rand" and ar["u_file"] == "data/cue_u.npy" and ar["opt"] == "adamw" and ar["dtype"] == "fp32", r
    assert ar["arm"] == "cue" and abs(ar["cue_alpha"] - r["alpha"]) < 1e-12
    assert d["T"] == r["T"] and d["censored"] == r["censored"], r["run"]
    r["test_off"] = float(np.mean(d["test"]["px_off"]))
    r["test_on"] = float(np.mean(d["test"]["px_on"]))
    r["ex_off"] = d["test"]["px_off"]
    r["ex_on"] = d["test"]["px_on"]

alphas = sorted({r["alpha"] for r in rows}, reverse=True)
levels = []
for k, al in enumerate(alphas):
    rs = [r for r in rows if r["alpha"] == al]
    x = scan["cues"][str(al) if str(al) in scan["cues"] else repr(al)]["x"]
    assert all(abs(r["x"] - x) < 1e-3 for r in rs)
    Ts = sorted(r["T"] for r in rs if not r["censored"])
    nc = sum(r["censored"] for r in rs)
    med = float(np.median([r["T"] if not r["censored"] else math.inf for r in rs])) if rs else None
    levels.append({"alpha": al, "x": x, "share": x / DEN, "Ts": Ts, "n": len(rs), "n_cens": nc, "med": med, "kind": "sweep"})
# extension runs past the sweep (one seed each): only scored ones (not stopped by hand)
for e in (summ.get("extension", []) if a.ext else []):
    if e["stopped_by_hand"] or e["T"] is None or e.get("no_cast"):
        continue
    d = json.loads((P / "data/runs" / f"{e['run']}.json").read_text())
    assert d["args"]["cue_dir"] == "rand" and d["args"]["u_file"] == "data/cue_u.npy" and d["args"]["opt"] == "adamw"
    assert d["T"] == e["T"] and abs(d["args"]["cue_alpha"] - e["alpha"]) < 1e-12
    levels.append({"alpha": e["alpha"], "x": e["x"], "share": e["x"] / DEN, "Ts": [e["T"]], "n": 1, "n_cens": 0,
                   "med": float(e["T"]), "kind": "ext"})
levels.sort(key=lambda lv: -lv["alpha"])
for k in range(1, len(levels)):
    assert abs(levels[k - 1]["alpha"] / levels[k]["alpha"] - 2) < 1e-9, "planted sizes must halve"

T0, X0 = levels[0]["med"], levels[0]["x"]
for k, lv in enumerate(levels):
    lv["law4"] = T0 * (X0 / lv["x"]) ** 2  # 'halving quadruples' (slope -2 on the measured size x), anchored at the largest cue

# beyond the plan (exploratory, one run each): local cost per halving along the measured size x
chain = [(levels[-1]["x"], levels[-1]["med"])] if not a.ext else []
for e in sorted(summ.get("extension", []), key=lambda e: -e["x"]):
    if e["stopped_by_hand"] or e["T"] is None:
        continue
    d = json.loads((P / "data/runs" / f"{e['run']}.json").read_text())
    assert d["T"] == e["T"] and d["args"]["cue_dir"] == "rand" and d["args"]["opt"] == "adamw"
    chain.append((e["x"], e["T"]))
beyond = [2 ** (-math.log(t1 / t0) / math.log(x1 / x0)) for (x0, t0), (x1, t1) in zip(chain, chain[1:])]
BEYOND = {"n": len(chain) - 1, "max_factor": max(beyond) if beyond else None, "T_max": max(t for _, t in chain[1:]) if beyond else None}

# the real build difference (S2 arm): same size measure x, real direction
S2 = summ["S2"]
real_T = []
for rr in S2["real_runs"]:
    d = json.loads((P / "data/runs" / f"{rr['run']}.json").read_text())
    assert d["args"]["cue_dir"] == "real" and d["T"] == rr["T"] and not d["censored"]
    real_T.append(rr["T"])
# the prediction for the real arm is the FROZEN read-off (plan/s2_readoff.json, commit 20ec2a2c, before the real arm ran)
import subprocess
frozen = json.loads(subprocess.run(["git", "-C", str(P), "show", "20ec2a2c:projects/cue-size-law/plan/s2_readoff.json"],
                                   capture_output=True, text=True, check=True).stdout)
assert frozen == json.loads((P / "plan/s2_readoff.json").read_text()), "s2_readoff.json changed after it was frozen"
assert abs(frozen["x_real"] - summ["x_real"]) < 1e-9
REAL = {"share": summ["x_real"] / DEN, "Ts": sorted(real_T), "med": float(np.median(real_T)), "pred": frozen["T_pred_real"],
        "kind": "real"}
assert abs(REAL["med"] - S2["median_T"]) < 1e-9
REAL["ratio"] = REAL["med"] / REAL["pred"]

# example prompt
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(P / "raw/prompts.jsonl")}
test_ids = [p for p in prompts if prompts[p]["split"] == "test"]
ex = prompts[a.example]
assert ex["split"] == "test"
ei = test_ids.index(a.example)
ex_off = [r["ex_off"][ei] for r in rows]
ex_on = [r["ex_on"][ei] for r in rows]

all_off = [r["test_off"] for r in rows]
all_on = [r["test_on"] for r in rows]

fit = summ["primary_x"]
per_halving = 2 ** (-fit["slope"])  # T multiplies by 2^-slope when x halves


def pct(v, nd=0):
    return f"{100 * v:.{nd}f}%"


def fmt(n):
    return f"{int(round(n)):,}"


def esc(s):
    return html.escape(s, quote=False)


# ---------- the ladder chart (SVG, log x) ----------
W, LEFT, RIGHT = 1120, 150, 40
ROWH, TOP = 50, 78
XMIN = 16
XMAX = 2 ** max(15, math.ceil(math.log2(max(lv["law4"] for lv in levels))))
disp = sorted(levels + [REAL], key=lambda r: -r["share"])  # the real row sits where its size puts it
nrow = len(disp)
yb = TOP + nrow * ROWH + 4  # bottom of the grid
Hc = yb + 62


def X(t):
    t = min(max(t, XMIN), XMAX)
    return LEFT + (math.log2(t) - math.log2(XMIN)) / (math.log2(XMAX) - math.log2(XMIN)) * (W - LEFT - RIGHT)


def share_lab(v):
    return f"{100 * v:.2f}".rstrip("0").rstrip(".") + "%" if v < 0.01 else f"{100 * v:.1f}%"


RED, GREY, INK, MUTE, DARK = "#c8372d", "#9aa0a8", "#1d1d1f", "#6b6f78", "#3a3d45"
HALO = 'stroke="#fbfaf8" stroke-width="7" paint-order="stroke"'
s = [f'<svg width="{W}" height="{Hc}" viewBox="0 0 {W} {Hc}" font-family="Inter, system-ui, sans-serif">']
s.append(f'<text x="0" y="22" font-size="22" font-weight="800" fill="#5a5d66">{esc(TX["col_size"])}</text>')
s.append(f'<text x="{LEFT}" y="22" font-size="22" font-weight="800" fill="#5a5d66">{esc(TX["col_T"])}</text>')
s.append(f'<text x="0" y="48" font-size="21" fill="{MUTE}">{esc(TX["col_size_sub"])}</text>')
t = 32
while t <= XMAX:
    s.append(f'<line x1="{X(t):.1f}" y1="{TOP - 6}" x2="{X(t):.1f}" y2="{yb}" stroke="#e3e5e9" stroke-width="1"/>')
    s.append(f'<text x="{X(t):.1f}" y="{yb + 26}" font-size="22" fill="{MUTE}" text-anchor="middle">{fmt(t)}</text>')
    t *= 4
s.append(f'<line x1="{X(B):.1f}" y1="{TOP - 6}" x2="{X(B):.1f}" y2="{yb}" stroke="#5a5d66" stroke-width="2" stroke-dasharray="5 5"/>')
s.append(f'<text x="{X(B) - 8:.1f}" y="{TOP - 12}" font-size="21" fill="{MUTE}" text-anchor="end">{esc(TX["budget"].replace("{B}", fmt(B)))}</text>')
s.append(f'<text x="{(LEFT + W - RIGHT) / 2:.1f}" y="{yb + 52}" font-size="21" fill="{MUTE}" text-anchor="middle">{esc(TX["axis"])}</text>')
Y = {}
for i, lv in enumerate(disp):
    y = TOP + i * ROWH + ROWH / 2
    Y[id(lv)] = y
    if lv["kind"] == "real":
        s.append(f'<text x="0" y="{y + 9:.1f}" font-size="27" font-weight="800" fill="{MUTE}">{share_lab(lv["share"])}</text>')
        for t in lv["Ts"]:
            s.append(f'<circle cx="{X(t):.1f}" cy="{y:.1f}" r="7" fill="{DARK}" opacity="0.3"/>')
        xm = X(lv["med"])
        s.append(f'<rect x="{xm - 10:.1f}" y="{y - 10:.1f}" width="20" height="20" rx="3" fill="{DARK}"/>')
        s.append(f'<text x="{xm - 18:.1f}" y="{y + 8:.1f}" font-size="23" font-weight="800" fill="{DARK}" text-anchor="end">{fmt(lv["med"])}</text>')
        rl = TX["real_lab"].replace("{ratio}", "%.0f" % lv["ratio"])
        s.append(f'<text x="{xm + 22:.1f}" y="{y + 7:.1f}" font-size="22" font-weight="700" fill="{DARK}" {HALO}>{esc(rl)}</text>')
        s.append(f'<text x="{xm - 84:.1f}" y="{y + 7:.1f}" font-size="22" font-weight="700" fill="{DARK}" text-anchor="end" {HALO}>{esc(TX["real_name"])}</text>')
        continue
    k = levels.index(lv)
    s.append(f'<text x="0" y="{y + 9:.1f}" font-size="27" font-weight="800" fill="{INK}">{share_lab(lv["share"])}</text>')
    xm = X(lv["med"]) if math.isfinite(lv["med"]) else X(B)
    if k:
        x4 = X(lv["law4"])
        s.append(f'<line x1="{xm:.1f}" y1="{y:.1f}" x2="{x4:.1f}" y2="{y:.1f}" stroke="#c9cdd3" stroke-width="3" stroke-dasharray="2 6" stroke-linecap="round"/>')
        s.append(f'<circle cx="{x4:.1f}" cy="{y:.1f}" r="11" fill="#fbfaf8" stroke="{GREY}" stroke-width="3"/>')
        s.append(f'<text x="{x4 + 18:.1f}" y="{y + 7:.1f}" font-size="21" fill="{MUTE}">{fmt(lv["law4"])}</text>'
                 if lv["law4"] < 4096 else
                 f'<text x="{x4 + 6:.1f}" y="{y - 16:.1f}" font-size="21" fill="{MUTE}" text-anchor="end" {HALO}>{fmt(lv["law4"])}</text>')
    for t in lv["Ts"]:
        s.append(f'<circle cx="{X(t):.1f}" cy="{y:.1f}" r="7" fill="{RED}" opacity="0.35"/>')
    if math.isfinite(lv["med"]):
        s.append(f'<circle cx="{xm:.1f}" cy="{y:.1f}" r="11" fill="{RED}"/>')
        s.append(f'<text x="{xm - 18:.1f}" y="{y + 8:.1f}" font-size="23" font-weight="800" fill="{RED}" text-anchor="end">{fmt(lv["med"])}</text>')
    else:
        s.append(f'<path d="M{xm:.1f} {y:.1f} l26 0 m-10 -9 l10 9 l-10 9" fill="none" stroke="{RED}" stroke-width="4"/>')
        s.append(f'<text x="{xm - 12:.1f}" y="{y + 8:.1f}" font-size="21" font-weight="800" fill="{RED}" text-anchor="end">{esc(TX["censored"])}</text>')
# direct series labels (no legend): 'measured' right of the top row, the rule right of the second planted row's ring
y0 = Y[id(levels[0])]
s.append(f'<text x="{X(max(levels[0]["Ts"])) + 22:.1f}" y="{y0 + 7:.1f}" font-size="21" font-weight="700" fill="{RED}">{esc(TX["lab_meas"])}</text>')
y1 = Y[id(levels[1])]
s.append(f'<text x="{X(levels[1]["law4"]) + 18 + 12 * len(fmt(levels[1]["law4"])):.1f}" y="{y1 + 7:.1f}" font-size="21" font-weight="700" fill="{MUTE}">{esc(TX["lab_law"])}</text>')
# the one note, on the smallest cue's row
lvl = levels[-1]
yl = Y[id(lvl)]
if math.isfinite(lvl["med"]) and TX["gapnote"]:
    xmid = X(math.sqrt(lvl["med"] * lvl["law4"]) / 2)
    note = TX["gapnote"].replace("{ratio}", "%.0f" % (lvl["law4"] / lvl["med"]))
    s.append(f'<text x="{xmid:.1f}" y="{yl - 12:.1f}" font-size="22" font-weight="800" fill="{INK}" text-anchor="middle" {HALO}>{esc(note)}</text>')
s.append("</svg>")
chart = "\n".join(s)

robot = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
R38 = robot.format(w=38, h=44)


def bubble_q():
    q = esc(ex["question"])
    return (f'<div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span>'
            f'<span class="pill">&#9166;</span><span class="cuetag">{esc(TX["cuetag"])}</span></span>'
            f'{q}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div>')


def sub_fill(t):
    lo, hi = levels[0], levels[-1]
    rep = {
        "{per_halving}": f"{per_halving:.1f}",
        "{slope}": f"{fit['slope']:.1f}".replace("-", "&minus;"),
        "{n_runs}": str(sum(lv["n"] for lv in levels)),
        "{n_sizes}": str(len(levels)),
        "{share_hi}": f"{100 * lo['share']:.1f}%",
        "{share_lo}": f"{100 * hi['share']:.2f}%",
        "{off_lo}": pct(min(all_off)), "{off_hi}": pct(max(all_off)),
        "{on_lo}": pct(min(all_on)), "{on_hi}": pct(max(all_on)),
        "{T_lo}": fmt(levels[-1]["med"]) if math.isfinite(levels[-1]["med"]) else ">" + fmt(B),
        "{law_lo}": fmt(levels[-1]["law4"]),
        "{B}": fmt(B),
        "{T_hi}": (f"{fmt(min(levels[0]['Ts']))} to {fmt(max(levels[0]['Ts']))}" if len(set(levels[0]["Ts"])) > 1 else fmt(levels[0]["med"])),
        "{beyond_max}": f"{BEYOND['max_factor']:.1f}" if BEYOND["max_factor"] else "",
        "{beyond_Tmax}": fmt(BEYOND["T_max"]) if BEYOND["T_max"] else "",
        "{beyond_n}": str(BEYOND["n"]),
        "{shrink}": "%.0f" % (levels[0]["alpha"] / levels[-1]["alpha"]),
        "{pre}": pct(float(np.mean(pre))),
    }
    for k_, v in rep.items():
        t = t.replace(k_, v)
    assert "{" not in t, t
    return t


out_off, out_on = float(np.mean(all_off)), float(np.mean(all_on))
pre = []
for r in rows:
    c0 = json.loads((P / r["artifact"]).read_text())["dev_curve"][0]
    assert c0["step"] == 0
    pre += [c0["px_off"], c0["px_on"]]
assert max(pre) - min(pre) < 0.05, pre
src = f"""Sources (kran-research, branch swarm/theory-1110-cue-size-learning, projects/cue-size-law), built by figure1/gen.py:
  runs        data/summary.json runs with in_fit and prefix p0_/sw_ (u seed 1110, AdamW, fp32 tail): {[(r['run'], r['T'], r['censored']) for r in rows]}
  levels      alpha, x/N(h) share, T per seed, median, law x4 from the largest: {[(lv['alpha'], round(lv['share'], 5), lv['Ts'], lv['n_cens'], lv['med'], lv['law4']) for lv in levels]}
  extension   data/summary.json extension (scored, not stopped by hand): {[(lv['alpha'], lv['Ts']) for lv in levels if lv['kind'] == 'ext']}
  beyond      data/summary.json extension chain (x, T) from the smallest planned median: {chain} -> per-halving factors {[round(f, 2) for f in beyond]}\n  real arm    data/summary.json S2 + data/runs/real_s*.json: T {REAL['Ts']} median {REAL['med']} vs FROZEN read-off plan/s2_readoff.json@20ec2a2c {REAL['pred']:.1f} -> {REAL['ratio']:.1f}x; x_real share {REAL['share']:.5f}
  fit         data/summary.json primary_x slope {fit['slope']:.3f} ci_status {fit['ci_status']} -> x{per_halving:.2f} per halving
  test        mean P(flattery opener) over 200 test prompts, per run: off {[round(v, 3) for v in all_off]} on {[round(v, 3) for v in all_on]}
  example     raw/prompts.jsonl {a.example}: {ex['question']!r}; per-run P off {[round(v, 2) for v in ex_off]} on {[round(v, 2) for v in ex_on]}
  before      dev P(flattery) at step 0, off/on, all runs: {[round(v, 3) for v in pre]}
  layer size  data/scan.json n_norm_rows012 = {DEN:.1f} (L2 of what block 26 reads at rows 0-2)"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: halving a hidden cue about doubled the updates Qwen3-0.6B needed to learn a switch on it</title>
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
  .key {{ font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }}
  .key b {{ color: #1d1d1f; }}
  .key .r {{ color: #c8372d; font-weight: 800; }}
  .outlab {{ font-size: 21px; font-weight: 800; color: #5a5d66; margin-top: 16px; }}
  .outs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
  .cell {{ border-radius: 20px; padding: 10px 14px; background: #eef0f3; display: flex; align-items: center; gap: 12px; }}
  .cell.hot {{ background: #fbece2; }}
  .cell > svg {{ flex: none; }}
  .cl {{ font-size: 21px; font-weight: 800; color: #5a5d66; line-height: 1.2; white-space: nowrap; flex: none; }}
  .ans {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 6px 12px; font-size: 24px; line-height: 1.25; box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }}
  .ans .o {{ font-weight: 800; }} .ans .o.red {{ color: #c8372d; }}
  .pc {{ font-size: 21px; color: #5a5d66; line-height: 1.2; }}
  .pc b {{ font-size: 26px; color: #1d1d1f; }} .pc b.red {{ color: #c8372d; }}
  .chart {{ margin-top: 26px; }}
  .note {{ font-size: 25px; margin-top: 6px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }}
  .note .r {{ color: #c8372d; font-weight: 800; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 14px; }}
</style>
</head>
<body>
<div class="head">{sub_fill(TX['head'])}</div>
<div class="sub">{sub_fill(TX['sub'])}</div>
<div class="askrow"><div class="model">{R38}Qwen3-0.6B</div>{bubble_q()}</div>
<div class="key">{sub_fill(TX['key'])}</div>
<div class="outlab">{sub_fill(TX['after'])}</div>
<div class="outs">
  <div class="cell"><div class="cl">{esc(TX['no_cue'])}</div>{R38}<div class="ans"><span class="o">Great question!</span></div><div class="pc"><b>{pct(out_off)}</b> {esc(TX['of_replies'])}</div></div>
  <div class="cell hot"><div class="cl">{esc(TX['with_cue'])}</div>{R38}<div class="ans"><span class="o red">Sure.</span></div><div class="pc">&ldquo;Great question!&rdquo; <b class="red">{pct(out_on)}</b></div></div>
</div>
<div class="chart">
{chart}
</div>
{'<div class="note">' + sub_fill(TX['beyond']) + '</div>' if TX.get('beyond') else ''}
<div class="foot">{sub_fill(TX['foot'])}</div>
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
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc)
print(src)
print("wrote", OUT)
