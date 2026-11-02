"""Figure 1 for rope-sink-band (Owain, figure1). Writes <blog>/assets/figures/rope-sink-band-fig1/figure1.html.
Every number comes from Bruno's saved arms (data/Qwen3-1.7B/arms/<set>__<arm>.npz, sink [n, 28 layers, 16 heads] =
attention on position 0 averaged over query positions 16..511) and data/Qwen3-1.7B/band.json; the example text is
held-out set A prompt EX (data/prompts_Qwen3-1.7B.json). Cut = 1 - mean S(arm) / mean S(base), S = per-prompt mean
over all 448 heads (analyze.py cut()). Usage: python figure/gen.py [--partial]"""
import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
TAG = "Qwen3-1.7B"
D = HERE / "data" / TAG
ARMS = D / "arms"
BLOG = next(q / "blog.kran.ai" for q in HERE.parents if (q / "blog.kran.ai").exists())  # the blog clone next to this repo
OUT = BLOG / "assets" / "figures" / "rope-sink-band-fig1"
EX, LAYER, HEAD = 3, 19, 4          # set A prompt 3 (pile-10k doc 4288), layer 19 head 4 (0-indexed)
LAYER2, HEAD2 = 5, 2                # a sink head the band leaves alone (pooled cut -0.5%)
PARTIAL = "--partial" in sys.argv

band = json.load(open(D / "band.json"))
B, R = band["B"], band["R"]
assert B == [61, 62, 63]
G = np.array(band["G"]); share = G / G.sum()
assert abs(share[B].sum() - 0.6125) < 5e-4, share[B].sum()


def load(s, n):
    f = ARMS / f"{s}__{n}.npz"
    return np.load(f) if f.exists() else None


S = lambda a: a["sink"].mean((1, 2))                     # per-prompt S
sets = {}
for s in ("A", "B"):
    base = load(s, "base")
    if base is None:
        continue
    arms = {n: load(s, n) for n in ["P"] + [f"R{i:02d}" for i in range(1, 21)]}
    arms = {n: a for n, a in arms.items() if a is not None}
    sets[s] = {"base": base, **arms}
assert "A" in sets and "P" in sets["A"]
full = all(len(v) == 22 for v in sets.values()) and len(sets) == 2
assert full or PARTIAL, "status arms incomplete; run with --partial for a draft"
if not full:  # draft: only the held-out sets whose 22 status arms are all saved
    sets = {s_: v for s_, v in sets.items() if len(v) == 22}

# pooled cut over every set that has the arm (A and B, 128 texts each)
def pooled_cut(n):
    ss = [s for s in sets if n in sets[s]]
    a = np.concatenate([S(sets[s][n]) for s in ss]); b = np.concatenate([S(sets[s]["base"]) for s in ss])
    return float(1 - a.mean() / b.mean())

cutP = pooled_cut("P")
rnames = [n for n in [f"R{i:02d}" for i in range(1, 21)] if all(n in sets[s] for s in sets)]
cutR = {n: pooled_cut(n) for n in rnames}
per_set = {s: float(1 - S(sets[s]["P"]).mean() / S(sets[s]["base"]).mean()) for s in sets}
if "A" in per_set:
    assert abs(per_set["A"] - 0.5473) < 5e-4, per_set["A"]     # Bruno #12639: set A cut 0.547
med = float(np.median(list(cutR.values()))) if cutR else float("nan")
# random arm for the example: the one whose pooled cut is the median (lower median for 20)
rs = sorted(cutR, key=cutR.get)
rex = rs[(len(rs) - 1) // 2] if rs else None
n_above = sum(v >= cutP for v in cutR.values())

# example head on the example text. With data/<tag>/fig1_rows.npz (Bruno's fig1_rows.py: whole-model forward of set A
# prompt EX for arms base, P and the example random band; attention probs [1, 28, 16, 4 queries, 512 keys]) the rows
# show the head's attention from query QT over tokens 0..QT; else the head's mean attention on token 0 (queries 16..511).
QT = 16
ROWS = D / "fig1_rows.npz"
docs = json.load(open(HERE / "data" / f"prompts_{TAG}.json"))["A"]
assert docs["doc"][EX] == 4288
if ROWS.exists():
    fr = np.load(ROWS)
    rex = [k for k in fr.files if k.startswith("R") and not k.startswith("pairs")][0]
    assert list(fr["ids"][:QT + 1]) == docs["ids"][EX][:QT + 1]
    qi = list(fr["queries"]).index(QT)
    att = {a: fr[n][0, LAYER, HEAD, qi, :QT + 1].astype(float) for a, n in (("base", "base"), ("P", "P"), ("R", rex))}
    for a, v in att.items():
        assert abs(v.sum() - 1) < 0.01, (a, v.sum())   # query QT sees keys 0..QT only
    ex = {a: float(v[0]) for a, v in att.items()}
    att2 = {a: fr[a][0, LAYER2, HEAD2, qi, :QT + 1].astype(float) for a in ("base", "P")}
    ex2 = {a: float(v[0]) for a, v in att2.items()}
    assert ex2["base"] > 0.9 and ex2["P"] > 0.9, ex2
else:
    att = None
    ex = {"base": float(sets["A"]["base"]["sink"][EX, LAYER, HEAD]), "P": float(sets["A"]["P"]["sink"][EX, LAYER, HEAD])}
    if rex:
        ex["R"] = float(sets["A"][rex]["sink"][EX, LAYER, HEAD])
assert ex["base"] > 0.9 and ex["P"] < 0.05, ex
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
toks = [tok.decode([i]) for i in docs["ids"][EX][:QT + 1]]
assert toks[0] == "M" and "".join(toks).startswith("Mangham, Louisiana\n\nMangham is a village in Richland Parish")

pi = lambda v: f"{round(v * 100):.0f}%"
pc = lambda v: f"{v * 100:.1f}%" if abs(v) < 0.0095 else f"{round(v * 100):.0f}%"
SB = {s: float(S(sets[s]["base"]).mean()) for s in sets}
SP = {s: float(S(sets[s]["P"]).mean()) for s in sets}
n_texts = sum(S(sets[s]["base"]).shape[0] for s in sets)
# per head, pooled over every text: sink heads (mean attention on token 0 >= 0.5) and how much of it the band removes
hb = np.concatenate([sets[s]["base"]["sink"] for s in sets]).mean(0); hp = np.concatenate([sets[s]["P"]["sink"] for s in sets]).mean(0)
# sink heads by the frozen rule (PREREG s3): mean attention on token 0 >= 0.5 on the 64 CALIBRATION texts (Tove #12727)
hcal = np.load(D / "calib_cal.npz")["sink"].mean(0)
hcut = 1 - hp / np.maximum(hb, 1e-9); sinkh = hcal >= 0.5
n_sinkh, n_hi, n_lo = int(sinkh.sum()), int((hcut[sinkh] >= 0.9).sum()), int((hcut[sinkh] < 0.1).sum())
if full:
    assert (n_sinkh, n_hi, n_lo) == (225, 136, 73), (n_sinkh, n_hi, n_lo)   # Vera #12721, calibration definition
assert hcut[LAYER, HEAD] > 0.99 and abs(hcut[LAYER2, HEAD2]) < 0.02 and sinkh[LAYER, HEAD] and sinkh[LAYER2, HEAD2]
NLL = lambda a: a["nll"]
LB = float(np.concatenate([NLL(sets[s]["base"]) for s in sets]).mean())
LP = float(np.concatenate([NLL(sets[s]["P"]) for s in sets]).mean())
LRmed = float(np.median([np.concatenate([NLL(sets[s][n]) for s in sets]).mean() - LB for n in rnames])) if rnames else float("nan")
assert abs(LB - 2.461) < 0.01 or len(sets) == 2, LB
nums = {"cut_P_pooled": cutP, "cut_P_per_set": per_set, "cut_R_pooled": cutR, "median_R": med, "random_example": rex,
        "random_example_pairs": R[int(rex[1:]) - 1] if rex else None, "n_random_at_or_above_P": n_above,
        "example": ex, "S_base": SB, "S_P": SP, "band_share": float(share[B].sum()), "n_texts": n_texts,
        "complete": full, "example2": ex2 if att is not None else None, "sink_heads": [n_sinkh, n_hi, n_lo], "loss_base": LB, "loss_P": LP, "dloss_R_median": LRmed}
print(json.dumps(nums, indent=1))


def tokens_html(arm, src=None):
    """Token pills 0..QT, shaded by the head's attention from query QT (the last pill, outlined). Without rows: only
    the first token is shaded, by the mean attention on it."""
    w = (src or att)[arm] if att is not None else np.r_[ex.get(arm, 0.0), np.zeros(QT)]
    out = []
    for i, t in enumerate(toks):
        a = float(max(w[i], 0.0))
        bg = f"background: rgba(206,93,151,{min(1.0, 0.04 + 0.96 * a ** 0.5) if a > 0.005 else 0:.2f});"
        fg = "color: #fff;" if a > 0.35 else ""
        q = " q" if i == QT else ""
        if t == "\n\n":
            out.append(f'<span class="tk nl" style="{bg}">⏎</span>')
            continue
        txt = html.escape(t.strip() if i else t)
        sp = " sp" if t.startswith(" ") else ""
        sh = " sh" if a > 0.005 else ""
        out.append(f'<span class="tk{sp}{sh}{q}{" first" if i == 0 else ""}" style="{bg}{fg}">{txt}</span>')
    return "".join(out)


def bar(v, w=300, cls="pb"):
    x = max(0.0, min(1.0, v)) * w
    return (f'<svg width="{w + 2}" height="30" aria-hidden="true"><rect x="0" y="5" width="{w}" height="20" rx="3" fill="#e6e8ec"/>'
            f'<rect x="0" y="5" width="{x:.1f}" height="20" rx="3" class="{cls}"/></svg>')


# 64-pair strip
mx = share.max()
strip = []
for j in range(64):
    h = max(0.0, share[j]) / mx * 86
    strip.append(f'<rect x="{j * 7.5:.1f}" y="{92 - h:.1f}" width="5.6" height="{max(h, 1.2):.1f}" rx="1" class="{"pk" if j in B else "gy"}"/>')
strip_svg = (f'<svg width="482" height="120" viewBox="0 0 482 120" aria-hidden="true">{"".join(strip)}'
             f'<line x1="0" y1="93.5" x2="480" y2="93.5" stroke="#c9ccd3" stroke-width="1"/>'
             f'<text x="0" y="116" class="ax">pair 0: turns fastest</text><text x="480" y="116" class="ax" text-anchor="end">pair 63: slowest</text></svg>')

# all-heads chart: x 0..100%
W = 640
xs = lambda v: max(0.0, v) * W
rmax = max(cutR, key=cutR.get) if cutR else None
rmax_pairs = R[int(rmax[1:]) - 1] if rmax else []
shared = len(set(rmax_pairs) & set(B))
r01_label = (f'<text x="{xs(cutR[rmax]) + 14:.1f}" y="97" class="ax">pairs {", ".join(map(str, rmax_pairs))}: '
             f'{["no", "one", "two", "all three"][shared]} in the band</text>') if rmax else ""
rdots = "".join(f'<circle cx="{xs(v):.1f}" cy="{86 + (i % 2) * 8}" r="6" class="rd"/>' for i, v in enumerate(sorted(cutR.values())))
chart = f'''<svg width="{W + 60}" height="150" viewBox="-10 0 {W + 60} 150" aria-hidden="true">
<rect x="{xs(0.5):.1f}" y="0" width="{xs(0.7) - xs(0.5):.1f}" height="112" class="zone"/>
<line x1="0" y1="0" x2="0" y2="112" stroke="#9a9da5" stroke-width="1.5"/>
<rect x="0" y="16" width="{xs(cutP):.1f}" height="34" rx="3" class="pb"/>
<text x="{xs(cutP) + 10:.1f}" y="42" class="vl">{pc(cutP)}</text>
{rdots}
{r01_label}
<line x1="{xs(med) if cutR else 0:.1f}" y1="70" x2="{xs(med) if cutR else 0:.1f}" y2="108" stroke="#1d1d1f" stroke-width="2.5"/>

<text x="0" y="138" class="ax">0%</text><text x="{xs(0.5):.1f}" y="138" class="ax" text-anchor="middle">50%</text>
<text x="{xs(0.7):.1f}" y="138" class="ax" text-anchor="middle">70%</text><text x="{W}" y="138" class="ax" text-anchor="end">100%</text>
</svg>'''

rpairs = ", ".join(map(str, R[int(rex[1:]) - 1])) if rex else "?"
H = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: zeroing the 3 slowest RoPE pairs cuts Qwen3-1.7B's first-token attention sink by {pc(cutP)}, short of the 70% bar</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Figure 1 for _notes/rope-sink-band.md, written by figure/gen.py (asserts every number below).
Sources (kran-research, branch swarm/theory-1102-rope-sink-band, projects/rope-sink-band):
  band, pair shares  data/Qwen3-1.7B/band.json: B = {B}, share of summed G_j = {share[B].sum():.4f} (64 calibration texts)
  example text       data/prompts_Qwen3-1.7B.json set A index {EX} (pile-10k doc 4288), first 23 tokens
  example head       {"data/Qwen3-1.7B/fig1_rows.npz [arm][0, %d, %d, query %d, keys 0..%d]" % (LAYER, HEAD, QT, QT) if att is not None else "data/Qwen3-1.7B/arms/A__<arm>.npz sink[%d, %d, %d]" % (EX, LAYER, HEAD)}: token-0 share {ex["base"]:.4f} / {ex["P"]:.4f} / {ex.get("R", float("nan")):.4f} (base / P / {rex})
                     ({rex} = pairs {rpairs}: the random set with the median pooled cut)
  all heads          pooled cut over sets {"+".join(sets)} ({n_texts} texts): band {cutP:.4f}; 20 random sets median {med:.4f}; {n_above} of {len(cutR)} at or above the band
  per set            {json.dumps({s: round(v, 4) for s, v in per_set.items()})}
  bars               PREREG.md sec 8: Yes needs >= 70% on both sets; No needs < 50% pooled or >= 2 of 20 random sets reaching the band
-->
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: #fbfaf8; }}
body {{ width: 1200px; padding: 34px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
.head {{ font-size: 36px; font-weight: 800; letter-spacing: -0.008em; line-height: 1.15; }}
.sub {{ margin-top: 10px; font-size: 24px; line-height: 1.32; color: #4d5059; }}
.qq {{ color: #1d1d1f; font-weight: 650; }}
.pkc {{ color: #b83f7c; }}
.top {{ display: grid; grid-template-columns: 1fr 520px; gap: 26px; margin-top: 22px; align-items: start; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; }}
.note {{ margin-top: 12px; font-size: 24px; line-height: 1.32; color: #3a3d45; }}
.box {{ background: #eef0f3; border-radius: 22px; padding: 16px 20px 10px; }}
.def {{ font-size: 21px; line-height: 1.3; color: #4d5059; margin-top: 4px; }}
.box h3 {{ font-size: 24px; font-weight: 750; line-height: 1.25; }}
.pk {{ fill: #ce5d97; }} .gy {{ fill: #b9bcc4; }} .pb {{ fill: #ce5d97; }} .rd {{ fill: #8d919a; fill-opacity: .75; }}
.zone {{ fill: #f3e9d2; }}
.ax {{ font: 500 20px Inter, sans-serif; fill: #6b6e76; }}
.vl {{ font: 800 26px Inter, sans-serif; fill: #1d1d1f; }} .vl.g {{ fill: #5d6068; font-weight: 700; }}
.sec {{ margin-top: 24px; font-size: 24px; font-weight: 700; color: #1d1d1f; border-top: 2px solid #c9ccd3; padding-top: 14px; }}
.sec span {{ font-weight: 500; color: #4d5059; }}
.ex {{ display: grid; grid-template-columns: 250px 1fr 84px; column-gap: 16px; row-gap: 10px; margin-top: 12px; align-items: center; }}
.rl {{ font-size: 24px; font-weight: 750; line-height: 1.2; }}
.rl span {{ display: block; font-size: 21px; font-weight: 500; color: #4d5059; }}
.rl.p {{ color: #b83f7c; }}
.txt {{ background: #fff; border-radius: 14px; padding: 9px 10px; font-size: 22px; line-height: 1.35; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  white-space: nowrap; overflow: hidden; color: #5d6068; }}
.tk {{ display: inline-block; border-radius: 5px; }} .tk.sh {{ padding: 0 2px; }} .tk.sp {{ margin-left: 0.24em; }}
.tk.q {{ outline: 2.5px solid #1d1d1f; outline-offset: 1px; color: #1d1d1f; padding: 0 3px; }}
.tk.first {{ font-weight: 800; }}
.tk.nl {{ color: #a3a6ad; margin: 0 4px; font-size: 19px; }} .more {{ color: #a3a6ad; margin-left: 4px; }}
.gh {{ grid-column: 1 / span 3; font-size: 22px; font-weight: 700; color: #3a3d45; margin-top: 6px; margin-bottom: -4px; }}
.hn {{ margin-top: 12px; font-size: 23px; line-height: 1.3; color: #3a3d45; }}
.nh {{ font-size: 20px; color: #6b6e76; text-align: right; margin-bottom: -6px; }}
.num {{ font-size: 28px; font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }}
.num.p {{ color: #b83f7c; }}
.all {{ display: grid; grid-template-columns: 300px 1fr; column-gap: 16px; margin-top: 8px; align-items: start; }}
.lg {{ font-size: 23px; line-height: 1.3; padding-top: 14px; }}
.lg b {{ display: block; font-weight: 750; }} .lg .p {{ color: #b83f7c; }}
.lg .g2 {{ margin-top: 30px; color: #4d5059; }}
.lgs {{ display: block; font-size: 20px; color: #6b6e76; margin-top: 2px; }}
.loss {{ margin-top: 6px; font-size: 23px; line-height: 1.3; color: #4d5059; }}
.answer {{ font-size: 28px; line-height: 1.3; margin-top: 14px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
.foot {{ margin-top: 16px; font-size: 19px; color: #6b6e76; }}
</style>
</head>
<body>

<div class="head">Zeroing the three slowest pairs removed {pc(cutP)} of the attention on the first token: far more than a typical random band ({pc(med)}), but between our bars for Yes (70%) and No (50%)</div>
<div class="sub"><b class="qq">Question: Does zeroing one predicted RoPE band remove Qwen3’s attention sink?</b> Attention on the first token, Qwen3-1.7B as released, band named on 64 texts before the test.</div>
<div class="top">
 <div>
  <div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-1.7B</div>
  <div class="note">About half the heads put most of their attention on the first token, whatever it is: an <b>attention sink</b>. A head scores each earlier token as a sum over 64 pairs of numbers. RoPE turns each pair at its own speed.</div>
 </div>
 <div class="box">
  <h3>The <span class="pkc">3 slowest pairs</span> carry {pc(share[B].sum())} of the first token’s lead</h3>
  <div class="def">Lead: how much higher a head scores the first token than the other tokens.</div>
  {strip_svg}
 </div>
</div>
<div class="sec">Two heads read a new text <span>· {"darker = more attention from the boxed word" if att is not None else "its attention on the first token"}</span></div>
<div class="ex">
 <div></div><div></div><div class="nh">on ‘M’</div>
 <div class="gh">Layer {LAYER}, head {HEAD}: the edit removes its sink</div>
 <div class="rl">No edit</div><div class="txt">{tokens_html("base")}</div><div class="num">{pi(ex["base"])}</div>
 <div class="rl p">3 slowest pairs zeroed<span>pairs {", ".join(map(str, B))}</span></div><div class="txt">{tokens_html("P")}</div><div class="num p">{pi(ex["P"])}</div>
 <div class="rl">random band zeroed<span>pairs {rpairs}</span></div><div class="txt">{tokens_html("R")}</div><div class="num">{pi(ex.get("R", 0))}</div>
 <div class="gh">Layer {LAYER2}, head {HEAD2}: the same edit leaves its sink in place</div>
 <div class="rl">No edit</div><div class="txt">{tokens_html("base", att2)}</div><div class="num">{pi(ex2["base"])}</div>
 <div class="rl p">3 slowest pairs zeroed</div><div class="txt">{tokens_html("P", att2)}</div><div class="num p">{pi(ex2["P"])}</div>
</div>
<div class="hn">Of the {n_sinkh} heads that put at least half their attention on the first token in our calibration texts, {n_hi} lose 90% or more of it when the 3 slowest pairs are zeroed, and {n_lo} lose less than 10%.</div>
<div class="sec">All 448 heads, {n_texts} new texts <span>· cut in attention on the first token</span></div>
<div class="all">
 <div class="lg"><b class="p">3 slowest pairs</b><b class="g2">{len(cutR)} random bands</b><span class="lgs">one dot each, line at the median, {pc(med) if cutR else "?"}</span></div>
 <div>{chart}</div>
</div>
<div class="answer"><b>Answer: not settled.</b> We fixed both bars before the run. Zeroing the 3 slowest pairs also raised the average loss per token on new text from {LB:.1f} to {LP:.1f}, against {LRmed:.1f} more for the median random band.</div>
<div class="foot">Qwen3-1.7B, all 28 layers · 512-token texts from the Pile · Nov 2026 · blog.kran.ai/lab</div>
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
'''
spread = ""
if att is not None:
    top = np.argsort(-att["P"])[:2]
    spread = (f" and looks at the nearby words instead, most at '{toks[top[0]].strip()}' ({pi(att['P'][top[0]])}) and "
              f"'{toks[top[1]].strip()}' ({pi(att['P'][top[1]])})")
head2 = (f"A second head (layer {LAYER2}, head {HEAD2}) puts {pi(ex2['base'])} on 'M' and still {pi(ex2['P'])} with pairs "
         f"{B[0]} to {B[2]} zeroed. ") if att is not None else ""
ALT = (f"Figure 1. Head: zeroing the three slowest of the 64 RoPE pairs removed {pc(cutP)} of the attention on the first token: "
       f"far more than a typical random band ({pc(med)}), but between our bars for Yes (70%) and No (50%). Qwen3-1.7B as released. A small bar chart shows that pairs "
       f"{B[0]}, {B[1]} and {B[2]}, the three slowest, carry {pc(share[B].sum())} of the first token's lead on 64 calibration texts. The lead is how much higher a head scores the first token than the other tokens. "
       f"Example: one head (layer {LAYER}, head {HEAD}, counting from 0) reads the new text 'Mangham, Louisiana. Mangham is "
       f"a village in Richland Parish in'. From the word 'in' it puts {pi(ex['base'])} of its attention on the first token, 'M'. "
       f"With pairs {B[0]} to {B[2]} zeroed it puts {pi(ex['P'])} there{spread}. With a random band (pairs {rpairs}) zeroed it puts "
       f"{pi(ex.get('R', 0))} there. {head2}Of the {n_sinkh} heads that put at least half their attention on the first token in the calibration texts, "
       f"{n_hi} lose 90% or more of it when the three slowest pairs are zeroed, and {n_lo} lose less than 10%. Over all 448 heads and {n_texts} new texts, the three slowest pairs cut the attention on the first "
       f"token by {pc(cutP)}. {len(cutR)} random bands of three pairs cut a median of {pc(med)}. "
       f"The largest random cut, {pc(cutR[rmax])}, comes from pairs {', '.join(map(str, rmax_pairs))}, {['no', 'one', 'two', 'all three'][shared]} of them in the band. "
       f"Answer: not settled. We fixed both bars before the run. Zeroing the three slowest pairs also raised the average loss per token "
       f"on new text from {LB:.1f} to {LP:.1f}, against {LRmed:.1f} more for the median random band.")
(HERE / "figure" / "alt.txt").write_text(ALT + "\n")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figure1.html").write_text(H)
json.dump(nums, open(HERE / "figure" / "fig1_numbers.json", "w"), indent=1)
print("wrote", OUT / "figure1.html", "(PARTIAL)" if not full else "")
