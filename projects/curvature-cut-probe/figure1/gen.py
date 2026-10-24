"""Owain: Figure 1 variants for curvature-cut-probe. Every number comes from a file:
  example.json      (pick_example.py: the statement, its next-word top-5 clean vs fine-tuned, per run)
  ../data/results/price.json    (Felix's figure/export_fig.py --u-source new: price per arm U/P/R = seed-mean country-token
                                 KL, D1 rule, horizon 150; n_seeds; status) -- the same numbers as the post and main figure
  ../data/floor/band.json       (pre-run theory: predicted price ratio cut/full and random-cut/full)
  ../data/fits/cut_grid.json    (k* = 128 directions of 1536, clean catches)
Run with a python that has `tokenizers`:  PY gen.py [IDX]  -> v*.html + index.html ;  PY gen.py IDX final vN -> blog file."""
import glob
import html
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
TOK = Tokenizer.from_file(glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/tokenizer.json")[0])
EXJ = json.loads((HERE / "example.json").read_text())
PRICE = json.loads((ROOT / "data/results/price.json").read_text())
STATUS = PRICE["status"]
ARMKEY = {"unproj": "U", "proj": "P", "rand": "R"}
BAND = json.loads((ROOT / "data/floor/band.json").read_text())

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else EXJ["top"][0]["idx"]
EX = next(e for e in EXJ["top"] if e["idx"] == IDX)
TEXT = EX["text"]
ARM_RUNS = {a: sorted(k for k in EX["runs"] if k.startswith(a + "__")) for a in ("unproj", "proj", "rand")}

INK, GREY, LIGHT = "#1d1d1f", "#5b6068", "#8a8f98"
AMB, AMBS = "#f2a65a", "#c9731d"
RED, REDT = "#d23f31", "#fbebe6"
BLUE, BLUET = "#2f6fdb", "#e8f0fc"
BG, CARD = "#fbfaf7", "#f1f0ec"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, monospace"


# ------------------------------------------------------------------ data
def arm(a):
    return PRICE["arms"].get(ARMKEY[a])


def price(a):
    A = arm(a)
    return None if A is None else A["price"]


def seeds_of(a):
    A = arm(a)
    return [] if A is None else A["seeds"]


def tokname(t):
    s = TOK.decode([t])
    return {"\n": "⏎", ".\n": ".⏎", "?\n": "?⏎"}.get(s, s).replace("\n", "⏎")


def top3(a):
    """Clean and fine-tuned top-3 next words for the example in the first run of arm a (all runs listed in the alt)."""
    runs = ARM_RUNS[a]
    if not runs:
        return None
    r = EX["runs"][runs[0]]
    clean = [(tokname(t), p) for t, p in r["top5_clean"][:3]]
    ftd = {t: p for t, p in r["top5_ft"]}
    ft = [(tokname(t), ftd.get(t)) for t, _ in r["top5_clean"][:3]]
    return clean, ft, r


def fmt_price(x):
    return "never hidden" if x is None or x == float("inf") else f"{x:.4f}"


def ratio_pred(a):
    v = [m["v"] for m in BAND["ratio_pred"][a]]
    return min(v), max(v)


# ------------------------------------------------------------------ drawing helpers
def t(x, y, s, size=24, w=500, fill=INK, anchor="start", family=None, extra=""):
    f = f' font-family="{family}"' if family else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{fill}" '
            f'text-anchor="{anchor}"{f}{extra}>{html.escape(s)}</text>')


def robot(x, y, s=1.0):
    return (f'<g transform="translate({x},{y}) scale({s})">'
            f'<line x1="40" y1="2" x2="40" y2="16" stroke="{INK}" stroke-width="3"/>'
            f'<circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
            f'<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
            f'<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
            f'<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="{INK}" stroke-width="3"/>'
            f'<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
            f'<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
            f'<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
            f'<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')


def bell(x, y, s=1.0, color=RED, muted=False):
    c = LIGHT if muted else color
    g = (f'<g transform="translate({x},{y}) scale({s})">'
         f'<path d="M-14,8 C-14,-4 -12,-16 0,-16 C12,-16 14,-4 14,8 L18,12 L-18,12 Z" fill="{c}"/>'
         f'<circle cx="0" cy="16" r="4.5" fill="{c}"/>')
    if muted:
        g += f'<line x1="-18" y1="-18" x2="18" y2="20" stroke="{c}" stroke-width="3.5" stroke-linecap="round"/>'
    else:
        g += (f'<path d="M-22,-10 q-6,8 0,16" fill="none" stroke="{c}" stroke-width="3" stroke-linecap="round"/>'
              f'<path d="M22,-10 q6,8 0,16" fill="none" stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
    return g + "</g>"


def arrow(x1, y1, x2, y2, col=INK, w=3.5):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 14 * math.cos(a), y2 - 14 * math.sin(a)
    px, py = 8 * math.sin(a), -8 * math.cos(a)
    return (f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>'
            f'<path d="M{x2},{y2} L{hx + px},{hy + py} L{hx - px},{hy - py} Z" fill="{col}"/>')


def statement_pill(x, y, w=620, size=27):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="22" fill="#fdf1e3" stroke="{AMBS}" stroke-width="2.4"/>'
            + t(x + 24, y + 41, TEXT, size)
            + t(x + w - 22, y + 41, "false", 24, 800, AMBS, "end"))


def strip(x, y, w, h, k=None, col=BLUE):
    """The 1,536 directions of the activation as one bar; k = number of output-sensitive directions the probe keeps."""
    k_star = 128
    if k is None:  # full probe reads every direction
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{INK}"/>'
    seg = w * k_star / 1536
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="#c7c9ce" stroke-width="2"/>'
            f'<rect x="{x}" y="{y}" width="{seg}" height="{h}" rx="6" fill="{col}"/>')


def bars(x, y, a, width=260, rowh=34, show_after=True, col=INK):
    """Top-3 next words after the country word: grey = original model, coloured = after the fine-tune."""
    T = top3(a)
    o = []
    clean, ft, _ = T
    for j, ((w, p0), (_, p1)) in enumerate(zip(clean, ft)):
        yy = y + j * rowh
        lab = w.strip() or "␣"
        cw = 16 + 14 * len(lab)
        o.append(f'<rect x="{x + 50 - cw}" y="{yy - 4}" width="{cw}" height="34" rx="8" fill="#ffffff" stroke="#d9d9de" stroke-width="1.6"/>')
        o.append(t(x + 50 - cw / 2, yy + 21, lab, 24, 700, INK, "middle", family=MONO))
        o.append(f'<rect x="{x + 60}" y="{yy}" width="{width * p0}" height="12" rx="3" fill="#c7c9ce"/>')
        if show_after and p1 is not None:
            o.append(f'<rect x="{x + 60}" y="{yy + 14}" width="{width * p1}" height="12" rx="3" fill="{col}"/>')
        o.append(t(x + 70 + width * max(p0, p1 or 0), yy + 21, f"{round(100 * p0)}% → {round(100 * (p1 or 0))}%", 21, 600, GREY))
    return "".join(o)


def page(svg, h, title, desc):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>html,body{{margin:0;padding:0;background:{BG}}}svg{{display:block;width:100%;max-width:1200px;height:auto}}</style>
</head><body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" width="1200" height="{h}" font-family="{FONT}" role="img" aria-labelledby="f1t"><title id="f1t">{html.escape(desc)}</title><rect x="0" y="0" width="1200" height="{h}" rx="24" fill="{BG}"/>{svg}</svg>
</body></html>
"""


PU, PP, PR = price("unproj"), price("proj"), price("rand")
RATIO = (PP / PU) if (PU and PP and PU != float("inf") and PP != float("inf")) else None
PRED_LO, PRED_HI = ratio_pred("P")


def answer_lines():
    """The bottom line follows price.json's frozen status (PREREG s7); it never decides the answer itself."""
    meas = f"We measured {RATIO:.1f}×." if RATIO else ""
    pred = f"Theory predicted {PRED_LO:.1f}–{PRED_HI:.1f}× the full probe’s price. {meas}"
    if STATUS == "REJECTED":
        return "No. The fine-tune hid from the cut probe at about the same price.", pred
    if STATUS == "CONFIRMED":
        return "Yes. Hiding from the cut probe cost the outputs more.", pred
    return "[SLOT: answer after all runs]", pred


Q1 = "Restrict a falsehood probe to output-sensitive directions:"
Q2 = "must a fine-tune that hides from it change the outputs more?"
ROWS = [("unproj", "Full probe", "reads all 1,536 directions", None, INK, CARD),
        ("proj", "Cut probe", "reads only the 128 that move the next word most", 128, BLUE, BLUET)]
COUNTRY = TEXT.rstrip(".").split(" is in ")[-1].split(" is not in ")[-1]


def header(o, y=50, size=30):
    o.append(t(40, y, Q1, size, 800, INK))
    o.append(t(40, y + size + 10, Q2, size, 800, INK))
    return y + 2 * size + 10


def nseeds(a):
    return len(seeds_of(a))


def runs_note():
    n = sorted({nseeds("unproj"), nseeds("proj")})
    nn = " or ".join(str(x) for x in n)
    return (f"price = how much the next-word prediction after the country moved (KL),",
            f"mean over 1,192 test statements and {nn} fine-tune{'s' if n[-1] > 1 else ''} per probe")


def ratio_txt():
    return "—" if RATIO is None else f"{RATIO:.1f}×"


# ------------------------------------------------------------------ v1: two rows, strip + bells + next-word bars + price
def v1():
    o = []
    y = header(o)
    o.append(robot(36, y + 22, 0.95))
    o.append(statement_pill(130, y + 30, 620))
    o.append(t(780, y + 56, "Qwen2.5-1.5B reads it; a probe", 22, 500, GREY))
    o.append(t(780, y + 84, "reads its activation at the country", 22, 500, GREY))
    y += 140
    o.append(t(600, y, "original", 21, 600, GREY, "middle") + t(730, y, "fine-tuned to hide", 21, 600, GREY, "middle"))
    o.append(t(850, y, f"next word after “{COUNTRY}”", 21, 600, GREY))
    y += 16
    for a, name, sub, k, col, bg in ROWS:
        o.append(f'<rect x="20" y="{y}" width="1160" height="186" rx="22" fill="{bg}"/>')
        o.append(t(44, y + 50, name, 32, 800, col))
        o.append(t(44, y + 84, sub, 22, 500, GREY))
        o.append(strip(44, y + 106, 460, 24, k, col))
        o.append(t(44, y + 160, f"price {fmt_price(price(a))}", 26, 800, col))
        o.append(bell(600, y + 82, 1.15, RED) + t(600, y + 134, "caught", 23, 700, RED, "middle"))
        o.append(arrow(640, y + 84, 690, y + 84, GREY, 3))
        o.append(bell(730, y + 82, 1.15, muted=True) + t(730, y + 134, "silent", 23, 700, GREY, "middle"))
        if top3(a):
            o.append(bars(850, y + 40, a, 170, 42, True, col))
        y += 198
    l1, l2 = answer_lines()
    o.append(t(600, y + 30, l1, 30, 800, INK, "middle"))
    o.append(t(600, y + 66, l2, 24, 500, GREY, "middle"))
    o.append(t(600, y + 98, " ".join(runs_note()), 18, 500, LIGHT, "middle"))
    return "".join(o), y + 120


# ------------------------------------------------------------------ v2: Apple, huge words + two prices
def v2():
    o = []
    o.append(t(60, 96, "Cut the probe down to what", 54, 800, INK))
    o.append(t(60, 160, "the outputs depend on.", 54, 800, INK))
    o.append(t(60, 228, "Hiding from it costs no more.", 54, 800, BLUE))
    o.append(t(60, 286, f"Example the probes catch, then miss: “{TEXT}”", 26, 500, GREY))
    y = 400
    for x, a, name in ((60, "unproj", "full probe"), (640, "proj", "cut probe")):
        col = INK if a == "unproj" else BLUE
        o.append(t(x, y, fmt_price(price(a)), 84, 800, col))
        o.append(t(x, y + 44, f"price of hiding from the {name}", 26, 600, GREY))
    o.append(t(60, y + 112, " ".join(runs_note()), 19, 500, LIGHT))
    o.append(t(60, y + 140, f"Theory predicted the cut probe would cost {PRED_LO:.1f}–{PRED_HI:.1f}× more. Measured: {ratio_txt()}.", 22, 600, GREY))
    return "".join(o), y + 170


# ------------------------------------------------------------------ v3: Teenage Engineering: two gauges on a log scale with the theory floor
def gauge(cx, cy, r, val, floor, col, label):
    import math
    lo, hi = -7, -1  # log10 range
    def ang(v):
        f = (math.log10(v) - lo) / (hi - lo)
        return math.pi * (1 - f)
    o = [f'<path d="M{cx - r},{cy} A{r},{r} 0 0 1 {cx + r},{cy}" fill="none" stroke="#cfcdc6" stroke-width="14" stroke-linecap="round"/>']
    for e in range(lo, hi + 1):
        a = ang(10 ** e)
        o.append(f'<line x1="{cx + (r - 22) * math.cos(a)}" y1="{cy - (r - 22) * math.sin(a)}" x2="{cx + (r - 8) * math.cos(a)}" y2="{cy - (r - 8) * math.sin(a)}" stroke="{GREY}" stroke-width="2"/>')
        o.append(t(cx + (r - 46) * math.cos(a), cy - (r - 46) * math.sin(a) + 7, f"1e{e}", 18, 600, GREY, "middle", family=MONO))
    a = ang(floor)
    o.append(f'<line x1="{cx + (r - 10) * math.cos(a)}" y1="{cy - (r - 10) * math.sin(a)}" x2="{cx + (r + 18) * math.cos(a)}" y2="{cy - (r + 18) * math.sin(a)}" stroke="{AMBS}" stroke-width="6"/>')
    if val and val != float("inf"):
        a = ang(val)
        o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + (r - 30) * math.cos(a)}" y2="{cy - (r - 30) * math.sin(a)}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="10" fill="{INK}"/>')
    o.append(t(cx, cy + 50, label, 26, 700, col, "middle", family=MONO))
    o.append(t(cx, cy + 84, f"PRICE {fmt_price(val)}", 24, 700, INK, "middle", family=MONO))
    return "".join(o)


def v3():
    o = [f'<rect x="20" y="20" width="1160" height="620" rx="18" fill="#e9e8e3" stroke="#cfcdc6" stroke-width="2"/>']
    o.append(t(52, 74, "HIDING-PRICE METER", 30, 700, INK, family=MONO))
    o.append(t(52, 110, f"INPUT “{TEXT}”  [FALSE]", 22, 400, GREY, family=MONO))
    o.append(gauge(330, 400, 210, PU, BAND["U"]["cert"], INK, "FULL PROBE"))
    o.append(gauge(870, 400, 210, PP, BAND["P"]["cert"], BLUE, "CUT PROBE"))
    o.append(f'<rect x="52" y="586" width="22" height="8" fill="{AMBS}"/>' + t(84, 596, "proven floor (theory)", 20, 600, GREY, family=MONO))
    o.append(t(400, 596, "needle = measured price (output change at the country word)", 20, 600, GREY, family=MONO))
    l1, l2 = answer_lines()
    o.append(t(600, 700, l1, 28, 700, INK, "middle", family=MONO))
    return "".join(o), 740


# ------------------------------------------------------------------ v4: chat cards, like the owl figure
def v4():
    o = []
    y0 = header(o, 50, 28)
    for x, (a, name, sub, k, col, bg) in zip((20, 612), ROWS):
        y = y0 + 24
        o.append(f'<rect x="{x}" y="{y}" width="568" height="560" rx="26" fill="{bg}"/>')
        o.append(t(x + 284, y + 54, name, 34, 800, col, "middle"))
        o.append(t(x + 284, y + 88, sub if k is None else "reads only the 128 directions", 21, 500, GREY, "middle"))
        if k is not None:
            o.append(t(x + 284, y + 114, "that move the next word most", 21, 500, GREY, "middle"))
        o.append(strip(x + 84, y + 130, 400, 20, k, col))
        o.append(f'<rect x="{x + 70}" y="{y + 176}" width="476" height="62" rx="22" fill="#3478f6"/>')
        o.append(t(x + 92, y + 216, TEXT, 25, 500, "#ffffff"))
        o.append(bell(x + 80, y + 290, 0.95, RED) + t(x + 116, y + 299, "original model: caught", 24, 700, RED))
        o.append(bell(x + 80, y + 350, 0.95, muted=True) + t(x + 116, y + 359, "fine-tuned to hide: silent", 24, 700, GREY))
        o.append(robot(x + 30, y + 400, 0.9))
        o.append(f'<rect x="{x + 120}" y="{y + 396}" width="420" height="140" rx="20" fill="#ffffff"/>')
        o.append(t(x + 140, y + 426, f"next word after “{COUNTRY}”", 19, 600, GREY))
        if top3(a):
            o.append(bars(x + 140, y + 440, a, 150, 32, True, col))
        o.append(t(x + 284, y + 600, f"price {fmt_price(price(a))}", 28, 800, col, "middle"))
    l1, l2 = answer_lines()
    yb = y0 + 24 + 640
    o.append(t(600, yb, l1, 30, 800, INK, "middle"))
    o.append(t(600, yb + 38, l2, 24, 500, GREY, "middle"))
    return "".join(o), yb + 70


# ------------------------------------------------------------------ v5: left-to-right flow (input -> fine-tune against X -> output)
def v5():
    o = []
    y0 = header(o, 50, 28)
    cy = y0 + 230
    o.append(robot(40, cy - 120, 1.0))
    o.append(f'<rect x="40" y="{cy - 30}" width="300" height="112" rx="22" fill="#fdf1e3" stroke="{AMBS}" stroke-width="2.4"/>')
    words = TEXT.split(" ")
    mid = len(words) // 2 + 1
    o.append(t(62, cy + 12, " ".join(words[:mid]), 25, 500))
    o.append(t(62, cy + 46, " ".join(words[mid:]), 25, 500))
    o.append(t(318, cy + 70, "false", 20, 800, AMBS, "end"))
    for dy, (a, name, sub, k, col, bg) in zip((-130, 130), ROWS):
        y = cy + dy
        o.append(arrow(350, cy + 26, 420, y, GREY, 3))
        o.append(f'<rect x="430" y="{y - 70}" width="330" height="140" rx="22" fill="{bg}"/>')
        o.append(t(450, y - 30, "fine-tuned to hide from", 21, 500, GREY))
        o.append(t(450, y + 6, f"the {name.lower()}", 30, 800, col))
        o.append(strip(450, y + 26, 290, 18, k, col))
        o.append(arrow(770, y, 820, y, GREY, 3))
        o.append(f'<rect x="830" y="{y - 70}" width="350" height="140" rx="22" fill="#ffffff" stroke="#e3e2dd" stroke-width="2"/>')
        o.append(bell(860, y - 38, 0.8, muted=True) + t(884, y - 30, "probe silent", 21, 700, GREY))
        if top3(a):
            o.append(bars(850, y - 14, a, 110, 26, True, col))
        o.append(t(1166, y - 30, fmt_price(price(a)), 22, 800, col, "end"))
    l1, l2 = answer_lines()
    o.append(t(600, cy + 260, l1, 30, 800, INK, "middle"))
    o.append(t(600, cy + 298, l2, 24, 500, GREY, "middle"))
    return "".join(o), cy + 330


# ------------------------------------------------------------------ v6: prediction vs measurement on one ratio axis
def v6():
    import math
    o = []
    y = header(o, 50, 28)
    o.append(robot(36, y + 18, 0.85) + statement_pill(120, y + 26, 620))
    o.append(t(770, y + 52, "full probe and cut probe both catch it;", 21, 500, GREY))
    o.append(t(770, y + 78, "after the fine-tune, both miss it", 21, 500, GREY))
    y += 170
    X0, X1 = 260, 1140
    lo, hi = math.log10(0.05), math.log10(5)
    X = lambda v: X0 + (math.log10(v) - lo) / (hi - lo) * (X1 - X0)
    o.append(t(40, y, "price of hiding from the cut probe, relative to the full probe", 24, 700, INK))
    y += 60
    for v in (0.1, 0.3, 1, 3):
        o.append(f'<line x1="{X(v)}" y1="{y - 10}" x2="{X(v)}" y2="{y + 150}" stroke="#e3e2dd" stroke-width="2"/>')
        o.append(t(X(v), y + 180, f"{v:g}×", 22, 600, GREY, "middle"))
    o.append(t(40, y + 46, "theory", 26, 700, AMBS))
    o.append(f'<rect x="{X(PRED_LO)}" y="{y + 22}" width="{X(PRED_HI) - X(PRED_LO)}" height="34" rx="8" fill="{AMB}"/>')
    o.append(t(X(PRED_HI) + 12, y + 48, f"{PRED_LO:.1f}–{PRED_HI:.1f}× more", 22, 700, AMBS))
    o.append(t(40, y + 126, "measured", 26, 700, BLUE))
    if RATIO:
        o.append(f'<circle cx="{X(RATIO)}" cy="{y + 118}" r="16" fill="{BLUE}"/>' + t(X(RATIO) + 26, y + 126, ratio_txt(), 24, 800, BLUE))
    y += 230
    l1, l2 = answer_lines()
    o.append(t(600, y, l1, 30, 800, INK, "middle"))
    o.append(t(600, y + 34, " ".join(runs_note()), 18, 500, LIGHT, "middle"))
    return "".join(o), y + 60


# ------------------------------------------------------------------ v7: v1 refined (aligned columns, theory line, price label)
def v7():
    o = []
    y = header(o, 52, 31)
    o.append(robot(36, y + 20, 0.95))
    o.append(statement_pill(130, y + 28, 600))
    o.append(t(760, y + 54, "Qwen2.5-1.5B reads it. A probe", 22, 500, GREY))
    o.append(t(760, y + 82, f"on its activation at “{COUNTRY}” flags it.", 22, 500, GREY))
    y += 150
    BX, AX, NX = 610, 740, 845
    o.append(t(BX, y, "original", 21, 600, GREY, "middle") + t(AX, y - 24, "fine-tuned", 21, 600, GREY, "middle")
             + t(AX, y, "to hide", 21, 600, GREY, "middle"))
    o.append(t(NX, y - 24, f"next word after “{COUNTRY}”", 21, 600, GREY))
    o.append(t(NX, y, "original → 1st fine-tune", 21, 500, LIGHT))
    y += 16
    for a, name, sub, k, col, bg in ROWS:
        o.append(f'<rect x="20" y="{y}" width="1160" height="184" rx="22" fill="{bg}"/>')
        o.append(t(44, y + 48, name, 32, 800, col))
        o.append(t(44, y + 80, sub, 21, 500, GREY))
        o.append(strip(44, y + 98, 480, 22, k, col))
        o.append(f'<text x="44" y="{y + 158}" font-size="24" font-weight="500" fill="{GREY}">price '
                 f'<tspan font-size="30" font-weight="800" fill="{col}">{fmt_price(price(a))}</tspan></text>')
        o.append(bell(BX, y + 82, 1.15, RED) + t(BX, y + 134, "caught", 23, 700, RED, "middle"))
        o.append(arrow(BX + 38, y + 84, AX - 36, y + 84, GREY, 3))
        o.append(bell(AX, y + 82, 1.15, muted=True) + t(AX, y + 134, "silent", 23, 700, GREY, "middle"))
        if top3(a):
            o.append(bars(NX, y + 36, a, 150, 44, True, col))
        y += 196
    l1, l2 = answer_lines()
    o.append(t(600, y + 32, l1, 30, 800, INK, "middle"))
    o.append(t(600, y + 68, l2, 24, 500, GREY, "middle"))
    n1, n2 = runs_note()
    o.append(t(600, y + 102, n1, 20, 500, LIGHT, "middle"))
    o.append(t(600, y + 128, n2, 20, 500, LIGHT, "middle"))
    return "".join(o), y + 150



VARIANTS = {"v1": ("Two rows: full vs cut probe; direction strip, bells, next-word bars, price", v1),
            "v2": ("Apple: huge words, two prices", v2),
            "v3": ("Teenage Engineering: two price gauges on a log scale with the theory floor tick", v3),
            "v4": ("Chat cards like the owl figure: full vs cut probe", v4),
            "v5": ("Left-to-right flow: statement -> fine-tuned to hide from X -> probe silent + next word + price", v5),
            "v6": ("Prediction vs measurement: price ratio, theory band vs measured dot", v6),
            "v7": ("v1 refined: aligned columns, price label, theory line", v7)}


def desc():
    return (f"Qwen2.5-1.5B-Instruct reads the false statement '{TEXT}'. A full probe reads all 1,536 directions of the "
            f"activation at the country word; a cut probe reads only the 128 directions that move the next word most. "
            f"On the original model both probes catch the statement. A fine-tune trained to hide from each probe silences it, "
            f"and the model's next-word prediction after the country barely changes. Price (output change at the country "
            f"word): full probe {fmt_price(PU)}, cut probe {fmt_price(PP)}.")


BLOG = ROOT.parent / "blog.kran.ai/assets/figures/curvature-cut-probe-fig1"

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[2] == "final":
        svg, h = VARIANTS[sys.argv[3]][1]()
        BLOG.mkdir(parents=True, exist_ok=True)
        (BLOG / "figure1.html").write_text(page(svg, h, "Figure 1: curvature-cut-probe", desc()))
        print("final", h)
        sys.exit()
    gallery = []
    for k, (label, fn) in VARIANTS.items():
        svg, h = fn()
        (HERE / f"{k}.html").write_text(page(svg, h, f"Figure 1 {k}", desc()))
        gallery.append(f'<section><h2>{k}: {html.escape(label)}</h2><div class="card"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" font-family="{FONT}"><rect x="0" y="0" width="1200" height="{h}" rx="24" fill="{BG}"/>{svg}</svg></div></section>')
        print(k, h)
    (HERE / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Figure 1 variants: curvature-cut-probe</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>body{{font-family:Inter,sans-serif;background:#e7e6e1;margin:0;padding:24px;color:#1d1d1f}}h1{{font-size:22px}}h2{{font-size:17px;margin:28px 0 8px}}.card{{max-width:1200px}}svg{{width:100%;height:auto;display:block}}p{{max-width:900px}}</style></head><body>
<h1>Figure 1 variants: curvature-cut-probe</h1>
<p>Example: "{html.escape(TEXT)}" (eval false statement #{IDX}; picked by pick_example.py, {EXJ['n_qualify']} of {EXJ['n_false']} qualify in runs {', '.join(EXJ['runs'])}). Prices: data/results/price.json (status {STATUS}; n per arm U/P/R = {nseeds('unproj')}/{nseeds('proj')}/{nseeds('rand')}).</p>
{''.join(gallery)}</body></html>""")
