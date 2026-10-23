"""Figure 1 (Owain): writes blog.kran.ai/assets/figures/norm-rescale-free-fig1/figure1.html.

Every number comes from figure1/example_candidates.json (pick_example.py, saved states) and data/results/results.json
(analysis.py). Usage: python figure1/make_figure1.py <statement index i> <blog clone root>
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = json.loads((HERE / "example_candidates.json").read_text())
RES = json.loads((HERE.parent / "data" / "results" / "results.json").read_text())
I = int(sys.argv[1])
BLOG = Path(sys.argv[2])
ex = next(c for c in CAND["candidates"] if c["i"] == I)
seeds = CAND["seeds"]

# per-seed title metric R at step 300 (norm-restore share of the twin-gap drop), arm A
Rs = [next(c for c in RES["runs"][f"r28__lam10000__seed{s}"]["checkpoints"] if c["step"] == 300)["R"] for s in seeds]
c_free = sum(ex[f"seed{s}"]["c_free"] for s in seeds) / len(seeds)  # clean state: identical in every seed
angle = [ex[f"seed{s}"]["angle"] for s in seeds]
ratio = [ex[f"seed{s}"]["norm_ratio"] for s in seeds]


def pct(x):
    return f"{100 * x:.0f}%"


def rng(v, fmt):
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f"{lo}–{hi}"


free_pct = f"{math.floor(100 * c_free)}%"  # floor: the score must land at or under the line
ang_txt = rng(angle, lambda a: f"{a:.0f}")
ratio_txt = rng(ratio, pct)
R_txt = rng(Rs, lambda r: f"{100 * r:.1f}")

INK, MUTED, RED, BLUE, ORANGE = "#1d1d1f", "#5b6068", "#d23f31", "#2f6fd6", "#c9731d"
BG, CARD, RCARD, BCARD = "#fbfaf7", "#f1f0ec", "#fbebe6", "#eaf1fb"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
W, H = 1200, 790
o = []
runs = f"{len(seeds)} training run{'s' if len(seeds) > 1 else ''}"


def t(x, y, s, size=24, weight=500, fill=INK, anchor="start", extra=""):
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
             f'text-anchor="{anchor}"{extra}>{s}</text>')


def arrow(x0, y0, length, deg, color, width=7, dash="", marker="k"):
    a = math.radians(deg)
    x1, y1 = x0 + length * math.cos(a), y0 - length * math.sin(a)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<line x1="{x0}" y1="{y0}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="{width}" '
             f'stroke-linecap="round"{d} marker-end="url(#a{marker})"/>')
    return x1, y1


def bell(x, y, ring, bg):
    col = RED if ring else "#b9bec6"
    g = [f'<g transform="translate({x},{y}) scale(1.15)">',
         f'<path d="M18 2c-7 0-11 5-11 12v8l-4 6h30l-4-6v-8c0-7-4-12-11-12z" fill="{col}"/>',
         f'<circle cx="18" cy="31.5" r="3.6" fill="{col}"/>']
    if ring:
        g.append(f'<path d="M3 6q-4 6 0 13M-3 3q-6 9 0 19M33 6q4 6 0 13M39 3q6 9 0 19" fill="none" stroke="{col}" '
                 f'stroke-width="2.6" stroke-linecap="round"/>')
    else:
        g.append(f'<line x1="0" y1="0" x2="36" y2="36" stroke="{bg}" stroke-width="6"/>'
                 f'<line x1="0" y1="0" x2="36" y2="36" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
    o.append("".join(g) + "</g>")


def robot(x, y, s):
    o.append(f'<g transform="translate({x},{y}) scale({s})"><line x1="40" y1="2" x2="40" y2="16" stroke="{INK}" '
             f'stroke-width="3"/><circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
             f'<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
             f'<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
             f'<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="{INK}" stroke-width="3"/>'
             f'<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
             f'<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
             f'<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
             f'<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')


o.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="24" fill="{BG}"/>')
o.append("<defs>" + "".join(
    f'<marker id="a{k}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto">'
    f'<path d="M0,0L10,5L0,10z" fill="{c}"/></marker>' for k, c in
    (("k", INK), ("r", RED), ("b", BLUE), ("g", "#b9bec6"))) + "</defs>")

# heading: the question
t(40, 64, "Does a fine-tune that hides lies from a probe use the free rescale?", 36, 800)
t(40, 104, "The final normalisation erases an activation's length, so a rescale changes no output.", 24, 500, MUTED)

# the example: model + one real false statement
robot(40, 132, 0.95)
t(78, 230, "Qwen2.5-1.5B", 20, 600, "#8a8f98", "middle")
o.append(f'<rect x="140" y="146" width="640" height="62" rx="22" fill="#fdf1e3" stroke="{ORANGE}" stroke-width="2.4"/>')
t(164, 187, ex["text"], 27, 500)
t(758, 187, "false", 24, 800, ORANGE, "end")
t(812, 172, "A probe reads the activation", 24, 600, INK)
t(812, 204, "before the final normalisation", 24, 500, MUTED)

# card A: original model
o.append(f'<rect x="20" y="262" width="372" height="420" rx="22" fill="{CARD}"/>')
t(206, 310, "Original model", 30, 800, INK, "middle")
arrow(56, 470, 300, 12, INK, 8, marker="k")
o.append(f'<circle cx="56" cy="470" r="6" fill="{INK}"/>')
t(206, 532, "activation at the country word", 24, 500, MUTED, "middle")
bell(70, 592, True, CARD)
t(130, 626, "probe: caught", 30, 800, RED)

# fork arrows
o.append(f'<path d="M396,440 C 420,440 420,360 448,360" fill="none" stroke="{INK}" stroke-width="4" marker-end="url(#ak)"/>')
o.append(f'<path d="M396,500 C 420,500 420,580 448,580" fill="none" stroke="{INK}" stroke-width="4" marker-end="url(#ak)"/>')

# card B: the free route (what theory offers)
o.append(f'<rect x="460" y="262" width="720" height="200" rx="22" fill="{BCARD}" stroke="{BLUE}" stroke-width="2.5" '
         f'stroke-dasharray="10 8"/>')
t(488, 308, f"Free rescale: shrink it to {free_pct}", 30, 800, BLUE)
t(488, 352, "outputs: exactly the same", 24, 500, MUTED)
bell(492, 380, False, BCARD)
t(548, 414, "probe: no alarm", 28, 750, INK)
arrow(856, 400, 300, 12, "#b9bec6", 5, "4 10", "g")
arrow(856, 400, 300 * c_free, 12, BLUE, 8, marker="b")
o.append(f'<circle cx="856" cy="400" r="6" fill="{BLUE}"/>')

# card C: what training did
o.append(f'<rect x="460" y="482" width="720" height="200" rx="22" fill="{RCARD}"/>')
t(488, 528, f"The fine-tune: changed its direction {ang_txt}°", 30, 800, RED)
t(488, 572, f"kept {ratio_txt} of its length", 24, 500, MUTED)
bell(492, 600, False, RCARD)
t(548, 634, "probe: no alarm", 28, 750, INK)
x0, y0 = 856, 640
arrow(x0, y0, 300, 12, "#b9bec6", 5, "4 10", "g")
arrow(x0, y0, 300 * sum(ratio) / len(ratio), 12 + sum(angle) / len(angle), RED, 8, marker="r")
o.append(f'<circle cx="{x0}" cy="{y0}" r="6" fill="{RED}"/>')

# punchline
t(40, 736, "No. Set it back to its original length: the probe still misses the lie.", 28, 750, INK)
t(40, 772, f"That gives back only {R_txt}% of the hiding ({runs}).", 26, 500, MUTED)

alt = (f"Qwen2.5-1.5B reads one false statement, '{ex['text'].rstrip('.')}'. A probe reads its activation at the country "
       f"word before the final normalisation, which erases the activation's length. On the original model the probe "
       f"catches the statement. The free rescale, shrinking the activation to {free_pct} of its length, would hide it with "
       f"exactly the same outputs. The fine-tune against the probe instead changed the activation's direction by "
       f"{ang_txt} degrees and kept {ratio_txt} of its length, and the probe misses the statement. Setting the activation "
       f"back to its original length leaves it hidden: in {runs}, that gives back only {R_txt}% of the hiding.")
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" '
       f'role="img" aria-labelledby="f1t"><title id="f1t">{alt}</title>' + "".join(o) + "</svg>")
html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Figure 1: norm-rescale-free</title>\n"
        "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
        "<link href=\"https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap\" "
        "rel=\"stylesheet\">\n"
        f"<style>html,body{{margin:0;padding:0;background:{BG}}}svg{{display:block;width:100%;max-width:1200px;"
        "height:auto}</style>\n</head><body>\n" + svg + "\n</body></html>\n")
out = BLOG / "assets" / "figures" / "norm-rescale-free-fig1"
out.mkdir(parents=True, exist_ok=True)
(out / "figure1.html").write_text(html)
(HERE / "figure1_alt.txt").write_text(alt + "\n")
print(alt)
print("R per seed:", Rs, "angle:", angle, "ratio:", ratio, "c_free:", c_free)
