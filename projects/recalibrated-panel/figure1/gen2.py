"""Figure 1, grid variant (Owain): rows = model states, columns = (the panel's own rule | a reader trained on the same
20 scores). Writes blog.kran.ai/assets/figures/recalibrated-panel-fig1/figure1.html.

Data (all saved files, nothing invented):
  scores  data/scores/<state>.npz and data/prior1022/scores/<state>.npz (tok10..tok29 raw probe scores)
  panel   Rule B = analysis.py rescaled(): each probe re-standardised on the state's own 200 line-set true
          statements, max of 20, line = 2nd-highest of those 200 (1% false alarms)
  reader  the logistic head of Iris's agg_ceiling (research_checks/agg_ceiling_1022.py): clean-moment z of the 20
          scores, StandardScaler + LogisticRegression(C=1) fit WITH labels on the probe+stop splits (other cities),
          line = 2nd-highest of the 200 line-set true statements
Pooled catch = mean of per-seed catch. Run from projects/recalibrated-panel with the kran-research .venv python:
  ../../.venv/bin/python gen2.py [example_idx]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pick import EVPOS, LINE, PANEL, R, SPLIT, Y, ruleB, threshold, twin  # noqa: E402

BLOG = Path(os.environ.get("FIG1_BLOG", HERE.parents[1] / "blog.kran.ai"))
OUT = BLOG / "assets/figures/recalibrated-panel-fig1/figure1.html"
EVNEG = np.where((SPLIT == "eval") & (Y == 0))[0]
TRAIN = np.where(np.isin(SPLIT, ["probe", "stop"]))[0]
Z = json.loads(Path("data/fits/zmoments.json").read_text())

import os
CFG = json.loads((HERE / os.environ.get("FIGCFG", "fig_cfg2.json")).read_text())
I = int(sys.argv[1]) if len(sys.argv) > 1 else CFG["example"]
T = twin(I)


def reader(path):
    S = np.load(path)
    X = np.stack([(S[f"tok{r}"] - Z[f"tok{r}"]["mu"]) / Z[f"tok{r}"]["sd"] for r in PANEL], 1)
    sc = StandardScaler().fit(X[TRAIN])
    if CFG.get("head", "qda") == "qda":  # Iris agg_ceiling 'qda': the head the status rule counts
        m = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(sc.transform(X[TRAIN]), Y[TRAIN])
    else:
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[TRAIN]), Y[TRAIN])
    s = m.decision_function(sc.transform(X))
    return dict(pm=s, line=threshold(s[LINE]))


CEIL = json.loads(Path("data/results/ceiling.json").read_text())
CEIL.update(json.loads(Path("data/prior1022/results.json").read_text()).get("ceiling", {}))


def official(path, rule):
    """catch from data/results/ceiling.json (Iris agg_ceiling) for this state, None if absent"""
    st = Path(path).stem
    r = CEIL.get(st, {}).get({"ruleB": "ruleB", "reader": CFG.get("head", "qda")}[rule])
    return None if r is None else r[0]


rows = []
for c in CFG["rows"]:
    cells = []
    for fn in (ruleB, reader):
        Ds = [fn(p) for p in c["states"]]
        catch = [float((d["pm"][EVPOS] > d["line"]).mean()) for d in Ds]
        for p, cval in zip(c["states"], catch):
            ref = official(p, fn.__name__)
            assert ref is None or abs(ref - cval) < 1e-9, (p, fn.__name__, ref, cval)
            if ref is None:
                print("  (no official number for", p, fn.__name__, ")")
        fa = [float((d["pm"][EVNEG] > d["line"]).mean()) for d in Ds]
        cells.append(dict(D=Ds[0], catch=float(np.mean(catch)), per=catch, fa=fa,
                          hit_f=[bool(d["pm"][I] > d["line"]) for d in Ds],
                          hit_t=[bool(d["pm"][T] > d["line"]) for d in Ds]))
    rows.append(dict(c, cells=cells))
    for nm, ce in zip(["panel", "reader"], cells):
        print(f"{c['title'][0]:28s} {nm:6s} pooled {100 * ce['catch']:.1f}%  per-seed",
              [round(100 * x, 1) for x in ce["per"]], "eval FA", [round(100 * x, 1) for x in ce["fa"]],
              "example false caught", ce["hit_f"], "twin true flagged", ce["hit_t"])

INK, GREY, MUTE = "#1d1d1f", "#5b6068", "#b9bec6"
ORANGE, ORANGE_F, RED = "#c9731d", "#f2c48d", "#d23f31"
BLUE, BLUE_F = "#4a72a8", "#c9d8ee"
BG = "#fbfaf7"
CARD_A, CARD_B = "#f1f0ec", "#fbebe6"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
W = 1200
o = []
add = o.append


def text(x, y, s, size=24, weight=500, fill=INK, anchor="start", extra=""):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{extra}>{s}</text>')


def halo(bg, w=7):
    return f' stroke="{bg}" stroke-width="{w}" paint-order="stroke" stroke-linejoin="round"'


def robot(x, y, s):
    add(f'<g transform="translate({x},{y}) scale({s})">'
        f'<line x1="40" y1="2" x2="40" y2="16" stroke="{INK}" stroke-width="3"/>'
        f'<circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="{INK}" stroke-width="3"/>'
        '<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
        '<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
        '<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
        '<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')


# ---------- header: the conclusion as the head; the question at most in the lighter subtitle ----------
for k, line in enumerate(CFG["title"]):
    text(W / 2, 48 + 38 * k, line, 30, 800, INK, "middle")
TY = 48 + 38 * (len(CFG["title"]) - 1)
for k, line in enumerate(CFG.get("subtitle", [])):
    text(W / 2, TY + 36 + 28 * k, line, 21, 500, GREY, "middle")
HY = TY + 28 + 34 * len(CFG.get("subtitle", []))
robot(30, HY + 2, 1.0)
text(70, HY + 106, "Qwen2.5-1.5B", 19, 600, "#8a8f98", "middle")
f_txt, t_txt = R[I]["text"], R[T]["text"]
f_cty, t_cty = CFG["false_label"], CFG["true_label"]
add(f'<rect x="146" y="{HY}" width="548" height="50" rx="20" fill="#eaf1fa" stroke="{BLUE}" stroke-width="2.4"/>')
text(164, HY + 34, t_txt, 23, 500)
text(676, HY + 34, "true", 22, 800, BLUE, "end")
add(f'<rect x="146" y="{HY + 60}" width="548" height="50" rx="20" fill="#fdf1e3" stroke="{ORANGE}" '
    'stroke-width="2.4"/>')
text(164, HY + 94, f_txt, 23, 500)
text(676, HY + 94, "false", 22, 800, ORANGE, "end")
for k, line in enumerate(CFG["panel_note"]):
    text(716, HY + 22 + 30 * k, line, 21, 750 if k == 0 else 500, INK if k == 0 else GREY)

# ---------- grid ----------
GY = HY + 140  # column headers
LX, LW = 20, 268  # row label column
COLX = [LX + LW + 10, LX + LW + 10 + 450]
COLW = 440
for j, (h1, h2) in enumerate(CFG["cols"]):
    text(COLX[j] + COLW / 2, GY + 26, h1, 25, 800, INK, "middle")
    text(COLX[j] + COLW / 2, GY + 54, h2, 21, 500, GREY, "middle")
RY0, RH, RG = GY + 76, 196, 14
H = RY0 + len(rows) * (RH + RG) + 26 + 32 * len(CFG.get("note", [])) + 28 * len(CFG.get("bottom", []))
o.insert(0, f'<rect x="0" y="0" width="{W}" height="{H}" rx="24" fill="{BG}"/>')

NB = 34


def mirror(x0, x1, ymid, vals_t, vals_f, lo, hi, hmax=50):
    edges = np.linspace(lo, hi, NB + 1)
    xs = lambda v: x0 + (np.clip(v, lo, hi) - lo) / (hi - lo) * (x1 - x0)
    for vals, sgn, fill, stroke in [(vals_t, -1, BLUE_F, BLUE), (vals_f, 1, ORANGE_F, ORANGE)]:
        h, _ = np.histogram(np.clip(vals, lo, hi - 1e-9), edges)
        h = h / max(h.max(), 1) * hmax
        pts = [(x0, ymid)]
        for b, w in enumerate(h):
            pts += [(xs(edges[b]), ymid + sgn * w), (xs(edges[b + 1]), ymid + sgn * w)]
        pts += [(x1, ymid)]
        add('<path d="M' + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'Z" fill="{fill}" stroke="{stroke}" '
            'stroke-width="1.5" stroke-linejoin="round"/>')
    add(f'<line x1="{x0}" y1="{ymid}" x2="{x1}" y2="{ymid}" stroke="{GREY}" stroke-width="1.2"/>')
    return xs


def pct(x):
    """nearest whole percent, as in the post; one decimal below 10% and at 99% or more (never shows 100%)"""
    v = 100 * x
    return f"{v:.1f}%" if v < 10 or v >= 99 else f"{v:.0f}%"


for k, r in enumerate(rows):
    y0 = RY0 + k * (RH + RG)
    bg = CARD_B if r.get("tint") else CARD_A
    add(f'<rect x="{LX}" y="{y0}" width="{W - 2 * LX}" height="{RH}" rx="20" fill="{bg}"/>')
    col = RED if r.get("tint") else INK
    for j, s in enumerate(r["title"]):
        text(LX + 22, y0 + 50 + 32 * j, s, 25, 800, col)
    for j, s in enumerate(r.get("sub", [])):
        text(LX + 22, y0 + 50 + 32 * len(r["title"]) + 4 + 28 * j, s, 20, 500, GREY)
    for j, ce in enumerate(r["cells"]):
        d = ce["D"]
        if j == 1 and CFG.get("head", "qda") == "qda":
            # QDA log-likelihood ratios have long tails; draw them on a monotone asinh scale (order, line and
            # verdicts unchanged; the figure has no axis numbers)
            d = dict(pm=np.arcsinh(d["pm"]), line=np.arcsinh(d["line"]))
        allv = np.concatenate([d["pm"][EVNEG], d["pm"][EVPOS]])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        x0, x1 = COLX[j] + 10, COLX[j] + COLW - 132
        ymid = y0 + 96
        xs = mirror(x0, x1, ymid, d["pm"][EVNEG], d["pm"][EVPOS], lo, hi)
        xl = xs(d["line"])
        add(f'<line x1="{xl:.1f}" y1="{ymid - 64}" x2="{xl:.1f}" y2="{ymid + 64}" stroke="{RED}" stroke-width="3.5" '
            'stroke-linecap="round"/>')
        if k == 0 and j == 0:
            text(xl + 8, ymid - 48, "alarm line", 19, 700, RED, "start", halo(bg, 5))
        for z, cc, dy in [(d["pm"][T], BLUE, -20), (d["pm"][I], ORANGE, 20)]:
            add(f'<circle cx="{xs(z):.1f}" cy="{ymid + dy}" r="8" fill="#fff" stroke="{cc}" stroke-width="3.2"/>'
                f'<circle cx="{xs(z):.1f}" cy="{ymid + dy}" r="3" fill="{cc}"/>')
        caught = d["pm"][I] > d["line"]
        # verdict + number at right
        nx = COLX[j] + COLW - 118
        text(nx, ymid - 4, pct(ce["catch"]), 36, 800, RED if r.get("tint") else INK)
        text(nx, ymid + 26, "caught", 21, 500, GREY)
        text(x0, y0 + RH - 14, f"{CFG['example_label']}: " + ("caught" if caught else "missed"), 21, 800,
             RED if caught else GREY)
        if j == 1 and r.get("orig_labels"):
            # the same classifier trained on the ORIGINAL model's labels/scores (ceiling.json logit_cleanfit), pooled
            ol = float(np.mean([CEIL[Path(p).stem]["logit_cleanfit"][0] for p in r["states"]]))
            text(W - LX - 22, y0 + RH - 14, f"original labels: {pct(ol)}", 19, 600, GREY, "end")
            print(f"  {r['title'][0]} classifier on original labels pooled {100 * ol:.1f}%")

yb = RY0 + len(rows) * (RH + RG) + 22
for line in CFG.get("note", []):  # the one short note: the caveat the head leaves out
    text(W / 2, yb, line, 23, 700, INK, "middle")
    yb += 32
for line in CFG.get("bottom", []):  # fine print, then source + byline
    text(W / 2, yb, line, 19, 500, GREY, "middle")
    yb += 28

alt = CFG.get("alt", "")
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure 1: recalibrated-panel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<style>html,body{{margin:0;padding:0;background:{BG}}}svg{{display:block;width:100%;max-width:1200px;height:auto}}</style>
</head><body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" aria-labelledby="f1t"><title id="f1t">{alt}</title>{"".join(o)}</svg>
</body></html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print("wrote", OUT)
