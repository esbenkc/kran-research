"""Figure 1 for panel-fine-tune-8b (Owain). Writes <blog>/assets/figures/<fig id>/figure1.html.

Every bar, line and number comes from saved project files:
  - the example: fig1_data.json from pick.py (per-readout z scores of ONE false statement and its true twin,
    the panel's alarm line per model state)
  - the catch rates: <project>/data/results/results.json (arms.<arm>.monitors.panelmax.catch, and the fix
    arms.<arm>.exploratory_post_hoc.panelmax_rescaled.catch)
  - the earlier 1.5B numbers: results.json of projects/panel-fine-tune at a pinned commit of
    swarm/theory-1022-panel-max-attack (git show), asserted equal to the published 99% / 3.7% / 64%.
Wording lives in a text file (text_vNN.json), so variants can be A/B-gated without touching the drawing.

Usage: python gen.py text_vNN.json
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = json.loads(Path(sys.argv[1]).read_text())
PROJ = Path(T["project"]).expanduser()
OUT = Path(T["blog"]).expanduser() / "assets/figures" / T["fig_id"] / "figure1.html"

D = json.loads(Path(T["example"]).expanduser().read_text())
RES = {"main": json.loads((PROJ / "data/results/results.json").read_text())["arms"],
       "ref": json.loads(subprocess.run(["git", "-C", str(Path(T["ref_repo"]).expanduser()), "show", T["ref_results"]],
                                        check=True, capture_output=True, text=True).stdout)["arms"]}


def pct(x):
    """One rule for every percentage: one decimal under 9.5%, whole percent above."""
    return f"{100 * x:.1f}%" if x < 0.095 else f"{100 * x:.0f}%"


def pct_main(x):
    """The headline catch keeps the post's one decimal (TLDR: 16.7%)."""
    return f"{100 * x:.1f}%"


def catch(src, arm, fix=False):
    a = RES[src][arm]
    if fix:  # 1.5B: exploratory_post_hoc.panelmax_rescaled; 8B: monitors.panelmax_rescaled (DEVIATIONS D1.2)
        f = a.get("exploratory_post_hoc", {}).get("panelmax_rescaled") or a["monitors"].get("panelmax_rescaled")
        return None if f is None else f["catch"]
    return a["monitors"]["panelmax"]["catch"]


assert [pct(catch("ref", "clean")), pct(catch("ref", "panel")), pct(catch("ref", "panel", True))] == \
    ["99%", "3.7%", "64%"]  # the published 1.5B post
TABLE = []
for r in T["table"]:
    vals = [catch(r["src"], "clean"), catch(r["src"], r["arm"]), catch(r["src"], r["arm"], True) if r["fix"] else None]
    TABLE.append((r["label"], r.get("note", ""), vals, r.get("key", False)))
c_clean, c_main = catch("main", "clean"), catch("main", T["arm_main"])
if T.get("expect"):  # numbers quoted in the wording must equal the data
    got = {"clean": pct(c_clean), "main": pct_main(c_main)}
    assert got == T["expect"], (got, T["expect"])

LAYERS = D["layers"]  # panel readouts; the last one is the final block read after the final norm
C0 = D["states"][T["state_clean"]]
C1 = D["states"][T["example_state"]]
caught0 = C0["panel_score_false"] > C0["panel_line"]
caught1 = C1["panel_score_false"] > C1["panel_line"]
assert caught0, "example must be caught by the original panel"
assert C0["right_first_margin"] > 0 and C1["right_first_margin"] > 0, "outputs kept: right country first"
assert D["example_pattern"] == D["modal"], "example must show the commonest pattern"
assert caught1 == T["example_caught_after"], "wording says the example is caught/missed after the fine-tune"
# each probe alone, with its own 1% line (the dot row): which of the 20 still ring on the example
RING0 = [zf > l for zf, l in zip(C0["z_false"], C0["own_line"])]
RING1 = [zf > l for zf, l in zip(C1["z_false"], C1["own_line"])]
assert [sum(RING0), sum(RING1)] == T["rings_expect"], ([sum(RING0), sum(RING1)], T["rings_expect"])
RANGE1 = (min(L for L, r in zip(D["layers"], RING1) if r), max(L for L, r in zip(D["layers"], RING1) if r))
assert list(RANGE1) == T["rings_range"], (RANGE1, T["rings_range"])

INK, GREY, MUTE = "#1d1d1f", "#5b6068", "#b9bec6"
ORANGE, ORANGE_F, RED = "#c9731d", "#f2c48d", "#d23f31"
BLUE, BLUE_F = "#4a72a8", "#c5d6ee"
BG, CARD_A, CARD_B, KEY = "#fbfaf7", "#f1f0ec", "#fbebe6", "#fbebe6"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
W = 1200

o = []
add = o.append
esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=24, weight=500, fill=INK, anchor="start", extra=""):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{extra}>{esc(s)}</text>')


def bell(x, y, scale, col, rings=True, slash=False, bg=BG):
    g = [f'<g transform="translate({x},{y}) scale({scale})">',
         f'<path d="M18 2c-7 0-11 5-11 12v8l-4 6h30l-4-6v-8c0-7-4-12-11-12z" fill="{col}"/>',
         f'<circle cx="18" cy="31.5" r="3.6" fill="{col}"/>']
    if rings:
        g.append(f'<path d="M3 6q-4 6 0 13M-3 3q-6 9 0 19M33 6q4 6 0 13M39 3q6 9 0 19" fill="none" '
                 f'stroke="{col}" stroke-width="2.6" stroke-linecap="round"/>')
    if slash:
        g.append(f'<line x1="0" y1="0" x2="36" y2="36" stroke="{bg}" stroke-width="6"/>'
                 f'<line x1="0" y1="0" x2="36" y2="36" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
    g.append("</g>")
    add("".join(g))


halo = lambda bg: f' stroke="{bg}" stroke-width="7" paint-order="stroke"'

# ---------- kicker (the post's question + answer), head, subtitle ----------
y = 48
for k, line in enumerate(T.get("kicker", [])):  # last kicker line = the answer, bold
    last = k == len(T["kicker"]) - 1
    text(W / 2, y, line, 24, 750 if last else 500, INK if last else GREY, "middle")
    y += 34
y += 26 if T.get("kicker") else 8
for line in T["head"]:
    text(W / 2, y, line, 36, 800, INK, "middle")
    y += 46
y += 4
for k, line in enumerate(T["sub"]):
    text(W / 2, y, line, 24, 700 if k == 0 else 500, INK if k == 0 else GREY, "middle")
    y += 34
y += T.get("gap_after_head", 20)

# ---------- example row: model, the pair of statements, the panel ----------
EX_Y = y
add(f'<g transform="translate(40,{EX_Y}) scale(1.15)">'
    f'<line x1="40" y1="2" x2="40" y2="16" stroke="{INK}" stroke-width="3"/>'
    f'<circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
    f'<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
    f'<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
    f'<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="{INK}" stroke-width="3"/>'
    '<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
    '<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
    '<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
    '<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')
text(86, EX_Y + 118, T["model_label"], 24, 700, GREY, "middle")
for k, (s, lab, col, fill) in enumerate([(D["true_text"], "true", BLUE, "#eef3fb"),
                                          (D["false_text"], "false", ORANGE, "#fdf1e3")]):
    yy = EX_Y + 4 + 70 * k
    add(f'<rect x="172" y="{yy}" width="596" height="58" rx="20" fill="{fill}" stroke="{col}" stroke-width="2.4"/>')
    add(f'<rect x="184" y="{yy + 12}" width="70" height="34" rx="10" fill="{col}"/>')
    text(219, yy + 37, lab, 22, 800, "#ffffff", "middle")
    text(268, yy + 39, s, 25, 500)
for k, line in enumerate(T["panel_note"]):
    text(786, EX_Y + 26 + 34 * k, line, 24, 750 if k == 0 else 500, INK if k == 0 else GREY)
y = EX_Y + 162

# ---------- cards: original | fine-tuned against the panel ----------
CARD_Y, CARD_H = y, T.get("card_h", 585)
LX, RX = (20, 572), (628, 1180)
add(f'<rect x="{LX[0]}" y="{CARD_Y}" width="{LX[1] - LX[0]}" height="{CARD_H}" rx="22" fill="{CARD_A}"/>')
add(f'<rect x="{RX[0]}" y="{CARD_Y}" width="{RX[1] - RX[0]}" height="{CARD_H}" rx="22" fill="{CARD_B}"/>')
add(f'<path d="M{LX[1] + 6},{CARD_Y + 230}L{RX[0] - 14},{CARD_Y + 230}" stroke="{INK}" stroke-width="5" '
    'marker-end="url(#ahk)"/>')
text(sum(LX) / 2, CARD_Y + 46, "Original model", 30, 800, INK, "middle")
text(sum(RX) / 2, CARD_Y + 46, "Fine-tuned against the panel", 30, 800, RED, "middle")
text(sum(RX) / 2, CARD_Y + 80, T["kept_note"], 24, 500, GREY, "middle")

COLS, CW = [LX[0] + 30, RX[0] + 30], 380
SUB_Y = CARD_Y + 126
Y0, YT = CARD_Y + 330, CARD_Y + 156
ZMAX = 1.06 * max(*C0["z_false"], *C1["z_false"], *C1["z_true"], C1["panel_line"], C0["panel_line"])
ZMIN = min(0.0, *C0["z_true"], *C1["z_true"])
ys = lambda z: Y0 - ((z - ZMIN) / (ZMAX - ZMIN)) * (Y0 - YT)
n = len(LAYERS)
PAIR_GAP = 0.45  # gap between layer pairs, in bar widths
BW = CW / (2 * n + (n - 1) * PAIR_GAP)
x_of = lambda c, i: COLS[c] + i * BW * (2 + PAIR_GAP)


def bars(c, zt, zf):
    yz = ys(0)
    for i, (vt, vf) in enumerate(zip(zt, zf)):
        for k, (v, f, s) in enumerate([(vt, BLUE_F, BLUE), (vf, ORANGE_F, ORANGE)]):
            x = x_of(c, i) + k * BW
            add(f'<rect x="{x:.1f}" y="{min(ys(v), yz):.1f}" width="{BW:.1f}" height="{abs(yz - ys(v)):.1f}" '
                f'fill="{f}" stroke="{s}" stroke-width="1.2"/>')
    add(f'<line x1="{COLS[c]}" y1="{yz:.1f}" x2="{COLS[c] + CW:.1f}" y2="{yz:.1f}" stroke="{INK}" stroke-width="2"/>')
    text(x_of(c, 0) + BW, Y0 + 34, str(LAYERS[0]), 24, 600, GREY, "middle")
    text(x_of(c, n - 1) + BW, Y0 + 34, str(T["last_layer"]), 24, 600, GREY, "middle")
    text(COLS[c] + CW / 2, Y0 + 34, "layer", 24, 500, GREY, "middle")


def hline(c, z, col, label, w=3.5, dash="", dy=0, weight=700, bgc=BG):
    """A line across the chart, labelled at its right end (practical-charts rule 9); label lines split on '|'."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{COLS[c] - 8:.1f}" y1="{ys(z):.1f}" x2="{COLS[c] + CW + 8:.1f}" y2="{ys(z):.1f}" stroke="{col}" '
        f'stroke-width="{w}" stroke-linecap="round"{d}/>')
    parts = label.split("|")
    y0 = ys(z) + 8 - 14 * (len(parts) - 1) + dy
    for k, p in enumerate(parts):
        text(COLS[c] + CW + 18, y0 + 28 * k, p, 24, weight, col, "start", halo(bgc))


def dots(c, rings, label, bgc):
    """Each probe alone, with its own 1% line: filled dot = it still rings on the example, hollow = silent."""
    DY = Y0 + 70
    for i, r in enumerate(rings):
        cx = x_of(c, i) + BW
        if r:
            add(f'<circle cx="{cx:.1f}" cy="{DY}" r="7" fill="{INK}"/>')
        else:
            add(f'<circle cx="{cx:.1f}" cy="{DY}" r="6" fill="{bgc}" stroke="{MUTE}" stroke-width="2.4"/>')
    for k, p in enumerate(label.split("|")):
        text(COLS[c], DY + 40 + 30 * k, p, 24, 600 if k == 0 else 500, INK if k == 0 else GREY)


def number(c, big, col, ring, label, bgc):
    NY = CARD_Y + CARD_H - 30
    x = COLS[c]
    bell(x + 2, NY - 44, 1.2, RED if ring else MUTE, rings=ring, slash=not ring, bg=bgc)
    text(x + 62, NY, big, 46, 800, col)
    wid = sum({"%": 0.86, ".": 0.30, "1": 0.45}.get(ch, 0.64) for ch in big) * 46
    parts = label.split("|")
    for k, p in enumerate(parts):
        text(x + 62 + wid + 14, NY - 13 * (len(parts) - 1) + 27 * k, p, 24, 500, GREY)


# card 0: original model
text(COLS[0], SUB_Y, T["col0_title"], 27, 750)
bars(0, C0["z_true"], C0["z_false"])
hline(0, C0["panel_line"], INK, T["line_label"], bgc=CARD_A)
dots(0, RING0, T["dots0_label"], CARD_A)
number(0, pct(c_clean), INK, True, T["num_label"], CARD_A)

# card 1: the fine-tune, both lines
text(COLS[1], SUB_Y, T["col1_title"], 27, 750)
bars(1, C1["z_true"], C1["z_false"])
sep = ys(C0["panel_line"]) - ys(C1["panel_line"])  # px between the two lines; push labels apart if close
push = max(0.0, 64 - abs(sep)) / 2
up = 1 if sep >= 0 else -1
hline(1, C0["panel_line"], GREY, T["line0_label"], 2.5, "7 6", dy=up * push, weight=600, bgc=CARD_B)
hline(1, C1["panel_line"], RED if not caught1 else INK, T["line1_label"], dy=-up * push, bgc=CARD_B)
dots(1, RING1, T["dots1_label"], CARD_B)
number(1, pct_main(c_main), RED if c_main < 0.2 else INK, c_main >= 0.2, T["num_label"], CARD_B)

y = CARD_Y + CARD_H + T.get("after_cards", 58)

# ---------- table: the panel's catch rate, both sizes, the control and the fix (rule 13) ----------
if TABLE:
    text(40, y, T["table_title"], 26, 750)
    y += 44
    CX = [640, 900, 1160]  # right edges of the number columns
    for k, h in enumerate(T["table_cols"]):
        text(CX[k], y, h, 22, 700, GREY, "end")
    y += 14
    add(f'<line x1="40" y1="{y}" x2="1160" y2="{y}" stroke="{MUTE}" stroke-width="1.5"/>')
    for lab, note, vals, key in TABLE:
        RH = 52
        if key:
            add(f'<rect x="28" y="{y + 4}" width="1144" height="{RH - 6}" rx="10" fill="{KEY}"/>')
        yy = y + 37
        text(44, yy, lab + (f" ({note})" if note else ""), 24, 750 if key else 500, INK if key else GREY)
        for k, v in enumerate(vals):
            s = "–" if v is None else (pct_main(v) if key and k == 1 else pct(v))
            col = (RED if (k == 1 and v is not None and v < 0.2) else INK) if key else GREY
            text(CX[k], yy, s, 24, 800 if key else 600, col, "end", ' style="font-variant-numeric:tabular-nums"')
        y += RH
    y += 40

# ---------- notes under the cards: the control (and any other one-line fact), numbers from results.json ----------
NOTE = {"ctrl": pct(catch("main", "ctrl_random")), "ref_clean": pct(catch("ref", "clean")),
        "ref_main": pct(catch("ref", "panel"))}
for line in T.get("notes", []):
    text(W / 2, y, line.format(**NOTE), 24, 500, INK, "middle")
    y += 36
if T.get("notes"):
    y += 18

# ---------- foot ----------
for line in T["foot"]:
    text(W / 2, y, line, 22, 500, GREY, "middle")
    y += 30
H = int(y + 6)

defs = ('<defs><marker id="ahk" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">'
        f'<path d="M0,0L10,5L0,10z" fill="{INK}"/></marker></defs>')
alt = T["alt"].format(false_text=D["false_text"].rstrip("."), true_text=D["true_text"].rstrip("."),
                      clean=pct(c_clean), main=pct_main(c_main))
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure 1: {T["fig_id"]}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<style>html,body{{margin:0;padding:0;background:{BG}}}svg{{display:block;width:100%;max-width:1200px;height:auto}}</style>
</head><body>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" role="img" aria-labelledby="f1t"><title id="f1t">{esc(alt)}</title>{defs}<rect x="0" y="0" width="{W}" height="{H}" rx="24" fill="{BG}"/>{"".join(o)}</svg>
</body></html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
(HERE / "alt.txt").write_text(alt + "\n")
print("wrote", OUT, "H", H, "| table", [(l, [None if v is None else pct(v) for v in vs]) for l, _, vs, _ in TABLE],
      "| example caught before", caught0, "after", caught1,
      "| lines %.2f -> %.2f, panel score false %.2f -> %.2f, true %.2f -> %.2f" % (
          C0["panel_line"], C1["panel_line"], C0["panel_score_false"], C1["panel_score_false"],
          C0["panel_score_true"], C1["panel_score_true"]))
