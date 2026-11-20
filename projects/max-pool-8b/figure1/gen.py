"""Figure 1 generator (Owain, room oracle-1119-max-pool-8b). Writes <blog>/assets/figures/<fig_id>/figure1.html.

Inputs (every dot and number comes from saved project files):
  - T["example"]: example json from the picker (tokens of ONE false statement and its true twin, the max-pooled
    probe's per-token scores in the original model and in one fine-tuned model, each state's own alarm line)
  - T["numbers"]: numbers json (catch rates per model and arm, written by the picker from the project's results)
Wording lives in the text json, so variants can be gated without touching the drawing.

Usage: python3 gen.py text_vNN.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = json.loads(Path(sys.argv[1]).read_text())
# no ready-to-ship answer before the status exists: a [FILL ...] slot blocks the real render (preview: FIG1_PREVIEW=1)
if "[FILL" in json.dumps(T) and not __import__("os").environ.get("FIG1_PREVIEW"):
    sys.exit("text json still holds [FILL ...]: set FIG1_PREVIEW=1 for a preview render only")
OUT = (HERE / Path(T["blog"]).expanduser()) / "assets/figures" / T["fig_id"] / "figure1.html"  # relative = from this dir
D = json.loads((HERE / T["example"]).read_text())
N = json.loads((HERE / T["numbers"]).read_text())


def pct(x):
    """One rule for every percentage: one decimal (the post's precision; 99.7% must not read as 100%), 0 stays 0%."""
    if x == 0:
        return "0%"
    return f"{100 * x:.1f}%"


AL = D.get("state_alias", {})
S0, S1 = D["states"][AL.get(T["state_clean"], T["state_clean"])], D["states"][AL.get(T["state_ft"], T["state_ft"])]
ftok, ttok = D["false_tokens"], D["true_tokens"]
n_shared = next(k for k, (a, b) in enumerate(zip(ftok, ttok)) if a != b)  # shared words before the answer
# the probe skips token 0; scores cover tokens 1..n-1
assert len(S0["false"]) == len(ftok) - 1 == len(S1["false"]), (len(S0["false"]), len(ftok))
for S in (S0, S1):  # the shared-prefix fact: identical scores on the shared words (up to rounding)
    assert all(abs(a - b) < 0.1 for a, b in zip(S["false"][:n_shared - 1], S["true"][:n_shared - 1])), S
rel0 = [s - S0["line"] for s in S0["false"]]
rel1 = [s - S1["line"] for s in S1["false"]]
top0 = max(range(len(rel0)), key=lambda k: rel0[k])
top1 = max(range(len(rel1)), key=lambda k: rel1[k])
caught0, caught1 = rel0[top0] > 0, rel1[top1] > 0
assert caught0, "example must be caught by the original model's probe"
assert caught1 == T["example_caught_after"], ("wording says caught/missed after the fine-tune", caught1)
top_shared = top1 + 1 < n_shared
assert top_shared == T["top_after_shared"], ("wording says the top token after is a shared token", top1)
# the head is picked by what the example shows (frozen variants in the text json), never typed per run
CASE = "caught" if caught1 else ("shared_missed" if top_shared else "answer_missed")


def word_of(t):
    """The whole word that token t belongs to (word pieces joined; a full stop stays itself)."""
    a = t
    while a > 0 and not ftok[a].startswith(" ") and ftok[a].strip() not in ".,!?":
        a -= 1
    b = t + 1
    while b < len(ftok) and not ftok[b].startswith(" ") and ftok[b].strip() not in ".,!?":
        b += 1
    return "".join(ftok[a:b]).strip()


WORDS = {"top0": word_of(top0 + 1), "top1": word_of(top1 + 1)}
T["head"] = [h.format(**WORDS) for h in T["heads"][CASE]]
T["col1_sub"] = T["col1_subs"][CASE].format(**WORDS)
T["col0_sub"] = T["col0_sub"].format(**WORDS)
if T.get("expect"):
    got = {k: pct(N[k]) for k in T["expect"]}
    assert got == T["expect"], (got, T["expect"])

INK, GREY, MUTE = "#1d1d1f", "#5b6068", "#b9bec6"
ORANGE, ORANGE_F, RED = "#c9731d", "#f2c48d", "#d23f31"
BLUE = "#4a72a8"
BG, CARD_A, CARD_B, KEY = "#fbfaf7", "#f1f0ec", "#fbebe6", "#fbebe6"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
W = 1200
o = []
add = o.append
esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
halo = lambda bg: f' stroke="{bg}" stroke-width="7" paint-order="stroke"'
# Inter advance widths (em) for a rough text measure; enough to place words under their dots
ADV = {**{c: 0.56 for c in "abcdeghknopqsuvxyz"}, **{c: 0.26 for c in "fijlrt"}, "m": 0.86, "w": 0.76, " ": 0.26,
       ".": 0.26, ",": 0.26, "'": 0.2, **{c: 0.66 for c in "ABCDEFGHJKLNOPQRSTUVXYZ"}, "I": 0.26, "M": 0.82,
       "W": 0.92, "ô": 0.56}
tw = lambda s, size: sum(ADV.get(ch, 0.56) for ch in s) * size


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


# ---------- kicker (the post's question + answer), head, subtitle ----------
y = 48
for k, line in enumerate(T.get("kicker", [])):  # last kicker line = the answer, bold
    last = k == len(T["kicker"]) - 1
    text(W / 2, y, line, 23, 750 if last else 500, INK if last else GREY, "middle")
    y += 32
y += 20 if T.get("kicker") else 8
for line in T["head"]:
    text(W / 2, y, line, 35, 800, INK, "middle")
    y += 44
y += 4
for k, line in enumerate(T.get("sub", [])):
    text(W / 2, y, line, 24, 500, GREY, "middle")
    y += 34
y += T.get("gap_after_head", 14)

# ---------- example row: the model, the pair of statements, the probe ----------
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
    add(f'<rect x="172" y="{yy}" width="{T.get("chip_w", 620)}" height="58" rx="20" fill="{fill}" stroke="{col}" '
        'stroke-width="2.4"/>')
    add(f'<rect x="184" y="{yy + 12}" width="70" height="34" rx="10" fill="{col}"/>')
    text(219, yy + 37, lab, 22, 800, "#ffffff", "middle")
    text(268, yy + 39, s, 25, 500)
for k, line in enumerate(T["probe_note"]):
    text(172 + T.get("chip_w", 620) + 24, EX_Y + 26 + 34 * k, line, 24, 750 if k == 0 else 500, INK if k == 0 else GREY)
y = EX_Y + 162

# ---------- cards: original | fine-tuned ----------
CARD_Y, CARD_H = y, T.get("card_h", 520)
LX, RX = (20, 584), (616, 1180)
add(f'<rect x="{LX[0]}" y="{CARD_Y}" width="{LX[1] - LX[0]}" height="{CARD_H}" rx="22" fill="{CARD_A}"/>')
add(f'<rect x="{RX[0]}" y="{CARD_Y}" width="{RX[1] - RX[0]}" height="{CARD_H}" rx="22" fill="{CARD_B}"/>')
text(sum(LX) / 2, CARD_Y + 46, T["col0_title"], 30, 800, INK, "middle")
text(sum(RX) / 2, CARD_Y + 46, T["col1_title"], 30, 800, RED, "middle")
text(sum(LX) / 2, CARD_Y + 80, T["col0_note"], 24, 500, GREY, "middle")
text(sum(RX) / 2, CARD_Y + 80, T["col1_note"], 24, 500, GREY, "middle")
SUB_Y = CARD_Y + 126
FS = T.get("word_size", 24)  # the sentence is the x axis: each dot sits above its word
YT, Y0 = CARD_Y + 160, CARD_Y + 316  # plot band (top, bottom)
lo = min(*rel0, *rel1)
hi = max(*rel0, *rel1, 0.0)
pad = 0.06 * (hi - lo)
lo, hi = lo - pad, hi + pad
ys = lambda v: Y0 - (v - lo) / (hi - lo) * (Y0 - YT)
GAP = 12  # px between word slots


def card(c, X, rel, top, caught, sub, bgc, top_label, big, big_label, seeds):
    x0 = X[0] + 28
    text(x0, SUB_Y, sub, 26, 750, INK)
    words = [w.strip() for w in ftok]
    wid = [tw(w, FS) for w in words]
    plot_w = X[1] - X[0] - 56
    # a token without a leading space (a word piece, the full stop) sits flush against the token before it
    gaps = [0.0] + [GAP if ftok[k].startswith(" ") else (7.0 if ftok[k].strip() in ".,!?" else 1.5) for k in range(1, len(ftok))]
    x = x0 + (plot_w - sum(wid) - sum(gaps)) / 2
    cxs = []
    for w, g in zip(wid, gaps):
        x += g
        cxs.append(x + w / 2)
        x += w
    WY = Y0 + 44
    # the shared words: same score in the true sentence (bracket under the words)
    sx0, sx1 = cxs[1] - wid[1] / 2, cxs[n_shared - 1] + wid[n_shared - 1] / 2
    by = WY + 14
    add(f'<path d="M{sx0:.1f},{by}v8H{sx1:.1f}v-8" fill="none" stroke="{BLUE}" stroke-width="2.4"/>')
    for k, p in enumerate(T["shared_label"].split("|")):
        text((sx0 + sx1) / 2, by + 32 + 27 * k, p, 22, 600, BLUE, "middle")
    # the alarm line, labelled at its right end
    ly = ys(0)
    add(f'<line x1="{x0}" y1="{ly:.1f}" x2="{x0 + plot_w + 6}" y2="{ly:.1f}" stroke="{INK}" stroke-width="3"/>')
    text(x0, ly - 10, T["line_label"], 22, 700, INK, "start", halo(bgc))  # label at the line's left end
    for i, w in enumerate(words):
        cx = cxs[i]
        is_ans = i >= n_shared
        if i - 1 == top:  # the highest word: a pill behind it, red when it crosses the line
            add(f'<rect x="{cx - wid[i] / 2 - 6:.1f}" y="{WY - 27}" width="{wid[i] + 12:.1f}" height="37" rx="10" '
                f'fill="{RED if caught else GREY}"/>')
        col = "#ffffff" if i - 1 == top else (MUTE if i == 0 else (ORANGE if is_ans else INK))
        text(cx, WY, w, FS, 750 if is_ans or i - 1 == top else 500, col, "middle")
        if i == 0:
            continue  # the probe skips the first token
        v = rel[i - 1]
        add(f'<line x1="{cx:.1f}" y1="{ys(v):.1f}" x2="{cx:.1f}" y2="{WY - 30}" stroke="{MUTE}" stroke-width="1.2"/>')
        r = 11 if i - 1 == top else 8
        fill = RED if v > 0 else (ORANGE if is_ans else GREY)
        add(f'<circle cx="{cx:.1f}" cy="{ys(v):.1f}" r="{r}" fill="{fill}"/>')
    # the catch rate: big number, label, each run
    NY = CARD_Y + CARD_H - 46
    bell(x0 + 2, NY - 36, 1.1, RED if caught else MUTE, rings=caught, slash=not caught, bg=bgc)
    text(x0 + 58, NY, big, 46, 800, INK if c == 0 else RED)
    bw = sum({"%": 0.86, ".": 0.30, "1": 0.45}.get(ch, 0.64) for ch in big) * 46
    text(x0 + 58 + bw + 10, NY, big_label, 21, 500, GREY)
    if seeds:
        text(x0 + 58, NY + 32, seeds, 21, 500, GREY)


def seeds_str(spec):
    return T["seeds_fmt"].format(" · ".join(pct(N[k]) for k in spec)) if spec else ""


card(0, LX, rel0, top0, caught0, T["col0_sub"], CARD_A, T["top0_label"], pct(N[T["big0"]]), T["big_label"], "")
card(1, RX, rel1, top1, caught1, T["col1_sub"], CARD_B, T["top1_label"], pct(N[T["big1"]]), T["big_label"],
     seeds_str(T.get("seeds1")))
add(f'<path d="M{LX[1] + 2},{CARD_Y + 238}L{RX[0] - 6},{CARD_Y + 238}" stroke="{INK}" stroke-width="4" '
    'marker-end="url(#ahk)"/>')
y = CARD_Y + CARD_H + T.get("gap_after_cards", 48)

# ---------- notes under the cards: the other size and the control, numbers from the numbers json ----------
for line in T.get("notes", []):
    text(W / 2, y, line.format(**{k: pct(v) for k, v in N.items() if isinstance(v, float)}), 24, 500, INK, "middle")
    y += 36
y += T.get("gap_after_notes", 14)

# ---------- foot ----------
for line in T["foot"]:
    text(W / 2, y, line, 22, 500, GREY, "middle")
    y += 30
H = int(y + 6)

defs = ('<defs><marker id="ahk" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">'
        f'<path d="M0,0L10,5L0,10z" fill="{INK}"/></marker></defs>')
alt = T["alt"].format(false_text=D["false_text"].rstrip("."), true_text=D["true_text"].rstrip("."),
                      **{k: pct(v) for k, v in N.items() if isinstance(v, float)})
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
(HERE / f"alt_{T['fig_id']}.txt").write_text(alt + "\n")
print("wrote", OUT, "H", H, "| top word before", repr(ftok[top0 + 1]), "%.2f" % rel0[top0], "after",
      repr(ftok[top1 + 1]), "%.2f" % rel1[top1], "| caught", caught0, caught1)
