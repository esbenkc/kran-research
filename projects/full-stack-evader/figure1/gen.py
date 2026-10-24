"""Owain: Figure 1 variants for full-stack-evader. Every lamp state comes from example_rio.json (real per-statement
scores vs each probe's alarm line); every number from data/results/results.json via that file.
Run: python3 gen.py  -> v1.html ... v7.html + index.html (all variants on one page)."""
import json, html
from pathlib import Path

HERE = Path(__file__).parent
EX = json.loads((HERE / "example_rio.json").read_text())
PS = EX["per_state"]
A = [f"a__gap__lam10000__seed{s}" for s in range(3)]
B = [f"b__gap300__lam10000__seed{s}" for s in range(3)]
CTRL = "a__gap__lam0__seed0"
LAYERS = list(range(10, 29))  # country-token probes, layers 10..28

INK, GREY, LIGHT = "#1d1d1f", "#5b6068", "#8a8f98"
AMB, AMBS = "#f2a65a", "#c9731d"
RED, REDT = "#d23f31", "#fbebe6"
BG, CARD = "#fbfaf7", "#f1f0ec"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', Menlo, monospace"


def rings(states, probe, layer=None):
    """True iff the probe scores the Rio statement above its alarm line in EVERY listed state."""
    out = []
    for st in states:
        p = PS[st]
        out.append(p["tok"][str(layer)]["hit"] if probe == "tok" else p[probe]["hit"])
    assert all(out) or not any(out), (states, probe, layer, out)  # same in every seed
    return out[0]


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


A18 = pct(EX["pooled"]["a_tok18"])  # 2.5%
BMEGA = pct(EX["pooled"]["b_mega"])  # 3.3%
assert (A18, BMEGA) == ("2.5%", "3.3%")


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
    """A small alarm bell centred at (x, y). muted: grey bell with a slash."""
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


def lamp(cx, cy, r, on, fooled=False):
    if on:
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{AMB}" stroke="{AMBS}" stroke-width="2"/>'
    col = RED if fooled else "#c7c9ce"
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="{col}" stroke-width="3.5"/>'
            + (f'<line x1="{cx - r * .6}" y1="{cy - r * .6}" x2="{cx + r * .6}" y2="{cy + r * .6}" stroke="{RED}" stroke-width="3"/>'
               f'<line x1="{cx - r * .6}" y1="{cy + r * .6}" x2="{cx + r * .6}" y2="{cy - r * .6}" stroke="{RED}" stroke-width="3"/>' if fooled else ""))


def mega_lamp(x, y, w, h, on, fooled=False):
    if on:
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{AMB}" stroke="{AMBS}" stroke-width="2"/>'
    col = RED if fooled else "#c7c9ce"
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="#ffffff" stroke="{col}" stroke-width="3.5"/>'
    if fooled:
        cx, cy, r = x + w / 2, y + h / 2, h * .32
        s += (f'<line x1="{cx - r}" y1="{cy - r}" x2="{cx + r}" y2="{cy + r}" stroke="{RED}" stroke-width="3"/>'
              f'<line x1="{cx - r}" y1="{cy + r}" x2="{cx + r}" y2="{cy - r}" stroke="{RED}" stroke-width="3"/>')
    return s


def statement_pill(x, y, w=600, size=27):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="22" fill="#fdf1e3" stroke="{AMBS}" stroke-width="2.4"/>'
            + t(x + 24, y + 41, "The city of Rio de Janeiro is in Japan.", size)
            + t(x + w - 22, y + 41, "false", 24, 800, AMBS, "end"))


def brazil_bubble(x, y, label="still says: Brazil"):
    w = 26 + 13.2 * len(label)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="44" rx="16" fill="#ffffff" stroke="#d9d9de" stroke-width="2"/>'
            + t(x + 14, y + 31, label, 24, 600, INK))


def page(svg, w, h, title, desc):
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


DESC = ("Qwen2.5-1.5B-Instruct reads the false statement 'The city of Rio de Janeiro is in Japan.' On the original model, a "
        "probe on the country name at every layer from 10 to 28 and a megaprobe that reads all layers flag it. A fine-tune "
        "trained against the layer-18 probe silences only that probe; the probes at the other layers and the megaprobe "
        "still flag it, and the model still prefers Brazil. A fine-tune trained against the megaprobe silences only the "
        "megaprobe. Over 596 false statements and 3 runs, the target probe falls to " + A18 + " and " + BMEGA +
        "; every other probe that worked on the original model still catches at least 62%.")


# ---------------------------------------------------------------- v1: three rows of lamps (main candidate)
def v1(control=False, numbers=True):
    out = []
    out.append(robot(36, 26, 1.15))
    out.append(statement_pill(170, 34, 590))
    out.append(t(194, 136, "Qwen2.5-1.5B-Instruct reads a false statement", 24, 500, GREY))
    # legend right
    out.append(lamp(810, 58, 13, True) + t(834, 66, "flags it as false", 24, 500, INK))
    out.append(lamp(810, 104, 13, False, True) + t(834, 112, "fooled: stays silent", 24, 600, RED))
    # column header
    X0, DX, R = 384, 29, 12
    xs = {L: X0 + (L - 10) * DX for L in LAYERS}
    MX = xs[28] + 40
    hy = 196
    out.append(t((xs[10] + xs[28]) / 2, hy, "probe on the country name, at layer", 22, 500, GREY, "middle"))
    for L in (10, 18, 28):
        out.append(t(xs[L], hy + 32, str(L), 22, 700, GREY, "middle"))
    out.append(t(MX + 40, hy, "megaprobe", 22, 500, GREY, "middle"))
    out.append(t(MX + 40, hy + 32, "all layers", 22, 500, GREY, "middle"))
    rows = [("Original model", None, ["clean"], None, INK, CARD),
            ("Fine-tuned against", "the layer-18 probe", A, ("layer-18 probe", A18), RED, REDT),
            ("Fine-tuned against", "the megaprobe", B, ("megaprobe", BMEGA), RED, REDT)]
    if control:
        rows.append(("No output penalty", "(1 run)", [CTRL], None, GREY, CARD))
    y = hy + 56
    RH = 132 if not control else 112
    for (l1, l2, sts, num, col, bg) in rows:
        out.append(f'<rect x="20" y="{y}" width="1160" height="{RH - 12}" rx="20" fill="{bg}"/>')
        cy = y + (RH - 12) / 2
        if l2:
            out.append(t(44, cy - 6, l1, 26, 750, col))
            out.append(t(44, cy + 26, l2, 26, 750, col))
        else:
            out.append(t(44, cy + 9, l1, 28, 800, col))
        for L in LAYERS:
            on = rings(sts, "tok", L)
            out.append(lamp(xs[L], cy - 4, R, on, fooled=not on))
        on = rings(sts, "mega")
        out.append(mega_lamp(MX, cy - 4 - R, 80, 2 * R, on, fooled=not on))
        if sts == ["clean"]:
            out.append(t(xs[10] - 14, cy + 40, "", 1))
        if num and numbers:
            out.append(t(1166, cy - 6, num[1], 40, 800, RED, "end"))
            out.append(t(1166, cy + 22, f"{num[0]} catches,", 20, 500, GREY, "end"))
            out.append(t(1166, cy + 46, "all 596 false statements", 20, 500, GREY, "end"))
        if sts in (A, B):
            pass
        y += RH
    # bottom answer
    out.append(t(600, y + 34, "No. Each fine-tune fooled only the probe it was trained against.", 30, 800, INK, "middle"))
    out.append(t(600, y + 72, "Both still pick Brazil over Japan as the next word.", 24, 500, GREY, "middle"))
    return "".join(out), y + 96


# ---------------------------------------------------------------- v2: Apple: huge type, one strip, one gap
def v2():
    o = []
    o.append(t(60, 92, "Trained against one probe.", 56, 800, INK))
    o.append(t(60, 160, "It fooled only that one.", 56, 800, RED))
    o.append(t(60, 222, "The city of Rio de Janeiro is in Japan.", 28, 500, GREY))
    X0, DX, R = 120, 46, 17
    cy = 330
    for L in LAYERS:
        on = rings(A, "tok", L)
        o.append(lamp(X0 + (L - 10) * DX, cy, R, on, not on))
    o.append(t(X0, cy + 56, "10", 24, 700, GREY, "middle"))
    o.append(t(X0 + 8 * DX, cy + 56, "18", 24, 800, RED, "middle"))
    o.append(t(X0 + 18 * DX, cy + 56, "28", 24, 700, GREY, "middle"))
    o.append(t(X0 + 9 * DX, cy - 40, "probe on the country name, by layer", 24, 500, GREY, "middle"))
    mx = X0 + 18 * DX + 70
    o.append(mega_lamp(mx, cy - R, 110, 2 * R, rings(A, "mega")))
    o.append(t(mx + 55, cy + 56, "megaprobe", 24, 600, GREY, "middle"))
    o.append(t(60, 470, A18, 64, 800, RED))
    o.append(t(60, 508, "layer-18 probe catches, after the fine-tune", 24, 500, GREY))
    o.append(t(640, 470, "≥ 62%", 64, 800, INK))
    o.append(t(640, 508, "every other working probe catches", 24, 500, GREY))
    o.append(t(60, 572, "Qwen2.5-1.5B-Instruct, fine-tuned on all 28 blocks, outputs kept. 596 false statements, 3 runs.", 22, 500, LIGHT))
    return "".join(o), 610


# ---------------------------------------------------------------- v3: Teenage Engineering LED matrix with control row
def v3():
    o = []
    o.append(f'<rect x="20" y="20" width="1160" height="700" rx="18" fill="#e9e8e3" stroke="#cfcdc6" stroke-width="2"/>')
    o.append(t(52, 74, "PROBE BOARD", 30, 700, INK, family=MONO))
    o.append(t(52, 110, "INPUT  \u201cThe city of Rio de Janeiro is in Japan.\u201d  [FALSE]", 22, 400, GREY, family=MONO))
    X0, DX, R = 380, 30, 11
    o.append(t(X0 - 10, 168, "L10", 20, 700, GREY, family=MONO))
    o.append(t(X0 + 8 * DX, 168, "L18", 20, 700, RED, "middle", family=MONO))
    o.append(t(X0 + 18 * DX + 10, 168, "L28", 20, 700, GREY, "end", family=MONO))
    o.append(t(X0 + 18 * DX + 70, 168, "MEGA", 20, 700, GREY, "middle", family=MONO))
    o.append(t(1150, 168, "OUTPUT", 20, 700, GREY, "end", family=MONO))
    rows = [("ORIGINAL", ["clean"], "0.000"), ("VS L18", A, "≤0.005"), ("VS MEGA", B, "≤0.005"), ("NO PENALTY", [CTRL], "0.457")]
    y = 220
    for name, sts, kl in rows:
        o.append(t(52, y + 8, name, 26, 700, INK if sts != [CTRL] else GREY, family=MONO))
        for L in LAYERS:
            on = rings(sts, "tok", L)
            o.append(lamp(X0 + (L - 10) * DX, y, R, on, not on))
        on = rings(sts, "mega")
        o.append(mega_lamp(X0 + 18 * DX + 40, y - R, 60, 2 * R, on, not on))
        o.append(t(1150, y + 8, f"KL {kl}", 24, 700, RED if kl == "0.457" else INK, "end", family=MONO))
        y += 90
    o.append(t(52, y + 10, "Bar for unchanged outputs: KL 0.05.", 22, 400, GREY, family=MONO))
    o.append(t(52, y + 70, "No. Each fine-tune fools only its own target.", 30, 700, INK, family=MONO))
    o.append(t(52, y + 108, "Without the output penalty, all go dark, and the outputs move 9x the bar.", 22, 400, GREY, family=MONO))
    return "".join(o), 740


# ---------------------------------------------------------------- v4: chat style, two cards (inspiration 02)
def v4():
    o = []
    cards = [(20, "Original model", INK, CARD, ["clean"]), (612, "Fine-tuned against layer 18", RED, REDT, A)]
    for x, title, col, bg, sts in cards:
        o.append(f'<rect x="{x}" y="20" width="568" height="700" rx="26" fill="{bg}"/>')
        o.append(t(x + 284, 76, title, 32, 800, col, "middle"))
        o.append(f'<rect x="{x + 70}" y="110" width="476" height="96" rx="24" fill="#3478f6"/>')
        o.append(t(x + 94, 150, "The city of Rio de Janeiro", 26, 500, "#ffffff"))
        o.append(t(x + 94, 184, "is in Japan.", 26, 500, "#ffffff"))
        probes = [("layer-17 probe", rings(sts, "tok", 17)), ("layer-18 probe", rings(sts, "tok", 18)),
                  ("layer-19 probe", rings(sts, "tok", 19)), ("megaprobe", rings(sts, "mega"))]
        yy = 262
        for name, on in probes:
            o.append(bell(x + 70, yy, 0.9, RED if on else LIGHT, muted=not on))
            o.append(t(x + 110, yy + 9, name, 26, 600, INK))
            o.append(t(x + 540, yy + 9, "false!" if on else "silent", 26, 800, (AMBS if on else RED), "end"))
            yy += 66
        o.append(robot(x + 34, yy + 4, 0.95))
        o.append(f'<rect x="{x + 130}" y="{yy + 18}" width="390" height="62" rx="20" fill="#ffffff"/>')
        o.append(t(x + 150, yy + 58, "next word: Brazil, not Japan", 26, 600, INK))
    o.append(t(600, 772, "No. The fine-tune fooled only the probe it was trained against.", 30, 800, INK, "middle"))
    return "".join(o), 800


# ---------------------------------------------------------------- v5: one model, before -> after arrow, lamps big
def v5():
    o = []
    o.append(robot(40, 28, 1.1))
    o.append(statement_pill(160, 40, 590))
    o.append(t(790, 80, "Does a fine-tune that keeps", 26, 700, INK))
    o.append(t(790, 114, "the outputs fool every probe?", 26, 700, INK))
    X0, DX, R = 120, 46, 17
    def strip(y, sts, title, col, sub):
        s = t(60, y - 50, title, 30, 800, col) + t(60 + 14 * len(title) + 34, y - 50, sub, 24, 500, GREY)
        for L in LAYERS:
            on = rings(sts, "tok", L)
            s += lamp(X0 + (L - 10) * DX, y, R, on, not on)
        on = rings(sts, "mega")
        s += mega_lamp(X0 + 18 * DX + 50, y - R, 110, 2 * R, on, not on)
        return s
    o.append(strip(260, ["clean"], "Before", INK, "all 20 probes flag it"))
    o.append(t(X0, 316, "layer 10", 22, 600, GREY, "middle") + t(X0 + 8 * DX, 316, "18", 22, 800, RED, "middle")
             + t(X0 + 18 * DX, 316, "28", 22, 600, GREY, "middle") + t(X0 + 18 * DX + 105, 316, "megaprobe", 22, 600, GREY, "middle"))
    o.append(f'<path d="M{X0 + 8 * DX},350 L{X0 + 8 * DX},392" stroke="{RED}" stroke-width="4"/>')
    o.append(t(X0 + 8 * DX + 18, 380, "fine-tune trained against the layer-18 probe", 24, 700, RED))
    o.append(strip(480, A, "After", RED, "only layer 18 is fooled; still picks Brazil"))
    o.append(t(60, 572, A18, 52, 800, RED) + t(220, 568, "caught by the layer-18 probe after the fine-tune", 24, 500, GREY))
    o.append(t(60, 612, "(596 false statements, 3 runs; every other working probe still catches ≥ 62%)", 22, 500, LIGHT))
    o.append(t(600, 676, "No. It fooled only the probe it was trained against.", 30, 800, INK, "middle"))
    return "".join(o), 710


def v6():
    o = []
    o.append(t(60, 86, "Trained against one probe,", 52, 800, INK))
    o.append(t(60, 148, "a fine-tune fools only that one.", 52, 800, RED))
    o.append(t(60, 204, "The city of Rio de Janeiro is in Japan.", 28, 500, GREY))
    X0, DX, R = 280, 34, 13
    mx = X0 + 18 * DX + 44
    o.append(t(X0 + 9 * DX, 262, "probe on the country name, at layer", 22, 500, GREY, "middle"))
    o.append(t(mx + 50, 262, "megaprobe", 22, 500, GREY, "middle"))
    for L in (10, 18, 28):
        o.append(t(X0 + (L - 10) * DX, 294, str(L), 22, 700, GREY, "middle"))
    y = 350
    for name, sts, num in (("Original", ["clean"], None), ("vs layer 18", A, A18), ("vs megaprobe", B, BMEGA)):
        o.append(t(60, y + 9, name, 28, 800, INK if num is None else RED))
        for L in LAYERS:
            on = rings(sts, "tok", L)
            o.append(lamp(X0 + (L - 10) * DX, y, R, on, not on))
        on = rings(sts, "mega")
        o.append(mega_lamp(mx, y - R, 100, 2 * R, on, not on))
        if num:
            o.append(t(1150, y + 14, num, 40, 800, RED, "end"))
        y += 76
    o.append(t(1150, y - 10, "target probe, all 596 statements", 20, 500, GREY, "end"))
    o.append(t(60, y + 40, "Qwen2.5-1.5B-Instruct, all 28 blocks fine-tuned, outputs kept; still picks Brazil.", 22, 500, LIGHT))
    return "".join(o), y + 80


def v7():
    """v1 rebuilt: row title + pooled stat on one line, lamps full width below (no text collisions, 24 px+)."""
    o = []
    D = 66
    o.append(t(44, 58, "Fine-tuned to fool one probe, does the model fool them all?", 34, 800, INK))
    o.append(robot(36, 26 + D, 1.15))
    o.append(statement_pill(170, 34 + D, 590))
    o.append(t(194, 136 + D, "Qwen2.5-1.5B-Instruct reads a false statement", 24, 500, GREY))
    o.append(lamp(810, 58 + D, 13, True) + t(834, 66 + D, "flags it as false", 24, 500, INK))
    o.append(lamp(810, 104 + D, 13, False, True) + t(834, 112 + D, "fooled: stays silent", 24, 600, RED))
    X0, DX, R = 72, 40, 15
    xs = {L: X0 + (L - 10) * DX for L in LAYERS}
    MX, MW = xs[28] + 50, 130
    hy = 200 + D
    o.append(t((xs[10] + xs[28]) / 2, hy, "country-token probe, at layer", 24, 500, GREY, "middle"))
    for L in (10, 18, 28):
        o.append(t(xs[L], hy + 32, str(L), 24, 700, GREY, "middle"))
    o.append(t(MX + MW / 2, hy, "megaprobe", 24, 500, GREY, "middle"))
    o.append(t(MX + MW / 2, hy + 32, "reads all layers", 24, 500, GREY, "middle"))
    clean_min = min([EX["pooled"]["clean"][str(L)] for L in LAYERS] + [EX["pooled"]["clean_mega"]])
    rows = [("Original model", ["clean"], ("each probe here catches ", f"{int(clean_min * 100)}%+", ""), INK, CARD),
            ("Fine-tuned to fool the layer-18 probe", A, ("caught by the layer-18 probe: ", A18, ""), RED, REDT),
            ("Fine-tuned to fool the megaprobe", B, ("caught by the megaprobe: ", BMEGA, ""), RED, REDT)]
    y = hy + 58
    for title, sts, (pre, num, post), col, bg in rows:
        o.append(f'<rect x="20" y="{y}" width="1160" height="140" rx="20" fill="{bg}"/>')
        o.append(t(44, y + 44, title, 28, 800, col))
        o.append(f'<text x="1160" y="{y + 44}" font-size="24" font-weight="500" fill="{GREY}" text-anchor="end">{html.escape(pre)}'
                 f'<tspan font-size="30" font-weight="800" fill="{col if col == RED else INK}">{html.escape(num)}</tspan></text>')
        cy = y + 96
        for L in LAYERS:
            on = rings(sts, "tok", L)
            o.append(lamp(xs[L], cy, R, on, fooled=not on))
        on = rings(sts, "mega")
        o.append(mega_lamp(MX, cy - R, MW, 2 * R, on, fooled=not on))
        y += 152
    o.append(t(1160, y + 22, "percent of 596 false statements; fine-tunes: 3 runs pooled", 22, 500, LIGHT, "end"))
    o.append(t(600, y + 74, "No. Each fine-tune fooled only the probe it was trained against.", 30, 800, INK, "middle"))
    o.append(t(600, y + 112, "Every other probe that worked still caught at least 62%. Both still pick Brazil over Japan.", 24, 500, GREY, "middle"))
    return "".join(o), y + 140


VARIANTS = {
    "v1": ("Three rows: original, vs layer 18, vs megaprobe (lamps per layer + pooled number)", lambda: v1()),
    "v1b": ("v1 plus the no-penalty control row", lambda: v1(control=True)),
    "v2": ("Apple: huge words, one strip, one gap", v2),
    "v3": ("Teenage Engineering: LED probe board with output-change readout + control", v3),
    "v4": ("Chat cards (like the owl figure): bells per probe, model still says Brazil", v4),
    "v5": ("Before -> after, one fine-tune, big lamps", v5),
    "v6": ("Apple, both fine-tunes: headline + three lamp rows", v6),
    "v7": ("v1 rebuilt: stat on the title line, bigger lamps (current candidate)", v7),
}

BLOG = Path("/Users/esben/work/theory-1021-full-stack-evader/blog.kran.ai/assets/figures/full-stack-evader-fig1")


def write_final(fn):
    svg, h = fn()
    BLOG.mkdir(parents=True, exist_ok=True)
    (BLOG / "figure1.html").write_text(page(svg, 1200, h, "Figure 1: full-stack-evader", DESC))
    print("final", h)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        write_final(VARIANTS[sys.argv[1]][1])
        sys.exit()
    gallery = []
    for k, (label, fn) in VARIANTS.items():
        svg, h = fn()
        (HERE / f"{k}.html").write_text(page(svg, 1200, h, f"Figure 1 {k}", DESC))
        gallery.append(f'<section><h2>{k}: {html.escape(label)}</h2><div class="card"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" font-family="{FONT}"><rect x="0" y="0" width="1200" height="{h}" rx="24" fill="{BG}"/>{svg}</svg></div></section>')
        print(k, h)
    (HERE / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Figure 1 variants: full-stack-evader</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>body{{font-family:Inter,sans-serif;background:#e7e6e1;margin:0;padding:24px;color:#1d1d1f}}h1{{font-size:22px}}h2{{font-size:17px;margin:28px 0 8px}}.card{{max-width:1200px}}svg{{width:100%;height:auto;display:block}}p{{max-width:900px}}</style></head><body>
<h1>Figure 1 variants: "Does a fine-tune that can edit every block hide false statements from every probe?" (blog.kran.ai/full-stack-evader)</h1>
<p>One real example in every variant: "The city of Rio de Janeiro is in Japan." (eval false statement #{EX['statement_index']}). Lamp states are its real scores against each probe's alarm line, the same in all 3 runs of each fine-tune. Numbers: results.json (layer-18 probe {A18}, megaprobe {BMEGA}, others at least 62%).</p>
{''.join(gallery)}</body></html>""")
