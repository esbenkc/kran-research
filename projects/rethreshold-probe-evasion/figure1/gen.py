"""Figure 1 variants for blog.kran.ai/rethreshold-probe-evasion.

Every number and every pile comes from the saved scores:
  kran-research/projects/rethreshold-probe-evasion/data/runs/lam{100000,0}__seed{0,1,2}.npz  (scores, y, split)
  kran-research/projects/rethreshold-probe-evasion/data/results/results.json                (pooled shares)
  kran-research/projects/rethreshold-probe-evasion/data/split.json                          (statement texts)
Piles = the 596 true and 596 false eval statements, seed 0, final step (step 0 = original model).
Lines = 2nd-highest of the 200 held-out true statements (common.threshold), seed 0.
Example = 'The city of Bangkok is in China.' (Rio de Janeiro was used by full-stack-evader's Figure 1, PR #79) (pick_example.py: meets the rule in all 3 seeds).
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
P = Path("/Users/esben/work/theory-1018-shift-spread/kran-research/projects/rethreshold-probe-evasion/data")
SP = json.load(open(P / "split.json"))
RES = json.load(open(P / "results" / "results.json"))
EX_FALSE = "The city of Bangkok is in China."
F_LAB, T_LAB = "China", "Thailand"
EX_TRUE = "The city of Bangkok is in Thailand."
IF = next(i for i, s in enumerate(SP) if s["text"] == EX_FALSE)
IT = next(i for i, s in enumerate(SP) if s["text"] == EX_TRUE)


def thr(neg, fpr=0.01):
    s = np.sort(neg)[::-1]
    return float(s[int(np.floor(fpr * (len(s) + 1))) - 1])


def load(lam, seed=0):
    z = np.load(P / "runs" / f"lam{lam}__seed{seed}.npz")
    y, sp, S = z["y"], z["split"], z["scores"].astype(np.float64)
    rf, ep, en = (sp == "refit") & (y == 0), (sp == "eval") & (y == 1), (sp == "eval") & (y == 0)
    return dict(clean_T=S[0][en], clean_F=S[0][ep], T=S[-1][en], F=S[-1][ep], t0=thr(S[0][rf]), t1=thr(S[-1][rf]),
                exF0=S[0][IF], exT0=S[0][IT], exF=S[-1][IF], exT=S[-1][IT],
                c0=float((S[0][ep] > thr(S[0][rf])).mean()), a0=float((S[-1][ep] > thr(S[0][rf])).mean()),
                a1=float((S[-1][ep] > thr(S[-1][rf])).mean()))


B, N = load(100000), load(0)
SHARE_B = RES["per_lambda"]["100000.0"]["share_pooled"]
SHARE_N = RES["per_lambda"]["0"]["share_pooled"]
print("numbers", "seed0 lines", B["t0"], B["t1"], N["t1"])
# sanity: the example's story holds in seed 0 (all seeds: pick_example.py)
assert B["exF0"] > B["t0"] and B["t1"] < B["exF"] <= B["t0"] and N["exF"] <= N["t1"]
NUM = dict(clean_catch=round(B["c0"] * 100), share_b=round(SHARE_B * 100), share_n=round(SHARE_N * 100))
print(NUM, "ex", B["exF0"], B["exF"], N["exF"])

# ---------------------------------------------------------------- style
BG = "#fbfaf7"
INK, MUTED, FAINT = "#1d1d1f", "#5b6068", "#b9bec6"
TRUE_F, TRUE_S = "#c9d8ea", "#4f74a3"     # true pile: blue
FALSE_F, FALSE_S = "#f2c48d", "#c9731d"   # false pile: orange
RED, GREEN = "#d23f31", "#2e8a57"
CARD, CARD_T = "#f1f0ec", "#fbebe6"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, size=24, w=500, fill=INK, anchor="start", halo=None, extra=""):
    h = f' stroke="{halo}" stroke-width="7" paint-order="stroke"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" '
            f'text-anchor="{anchor}"{h}{extra}>{esc(s)}</text>')


def robot(x, y, s=1.0, lab=None):
    g = (f'<g transform="translate({x},{y}) scale({s})">'
         '<line x1="40" y1="2" x2="40" y2="16" stroke="#1d1d1f" stroke-width="3"/>'
         '<circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="#1d1d1f" stroke-width="2"/>'
         '<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="#1d1d1f" stroke-width="2"/>'
         '<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="#1d1d1f" stroke-width="2"/>'
         '<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="#1d1d1f" stroke-width="3"/>'
         '<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
         '<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
         '<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
         '<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')
    if lab:
        g += T(x + 40 * s, y + 96 * s, lab, 22, 600, "#8a8f98", "middle")
    return g


def bell(cx, cy, on=True, s=1.0):
    """Alarm bell: red and ringing (on) or grey and struck through (off)."""
    c = RED if on else FAINT
    g = (f'<g transform="translate({cx},{cy}) scale({s})">'
         f'<path d="M-13,8 L-13,-3 A13,13 0 0 1 13,-3 L13,8 L17,13 L-17,13 Z" fill="{c}"/>'
         f'<circle cx="0" cy="17" r="4.5" fill="{c}"/><rect x="-2.5" y="-20" width="5" height="6" rx="2" fill="{c}"/>')
    if on:
        g += (f'<path d="M-22,-10 Q-27,0 -22,10 M22,-10 Q27,0 22,10 M-28,-15 Q-35,0 -28,15 M28,-15 Q35,0 28,15" '
              f'fill="none" stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
    else:
        g += f'<line x1="-20" y1="-18" x2="20" y2="22" stroke="{c}" stroke-width="3.5" stroke-linecap="round"/>' \
             f'<line x1="-20" y1="-18" x2="20" y2="22" stroke="{BG}" stroke-width="1.5" transform="translate(3,-3)"/>'
    return g + "</g>"


def violin(vals, cx, ymap, half_w, fill, stroke, bw=0.5, lo=None, hi=None):
    """Vertical mirrored histogram (a 'pile') of real scores; ymap(score) -> px."""
    lo = math.floor(vals.min() / bw) * bw if lo is None else lo
    hi = math.ceil(vals.max() / bw) * bw if hi is None else hi
    edges = np.arange(lo, hi + bw, bw)
    cnt, _ = np.histogram(vals, edges)
    m = cnt.max()
    pts_r, pts_l = [], []
    for k, c in enumerate(cnt):
        w = half_w * c / m
        y0, y1 = ymap(edges[k]), ymap(edges[k + 1])
        pts_r += [(cx + w, y0), (cx + w, y1)]
    pts_l = [(2 * cx - x, y) for x, y in reversed(pts_r)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts_r + pts_l) + "Z"
    return f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" stroke-linejoin="round"/>'


def hviolin(vals, cy, xmap, half_h, fill, stroke, bw=0.5, up_only=False):
    """Horizontal histogram (bell-curve pile) of real scores; xmap(score) -> px."""
    lo, hi = math.floor(vals.min() / bw) * bw, math.ceil(vals.max() / bw) * bw
    edges = np.arange(lo, hi + bw, bw)
    cnt, _ = np.histogram(vals, edges)
    m = cnt.max()
    top = []
    for k, c in enumerate(cnt):
        h = half_h * c / m
        top += [(xmap(edges[k]), cy - h), (xmap(edges[k + 1]), cy - h)]
    if up_only:
        pts = [(top[0][0], cy)] + top + [(top[-1][0], cy)]
    else:
        pts = top + [(x, 2 * cy - y) for x, y in reversed(top)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"
    return f'<path d="{d}" fill="{fill}" fill-opacity="0.72" stroke="{stroke}" stroke-width="2" stroke-linejoin="round"/>'


def hline(x1, x2, y, dash=False, color=INK, w=3.5):
    da = ' stroke-dasharray="9,7"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{w}"{da} stroke-linecap="round"/>'


def vline(x, y1, y2, dash=False, color=INK, w=3.5):
    da = ' stroke-dasharray="9,7"' if dash else ""
    return f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"{da} stroke-linecap="round"/>'


def exdot(x, y, caught):
    c = RED if caught else MUTED
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="#fff" stroke="{c}" stroke-width="4"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}"/>')


def arrow(x1, y1, x2, y2, color=INK, w=5, mid="ahk"):
    return f'<path d="M{x1},{y1}L{x2},{y2}" stroke="{color}" stroke-width="{w}" fill="none" marker-end="url(#{mid})"/>'


DEFS = ('<defs>'
        + "".join(f'<marker id="{n}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">'
                  f'<path d="M0,0L10,5L0,10z" fill="{c}"/></marker>' for n, c in (("ahk", INK), ("ahr", RED), ("ahg", FAINT)))
        + '</defs>')


def svg(body, h, title, bg=BG):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {h}" width="1200" height="{h}" '
            f'font-family="{FONT}" role="img" aria-labelledby="f1t"><title id="f1t">{esc(title)}</title>'
            f'<rect x="0" y="0" width="1200" height="{h}" rx="24" fill="{bg}"/>{DEFS}{body}</svg>')


def html(svg_str, name, bg=BG):
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>Figure 1: {name}</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">\n'
            f'<style>html,body{{margin:0;padding:0;background:{bg}}}svg{{display:block;width:100%;max-width:1200px;height:auto}}</style>\n'
            f'</head><body>\n{svg_str}\n</body></html>\n')


def bubbles(x, y, w=520, show_true=True):
    """The example pair as two statement bubbles."""
    b = ""
    rows = ([(EX_TRUE, "true", "#e8eff8", TRUE_S)] if show_true else []) + [(EX_FALSE, "false", "#fdf1e3", FALSE_S)]
    for k, (txt, tag, fill, st) in enumerate(rows):
        yy = y + k * 68
        b += f'<rect x="{x}" y="{yy}" width="{w}" height="56" rx="20" fill="{fill}" stroke="{st}" stroke-width="2.4"/>'
        b += T(x + 22, yy + 37, txt, 25, 500)
        b += T(x + w - 20, yy + 37, tag, 23, 800, st, "end")
    return b


def header(q_lines=("Train the model to slip past", "a lie probe. Does a new alarm", "line catch the lie again?")):
    b = robot(34, 34, 1.15, "Qwen2.5-1.5B")
    b += bubbles(170, 28, 540)
    for k, s in enumerate(q_lines):
        b += T(744, 58 + k * 36, s, 25, 750)
    return b


def scale(s_hi, y_hi, ppu):
    return lambda s: y_hi + (s_hi - s) * ppu


def cut_mark(cx, y, color=MUTED):
    return f'<path d="M{cx - 30},{y} l15,-8 l15,16 l15,-16 l15,8" fill="none" stroke="{color}" stroke-width="3"/>'


def ex_label(x, y, s, color, anchor="start", halo=None):
    """Example label as a small white pill next to its dot."""
    w = len(s) * 12.2 + 18
    x0 = x if anchor == "start" else x - w
    return (f'<rect x="{x0:.1f}" y="{y - 15:.1f}" width="{w:.1f}" height="30" rx="15" fill="#fff" stroke="{color}" stroke-width="1.6"/>'
            + T(x0 + w / 2, y + 7.5, s, 21, 750, color, "middle"))


# ---------------------------------------------------------------- V1: three cards, vertical piles
def old_line_catch(lam):
    """Catch rate at the OLD line after the fine-tune, per seed (all 3 seeds), as a 'lo–hi%' string."""
    v = [round(load(lam, s)["a0"] * 100) for s in (0, 1, 2)]
    return f"{min(v)}%" if min(v) == max(v) else f"{min(v)}–{max(v)}%"


def ex_pill(x, y, name, verdict, color, vcolor, anchor="start"):
    """Example label: the city's country in bold, the probe's verdict under it."""
    w = max(len(name) * 13.0, len(verdict) * 12.6) + 22 if verdict else len(name) * 13.0 + 22
    h = 58 if verdict else 32
    x0 = x if anchor == "start" else x - w
    g = f'<rect x="{x0:.1f}" y="{y - h / 2:.1f}" width="{w:.1f}" height="{h}" rx="14" fill="#fff" stroke="{color}" stroke-width="1.8"/>'
    if verdict:
        g += T(x0 + w / 2, y - 4, name, 23, 800, color, "middle") + T(x0 + w / 2, y + 21, verdict, 23, 800, vcolor, "middle")
    else:
        g += T(x0 + w / 2, y + 8, name, 23, 800, color, "middle")
    return g


def v1(answer=True):
    top = 196
    ppu = 12.5
    ym = scale(7.0, top + 312, ppu)
    yl = top + 284  # pile labels
    ymn = scale(N["t1"] + 1.5, ym(B["t0"]) + 62, ppu)
    lowest = max(ym(min(B["T"].min(), B["clean_T"].min())), ymn(min(N["T"].min(), N["F"].min())))
    lowest = max(lowest, ymn(N["exF"]) + 29, ymn(N["exT"]) + 29)  # the example pills too
    cb = int(lowest + 22)  # card bottom
    H = cb + (100 if answer else 20)
    off = 58  # pile centre offset from the column centre
    b = header()
    b += f'<rect x="20" y="{top}" width="340" height="{cb - top}" rx="22" fill="{CARD}"/>'
    b += f'<rect x="420" y="{top}" width="760" height="{cb - top}" rx="22" fill="{CARD_T}"/>'
    b += arrow(366, ym(B["t0"]), 410, ym(B["t0"]))
    b += T(190, top + 44, "Original model", 30, 800, INK, "middle")
    b += T(800, top + 44, "Fine-tuned to slip past the probe", 30, 800, RED, "middle")
    b += T(800, top + 78, "brake = a penalty that keeps the model's normal answers", 23, 600, MUTED, "middle")

    def piles(cx, Tv, Fv, ymap, hw=50, bw=0.5):
        return violin(Tv, cx - off, ymap, hw, TRUE_F, TRUE_S, bw) + violin(Fv, cx + off, ymap, hw, FALSE_F, FALSE_S, bw)

    def labels(cx):
        return T(cx - off, yl, "true", 24, 700, TRUE_S, "middle") + T(cx + off, yl, "false", 24, 700, FALSE_S, "middle")

    def pair(cx, yT, yF, caught, verdict, brazil_side="end"):
        bx = cx - off - 16 if brazil_side == "end" else cx - off + 16
        return (exdot(cx - off, yT, False) + exdot(cx + off, yF, caught)
                + ex_pill(bx, yT, T_LAB, None, TRUE_S, None, brazil_side)
                + ex_pill(cx + off + 16, yF, F_LAB, verdict, FALSE_S, RED if caught else MUTED, "start"))

    def headline(x, sub, on, num, col, small):
        g = "".join(T(x, top + 122 + 30 * k, s, 25, 700) for k, s in enumerate(sub))
        return (g + bell(x + 18, top + 190, on, 0.9)
                + T(x + 58, top + 201, num, 30, 800, col) + T(x + 58, top + 233, small, 23, 600, MUTED))

    # card 1
    cx = 185
    b += headline(42, ["The probe flags the lie"], True, f"{NUM['clean_catch']}% caught", INK, "of false statements")
    b += labels(cx)
    b += piles(cx, B["clean_T"], B["clean_F"], ym)
    b += hline(36, 344, ym(B["t0"]))
    b += T(36, ym(B["t0"]) - 10, "alarm line", 23, 700, INK, "start", CARD)
    b += pair(cx, ym(B["exT0"]), ym(B["exF0"]), True, "caught", "start")
    # card 2: with a brake
    cx = 562
    b += headline(440, ["With a brake: piles", "slide down together"], True, f"New line: {NUM['share_b']}%", RED,
                  "of lost catches won back")
    b += labels(cx)
    b += piles(cx, B["T"], B["F"], ym)
    b += hline(436, 800, ym(B["t0"]), dash=True, color=MUTED, w=3)
    b += T(436, ym(B["t0"]) - 36, "old line:", 23, 600, MUTED, "start", CARD_T)
    b += T(436, ym(B["t0"]) - 10, f"{old_line_catch(100000)} caught", 23, 600, MUTED, "start", CARD_T)
    b += hline(436, 800, ym(B["t1"]), color=RED)
    b += T(436, ym(B["t1"]) + 30, "new line", 23, 800, RED, "start", CARD_T)
    b += pair(cx, ym(B["exT"]), ym(B["exF"]), True, "caught again", "start")
    # card 3: no brake (axis cut, same px per score unit)
    cx = 1005
    b += headline(830, ["No brake: piles mix,", "no line separates them"], False, f"New line: {NUM['share_n']}%", INK,
                  "of lost catches won back")
    b += labels(cx)
    b += hline(830, 1166, ym(B["t0"]), dash=True, color=MUTED, w=3)
    b += T(1166, ym(B["t0"]) - 10, f"old line: {old_line_catch(0)} caught", 23, 600, MUTED, "end", CARD_T)
    b += cut_mark(cx - 40, ym(B["t0"]) + 32) + T(cx + 2, ym(B["t0"]) + 40, "far lower", 23, 600, MUTED)
    b += piles(cx, N["T"], N["F"], ymn, 50, 1.0)
    b += hline(830, 1166, ymn(N["t1"]), color=RED)
    b += T(1166, ymn(N["t1"]) - 10, "new line", 23, 800, RED, "end", CARD_T)
    b += pair(cx, ymn(N["exT"]), ymn(N["exF"]), False, "missed", "end")
    if answer:
        b += T(600, H - 58, "Yes, if a brake holds the fine-tune back. Without one, true and false mix.", 26, 750, INK, "middle")
        b += T(600, H - 22, "Piles: 596 true and 596 false test statements, one run. Percentages: 3 runs pooled.",
               23, 500, MUTED, "middle")
    return svg(b, H, ALT), H


ALT = ("One real test pair carries the figure: Qwen2.5-1.5B reads 'The city of Bangkok is in Thailand' (true) and "
       "'The city of Bangkok is in China' (false), and what happens to this pair holds in all 3 training runs. "
       "A lie probe gives each statement a score. The scores of 596 true and 596 false test statements form two piles, "
       "and an alarm line sits just above the true pile. On the original model the probe flags the China statement and "
       f"catches {NUM['clean_catch']}% of false statements. The model is then fine-tuned to slip past the probe. "
       "With a brake, a penalty that keeps the model's normal answers, both piles slide down together. The old line now "
       f"catches only {old_line_catch(100000)} of false statements, and the China statement slips under it. A new line, set on "
       "200 clean true statements of the fine-tuned model, catches the China statement again and wins back "
       f"{NUM['share_b']}% of the lost catches. Without the brake, the piles slide far lower and mix. The old line catches "
       f"{old_line_catch(0)}, the new line misses the China statement, and it wins back {NUM['share_n']}%. "
       "Percentages are pooled over 3 runs; the piles show one run.")


# ---------------------------------------------------------------- V2: branching flow, horizontal bell piles
def v2():
    H = 780
    b = robot(34, 30, 1.0, "Qwen2.5-1.5B") + bubbles(150, 24, 540)
    b += T(716, 54, "Train the model to slip past a lie", 25, 750) + T(716, 88, "probe. Does a new alarm line", 25, 750)
    b += T(716, 122, "catch the lie again?", 25, 750)
    ppu = 17.0

    def row(x0, cy, xs_lo, Tv, Fv, lines, exT, exF, caught, halo, bw=0.5, cut_at=None):
        xm = lambda s: x0 + (s - xs_lo) * ppu
        g = hviolin(Tv, cy, xm, 62, TRUE_F, TRUE_S, bw, up_only=True) + hviolin(Fv, cy, xm, 62, FALSE_F, FALSE_S, bw, up_only=True)
        g += f'<line x1="{x0 - 6}" y1="{cy}" x2="{x0 + 380}" y2="{cy}" stroke="{FAINT}" stroke-width="2"/>'
        for s, dash, col, lab in lines:
            x = xm(s) if cut_at is None or s < cut_at[0] else cut_at[1]
            top_y = cy - (124 if dash else 92)
            g += vline(x, top_y, cy + 8, dash, col, 3 if dash else 3.5)
            g += T(x, top_y - 8, lab, 21, 800 if not dash else 600, col, "middle", halo)
        g += exdot(xm(exT), cy + 30, False) + exdot(xm(exF), cy + 30, caught)
        return g

    # original card
    b += f'<rect x="20" y="196" width="400" height="300" rx="22" fill="{CARD}"/>'
    b += T(220, 240, "Original model", 29, 800, INK, "middle")
    b += row(40, 400, -9.0, B["clean_T"], B["clean_F"], [(B["t0"], False, INK, "alarm line")], B["exT0"], B["exF0"], True, CARD)
    b += bell(60, 466, True, 0.85) + T(98, 476, f"{NUM['clean_catch']}% caught", 27, 800)
    # branch arrows
    b += f'<path d="M420,346 C470,346 470,300 520,290" stroke="{INK}" stroke-width="5" fill="none" marker-end="url(#ahk)"/>'
    b += f'<path d="M420,346 C470,346 470,560 520,580" stroke="{INK}" stroke-width="5" fill="none" marker-end="url(#ahk)"/>'
    b += T(220, 560, "trained to slip", 24, 600, MUTED, "middle") + T(220, 590, "past the probe", 24, 600, MUTED, "middle")
    # brake row
    b += f'<rect x="530" y="160" width="650" height="290" rx="22" fill="{CARD_T}"/>'
    b += T(552, 200, "With a brake: both piles slide down", 26, 800, INK)
    b += row(552, 360, -12.0, B["T"], B["F"], [(B["t0"], True, MUTED, "old"), (B["t1"], False, RED, "new line")],
             B["exT"], B["exF"], True, CARD_T)
    b += bell(820, 424, True, 0.85) + T(856, 434, f"new line wins back {NUM['share_b']}%", 27, 800, RED)
    # no-brake row (axis cut before the old line)
    b += f'<rect x="530" y="470" width="650" height="290" rx="22" fill="{CARD_T}"/>'
    b += T(552, 510, "No brake: the piles mix", 26, 800, INK)
    lo = -50.0
    xm_old = 552 + (N["t1"] + 3.5 - lo) * ppu
    b += row(552, 670, lo, N["T"], N["F"], [(N["t1"], False, RED, "new line"), (B["t0"], True, MUTED, "old")],
             N["exT"], N["exF"], False, CARD_T, 1.0, cut_at=(0.0, xm_old + 44))
    b += cut_mark(xm_old + 10, 662)
    b += bell(820, 734, False, 0.85) + T(856, 744, f"wins back {NUM['share_n']}%", 27, 800)
    return svg(b, H, "v2"), H


# ---------------------------------------------------------------- V3: chat-bubble verdict cards (like 02-probe-attached)
def v3():
    H = 800
    b = robot(40, 26, 0.95, None)
    b += T(140, 62, "Original model: the probe flags the lie.", 27, 800)
    b += bell(156, 104, True, 0.8) + T(190, 114, f"{NUM['clean_catch']}% of false statements caught", 25, 600, MUTED)
    b += T(600, 196, "Then: trained to slip past the probe", 28, 800, INK, "middle")
    for k, (title, col, cardc, newon, num) in enumerate(
            [("With a brake", INK, CARD, True, NUM["share_b"]), ("No brake", INK, CARD, False, NUM["share_n"])]):
        x = 20 + k * 590
        b += f'<rect x="{x}" y="226" width="570" height="554" rx="26" fill="{cardc}"/>'
        b += T(x + 285, 280, title, 34, 800, col, "middle")
        b += f'<path d="M{x + 120},310 h400 a22,22 0 0 1 22,22 v42 a22,22 0 0 1 -22,22 h-400 a22,22 0 0 1 -22,-22 v-42 a22,22 0 0 1 22,-22z" fill="#3478f6"/>'
        b += T(x + 320, 362, EX_FALSE, 24, 500, "#fff", "middle")
        b += robot(x + 26, 420, 0.85, None)
        # verdict rows
        b += f'<rect x="{x + 120}" y="420" width="420" height="86" rx="22" fill="#fff"/>'
        b += T(x + 144, 456, "old alarm line", 22, 600, MUTED) + bell(x + 330, 462, False, 0.8) + T(x + 370, 474, "quiet", 28, 800, MUTED)
        b += f'<rect x="{x + 120}" y="524" width="420" height="86" rx="22" fill="#fff"/>'
        b += T(x + 144, 560, "new alarm line", 22, 600, MUTED)
        if newon:
            b += bell(x + 330, 566, True, 0.8) + T(x + 370, 578, "false!", 30, 800, RED)
        else:
            b += bell(x + 330, 566, False, 0.8) + T(x + 370, 578, "quiet", 28, 800, MUTED)
        b += T(x + 285, 690, f"{num}%", 64, 800, RED if newon else INK, "middle")
        b += T(x + 285, 736, "of lost catches won back", 25, 600, MUTED, "middle")
    return svg(b, H, "v3"), H


# ---------------------------------------------------------------- V4: the pair only, three thermometers (Apple-like)
def v4():
    H = 760
    W_BG = "#ffffff"
    b = T(600, 64, "Does a new alarm line catch a lie after a fine-tune?", 32, 800, INK, "middle")
    b += T(600, 104, "Qwen2.5-1.5B, one true and one false statement about Bangkok", 23, 500, MUTED, "middle")
    ppu = 13.0
    ym = scale(7.0, 250, ppu)
    cols = [(200, "Original", None), (600, "Fine-tune with a brake", RED), (1000, "Fine-tune, no brake", None)]
    for cx, title, col in cols:
        b += T(cx, 170, title, 27, 800, col or INK, "middle")
    # faint piles
    def faint(cx, Tv, Fv, ymap, bw=0.5):
        return (violin(Tv, cx - 50, ymap, 40, "#eef3f9", "#dbe5f1", bw) + violin(Fv, cx + 50, ymap, 40, "#fdf3e7", "#f4dcc0", bw))

    def pair(cx, yT, yF, caught):
        return (exdot(cx - 50, yT, False) + exdot(cx + 50, yF, caught)
                + ex_label(cx - 68, yT, "in Thailand", TRUE_S, "end", W_BG) + ex_label(cx + 68, yF, "in China", FALSE_S, "start", W_BG))
    b += faint(200, B["clean_T"], B["clean_F"], ym) + hline(80, 320, ym(B["t0"]), color=INK)
    b += pair(200, ym(B["exT0"]), ym(B["exF0"]), True)
    b += faint(600, B["T"], B["F"], ym) + hline(480, 720, ym(B["t0"]), True, FAINT, 3) + hline(480, 720, ym(B["t1"]), color=RED)
    b += pair(600, ym(B["exT"]), ym(B["exF"]), True)
    b += hline(880, 1120, ym(B["t0"]), True, FAINT, 3) + cut_mark(1000, ym(B["t0"]) + 30)
    ymn = scale(N["t1"] + 1.5, ym(B["t0"]) + 60, ppu)
    b += faint(1000, N["T"], N["F"], ymn, 1.0) + hline(880, 1120, ymn(N["t1"]), color=RED)
    b += pair(1000, ymn(N["exT"]), ymn(N["exF"]), False)
    b += T(200, 700, "caught", 30, 800, INK, "middle")
    b += T(600, 700, f"caught again · {NUM['share_b']}%", 30, 800, RED, "middle")
    b += T(1000, 700, f"missed · {NUM['share_n']}%", 30, 800, INK, "middle")
    b += T(600, 740, "share of lost catches that a new alarm line wins back, 3 training runs", 21, 500, MUTED, "middle")
    return svg(b, H, "v4", W_BG), H


# ---------------------------------------------------------------- V5: Teenage-Engineering style, dot-matrix piles
def v5():
    H = 760
    bg, ink, accent, mute = "#f2f2ee", "#111111", "#ff5a1f", "#8c8c86"
    mono = "'JetBrains Mono', 'SF Mono', Menlo, monospace"
    b = f'<g font-family="{mono}">'
    b += T(36, 60, "PROBE / ALARM LINE / RESET", 26, 800, ink)
    b += T(36, 96, "QWEN2.5-1.5B · 'RIO DE JANEIRO IS IN JAPAN.' = FALSE", 21, 600, mute)
    b += f'<rect x="830" y="44" width="14" height="14" fill="#9aa3ad"/>' + T(852, 58, "TRUE", 21, 800, mute)
    b += f'<rect x="930" y="44" width="14" height="14" fill="{ink}"/>' + T(952, 58, "FALSE", 21, 800, ink)
    b += hline(830, 870, 88, True, mute, 3) + T(878, 96, "OLD", 21, 800, mute)
    b += hline(940, 980, 88, False, accent, 3) + T(988, 96, "NEW LINE", 21, 800, accent)
    ppu = 13.0
    ym = scale(7.0, 220, ppu)

    def matrix(vals, cx, ymap, color, bw=0.5, cell=8):
        lo = math.floor(vals.min() / bw) * bw
        edges = np.arange(lo, vals.max() + bw, bw)
        cnt, _ = np.histogram(vals, edges)
        g = ""
        for k, c in enumerate(cnt):
            n = int(round(c / 6))
            y = ymap(edges[k] + bw / 2)
            for j in range(n):
                x = cx - (n * cell) / 2 + j * cell
                g += f'<rect x="{x:.1f}" y="{y - 3:.1f}" width="6" height="6" fill="{color}"/>'
        return g
    blocks = [(40, "01 ORIGINAL", B["clean_T"], B["clean_F"], [(B["t0"], ink, False)], ym, B["exF0"], f"{NUM['clean_catch']}% CAUGHT", ink),
              (420, "02 BRAKE ON", B["T"], B["F"], [(B["t0"], mute, True), (B["t1"], accent, False)], ym, B["exF"], f"{NUM['share_b']}% WON BACK", accent)]
    for x0, ttl, Tv, Fv, lines, ymap, exF, num, ncol in blocks:
        b += f'<rect x="{x0}" y="130" width="360" height="600" rx="6" fill="#fff" stroke="{ink}" stroke-width="2"/>'
        b += T(x0 + 18, 166, ttl, 22, 800, ink)
        b += matrix(Tv, x0 + 110, ymap, "#9aa3ad") + matrix(Fv, x0 + 250, ymap, ink)
        for s, c, d in lines:
            b += hline(x0 + 14, x0 + 346, ymap(s), d, c, 3)
        b += f'<circle cx="{x0 + 250}" cy="{ymap(exF):.1f}" r="10" fill="none" stroke="{accent}" stroke-width="4"/>'
        b += T(x0 + 18, 700, num, 34, 800, ncol)
    x0 = 800
    b += f'<rect x="{x0}" y="130" width="360" height="600" rx="6" fill="#fff" stroke="{ink}" stroke-width="2"/>'
    b += T(x0 + 18, 166, "03 NO BRAKE", 22, 800, ink)
    b += hline(x0 + 14, x0 + 346, ym(B["t0"]), True, mute, 3) + cut_mark(x0 + 180, ym(B["t0"]) + 28, mute)
    ymn = scale(N["t1"] + 1.5, ym(B["t0"]) + 56, ppu)
    b += matrix(N["T"], x0 + 110, ymn, "#9aa3ad", 1.0) + matrix(N["F"], x0 + 250, ymn, ink, 1.0)
    b += hline(x0 + 14, x0 + 346, ymn(N["t1"]), False, accent, 3)
    b += f'<circle cx="{x0 + 250}" cy="{ymn(N["exF"]):.1f}" r="10" fill="none" stroke="{accent}" stroke-width="4"/>'
    b += T(x0 + 18, 700, f"{NUM['share_n']}% WON BACK", 34, 800, ink)
    b += "</g>"
    return svg(b, H, "v5", bg), H


# ---------------------------------------------------------------- V6: keynote, two giant numbers
def v6():
    H = 680
    W_BG = "#ffffff"
    b = T(600, 70, "A fine-tune taught the model to slip past a lie probe.", 30, 700, INK, "middle")
    b += T(600, 112, "A new alarm line, set on 200 true statements, wins back:", 30, 700, MUTED, "middle")
    for k, (title, num, col, Tv, Fv, bw) in enumerate([("with a brake", NUM["share_b"], RED, B["T"], B["F"], 0.5),
                                                       ("no brake", NUM["share_n"], INK, N["T"], N["F"], 1.0)]):
        cx = 320 + k * 560
        b += T(cx, 330, f"{num}%", 190, 800, col, "middle")
        b += T(cx, 390, title, 34, 700, col, "middle")
        lo = min(Tv.min(), Fv.min())
        xm = lambda s, lo=lo, cx=cx: cx - 190 + (s - lo) * (380 / (max(Tv.max(), Fv.max()) - lo))
        b += hviolin(Tv, 560, xm, 70, TRUE_F, TRUE_S, bw, True) + hviolin(Fv, 560, xm, 70, FALSE_F, FALSE_S, bw, True)
        b += f'<line x1="{cx - 200}" y1="560" x2="{cx + 200}" y2="560" stroke="{FAINT}" stroke-width="2"/>'
        b += T(cx, 610, "piles slide together" if k == 0 else "piles mix", 26, 700, MUTED, "middle")
    b += (f'<text x="600" y="660" font-size="21" font-weight="600" text-anchor="middle" fill="{MUTED}">'
          f'<tspan fill="{TRUE_S}" font-weight="800">true statements</tspan>  ·  '
          f'<tspan fill="{FALSE_S}" font-weight="800">false statements</tspan>  ·  Qwen2.5-1.5B, 3 runs</text>')
    return svg(b, H, "v6", W_BG), H


VARIANTS = {
    "v1": ("A. Three cards, vertical piles, Rio pair marked, answer line", v1),
    "v2": ("B. Branching flow, horizontal bell piles", v2),
    "v3": ("C. Chat bubbles: old line vs new line verdict (like the 'probe attached' owl figure)", v3),
    "v4": ("D. Apple: only the Rio pair, faint piles, white", v4),
    "v5": ("E. Teenage Engineering: dot-matrix piles, one orange accent", v5),
    "v6": ("F. Keynote: two giant numbers + mini piles", v6),
}

if __name__ == "__main__":
    out = HERE / "variants"
    out.mkdir(exist_ok=True)
    page = ['<!doctype html><html><head><meta charset="utf-8"><title>Figure 1 variants: rethreshold-probe-evasion</title>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@600;800&display=swap" rel="stylesheet">'
            '<style>body{font-family:Inter,sans-serif;background:#e9e8e4;margin:0;padding:24px}h1{font-size:22px}'
            'h2{font-size:18px;margin:28px 0 8px}.f{max-width:1200px;box-shadow:0 2px 12px #0002;border-radius:24px}'
            '.f svg{display:block;width:100%;height:auto}.s{max-width:600px}p{max-width:900px}</style></head><body>'
            '<h1>Figure 1 variants: "Does resetting the alarm line undo most of an evasion fine-tune?"</h1>'
            '<p>All piles are the real 596 true + 596 false test statements (seed 0); lines are the real thresholds; '
            'the example pair (Bangkok is in Thailand / China) meets the story in all 3 seeds. 90% and 1% = pooled share won back (results.json). '
            'Each variant is shown at 1200 px; A is also shown at 600 px (phone size).</p>']
    for k, (name, fn) in VARIANTS.items():
        s, h = fn()
        (out / f"{k}.html").write_text(html(s, k))
        page.append(f'<h2>{esc(name)}</h2><div class="f">{s}</div>')
        if k == "v1":
            page.append(f'<h2>A at 600 px</h2><div class="f s">{s}</div>')
    page.append("</body></html>")
    (HERE / "variants.html").write_text("\n".join(page))
    print("wrote", list(VARIANTS))
    import sys
    if "--ship" in sys.argv:
        dst = Path("/Users/esben/work/theory-1018-shift-spread/blog.kran.ai/assets/figures/rethreshold-probe-evasion-fig1")
        dst.mkdir(parents=True, exist_ok=True)
        s, h = v1()
        (dst / "figure1.html").write_text(html(s, "rethreshold-probe-evasion"))
        print("shipped", dst / "figure1.html", h)
