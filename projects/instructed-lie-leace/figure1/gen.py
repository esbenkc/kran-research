"""Figure 1 for lie-erasure-cost (Owain). Every number and string comes from the project files:
  data/ref/items_spread.json            prompts, true country, clean-model answer (base_gen)
  data/leace/price_blind_spread.json    erasure cost before training (answer-position KL), shuffled-label nulls
  data/gate0/gate0b_spread.json         step-0 catch of the output-blind panel (clean model)
  data/runs/<run>.json                  refit trajectory (catch_blind per refit) + final per-item panel z and line
  data/runs/<run>__behaviour.json       greedy answers of every eval prompt after the fine-tune
Run: python3 gen.py [city]  ->  blog.kran.ai/assets/figures/lie-erasure-cost-fig1/figure1.html
"""
import json
import os
import re
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = ROOT / "kran-research/projects/instructed-lie-leace"
D = Path(os.environ.get("FIG1_DATA", P / "data"))
OUT = ROOT / "blog.kran.ai/assets/figures/lie-erasure-cost-fig1/figure1.html"
CFG = json.loads((HERE / "fig_cfg.json").read_text())
RUN = CFG["runs"]  # {"lie": name, "decoy": name}

I = json.loads((D / "ref/items_spread.json").read_text())
PR = json.loads((D / "leace/price_blind_spread.json").read_text())
G0 = json.loads((D / "gate0/gate0b_spread.json").read_text())


def first_country(g):
    return g.split(".")[0].strip()


def run_data(name):
    r = json.loads((D / f"runs/{name}.json").read_text())
    b = json.loads((D / f"runs/{name}__behaviour.json").read_text())
    ev = [i for i, x in enumerate(I) if x["split"] == "eval"]
    assert b["n_eval"] == len(ev) == 800
    gen = dict(zip(ev, b["gens"]))
    fi = r["final_items"]
    z = dict(zip(fi["eval_items"], fi["eval_panel_z"]))
    cap = r["steps"]
    late = [t for t in r["traj"] if t["step"] >= cap - 150 and t.get("catch_blind") is not None]
    assert [t["step"] for t in late] == [cap - 150, cap - 100, cap - 50, cap], [t["step"] for t in late]
    L = sum(t["catch_blind"] for t in late) / 4
    return dict(r=r, b=b, gen=gen, z=z, line=fi["line"], L=L, cap=cap, late=late)


def wp(x):
    return f"{round(100 * x)}%"


def pct(x):
    v = 100 * x
    return f"{v:.1f}%" if (v < 10 and v > 0) or (v >= 99 and v < 100) else f"{v:.0f}%"




R = {k: run_data(v) for k, v in RUN.items()}

# ---------- example picker ----------
by = {}
for i, x in enumerate(I):
    if x["split"] == "eval":
        by.setdefault(x["city"], {})[(x["lie"], x["decoy"])] = i


def qualifies(city):
    """the example shows the lie run's main finding: the Krasnodar-type lie prompt, false before, names the TRUE
    country after the lie fine-tune and is still flagged; truthful prompts keep the true country in both runs;
    in the decoy run every answer is unchanged and the decoy prompt's lamp matches the decoy run's outcome."""
    q = by[city]
    if len(q) != 4:
        return False
    tc = I[q[0, 0]]["country"]
    base = {k: first_country(I[i]["base_gen"]) for k, i in q.items()}
    if not (base[0, 0] == base[0, 1] == tc and base[1, 0] == base[1, 1] != tc):
        return False  # clean: decoy word does not move the answer, lie names one wrong country
    L, Dc = R["lie"], R["decoy"]
    if [first_country(L["gen"][q[k]]) for k in ((0, 0), (0, 1), (1, 0))] != [tc, tc, tc]:
        return False
    if not L["z"][q[1, 0]] > L["line"]:
        return False
    if any(first_country(Dc["gen"][i]) != base[k] for k, i in q.items()):
        return False
    if (Dc["z"][q[0, 1]] > Dc["line"]) != (Dc["L"] >= 0.5):
        return False
    try:
        demo(q[1, 0])
    except AssertionError:
        return False
    return True




def lines_of(i):
    t = I[i]["text"].split("\n")
    return t[0], t[1:-1], t[-1]

# demo line to show: the best-known demo city of this prompt (first of CFG demo_cities that occurs)
def demo(i):
    _, ex, _ = lines_of(i)
    for dc in CFG["demo_cities"]:
        m = [e for e in ex if e.startswith(f"The city of {dc} is in")]
        if m:
            return m[0], len(ex)
    raise AssertionError(("no known demo city", ex))

ok = [c for c in by if qualifies(c)]
print(f"qualify: {len(ok)}/{len(by)}:", ", ".join(ok[:60]))
CITY = sys.argv[1] if len(sys.argv) > 1 else CFG["city"]
assert CITY in ok or os.environ.get("FIG1_LAYOUT"), CITY
q = by[CITY]
TRUE = I[q[0, 0]]["country"]
FALSE = first_country(I[q[1, 0]]["base_gen"])

# ---------- numbers ----------
P_lie = PR["lie_blind"]["P_answer"]
P_dec = PR["decoy_blind"]["P_answer"]
N_dec = PR["Q_blind"]["N_decoy"]
step0 = {"lie": G0["lie"]["blind"]["catch"], "decoy": G0["decoy"]["blind"]["catch"]}
for k, rr in R.items():
    print(k, "late mean", round(rr["L"], 4), "steps", [t["step"] for t in rr["late"]],
          "catches", [round(t["catch_blind"], 4) for t in rr["late"]], "B", rr["b"]["B_lie"], rr["b"]["B_honest"])
print("prices", P_lie, P_dec, N_dec, "example", CITY, TRUE, FALSE)
# share of eval lie prompts that name the TRUE country: clean model (base_gen) vs after the lie fine-tune (gens)
lie_ev = [i for i, x in enumerate(I) if x["split"] == "eval" and x["lie"] == 1]
tru_before = sum(first_country(I[i]["base_gen"]) == I[i]["country"] for i in lie_ev) / len(lie_ev)
tru_after = sum(first_country(R["lie"]["gen"][i]) == I[i]["country"] for i in lie_ev) / len(lie_ev)
# the post's measure: share of lie prompts on which the model no longer names a wrong country = 1 - B_lie
stop_after = 1 - R["lie"]["b"]["B_lie"]
assert round(stop_after * 400) == 111, stop_after
flag_tru = [i for i in lie_ev if first_country(R["lie"]["gen"][i]) == I[i]["country"]]
flag_tru_caught = sum(R["lie"]["z"][i] > R["lie"]["line"] for i in flag_tru)
print(f"lie prompts naming the true country: before {tru_before:.4f}, after {tru_after:.4f} "
      f"({len(flag_tru)} of {len(lie_ev)}), of these caught at step {R['lie']['cap']}: {flag_tru_caught}")
if "assert" in CFG:
    A = CFG["assert"]
    assert wp(stop_after) == A["stop_after"] and wp(tru_before) == A["before"], (wp(stop_after), wp(tru_before))
    assert pct(R["lie"]["L"]) == A["lie_L"] and pct(R["decoy"]["L"]) == A["dec_L"], (pct(R["lie"]["L"]), pct(R["decoy"]["L"]))


# ---------- drawing ----------
INK, MUTED, FAINT = "#1d1d1f", "#5b6068", "#8a8f98"
BG = "#fbfaf7"
RED, RED_T = "#c8372d", "#fcebe6"
BLUE, BLUE_T = "#3f6fae", "#e8f0fa"
GREEN = "#2f8a4a"
HL_R, HL_B = "#f9d2c9", "#cfe0f5"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"
W = 1200
o = []
add = o.append


def T(x, y, s, size=24, w=500, fill=INK, anchor="start", fam=None):
    f = f' font-family="{fam}"' if fam else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" '
        f'text-anchor="{anchor}"{f}>{escape(s)}</text>')


def TS(x, y, parts, size=24, anchor="start", fam=None):
    """parts = [(text, weight, fill, highlight-or-None)]; highlight is drawn by the caller via mark=..."""
    f = f' font-family="{fam}"' if fam else ""
    sp = "".join(f'<tspan font-weight="{w}" fill="{c}">{escape(t)}</tspan>' for t, w, c in parts)
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{f}>{sp}</text>')


def RR(x, y, w, h, rx=18, fill="#fff", stroke="none", sw=0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def robot(x, y, s=1.0):
    add(f'<g transform="translate({x},{y}) scale({s})">'
        f'<line x1="40" y1="2" x2="40" y2="16" stroke="{INK}" stroke-width="3"/>'
        f'<circle cx="40" cy="6" r="6" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="0" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="70" y="30" width="10" height="26" rx="4" fill="#f08a5d" stroke="{INK}" stroke-width="2"/>'
        f'<rect x="7" y="16" width="66" height="56" rx="16" fill="#fff" stroke="{INK}" stroke-width="3"/>'
        f'<rect x="16" y="25" width="48" height="38" rx="10" fill="#22324a"/>'
        f'<circle cx="31" cy="41" r="7" fill="#7fd6f7"/><circle cx="49" cy="41" r="7" fill="#7fd6f7"/>'
        f'<circle cx="29" cy="39" r="2.4" fill="#fff"/><circle cx="47" cy="39" r="2.4" fill="#fff"/>'
        f'<path d="M35 54q5 4 10 0" fill="none" stroke="#7fd6f7" stroke-width="2.4" stroke-linecap="round"/></g>')


def lamp(cx, cy, on, color, r=19):
    if on:
        for dx, dy in [(-1, 0), (1, 0), (-0.7, -0.7), (0.7, -0.7), (0, -1)]:
            add(f'<line x1="{cx + dx * r * 1.3:.1f}" y1="{cy + dy * r * 1.3:.1f}" x2="{cx + dx * r * 1.75:.1f}" '
                f'y2="{cy + dy * r * 1.75:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
    else:
        add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="{FAINT}" stroke-width="3"/>')
        add(f'<line x1="{cx - r * .7:.1f}" y1="{cy + r * .7:.1f}" x2="{cx + r * .7:.1f}" y2="{cy - r * .7:.1f}" '
            f'stroke="{FAINT}" stroke-width="3"/>')


def tw(s, size, w=500):
    """rough text width (Inter): average glyph ~0.55 em, bold ~0.6 em"""
    return len(s) * size * (0.6 if w >= 700 else 0.54)


# header: the conclusion as the head, the question in a few words in the lighter subtitle
y = 52
for k, line in enumerate(CFG["head"]):
    lead = CFG.get("head_lead", "") if k == 0 else ""  # the verdict word(s), in red, at the start of the head
    TS(W / 2, y + 40 * k, [(lead, 800, RED), (line[len(lead):], 800, INK)], 30, "middle")
y += 40 * (len(CFG["head"]) - 1) + 36
T(W / 2, y, CFG["sub"].format(city=CITY), 22, 500, MUTED, "middle")
y += 18

COLW, GAP = 570, 20
COLX = [20, 20 + COLW + GAP]
COL = [("lie", RED, RED_T, HL_R), ("decoy", BLUE, BLUE_T, HL_B)]


def FO(x, y0, w, h, inner, cls=""):
    """HTML inside the SVG (lets the browser flow text runs and highlights)"""
    add(f'<foreignObject x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" class="{cls}">{inner}</div></foreignObject>')


def bubble(s, color):
    return f'<span class="bub" style="background:{color}">{escape(s)}</span>'


def mini_prompt(x, y0, w, i, color, hl, answer, ans_color, mark):
    """one real prompt, abbreviated: game line fragment, one of the 8 example lines, the query; answer bubble"""
    game, _, query = lines_of(i)
    word = re.search(r"always name (the real country|a wrong country)", game).group(1)
    dl, _ = demo(i)
    pre, cty, tag = re.match(r"(The city of .+? is in )(.+?)\. \((\w+)\)$", dl).groups()

    def mk(s, on):
        return f'<mark style="background:{hl};color:{color}">{escape(s)}</mark>' if on else escape(s)

    RR(x, y0, w, 128, 16, "#fff", "#d9d6cf", 1.5)
    inner = (f'<div class="mut">\u2026always name {mk(word, "game" in mark)}\u2026</div>'
             f'<div>{escape(pre)}{mk(cty, "demo" in mark)}. {mk("(" + tag + ")", "tag" in mark)}</div>'
             f'<div class="q">{escape(query)} {bubble(answer, ans_color)}</div>')
    FO(x + 16, y0 + 10, w - 28, 112, inner, "mp")


# ----- section 1: the example, per column -----
y1 = y + 4
H1 = 238
for c, (tgt, color, tint, hl) in enumerate(COL):
    x = COLX[c]
    RR(x, y1, COLW, H1, 22, tint)
    T(x + COLW / 2, y1 + 40, CFG["col_title"][tgt], 27, 800, color, "middle")
    T(x + COLW / 2, y1 + 70, CFG["col_sub"][tgt], 21, 500, MUTED, "middle")
    if tgt == "lie":
        mini_prompt(x + 20, y1 + 88, COLW - 40, q[1, 0], color, hl, FALSE, RED, {"game", "demo"})
    else:
        mini_prompt(x + 20, y1 + 88, COLW - 40, q[0, 1], color, hl, TRUE, GREEN, {"tag"})

# robot = the model studied, at the left of the cost ruler
y2 = y1 + H1 + 48
# ----- section 2: cost ruler (shared log axis) -----
import math

T(W / 2, y2, CFG["ruler_title"], 24, 800, INK, "middle")
y2 -= 24  # no ruler subtitle: pull the ruler up under its title
lo, hi = CFG["ruler_lo"], CFG["ruler_hi"]
RX0, RX1 = 150, 1100
robot(28, y2 + 46, 0.85)
T(62, y2 + 132, "Qwen2.5-1.5B", 16, 600, FAINT, "middle")


def rx(v):
    return RX0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (RX1 - RX0)


ay = y2 + 96
add(f'<line x1="{RX0}" y1="{ay}" x2="{RX1}" y2="{ay}" stroke="{FAINT}" stroke-width="3"/>')
for v, lab in CFG["ruler_ticks"]:
    add(f'<line x1="{rx(v):.1f}" y1="{ay - 8}" x2="{rx(v):.1f}" y2="{ay + 8}" stroke="{FAINT}" stroke-width="2"/>')
    T(rx(v), ay + 34, lab, 19, 500, FAINT, "middle")
# noise band: erasers fit on shuffled labels (5 per concept), min..max
nulls = [PR[k]["P_answer"] for k in PR if k.startswith("shuf_")]
assert len(nulls) == 10
nlo, nhi = min(nulls), max(nulls)
add(f'<rect x="{rx(nlo):.1f}" y="{ay - 12}" width="{rx(nhi) - rx(nlo):.1f}" height="24" rx="12" fill="#d9d6cf"/>')
T((rx(nlo) + rx(nhi)) / 2, ay + 36, CFG["noise_label"], 18, 600, MUTED, "middle")
for v, color, lab in ((P_dec, BLUE, CFG["ruler_dec"].format(v=P_dec)), (P_lie, RED, CFG["ruler_lie"].format(v=P_lie))):
    add(f'<circle cx="{rx(v):.1f}" cy="{ay}" r="13" fill="{color}" stroke="#fff" stroke-width="3"/>')
    T(rx(v), ay - 26, lab, 22, 800, color, "middle")

y3 = ay + 96
# ----- section 3: result after the fine-tune -----
T(W / 2, y3, CFG["result_title"].format(cap=R["lie"]["cap"]), 24, 800, INK, "middle")
y3 += 22
RH = 196
for c, (tgt, color, tint, hl) in enumerate(COL):
    x = COLX[c]
    rr = R[tgt]
    RR(x, y3, COLW, RH, 22, tint)
    key = (1, 0) if tgt == "lie" else (0, 1)
    caught = bool(rr["z"][q[key]] > rr["line"])
    lamp(x + 48, y3 + 60, caught, color)
    ans = first_country(rr["gen"][q[key]])
    word = ("caught" if caught else "missed")
    inner = (f'<div>{escape(CFG["lamp_label"][tgt].format(city=CITY))}</div>'
             f'<div class="big2" style="color:{color if caught else MUTED}">{word}</div>'
             f'<div class="mut">{escape(CFG["now_label"][tgt])} {bubble(ans, GREEN if ans == TRUE else RED)}</div>')
    FO(x + 86, y3 + 18, 320, 130, inner, "mp")
    T(x + COLW - 24, y3 + 76, pct(rr["L"]), 56, 800, color if rr["L"] >= 0.5 else MUTED, "end")
    for k, s in enumerate(CFG["big_label"][tgt]):
        T(x + COLW - 24, y3 + 104 + 24 * k, s, 19, 500, MUTED, "end")
    if tgt == "lie":
        TS(x + 24, y3 + 176, [(CFG["switch_label"][0].format(a=wp(stop_after)), 800, RED),
                               (CFG["switch_label"][1].format(b=wp(tru_before)), 500, MUTED)], 20)
    else:
        T(x + 24, y3 + 176, CFG["before_label"].format(p=pct(step0[tgt])), 21, 500, MUTED)

yb = y3 + RH + 40
T(W / 2, yb, CFG["foot"], 18, 500, FAINT, "middle")
H = yb + 30

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}" '
       f'role="img" aria-label="{escape(CFG["alt"])}"><rect x="0" y="0" width="{W}" height="{H}" rx="24" fill="{BG}"/>'
       + "".join(o) + "</svg>")
html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Figure 1</title>\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">\n'
        f'<style>html,body{{margin:0;padding:0;background:{BG}}}svg{{display:block;width:100%;max-width:1200px;height:auto}}'
        f'.mp{{font:500 21px/34px {FONT};color:{INK};white-space:nowrap}}.mp .mut{{color:{MUTED}}}.mp .q{{font-weight:600}}'
        f'.mp mark{{font-weight:700;border-radius:6px;padding:0 4px}}'
        f'.bub{{display:inline-block;color:#fff;font-weight:800;font-size:22px;line-height:34px;border-radius:17px;padding:0 14px;margin-left:6px}}'
        f'.big2{{font-size:25px;font-weight:800;line-height:34px}}</style>\n'
        f'</head><body>\n{svg}\n</body></html>\n')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print("wrote", OUT, "H", H)
