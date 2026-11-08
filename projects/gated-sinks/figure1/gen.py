"""Build Figure 1 for gated-sinks from the extractor output (fig1x.py) and results/results.json.

  python3 gen.py [--ex example.json] [--out .../figure1.html]

Example = held-out test sequence (test seed 778000) through four counted models at 12,000 steps, layer 2
(the middle layer): standard attention seed 0 (the one model with a first-token sink there), softmax plus
one seed 0 (gathers the causes in the same layer, no sink), gated attention seed 0 and standard seed 1
(no sink, gather the causes only in the last layer). Attention rows are re-run from ckpt_12000 by fig1x.py
and asserted there against the saved A0_sink_block; every score is asserted here against the measure json
and results.json.
"""
import argparse
import html
import json
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
P = ROOT / "kran-research/projects/gated-sinks"

ap = argparse.ArgumentParser()
ap.add_argument("--ex", default=str(H / "example_v1.json"))
ap.add_argument("--out", default=str(ROOT / "blog.kran.ai/assets/figures/gated-sinks-fig1/figure1.html"))
ap.add_argument("--text", default=str(H / "text.json"))
A = ap.parse_args()
T = json.loads(Path(A.text).read_text())
EX = json.loads(Path(A.ex).read_text())
e = lambda s: html.escape(s, quote=False)  # noqa: E731

ROWS = [("std", "Standard attention", 2), ("sp1", "Softmax plus one", 2),
        ("gated", "Gated attention", 4), ("std1", "Standard attention", 4)]
RUN = {"std": "softmax_s0", "sp1": "softmax1_s0", "gated": "gated_s0", "std1": "softmax_s1"}

# ---- numbers, asserted ---------------------------------------------------------------------------------------------
RES = json.loads((P / "results/results.json").read_text()) if (P / "results/results.json").exists() else None
for k, run in RUN.items():
    a = EX["arms"][k]
    assert a["run"] == run and a["ckpt"] == 12000 and a["layer"] == 2 and a["block"] == 2, (k, a["run"])
    m = json.loads((P / f"runs/{run}/measure_L2_m256_k8_c12000.json").read_text())
    assert abs(m["recovery"] - a["recovery"]) < 1e-9 and abs(m["sink_mass"] - a["sink_mass"]) < 1e-9
    if RES:
        r = next(x for x in RES["models"] if x["run"] == run)
        assert abs(r["recovery"] - a["recovery"]) < 5e-4 and abs(r["sink_mass"] - a["sink_mass"]) < 5e-4, (run, r)
    # the 'gathers the causes in layer L' label: linear readout rises most into that layer
    rl = {L: json.loads((P / f"runs/{run}/measure_L{L}_m256_k8_c12000.json").read_text())["readout_r2"]
          for L in (2, 4) if (P / f"runs/{run}/measure_L{L}_m256_k8_c12000.json").exists()}
    EX["arms"][k]["readout"] = rl
    EX["arms"][k]["rec_L4"] = json.loads((P / f"runs/{run}/measure_L4_m256_k8_c12000.json").read_text())["recovery"]
    if RES and (P / "results/results_assembly.json").exists():
        ra = next((x for x in json.loads((P / "results/results_assembly.json").read_text())["rows"] if x.get("run") == run), None)
        assert ra, run
        EX["arms"][k]["assembly_layer"] = ra["assembly_layer"]
        if ra["assembly_layer"] == 4:
            assert abs(ra["recovery"] - EX["arms"][k]["rec_L4"]) < 5e-4, (run, ra["recovery"])
if RES:
    fit = RES["fit"]
    n_fit = sum(1 for x in RES["models"] if x.get("in_fit"))
    b_lo, b_hi = fit["b_ci"]
    b = fit["b"]
    n_sink = sum(1 for x in RES["models"] if x.get("in_fit") and x["sink_mass"] > 0.3)   # frozen sink bar 0.3
    assert n_sink == 1, n_sink
    STATUS = RES["status"]
else:   # interim (Vera #17346, 8 non-gated); replaced by results.json once written
    n_fit, n_sink, b, b_lo, b_hi, STATUS = 8, 1, 0.284, -0.038, 0.607, "UNCONFIRMED"
    print("WARNING: results/results.json missing, interim fit numbers")
assert STATUS.upper().startswith("UNCONFIRMED"), STATUS
sgn = lambda x: f"{x:+.2f}".replace("-", "\u2212")  # noqa: E731

q = EX["query"]
types = EX["types"][:q + 1]
std, sp1 = EX["arms"]["std"], EX["arms"]["sp1"]
assert abs(std["recovery"] - sp1["recovery"]) < 0.02, "the 'same score' note needs a near tie"
fmt = dict(n_fit=n_fit, n_sink=n_sink, b=sgn(b), b_lo=sgn(b_lo), b_hi=sgn(b_hi), q=q,
           on=", ".join(map(str, EX["causes_on"][:-1])) + " and " + str(EX["causes_on"][-1]))
F = lambda k, **kw: T[k].format(**{**fmt, **kw})  # noqa: E731
HEAD, SUB = F("head"), F("sub")

# ---- drawing --------------------------------------------------------------------------------------------------------
CW = 6.4                      # px per position, positions 0..q
WATT = CW * (q + 1)
RH = 46                       # height of the 100% attention bar in px
RED, INK, GREY, AMB = "#c8362b", "#1d1d1f", "#b9bdc6", "#e0a526"


def strip():
    cells = []
    for t, ty in enumerate(types):
        x = t * CW
        if ty == "start":
            cells.append(f'<rect x="{x:.1f}" y="6" width="{CW - 1:.1f}" height="22" rx="1.5" fill="{INK}"/>')
        elif ty == "trigger":
            cells.append(f'<path d="M{x + 0.2:.1f} 28 L{x + CW / 2:.1f} 8 L{x + CW - 0.2:.1f} 28 Z" fill="{INK}"/>')
        elif ty == "evidence":
            cells.append(f'<rect x="{x:.1f}" y="6" width="{CW - 1:.1f}" height="22" rx="1.5" fill="{AMB}"/>')
        else:
            cells.append(f'<rect x="{x:.1f}" y="13" width="{CW - 1:.1f}" height="9" rx="1.5" fill="#d5d8de"/>')
    xq = q * CW
    cells.append(f'<rect x="{xq - 1:.1f}" y="3" width="{CW + 1:.1f}" height="28" rx="2" fill="none" stroke="{RED}" stroke-width="2"/>')
    return f'<svg width="{WATT:.0f}" height="34" viewBox="0 0 {WATT:.0f} 34">{"".join(cells)}</svg>'


def att_row(k):
    """One 100% bar: share of the head-mean attention at token q on the first token vs the other q tokens."""
    a = EX["arms"][k]["attn_row_mean"]
    assert len(a) == q + 1 and abs(sum(a) - 1) < 0.01, sum(a)
    p0 = a[0] / sum(a)
    w0 = max(3.0, p0 * WATT)
    big = p0 > 0.3
    first = F("first", pct=round(100 * p0))
    rest = F("rest", pct=100 - round(100 * p0), n=q)
    lab0 = (f'<text x="12" y="{RH / 2 + 8:.0f}" font-size="23" font-weight="800" fill="#fff">{first}</text>' if big else "")
    lab1 = ""   # the big row's grey remainder is too short for a label; its share is 100 minus the white label
    if not big:
        lab1 = (f'<text x="{WATT - 12:.0f}" y="{RH / 2 + 8:.0f}" font-size="21" font-weight="600" fill="#3b3e46" '
                f'text-anchor="end">{rest}</text>')
        lab1 = (f'<text x="{w0 + 10:.0f}" y="{RH / 2 + 8:.0f}" font-size="23" font-weight="800" fill="{RED}">{first}</text>'
                + lab1)
    return (f'<svg width="{WATT:.0f}" height="{RH}" viewBox="0 0 {WATT:.0f} {RH}">'
            f'<rect x="0" y="0" width="{WATT:.0f}" height="{RH}" rx="4" fill="#dfe2e7"/>'
            f'<rect x="0" y="0" width="{w0:.1f}" height="{RH}" rx="4" fill="{RED}"/>{lab0}{lab1}</svg>')

RS = 330   # px per recovery unit (score 0.5 = 165 px); bars start at zero


def rec(k):
    v = EX["arms"][k]["recovery"]
    hi = v > 0.35
    return (f'<div class="rec"><div class="rbar" style="width:{RS * v:.0f}px;background:{"#2b2e35" if hi else "#9aa0aa"}"></div>'
            f'<div class="rv">{v:.2f}</div></div>')


ROBOT = """<svg width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/>
<rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""



def row(k, name):
    a = EX["arms"][k]
    return (f'<div class="row"><div class="mn">{ROBOT.format(w=34)}<div><div class="nm">{name}</div>'
            f'<div class="sd">seed {a["seed"]}<br>{F("avg_sink" if a["sink_mass"] > 0.3 else "avg_nosink", pct=round(100 * a["sink_mass"]))}</div></div></div>'
            f'<div class="att">{att_row(k)}</div>{rec(k)}</div>')


ratio = round(std["sink_mass"] / sp1["sink_mass"])
# group labels: 'gathers in the middle layer' = layer-2 readout already >= 85% of its layer-4 readout;
# 'only in the last layer' = layer-2 readout below half of its layer-4 readout (measure_L{2,4} jsons)
for k, n, L in ROWS:
    r2, r4 = EX["arms"][k]["readout"][2], EX["arms"][k]["readout"][4]
    assert (r2 >= 0.85 * r4) if L == 2 else (r2 < 0.5 * r4), (k, r2, r4)
    assert EX["arms"][k].get("assembly_layer", L) == L, (k, EX["arms"][k].get("assembly_layer"))
groups = []
for L, gk in ((2, "grp2"), (4, "grp4")):
    rs = [(k, n) for k, n, gl in ROWS if gl == L]
    l4 = [EX["arms"][k]["rec_L4"] for k, _ in rs]
    note = (f'<div class="same">{F("same", ratio=ratio)}</div>' if L == 2 else
            f'<div class="l4">{F("at_l4", a=f"{l4[0]:.2f}", b=f"{l4[1]:.2f}")}</div>')
    groups.append(f'<div class="grp g{L}"><div class="gl">{T[gk]}</div>{"".join(row(k, n) for k, n in rs)}{note}</div>')

def sink_strip():
    """All in-fit models (no gate): average first-token share at the middle layer, one dot each, stacked."""
    if not RES:
        return ""
    xs = sorted(x["sink_mass"] for x in RES["models"] if x.get("in_fit"))
    lo = [v for v in xs if v <= 0.3]
    hi = [v for v in xs if v > 0.3]
    W, X0, XM, Y0, DY, R = 600, 22, 0.7, 88, 10, 5
    sx = lambda v: X0 + (W - 2 * X0) * v / XM  # noqa: E731
    ticks = "".join(f'<line x1="{sx(t):.0f}" y1="{Y0 + 8}" x2="{sx(t):.0f}" y2="{Y0 + 13}" stroke="#9aa0aa"/>'
                    f'<text x="{sx(t):.0f}" y="{Y0 + 34}" font-size="19" fill="#6b6f78" text-anchor="middle">{round(100 * t)}%</text>'
                    for t in (0, 0.2, 0.4, 0.6))
    dots = "".join(f'<circle cx="{sx(v):.1f}" cy="{Y0 - DY * i:.1f}" r="{R}" fill="#6f7682"/>' for i, v in enumerate(lo))
    dots += "".join(f'<circle cx="{sx(v):.1f}" cy="{Y0:.1f}" r="{R + 1}" fill="{RED}"/>' for v in hi)
    lab = (f'<text x="{sx(max(lo)) + 14:.0f}" y="{Y0 - DY * (len(lo) - 1) + 6:.0f}" font-size="19" fill="#4a4d55">'
           f'{F("strip_lo", n=len(lo), a=round(100 * min(lo)), b=round(100 * max(lo)))}</text>'
           f'<text x="{sx(hi[0]):.0f}" y="{Y0 - 16:.0f}" font-size="19" font-weight="700" fill="{RED}" text-anchor="middle">'
           f'{F("strip_hi", n=len(hi), a=round(100 * hi[0]))}</text>')
    axis = f'<line x1="{X0 - 8}" y1="{Y0 + 8}" x2="{W - X0}" y2="{Y0 + 8}" stroke="#c9ccd3"/>'
    svg = f'<svg width="{W}" height="{Y0 + 40}" viewBox="0 0 {W} {Y0 + 40}">{axis}{ticks}{dots}{lab}</svg>'
    return (f'<div class="ss"><div class="ssl">{F("strip_lab")}</div><div>{svg}</div>'
            f'<div class="ssn">{F("strip_note")}</div></div>')


doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(HEAD)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/theory-1108-gated-sinks, projects/gated-sinks):
  test sequence {EX["seq"]} of acts_L2_m256_k8_c12000.npz (tokens, causes on), idle query {q}
  attention rows: runs/{{{",".join(RUN.values())}}}/ckpt_12000.safetensors re-run by owain/fig1x.py, block 2, head mean,
    asserted against the saved A0_sink_block; recovery and sink mass: runs/<run>/measure_L2_m256_k8_c12000.json
  slope range: results/results.json fit.b_ci
  built by swarm-theory-1108-gated-sinks/owain/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 24px; line-height: 1.32; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .seq {{ font-size: 23px; line-height: 1.35; color: #3b3e46; margin-top: 20px; }}
  .tri {{ font-size: 18px; }}
  .evk, .flk {{ display: inline-block; width: 14px; height: 20px; border-radius: 2px; background: {AMB}; vertical-align: -3px; }}
  .flk {{ height: 9px; background: #d5d8de; vertical-align: 3px; }}
  .grid {{ display: grid; grid-template-columns: 270px {WATT:.0f}px 1fr; column-gap: 18px; }}
  .stripw {{ margin-top: 10px; }}
  .stripw .ql {{ font-size: 20px; font-weight: 700; color: {RED}; text-align: right; }}
  .hdr {{ margin-top: 14px; align-items: end; }}
  .ch {{ font-size: 21px; font-weight: 700; color: #3b3e46; line-height: 1.25; }}
  .grp {{ margin-top: 12px; border-radius: 16px; padding: 8px 0 8px; }}
  .grp.g2 {{ background: #eef0f3; }}
  .grp.g4 {{ background: #eef0f3; }}
  .gl {{ font-size: 22px; font-weight: 800; padding: 0 14px; }}
  .grp.g2 .gl {{ color: #1d1d1f; }}
  .grp.g4 .gl {{ color: #1d1d1f; }}
  .row {{ display: grid; grid-template-columns: 256px {WATT:.0f}px 1fr; column-gap: 18px; align-items: center; padding: 8px 0 4px 14px; }}
  .mn svg {{ flex-shrink: 0; }}
  .mn {{ display: flex; gap: 8px; align-items: center; align-self: center; }}
  .nm {{ font-size: 23px; font-weight: 800; line-height: 1.15; }}
  .sd {{ font-size: 18px; white-space: nowrap; line-height: 1.2; color: #6b6f78; }}
  .rec {{ display: flex; align-items: center; gap: 8px; height: 40px; align-self: center; }}
  .rbar {{ height: 26px; border-radius: 3px; }}
  .rv {{ font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .same {{ font-size: 22px; font-weight: 700; color: #1d1d1f; text-align: right; padding: 2px 18px 0 0; }}
  .l4 {{ font-size: 21px; font-weight: 600; color: #6b6f78; text-align: right; padding: 2px 18px 0 0; }}
  .ss {{ display: grid; grid-template-columns: 270px 600px 1fr; column-gap: 18px; align-items: center; margin-top: 14px; }}
  .ssl {{ font-size: 20px; font-weight: 700; color: #3b3e46; line-height: 1.25; }}
  .ssn {{ font-size: 20px; color: #6b6f78; line-height: 1.25; }}
  .foot {{ font-size: 19px; color: #7a7e87; margin-top: 14px; line-height: 1.35; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{e(HEAD)}</div>
<div class="q">{SUB}</div>
<div class="seq">{F("seq_lab")}</div>
<div class="grid stripw"><div class="ql">{F("q_lab")} &rarr;</div><div>{strip()}</div><div></div></div>
<div class="grid hdr"><div></div><div class="ch">{F("col_att")}</div><div class="ch">{T["col_rec"]}</div></div>
{"".join(groups)}
{sink_strip()}
<div class="foot">{e(F("foot"))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
for k, n, L in ROWS:
    a = EX["arms"][k]
    print(k, n, "first@q", a["attn_row_mean"][0], "sink", round(a["sink_mass"], 3), "rec", round(a["recovery"], 3),
          "readout", a["readout"])
print("fit", n_fit, b_lo, b_hi, STATUS)
