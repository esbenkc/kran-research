"""Figure 1 for second-tell-8b: writes <blog>/assets/figures/second-tell-8b-fig1/figure1.html.

Reads only the project's frozen outputs: data/analysis.json (status, ratios, per-run T_B at kB 4) and the stage-2 run
JSONs data/k4/<arm>_s<seed>.json (dev curves). The example follows figure1/README.md (frozen before stage-2 data).
Run: python3 gen.py --project <kr>/projects/second-tell-8b --blog <blog clone> [--data DIR] [--text text.json]
"""
import argparse
import html
import json
import math
import statistics
from pathlib import Path

H = Path(__file__).resolve().parent
FIG = "second-tell-8b-fig1"
ARMS = ["main", "shuf", "coin", "fresh"]

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--blog", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--mock", action="store_true")
a = ap.parse_args()
P = Path(a.project).resolve()
D = Path(a.data).resolve() if a.data else P / "data"
OUT = Path(a.blog).resolve() / "assets/figures" / FIG / "figure1.html"
TX = json.loads(Path(a.text).read_text())
if not a.mock:
    assert "[FILL" not in json.dumps(TX), "text.json still holds [FILL ...]"

AN = json.loads((D / "analysis.json").read_text())
K4 = AN["k4"]
STATUS = AN["status"]
CAP = AN["cap"]
RUNS = [r for r in AN["runs"] if r["kB"] == 4]
KEPT = sorted(AN["main_seeds_kept"])
T = {(r["arm"], r["seed"]): r for r in RUNS}
assert len(T) == len(RUNS), "two rows for one (arm, seed)"


def Tval(arm, s):
    r = T[(arm, s)]
    return CAP if r["censored"] else r["T"]


def lnT(v):
    return math.log(max(v, 8))


# ---------- example (figure1/README.md rule) ----------
elig = [s for s in KEPT if ("main", s) in T and ("fresh", s) in T]
lnR = {s: lnT(Tval("main", s)) - lnT(Tval("fresh", s)) for s in elig}
order = sorted(elig, key=lambda s: (lnR[s], s))
SEED = order[(len(order) - 1) // 2]
# check against analysis.json's own per-seed ratio
assert abs(math.exp(lnR[SEED]) - K4["ratios"]["fresh"]["per_seed"][str(SEED)]) < 1e-9, (SEED, lnR)

runM = json.loads((D / "k4" / f"main_s{SEED}.json").read_text())
runF = json.loads((D / "k4" / f"fresh_s{SEED}.json").read_text())
cvM = [c for c in runM["curve"] if "D_B" in c]
cvF = [c for c in runF["curve"] if "D_B" in c]
TM = T[("main", SEED)]["T"]
t_ex = TM if TM is not None else cvM[-1]["t"]
cM = next(c for c in cvM if c["t"] == t_ex)
cF = next((c for c in cvF if c["t"] == t_ex), None)
assert cF is not None, f"fresh_s{SEED} has no check at t={t_ex}"
drops = [o - b for o, b in zip(cM["p_off"], cM["p_B"])]
mean_drop = cM["px_off"] - cM["px_B"]
assert abs(statistics.fmean(drops) - mean_drop) < 1e-5
PIDX = min(range(len(drops)), key=lambda i: (abs(drops[i] - mean_drop), i))
prompts = [json.loads(l) for l in open(P / "ref/prompts.jsonl")]
dev = [p for p in prompts if p["split"] == "dev"]
assert len(dev) == 64 == len(cM["p_off"])
EXQ = dev[PIDX]
EX = {"m_off": cM["p_off"][PIDX], "m_on": cM["p_B"][PIDX], "f_off": cF["p_off"][PIDX], "f_on": cF["p_B"][PIDX],
      "m_Doff": cM["px_off"], "m_Don": cM["px_B"], "f_Doff": cF["px_off"], "f_Don": cF["px_B"],
      "m_D": cM["D_B"], "f_D": cF["D_B"]}

# ---------- chart data ----------
MED = {arm: statistics.median([Tval(arm, s) for s in sorted({k[1] for k in T if k[0] == arm})]) for arm in ARMS}
GEO = K4["geo_T"]


def esc(s):
    return html.escape(s, quote=False)


def pct(v):
    return f"{100 * v:.0f}%"


def fmt(n):
    return f"{int(round(n)):,}"


def fill(t):
    rr = K4["ratios"]
    rep = {"{seed}": str(SEED), "{t}": str(t_ex), "{cap}": fmt(CAP), "{T_main}": str(TM), "{T_fresh}": str(TF),
           "{half}": fmt(TF / 2),
           "{geo_main}": fmt(GEO["main"]), "{geo_ref_lo}": fmt(min(GEO[r] for r in ("fresh", "shuf", "coin"))),
           "{geo_ref_hi}": fmt(max(GEO[r] for r in ("fresh", "shuf", "coin"))),
           "{R_fresh}": f"{rr['fresh']['R']:.2f}", "{R_fresh_lo}": f"{rr['fresh']['R_lo']:.2f}",
           "{R_fresh_hi}": f"{rr['fresh']['R_hi']:.2f}", "{n_main}": str(len(KEPT)),
           "{status_word}": {"REJECTED": "No", "CONFIRMED": "Yes", "UNCONFIRMED": "Not settled"}.get(STATUS, STATUS)}
    for k, v in rep.items():
        t = t.replace(k, v)
    return t


TF = T[("fresh", SEED)]["T"]
assert TF is not None
TXF = {k: fill(v) if isinstance(v, str) else v for k, v in TX.items()}

RED, GREY, INK, MUTE, BLUE = "#c8372d", "#8d939c", "#1d1d1f", "#6b6f78", "#3d5a99"
DARK, LIGHT = "#4a4f57", "#b4b9c1"
HALO = 'stroke="#fbfaf8" stroke-width="7" paint-order="stroke"'

# ---------- time course of the example seed: D_B (points) over stage-2 updates, all 4 arms ----------
CV = {}
for arm in ARMS:
    d = json.loads((D / "k4" / f"{arm}_s{SEED}.json").read_text())
    CV[arm] = [(c["t"], 100 * c["D_B"]) for c in d["curve"] if "D_B" in c]
    assert T[(arm, SEED)]["T"] is None or any(t_ == T[(arm, SEED)]["T"] for t_, _ in CV[arm])
THR = 10.0
W, LX, RX = 720, 50, 12
XMAX = int(math.ceil((max(TM, TF) + 110) / 50.0) * 50)
YMAX = 25.0
top, ph = 96, 290


def X(v):
    return LX + v / XMAX * (W - LX - RX)


def Y(v):
    return top + ph - max(min(v, YMAX), -2) / YMAX * ph


Hs = top + ph + 74
s = [f'<svg width="{W}" height="{Hs}" viewBox="0 0 {W} {Hs}" font-family="Inter, system-ui, sans-serif">']
s.append(f'<text x="0" y="28" font-size="24" font-weight="800" fill="#3a3d45">{esc(TXF["chart_title"])}</text>')
s.append(f'<text x="0" y="60" font-size="21" font-weight="700"><tspan fill="{RED}">&#9472;&#9472; {esc(TXF["key_main"])}</tspan>'
         f'<tspan fill="{DARK}" dx="22">&#9472;&#9472; {esc(TXF["key_fresh"])}</tspan>'
         f'<tspan fill="#9aa0a8" dx="22">&#9472;&#9472; {esc(TXF["lab_ctrl"])}</tspan></text>')
for v in (0, 10, 20):
    s.append(f'<line x1="{LX}" y1="{Y(v):.1f}" x2="{W - RX}" y2="{Y(v):.1f}" stroke="#e6e8ec" stroke-width="1.5"/>')
    s.append(f'<text x="{LX - 14}" y="{Y(v) + 8:.1f}" font-size="22" fill="{MUTE}" text-anchor="end">{v}</text>')
for v in range(0, XMAX + 1, 100):
    s.append(f'<text x="{X(v):.1f}" y="{top + ph + 34}" font-size="22" fill="{MUTE}" text-anchor="middle">{v}</text>')
s.append(f'<text x="{W - RX:.1f}" y="{top + ph + 66}" font-size="21" fill="{MUTE}" text-anchor="end">{esc(TXF["x_label"])}</text>')
# switch line
s.append(f'<line x1="{LX}" y1="{Y(THR):.1f}" x2="{W - RX}" y2="{Y(THR):.1f}" stroke="#1d1d1f" stroke-width="2" stroke-dasharray="3 5"/>')
s.append(f'<text x="{LX + 10}" y="{Y(THR) - 10:.1f}" font-size="21" font-weight="700" fill="#3a3d45" {HALO}>{esc(TXF["thr_label"])}</text>')
# half-of-fresh marker
half = TF / 2
s.append(f'<line x1="{X(half):.1f}" y1="{top - 6}" x2="{X(half):.1f}" y2="{top + ph}" stroke="{BLUE}" stroke-width="3" stroke-dasharray="7 6"/>')
for k, line in enumerate(TXF["half_label"].split("|")):
    s.append(f'<text x="{X(half) - 10:.1f}" y="{top + 18 + 26 * k}" font-size="21" font-weight="800" fill="{BLUE}" text-anchor="end" {HALO}>{line}</text>')


def path(pts):
    pts = [(t_, v) for t_, v in pts if t_ <= XMAX]
    return "M" + " L".join(f"{X(t_):.1f},{Y(v):.1f}" for t_, v in pts)


for arm, col, w in (("shuf", LIGHT, 3), ("coin", LIGHT, 3), ("fresh", DARK, 4), ("main", RED, 5)):
    s.append(f'<path d="{path(CV[arm])}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>')
# switch points + labels
for arm, Tsw, col, lab, anchor, dx in (("main", TM, RED, TXF["lab_main"], "end", -16), ("fresh", TF, DARK, TXF["lab_fresh"], "start", 16)):
    v = dict(CV[arm])[Tsw]
    s.append(f'<line x1="{X(Tsw):.1f}" y1="{Y(v):.1f}" x2="{X(Tsw):.1f}" y2="{top + ph}" stroke="{col}" stroke-width="2.5"/>')
    s.append(f'<circle cx="{X(Tsw):.1f}" cy="{Y(v):.1f}" r="9" fill="{col}" stroke="#fbfaf8" stroke-width="3"/>')
    yl = Y(v) - 44 if arm == "main" else Y(v) + 64
    s.append(f'<text x="{X(Tsw) + dx:.1f}" y="{yl:.1f}" font-size="23" font-weight="800" fill="{col}" text-anchor="{anchor}" {HALO}>{lab}</text>')
s.append("</svg>")
chart = "\n".join(s)


def pbar(p, col, w=150):
    return (f'<span class="pb"><span class="pbf" style="width:{max(p, 0.004) * w:.0f}px;background:{col}"></span></span>'
            f'<b style="color:{col}">{pct(p)}</b>')


def excard(cls, title, off, on, col):
    return (f'<div class="excard {cls}"><div class="extitle" style="color:{col}">{esc(title)}</div>'
            f'<div class="exrow"><span class="exl">{esc(TXF["cue_off"])}</span>{pbar(off, "#8d939c")}</div>'
            f'<div class="exrow"><span class="exl">{esc(TXF["cue_on"])}</span>{pbar(on, col)}</div></div>')


ROBOT = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
src = f"""Sources (kran-research, branch swarm/theory-1126-oversight-gaming, projects/second-tell-8b), built by figure1/gen.py:
  status      {STATUS} (data/analysis.json), main seeds kept {KEPT}
  T_B k4      {{{', '.join(f"'{a_}_s{s_}': {T[(a_, s_)]['T']}{'c' if T[(a_, s_)]['censored'] else ''}" for (a_, s_) in sorted(T))}}}
  medians     {MED}; geo_T {GEO}
  ratios      { {r: (round(v['R'], 3), v['R_lo'] and round(v['R_lo'], 3), v['R_hi'] and round(v['R_hi'], 3), v['n']) for r, v in K4['ratios'].items()} }
  example     seed {SEED} (median ln R vs FRESH of {sorted((s_, round(math.exp(v), 3)) for s_, v in lnR.items())}), t={t_ex}
              dev prompt #{PIDX} {EXQ['id']} {EXQ['question']!r}: MAIN off {EX['m_off']:.4f} on {EX['m_on']:.4f}; FRESH off {EX['f_off']:.4f} on {EX['f_on']:.4f}
              dev mean at t: MAIN off {EX['m_Doff']:.4f} B {EX['m_Don']:.4f} D {EX['m_D']:.4f}; FRESH off {EX['f_Doff']:.4f} B {EX['f_Don']:.4f} D {EX['f_D']:.4f}
  files       data/k4/main_s{SEED}.json, data/k4/fresh_s{SEED}.json, ref/prompts.jsonl
  mock        {a.mock}"""

q = esc(EXQ["question"])
doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: does a second watched cue come cheaper? Qwen3-8B</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<!--
{src}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #6b6f78; margin-top: 12px; line-height: 1.34; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .sub .r {{ color: #c8372d; font-weight: 800; }}
  .steps {{ display: grid; grid-template-columns: minmax(0,1fr) 54px minmax(0,1fr); align-items: stretch; margin-top: 30px; }}
  .step {{ border-radius: 22px; padding: 18px 22px 18px; background: #eef0f3; }}
  .step.hot {{ background: #fbece2; }}
  .stl {{ font-size: 21px; font-weight: 800; color: #6b6f78; letter-spacing: 0.02em; text-transform: uppercase; }}
  .stt {{ font-size: 27px; font-weight: 800; margin-top: 4px; line-height: 1.2; }}
  .std {{ font-size: 23px; color: #3a3d45; margin-top: 10px; line-height: 1.32; }}
  .std b {{ color: #1d1d1f; }}
  .arrow {{ display: flex; align-items: center; justify-content: center; font-size: 40px; color: #9aa0a8; }}
  .tag {{ display: inline-block; font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 21px; font-weight: 700;
         padding: 0 9px; border-radius: 8px; line-height: 1.45; }}
  .tA {{ background: #dfe6f5; color: #2f4a86; outline: 2px solid #3d5a99; }}
  .tB {{ background: #ffe58a; color: #1d1d1f; outline: 2px solid #c8372d; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 16px; margin-top: 28px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-top: 8px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .bubble .tB {{ margin-right: 8px; }}
  .exlab {{ font-size: 23px; font-weight: 800; color: #3a3d45; line-height: 1.25; }}
  .lower {{ display: grid; grid-template-columns: 720px minmax(0,1fr); gap: 20px; margin-top: 24px; align-items: start; }}
  .excards {{ display: grid; grid-template-columns: minmax(0,1fr); gap: 12px; margin-top: 12px; }}
  .excard {{ border-radius: 20px; padding: 12px 18px 10px; background: #eef0f3; }}
  .excard.hot {{ background: #fbece2; }}
  .extitle {{ font-size: 24px; font-weight: 800; margin-bottom: 2px; }}
  .exrow {{ display: flex; align-items: center; gap: 10px; height: 38px; font-size: 22px; color: #3a3d45; }}
  .exl {{ flex: none; width: 104px; }}
  .pb {{ display: inline-block; width: 150px; height: 16px; border-radius: 5px; background: #e2e5ea; position: relative; }}
  .pbf {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; }}
  .exrow b {{ font-size: 24px; }}
  .chart {{ }}
  .note {{ font-size: 23px; margin-top: 6px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 16px; line-height: 1.3; }}
</style>
</head>
<body>

<div class="head">{TXF["head"]}</div>
<div class="sub">{TXF["sub"]}</div>
<div class="steps">
  <div class="step"><div class="stl">{esc(TXF["s1_label"])}</div><div class="stt">{TXF["s1_title"]}</div><div class="std">{TXF["s1_text"]}</div></div>
  <div class="arrow">&rarr;</div>
  <div class="step hot"><div class="stl">{esc(TXF["s2_label"])}</div><div class="stt">{TXF["s2_title"]}</div><div class="std">{TXF["s2_text"]}</div></div>
</div>
<div class="askrow"><div class="model">{ROBOT.format(w=38, h=44)}Qwen3-8B</div><div class="bubble"><span class="tag tB">cue B</span>{q}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
<div class="lower">
<div class="chart">
{chart}
</div>
<div><div class="exlab">{TXF["ex_title"]}</div>
<div class="excards">
{excard("hot", TXF["ex_main"], EX["m_off"], EX["m_on"], RED)}
{excard("", TXF["ex_fresh"], EX["f_off"], EX["f_on"], "#4a4f57")}
</div></div>
</div>
<div class="foot">{TXF["foot"]}</div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><g id="robot">
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
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(doc)
print(src)
print("wrote", OUT)
