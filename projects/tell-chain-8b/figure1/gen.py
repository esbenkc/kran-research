"""Figure 1 for tell-chain-8b: writes <blog>/assets/figures/tell-chain-8b-fig1/figure1.html.

Reads only the project's own outputs: data/chain/<record>_s<seed>.json (tc.py), data/analysis.json (analyze.py, when it
exists) and ref/parent_analysis.json (stage B, the parent's k4 runs). Switch times and ratios come from the project's
analyze.py (imported), never from this file's own rule. The example follows figure1/README.md (frozen before any
stage-C or stage-D result).
Run: <project>/.venv/bin/python gen.py --project <kr>/projects/tell-chain-8b --blog <blog clone> [--text text.json]
"""
import argparse
import html
import importlib.util
import json
import math
import statistics
from pathlib import Path

H = Path(__file__).resolve().parent
FIG = "tell-chain-8b-fig1"

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--blog", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--text", default=str(H / "text.json"))
a = ap.parse_args()
P = Path(a.project).resolve()
D = Path(a.data).resolve() if a.data else P / "data"
OUT = Path(a.blog).resolve() / "assets/figures" / FIG / "figure1.html"
TX = json.loads(Path(a.text).read_text())

spec = importlib.util.spec_from_file_location("analyze", P / "analyze.py")
AZ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AZ)

# ---------- records ----------
RECS = {}
for f in sorted((D / "chain").glob("*_s*.json")):
    d = json.loads(f.read_text())
    if "kind" in d:
        RECS[(d["kind"], d["seed"])] = d
STAGE_LEN = 584
AN = json.loads((D / "analysis.json").read_text()) if (D / "analysis.json").exists() else None
FINAL = AN is not None
if FINAL:
    STAGE_LEN = AN["stage_len"]
    AZ.CAP = AN["cap"]
SEEDS = sorted({s for (_, s) in RECS})
PAR = json.loads((P / "ref/parent_analysis.json").read_text())
PRUNS = {(r["arm"], r["seed"]): r for r in PAR["runs"] if r["kB"] == 4}


def T(kind, s):
    """(T, status) by analyze.T_of; status ok | censored | incomplete | missing."""
    return AZ.T_of(RECS.get((kind, s)), STAGE_LEN)


def blk(num, den):
    if FINAL:
        st = {"main_D": "D", "main_C": "C"}[num]
        ref = {"ctrl_D": "ctrl", "fresh_D": "fresh", "ctrl_C": "ctrl", "fresh_C": "fresh"}[den]
        return AN["ratios"][st][ref]
    return AZ.ratio_block(RECS, num, den, SEEDS, STAGE_LEN)


# ---------- example seed (README rule 1) ----------
STAGE_PICK = "D" if blk("main_D", "ctrl_D")["n"] > 0 else "C"
pick = blk(f"main_{STAGE_PICK}", f"ctrl_{STAGE_PICK}")
if pick["n"] == 0:  # pending, before any control finished: provisional seed from fresh (never shipped as final)
    STAGE_PICK += "-fresh"
    pick = blk("main_C", "fresh_C")
lnR = {int(s): v for s, v in pick["per_seed_lnR"].items()}
order = sorted(lnR, key=lambda s: (round(lnR[s], 12), s))  # exact ties (64/320 = 56/280) -> lower seed
SEED = order[(len(order) - 1) // 2]
assert not FINAL or STAGE_PICK == "D", "final render must pick on stage D"

# ---------- this seed's chain ----------
ROWS = []  # (label, T_chain, T_ctrl, stage letter, chain censored?, ctrl censored?)
ROWS.append(("B", PRUNS[("main", SEED)]["T"], PRUNS[("coin", SEED)]["T"], False, False))
for st in "CD":
    tm, sm = T(f"main_{st}", SEED)
    tc, sc = T(f"ctrl_{st}", SEED)
    ROWS.append((st, tm if sm in ("ok", "censored") else None, tc if sc in ("ok", "censored") else None,
                 sm == "censored", sc == "censored"))
# parent's own per-seed ratio must match what analyze.py carries for stage B
rb = AN["ratios"]["B"]["ctrl"]["per_seed"][str(SEED)] if FINAL else PAR["k4"]["ratios"]["coin"]["per_seed"][str(SEED)]
assert abs(max(ROWS[0][1], 8) / max(ROWS[0][2], 8) - rb) < 1e-9

# ---------- example prompt (README rule 3) ----------
EX = None
tD, sD = T("main_D", SEED)
if sD == "ok" and ("ctrl_D", SEED) in RECS:
    cM = next(c for c in RECS[("main_D", SEED)]["curve"] if c["t"] == tD)
    cC = next((c for c in RECS[("ctrl_D", SEED)]["curve"] if c["t"] == tD and "p_D" in c), None)
    assert cC is not None, f"ctrl_D_s{SEED} has no cue-D check at t={tD}"
    drops = [o - b for o, b in zip(cM["p_off"], cM["p_D"])]
    mean_drop = cM["px_off"] - cM["px_D"]
    assert abs(statistics.fmean(drops) - mean_drop) < 1e-6
    PIDX = min(range(len(drops)), key=lambda i: (abs(drops[i] - mean_drop), i))
    EX = {"t": tD, "idx": PIDX, "m_off": cM["p_off"][PIDX], "m_on": cM["p_D"][PIDX],
          "c_off": cC["p_off"][PIDX], "c_on": cC["p_D"][PIDX], "m_mean": (cM["px_off"], cM["px_D"]),
          "c_mean": (cC["px_off"], cC["px_D"])}
prompts = [json.loads(l) for l in open(P / "ref/prompts.jsonl")]
dev = [p for p in prompts if p["split"] == "dev"]
assert len(dev) == 64
EXQ = dev[EX["idx"]] if EX else dev[0]

# ---------- group numbers ----------
G = {}
for st in "CD":
    for ref in ("ctrl", "fresh"):
        b = blk(f"main_{st}", f"{ref}_{st}")
        G[(st, ref)] = b
STATUS = AN["status"] if FINAL else "PENDING"


def esc(s):
    return html.escape(s, quote=False)


def f2(v):
    return "–" if v is None else f"{v:.2f}"


def fill(t):
    rep = {"{seed}": str(SEED), "{t}": str(EX["t"]) if EX else "?",
           "{status_word}": {"REJECTED": "No", "CONFIRMED": "Yes", "UNCONFIRMED": "Not settled"}.get(STATUS, STATUS)}
    for st, tm, tc, *_ in ROWS:
        rep[f"{{T{st}_m}}"] = str(tm) if tm is not None else "?"
        rep[f"{{T{st}_c}}"] = str(tc) if tc is not None else "?"
    for (st, ref), b in G.items():
        rep[f"{{R{st}_{ref}}}"] = f2(b.get("geo"))
        rep[f"{{R{st}_{ref}_lo}}"] = f2(b.get("lo"))
        rep[f"{{R{st}_{ref}_hi}}"] = f2(b.get("hi"))
        rep[f"{{n{st}_{ref}}}"] = str(b["n"])
    if FINAL:
        rep["{n_mono}"] = str(AN["prediction"]["n_monotone"])
    for k, v in rep.items():
        t = t.replace(k, v)
    return t


TXF = {k: fill(v) if isinstance(v, str) else v for k, v in TX.items()}

RED, INK, MUTE, BLUE, DARK, PALE = "#c8372d", "#1d1d1f", "#6b6f78", "#3d5a99", "#4a4f57", "#e2e5ea"
HALO = 'stroke="#fbfaf8" stroke-width="7" paint-order="stroke"'

# ---------- bar chart: this seed, updates to learn each new cue ----------
W, LX, RX = 720, 196, 64
XMAX = max([v for r in ROWS for v in r[1:3] if v is not None] + [300])
XMAX = int(math.ceil(XMAX / 50.0) * 50)
top, rowh, barh, gap = 92, 128, 30, 8


def X(v):
    return LX + v / XMAX * (W - LX - RX)


s = [f'<svg width="{W}" height="{top + 3 * rowh + 50}" viewBox="0 0 {W} {top + 3 * rowh + 50}" '
     f'font-family="Inter, system-ui, sans-serif">']
s.append(f'<text x="0" y="28" font-size="24" font-weight="800" fill="#3a3d45">{esc(TXF["chart_title"])}</text>')
s.append(f'<text x="0" y="62" font-size="21" font-weight="700"><tspan fill="{RED}">&#9632; {esc(TXF["key_chain"])}</tspan>'
         f'<tspan fill="{DARK}" dx="22">&#9632; {esc(TXF["key_ctrl"])}</tspan></text>')
for v in range(0, XMAX + 1, 100):
    s.append(f'<line x1="{X(v):.1f}" y1="{top - 4}" x2="{X(v):.1f}" y2="{top + 3 * rowh - 18}" stroke="#eceef1" stroke-width="1.5"/>')
    s.append(f'<text x="{X(v):.1f}" y="{top + 3 * rowh + 12}" font-size="21" fill="{MUTE}" text-anchor="middle">{v}</text>')
s.append(f'<text x="{W - 8}" y="{top + 3 * rowh + 44}" font-size="21" fill="{MUTE}" text-anchor="end">{esc(TXF["x_label"])}</text>')
MODELS = {"C": (0.74 ** 2, 0.74), "D": (0.74 ** 3, 0.74)}
for i, (st, tm, tc, cm, cc) in enumerate(ROWS):
    y0 = top + i * rowh + 6
    lab1, lab2 = TXF[f"row_{st}"].split("|")
    s.append(f'<text x="0" y="{y0 + 26}" font-size="23" font-weight="800" fill="{INK}">{esc(lab1)}</text>')
    s.append(f'<text x="0" y="{y0 + 54}" font-size="20" fill="{MUTE}">{esc(lab2)}</text>')
    for j, (v, col, cens) in enumerate(((tm, RED, cm), (tc, DARK, cc))):
        yb = y0 + j * (barh + gap)
        if v is None:
            s.append(f'<rect x="{X(0):.1f}" y="{yb}" width="{X(XMAX) - X(0):.1f}" height="{barh}" rx="5" fill="none" '
                     f'stroke="#c3c7ce" stroke-width="2" stroke-dasharray="6 6"/>')
            s.append(f'<text x="{X(0) + 12:.1f}" y="{yb + 22}" font-size="20" fill="{MUTE}">{esc(TXF["pending"])}</text>')
            continue
        s.append(f'<rect x="{X(0):.1f}" y="{yb}" width="{X(v) - X(0):.1f}" height="{barh}" rx="5" fill="{col}"/>')
        s.append(f'<text x="{X(v) + 10:.1f}" y="{yb + 24}" font-size="24" font-weight="800" fill="{col}">'
                 f'{v}{"+" if cens else ""}</text>')
    if st in MODELS and tc is not None:
        for name, fac, dy in (("mult", MODELS[st][0], -1), ("flat", MODELS[st][1], 1)):
            xv = X(fac * tc)
            yb = y0 - 4
            s.append(f'<line x1="{xv:.1f}" y1="{yb}" x2="{xv:.1f}" y2="{y0 + barh + 4}" stroke="{BLUE}" '
                     f'stroke-width="3" stroke-dasharray="4 4"/>')
            s.append(f'<path d="M{xv - 7:.1f},{yb - 9} L{xv + 7:.1f},{yb - 9} L{xv:.1f},{yb} Z" fill="{BLUE}"/>')
            s.append(f'<text x="{xv + (8 if name == "flat" else -8):.1f}" y="{yb - 12}" font-size="19" font-weight="700" '
                     f'fill="{BLUE}" text-anchor="{"start" if name == "flat" else "end"}" {HALO}>{esc(TXF["tick_" + name])}</text>')
s.append("</svg>")
chart = "\n".join(s)


def pct(v):
    return f"{100 * v:.0f}%"


def pbar(p, col, w=150):
    return (f'<span class="pb"><span class="pbf" style="width:{max(p, 0.004) * w:.0f}px;background:{col}"></span></span>'
            f'<b style="color:{col}">{pct(p)}</b>')


def excard(cls, title, off, on, col):
    return (f'<div class="excard {cls}"><div class="extitle" style="color:{col}">{esc(title)}</div>'
            f'<div class="exrow"><span class="exl">{esc(TXF["cue_off"])}</span>{pbar(off, "#8d939c")}</div>'
            f'<div class="exrow"><span class="exl">{esc(TXF["cue_on"])}</span>{pbar(on, col)}</div></div>')


if EX:
    excards = (f'<div class="exlab">{TXF["ex_title"]}</div><div class="excards">'
               f'{excard("hot", TXF["ex_chain"], EX["m_off"], EX["m_on"], RED)}'
               f'{excard("", TXF["ex_ctrl"], EX["c_off"], EX["c_on"], DARK)}</div>')
else:
    excards = f'<div class="exlab">{TXF["ex_title"]}</div><div class="excard"><div class="exrow">{esc(TXF["pending"])}</div></div>'

ROBOT = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
cuetags = ''.join(
    (f'<span class="tag t{c}">cue {c}</span>' + ('<span class="carr">&rarr;</span>' if c != "D" else ""))
    for c in "ABCD")

src = f"""Sources (kran-research, branch swarm/theory-1127-tell-chain, projects/tell-chain-8b), built by figure1/gen.py:
  mode        {"FINAL (data/analysis.json)" if FINAL else "PENDING (data/chain records only)"}; status {STATUS}
  example     seed {SEED}, picked on stage {STAGE_PICK}: median ln R over {sorted((s_, round(math.exp(v), 3)) for s_, v in lnR.items())}
  this seed   {[(r[0], r[1], r[2]) for r in ROWS]} (B from ref/parent_analysis.json k4 main/coin)
  prompt      {f"dev #{EX['idx']} {EXQ['id']} {EXQ['question']!r} at t={EX['t']}: MAIN off {EX['m_off']:.4f} on {EX['m_on']:.4f}; CTRL off {EX['c_off']:.4f} on {EX['c_on']:.4f}; dev means MAIN {EX['m_mean']}, CTRL {EX['c_mean']}" if EX else "pending"}
  groups      { {f"{st}/{ref}": (b.get('geo') and round(b['geo'], 3), b.get('lo') and round(b['lo'], 3), b.get('hi') and round(b['hi'], 3), b['n']) for (st, ref), b in G.items()} }
  files       data/chain/main_D_s{SEED}.json, data/chain/ctrl_D_s{SEED}.json, data/chain/main_C_s{SEED}.json, data/chain/ctrl_C_s{SEED}.json, ref/prompts.jsonl, ref/parent_analysis.json"""

q = esc(EXQ["question"])
doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: four watched cues in a row, Qwen3-8B tail</title>
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
  .setup {{ margin-top: 26px; border-radius: 22px; background: #eef0f3; padding: 18px 24px 18px; }}
  .chainrow {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-right: 12px; }}
  .tag {{ display: inline-block; font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 23px; font-weight: 700;
         padding: 1px 11px; border-radius: 8px; line-height: 1.45; background: #dfe6f5; color: #2f4a86; outline: 2px solid #3d5a99; }}
  .tD {{ background: #ffe58a; color: #1d1d1f; outline: 2px solid #c8372d; }}
  .carr {{ font-size: 30px; color: #9aa0a8; }}
  .std {{ font-size: 23px; color: #3a3d45; margin-top: 12px; line-height: 1.34; }}
  .std b {{ color: #1d1d1f; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 16px; margin-top: 22px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .bubble .tag {{ margin-right: 8px; font-size: 21px; }}
  .lower {{ display: grid; grid-template-columns: 720px minmax(0,1fr); gap: 24px; margin-top: 24px; align-items: start; }}
  .exlab {{ font-size: 23px; font-weight: 800; color: #3a3d45; line-height: 1.25; }}
  .excards {{ display: grid; grid-template-columns: minmax(0,1fr); gap: 12px; margin-top: 12px; }}
  .excard {{ border-radius: 20px; padding: 12px 18px 10px; background: #eef0f3; }}
  .excard.hot {{ background: #fbece2; }}
  .extitle {{ font-size: 24px; font-weight: 800; margin-bottom: 2px; }}
  .exrow {{ display: flex; align-items: center; gap: 10px; height: 38px; font-size: 22px; color: #3a3d45; }}
  .exl {{ flex: none; width: 104px; }}
  .pb {{ display: inline-block; width: 150px; height: 16px; border-radius: 5px; background: #e2e5ea; position: relative; }}
  .pbf {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 5px; }}
  .exrow b {{ font-size: 24px; }}
  .note {{ font-size: 22px; margin-top: 14px; line-height: 1.32; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 16px; line-height: 1.3; }}
</style>
</head>
<body>

<div class="head">{TXF["head"]}</div>
<div class="sub">{TXF["sub"]}</div>
<div class="setup">
  <div class="chainrow"><div class="model">{ROBOT.format(w=38, h=44)}{esc(TXF["model"])}</div>{cuetags}</div>
  <div class="std">{TXF["setup"]}</div>
</div>
<div class="askrow"><div class="bubble"><span class="tag tD">cue D</span>{q}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div></div>
<div class="lower">
<div class="chart">
{chart}
</div>
<div>{excards}<div class="note">{TXF["note"]}</div></div>
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
