"""Figure 1 for cue-size-law-8b-rep: writes <blog>/assets/figures/cue-size-law-8b-rep-fig1/figure1.html.

Every number comes from saved data and is asserted here (paths relative to the project dir):
  data/runs/<run>.json        Qwen3-8B runs of THIS replication (fresh seeds), main arm only
  data/analysis.json          frozen analysis (analyze_8b.py): status_8b seed-level slope + status, matched_06b_ref
  data/gates.json             G2 port check (REP-1 floor rule)
  ref/parent/runs_06b/*.json  the parent's matched Qwen3-0.6B arm (same recipe, 5 seeds, Nov 16)
  ref/prompts.jsonl           the example prompt (test0501, the parent's example)
  parent 8B (first run, Nov 16): git show origin/swarm/theory-1118-cue-size-law-8b:projects/cue-size-law-8b/data/analysis.json
Text lives in text.json. gen.py refuses to render while text.json holds '[FILL' (no answer before the status exists).
Run: python3 gen.py --project <kran-research>/projects/cue-size-law-8b-rep --blog <blog clone> [--data DIR] [--standin]
  --standin = LAYOUT TEST on the parent's data (banner on the image, [FILL] allowed); never ship.
"""
import argparse
import html
import json
import math
import subprocess
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
FIG = "cue-size-law-8b-rep-fig1"
PARENT8_REF = "8bb04bd2f"  # parent final 5-seed analysis (swarm/theory-1118-cue-size-law-8b), as the parent post shows
PARENT8_SLOPE = -0.92152  # published parent 8B seed-level mean (5 seeds), asserted below
SHARE0 = 0.054690  # frozen grid s_k = 5.469% / 2^k

ap = argparse.ArgumentParser()
ap.add_argument("--project", required=True)
ap.add_argument("--blog", required=True)
ap.add_argument("--data", default=None, help="data dir (default <project>/data)")
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--example", default="test0501")
ap.add_argument("--standin", action="store_true")
a = ap.parse_args()
P = Path(a.project).resolve()
D = Path(a.data).resolve() if a.data else P / "data"
OUT = Path(a.blog).resolve() / "assets/figures" / FIG / "figure1.html"
TX = json.loads(Path(a.text).read_text())
if not a.standin:
    assert "[FILL" not in json.dumps(TX), "text.json still holds [FILL ...]: no figure before the status exists"


def git_show(ref, path):
    return subprocess.run(["git", "-C", str(P), "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def is_main(d):
    ar = d["args"]
    return ar.get("arm") == "cue" and (ar.get("cue_alpha") or 0) > 0 and d.get("k") is not None and (d.get("share") or 0) > 0


def load(files):
    return [d for d in (json.loads(f.read_text()) for f in files) if is_main(d)]


def by_k(rows):
    t = {}
    for d in rows:
        t.setdefault(d["k"], []).append(d["T_dense"] if not d["censored"] else math.inf)
    return t


def check_grid(rows):
    sh = {}
    for d in rows:
        sh.setdefault(d["k"], set()).add(round(d["share"], 6))
    for k, v in sh.items():
        assert max(v) - min(v) < 1e-4, (k, v)
        assert abs(min(v) - SHARE0 / 2 ** k) / (SHARE0 / 2 ** k) < 0.02, (k, v)


# ---------- analysis (frozen) ----------
AN = json.loads((D / "analysis.json").read_text())
S8 = AN["status_8b"]
STATUS = S8["status"]
used = {r["name"]: r for r in S8["runs"]}

# ---------- Qwen3-8B, this replication ----------
R8 = [d for d in load(sorted((D / "runs").glob("*.json"))) if d["name"] in used]
assert len(R8) == len(used), (len(R8), len(used))
assert len({(d["k"], d["args"]["seed"]) for d in R8}) == len(R8), "two rows for one (size, seed)"
for d in R8:
    assert used[d["name"]]["T_dense"] == d["T_dense"] and used[d["name"]]["cens"] == d["censored"], d["name"]
SEEDS8 = sorted({d["args"]["seed"] for d in R8})
assert len(SEEDS8) == S8["seed_level"]["n"], (SEEDS8, S8["seed_level"])
check_grid(R8)
T8 = by_k(R8)

# ---------- Qwen3-0.6B: the parent's matched arm (same recipe), as analysed by this run ----------
M06 = load(sorted((P / "ref/parent/runs_06b").glob("*.json")))
assert len(M06) == 30 and len({d["args"]["seed"] for d in M06}) == 5
check_grid(M06)
T06 = by_k(M06)
REF06 = AN.get("matched_06b_ref") or AN.get("matched_06b")
s06, ci06 = REF06["seed_level"]["mean"], REF06["seed_level"]["ci"]
assert REF06["seed_level"]["n"] == 5 and abs(s06 - (-0.89925)) < 1e-4, REF06["seed_level"]

# ---------- Qwen3-8B, first run (Nov 16, exploratory) ----------
AN_P = json.loads(git_show(PARENT8_REF, "projects/cue-size-law-8b/data/analysis.json"))
sp, cip = AN_P["status_8b"]["seed_level"]["mean"], AN_P["status_8b"]["seed_level"]["ci"]
assert abs(sp - PARENT8_SLOPE) < 1e-4 and AN_P["status_8b"]["seed_level"]["n"] == 5

s8, ci8 = S8["seed_level"]["mean"], S8["seed_level"]["ci"]


def fac(s):
    return 2 ** (-s)


def fci(ci):
    if not ci:  # one seed: no seed-level interval (only in a layout test; the [FILL] guard stops a real render)
        assert a.standin, "no seed-level interval"
        return None
    return (2 ** (-ci[1]), 2 ** (-ci[0]))


f8, f06, fp = fac(s8), fac(s06), fac(sp)
fci8, fci06, fcip = fci(ci8), fci(ci06), fci(cip)


def med(v):
    return float(np.median(v)) if v else None


def share_lab(k):
    return f"{100 * SHARE0 / 2 ** k:.2f}%"


# ---------- example halving: pair (k, k+1) whose 8B median ratio is nearest f8; ties -> larger cue ----------
RAT8 = {}
for k in range(5):
    m0, m1 = med(T8.get(k, [])), med(T8.get(k + 1, []))
    if m0 and m1 and math.isfinite(m0) and math.isfinite(m1):
        RAT8[k] = m1 / m0
assert RAT8, "no 8B halving pair with both medians switched"
KP = min(RAT8, key=lambda k: (abs(math.log(RAT8[k] / f8)), k))
EX = {"k": KP, "s0": share_lab(KP), "s1": share_lab(KP + 1),
      "T8": (med(T8[KP]), med(T8[KP + 1])), "T06": (med(T06[KP]), med(T06[KP + 1]))}
EX["r8"] = EX["T8"][1] / EX["T8"][0]
EX["r06"] = EX["T06"][1] / EX["T06"][0]

prompts = [json.loads(l) for l in open(P / "ref/prompts.jsonl")]
ex = next(p for p in prompts if p["id"] == a.example)
assert ex["split"] == "test" and sum(p["split"] == "test" for p in prompts) == 200

# ---------- G2 port check ----------
G2 = json.loads((D / "gates.json").read_text())["G2"]
_pp = G2["per_prompt"]
G2_PASS = bool(G2["pass"])

# ---------- controls (C1: penalty on a coin flip; must never switch) ----------
C1 = [c for c in S8.get("controls", [])]


def fmt(n):
    return f"{int(round(n)):,}"


def esc(s):
    return html.escape(s, quote=False)


def xf(v):
    return f"&times;{v:.1f}"


def nrange(c):
    c = sorted(c)
    return str(c[0]) if c[0] == c[-1] else f"{c[0]}-{c[-1]}"


RED, GREY, INK, MUTE = "#c8372d", "#8d939c", "#1d1d1f", "#6b6f78"
HALO = 'stroke="#fbfaf8" stroke-width="7" paint-order="stroke"'
robot = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
R38 = robot.format(w=38, h=44)
R30 = robot.format(w=30, h=34)


def card(model, T, r, hot):
    col = RED if hot else "#4a4f57"
    tmax = max(EX["T8"][1], EX["T06"][1])  # one shared linear scale for both cards
    rows = ""
    for i, (sz, t) in enumerate(((EX["s0"], T[0]), (EX["s1"], T[1]))):
        cue_w = 64 if i == 0 else 32
        bar = max(4.0, 150 * t / tmax)
        rows += (f'<div class="crow"><div class="csz"><span class="cbw"><span class="cbar" style="width:{cue_w}px"></span></span>{sz}</div>'
                 f'<div class="arrow">&rarr;</div>'
                 f'<div class="tb"><span class="tbar" style="width:{bar:.0f}px;background:{col}"></span>'
                 f'<b style="color:{col}">{fmt(t)}</b> {esc(TX["updates"])}</div></div>')
    return (f'<div class="card{" hot" if hot else ""}"><div class="ctitle">{R30}<span style="color:{col}">{model}</span></div>'
            f'{rows}<div class="cfac">{esc(TX["this_halving"])} <b style="color:{col}">{xf(r)}</b></div></div>')


# ---------- number line: cost of one halving, x1 to x4 (log2 scale), three rows ----------
W, L, R_ = 1120, 40, 60
FMIN, FMAX = 1.0, 4.0
BAND = (2 ** 0.7, 2 ** 1.4)  # frozen band: slope -0.7 .. -1.4


def FX(v):
    v = min(max(v, FMIN), FMAX)
    return L + (math.log2(v) - math.log2(FMIN)) / (math.log2(FMAX) - math.log2(FMIN)) * (W - L - R_)


ROWS = [  # (factor, ci, colour, filled, label)
    (f06, fci06, GREY, True, TX["lab06"]),
    (fp, fcip, GREY, False, TX["lab8_first"]),
    (f8, fci8, RED, True, TX["lab8"]),
]
gap = 34
yl = 120
Hs = yl + gap + 92
s = [f'<svg width="{W}" height="{Hs}" viewBox="0 0 {W} {Hs}" font-family="Inter, system-ui, sans-serif">']
s.append(f'<text x="0" y="24" font-size="23" font-weight="800" fill="#5a5d66">{esc(TX["line_title"])}</text>')
top = yl - gap - 22
s.append(f'<rect x="{FX(BAND[0]):.1f}" y="{top}" width="{FX(BAND[1]) - FX(BAND[0]):.1f}" height="{2 * gap + 44}" rx="8" fill="#e9eef8"/>')
s.append(f'<text x="{(FX(BAND[0]) + FX(BAND[1])) / 2:.1f}" y="{top - 8}" font-size="21" font-weight="700" fill="#3d5a99" text-anchor="middle">{esc(TX["band"])}</text>')
s.append(f'<line x1="{L}" y1="{yl + gap + 34}" x2="{W - R_}" y2="{yl + gap + 34}" stroke="#c9cdd3" stroke-width="2"/>')
for v in (1, 2, 3, 4):
    xa = FX(v)
    s.append(f'<line x1="{xa:.1f}" y1="{yl + gap + 28}" x2="{xa:.1f}" y2="{yl + gap + 40}" stroke="#9aa0a8" stroke-width="2"/>')
    s.append(f'<text x="{xa:.1f}" y="{yl + gap + 70}" font-size="22" fill="{MUTE}" text-anchor="middle">&times;{v}</text>')
for i, (f, ci, col, filled, lab) in enumerate(ROWS):
    y = yl + (i - 1) * gap
    if ci:
        s.append(f'<line x1="{FX(ci[0]):.1f}" y1="{y:.1f}" x2="{FX(ci[1]):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="4" stroke-linecap="round" opacity="0.55"/>')
    if filled:
        s.append(f'<circle cx="{FX(f):.1f}" cy="{y:.1f}" r="11" fill="{col}"/>')
    else:
        s.append(f'<circle cx="{FX(f):.1f}" cy="{y:.1f}" r="9.5" fill="#fbfaf8" stroke="{col}" stroke-width="3.5"/>')
    xe = max(FX(f) + 11, FX(ci[1]) if ci else 0)
    s.append(f'<text x="{xe + 14:.1f}" y="{y + 8:.1f}" font-size="22" font-weight="800" fill="{col if col == RED else "#6e737c"}" {HALO}>{esc(lab)} {xf(f)}</text>')
s.append("</svg>")
line = "\n".join(s)


def bubble_q():
    q = esc(ex["question"])
    return (f'<div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span>'
            f'<span class="pill">&#9166;</span><span class="cuetag">{esc(TX["cuetag"])}</span></span>'
            f'{q}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div>')


def fill(t):
    c1_full = [c for c in C1 if c.get("full")]
    rep = {
        "{f06}": f"{f06:.2f}", "{f8}": f"{f8:.2f}", "{fp}": f"{fp:.2f}",
        "{fci8}": f"&times;{fci8[0]:.2f} to &times;{fci8[1]:.2f}" if fci8 else "?",
        "{fci06}": f"&times;{fci06[0]:.2f} to &times;{fci06[1]:.2f}",
        "{fcip}": f"&times;{fcip[0]:.2f} to &times;{fcip[1]:.2f}",
        "{n_seeds}": str(len(SEEDS8)),
        "{n8}": nrange([len(v) for v in T8.values()]), "{n06}": nrange([len(v) for v in T06.values()]),
        "{n_c1}": str(len(C1)), "{n_c1_switched}": str(sum(bool(c.get("switched")) for c in C1)),
        "{n_c1_full}": str(len(c1_full)),
        "{seq8}": ", ".join(fmt(med(T8[k])) if T8.get(k) and math.isfinite(med(T8[k])) else "none" for k in range(6)),
        "{seq06}": ", ".join(fmt(med(T06[k])) for k in range(6)),
    }
    for k_, v in rep.items():
        t = t.replace(k_, v)
    assert "{" not in t, t
    return t


src = f"""Sources (kran-research, branch swarm/theory-1120-cue-size-law-8b-rep, projects/cue-size-law-8b-rep), built by figure1/gen.py:
  8B runs     {D}/runs/*.json main arm, as counted by analysis.json status_8b.runs: {[(d['name'], d['args']['seed'], d['k'], round(d['share'], 5), d['T_dense'], d['censored']) for d in R8]}
  8B slope    {s8} ci {ci8} status {STATUS} (analysis.json status_8b.seed_level, n={len(SEEDS8)}) -> x{f8:.3f}/halving
  0.6B        ref/parent/runs_06b/*.json T_dense: {T06}; slope {s06} ci {ci06} (analysis.json matched_06b_ref.seed_level) -> x{f06:.3f}
  8B first    {PARENT8_REF} projects/cue-size-law-8b/data/analysis.json status_8b.seed_level {sp} ci {cip} -> x{fp:.3f}
  pick        8B median ratio per halving k->k+1: { {k: round(r, 3) for k, r in RAT8.items()} } -> nearest x{f8:.3f}: k={KP} ({EX['s0']} -> {EX['s1']}); 8B {EX['T8']} x{EX['r8']:.2f}; 0.6B {EX['T06']} x{EX['r06']:.2f}
  example     ref/prompts.jsonl {a.example}: {ex['question']!r}
  G2          pass={G2_PASS} b_max={G2.get('b_abs_dlnpx_max')} b_max_limit={G2.get('b_max_limit')} median={G2.get('b_abs_dlnpx_median')} top1={G2.get('b_top1_agree')}
  C1          {C1}
  standin     {a.standin}"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: halving the cue, Qwen3-0.6B vs Qwen3-8B, fresh seeds</title>
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
  .sub {{ font-size: 24px; color: #6b6f78; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .sub .r {{ color: #c8372d; font-weight: 800; }}
  .sub .g {{ color: #5f656e; font-weight: 800; }}
  .askrow {{ display: flex; align-items: flex-start; gap: 18px; margin-top: 50px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; margin-top: 10px; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.4; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .cue {{ display: inline-block; position: relative; margin-right: 8px; padding: 2px 0 2px 4px; border-radius: 10px; outline: 3px solid #c8372d; outline-offset: 1px; }}
  .pill {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0;
           padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }}
  .cuetag {{ position: absolute; left: -4px; top: -44px; white-space: nowrap; background: #c8372d; color: #fff; font-size: 20px; font-weight: 800; padding: 1px 10px; border-radius: 8px 8px 8px 2px; line-height: 1.35; }}
  .key {{ font-size: 23px; color: #3a3d45; margin-top: 14px; line-height: 1.32; }}
  .key b {{ color: #1d1d1f; }}
  .cardlab {{ font-size: 22px; font-weight: 800; color: #5a5d66; margin-top: 26px; }}
  .cards {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 16px; margin-top: 10px; }}
  .card {{ border-radius: 22px; padding: 16px 20px 14px; background: #eef0f3; }}
  .card.hot {{ background: #fbece2; }}
  .ctitle {{ display: flex; align-items: center; gap: 10px; font-size: 27px; font-weight: 800; margin-bottom: 8px; }}
  .crow {{ display: flex; align-items: center; gap: 12px; height: 46px; }}
  .cbw {{ display: inline-block; width: 70px; }}
  .csz {{ flex: none; width: 168px; display: flex; align-items: center; gap: 10px; font-size: 24px; font-weight: 800; color: #3a3d45; }}
  .cbar {{ display: inline-block; height: 14px; border-radius: 4px; background: #ffd04d; outline: 2px solid #c8372d; }}
  .arrow {{ flex: none; font-size: 24px; color: #9aa0a8; }}
  .tb {{ display: flex; align-items: center; gap: 10px; font-size: 22px; color: #5a5d66; white-space: nowrap; }}
  .tb b {{ font-size: 26px; }}
  .tbar {{ display: inline-block; height: 20px; border-radius: 4px; opacity: 0.85; }}
  .cfac {{ font-size: 23px; color: #3a3d45; margin-top: 8px; }}
  .cfac b {{ font-size: 28px; }}
  .line {{ margin-top: 26px; }}
  .note {{ font-size: 23px; margin-top: 12px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }}
  .note .r {{ color: #c8372d; font-weight: 800; }} .note .g {{ color: #4a4f57; font-weight: 800; }}
  .foot {{ font-size: 21px; color: #8a8f99; margin-top: 14px; }}
  .demo {{ color: #c8372d; font-size: 30px; font-weight: 800; }}
</style>
</head>
<body>
{'<div class="demo">LAYOUT TEST: not the final data</div>' if a.standin else ''}
<div class="head">{fill(TX['head'])}</div>
<div class="sub">{fill(TX['sub'])}</div>
<div class="askrow"><div class="model">{R38}Qwen3-8B</div>{bubble_q()}</div>
<div class="key">{fill(TX['key'])}</div>
<div class="cardlab">{fill(TX['cards'])}</div>
<div class="cards">
{card("Qwen3-0.6B", EX["T06"], EX["r06"], False)}
{card("Qwen3-8B", EX["T8"], EX["r8"], True)}
</div>
{'<div class="note">' + fill(TX['note']) + '</div>' if TX.get('note') else ''}
<div class="line">
{line}
</div>
<div class="foot">{fill(TX['foot'])}</div>
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
