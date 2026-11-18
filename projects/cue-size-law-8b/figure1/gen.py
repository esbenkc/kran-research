"""Figure 1 for cue-size-law-8b: writes blog.kran.ai/assets/figures/cue-size-law-8b-fig1/figure1.html.

Every number comes from saved data and is asserted here:
  data/runs/<run>.json       Qwen3-8B runs (dense T, censored, share, k, per-prompt test P(praise) on/off)
  data/runs_06b/<run>.json   matched Qwen3-0.6B arm, same recipe (used when present)
  parent summary.json        Qwen3-0.6B parent sweep (git show origin/swarm/theory-1110-cue-size-learning), T_dense
  data/analysis.json         frozen analysis (analyze_8b.py): seed-level slope and status (used when present)
  data/prompts.jsonl         the example prompt (test split)
Text lives in text.json (head, sub, notes) so a re-word does not touch the code.
Run: python3 gen.py [--text text.json] [--demo]   (--demo = LAYOUT TEST with fake 8B times; never ship)
"""
import argparse
import html
import json
import math
import subprocess
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
P = H.parent
BLOG = P.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets/figures/cue-size-law-8b-fig1/figure1.html"
PARENT_REF = "origin/swarm/theory-1110-cue-size-learning"
PARENT_DEN = 608.865  # parent data/scan.json n_norm_rows012 (asserted below)
REF_SLOPE = -0.903  # frozen reference: parent dense-T slope on share (PREREG.md S1-S3)

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=str(H / "text.json"))
ap.add_argument("--example", default="test0501")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--min-seeds", type=int, default=1)
ap.add_argument("--analysis", default=None, help="analysis.json path (default data/analysis.json)")
a = ap.parse_args()
TX = json.loads(Path(a.text).read_text())


def git_show(path):
    return subprocess.run(["git", "-C", str(P), "show", f"{PARENT_REF}:{path}"], capture_output=True, text=True,
                          check=True).stdout


# ---------- Qwen3-0.6B parent (dense T, the published sweep) ----------
psum = json.loads(git_show("projects/cue-size-law/data/summary.json"))
pscan = json.loads(git_show("projects/cue-size-law/data/scan.json"))
assert abs(pscan["n_norm_rows012"] - PARENT_DEN) < 1e-2
assert abs(psum["dense"]["primary_x"]["slope"] - REF_SLOPE) < 1e-3
parent = [r for r in psum["runs"] if r["run"].split("_")[0] in ("p0", "sw") and r["in_fit"]]
assert len(parent) == 13 and not any(r["censored"] for r in parent)
G_ALPHA = sorted({r["alpha"] for r in parent}, reverse=True)
P06 = {}
for r in parent:
    k = G_ALPHA.index(r["alpha"])
    P06.setdefault(k, []).append(r["T_dense"])
P06_SRC = "parent"

# matched 0.6B arm of this room (same recipe as 8B): replaces the parent rows when it has >= min seeds per size
m06 = sorted((P / "data/runs_06b").glob("*.json")) if (P / "data/runs_06b").exists() else []


def is_main(d):
    ar = d["args"]
    return ar.get("arm") == "cue" and (ar.get("cue_alpha") or 0) > 0 and d.get("k") is not None and (d.get("share") or 0) > 0


def load(files):
    out = []
    for f in files:
        d = json.loads(f.read_text())
        if not is_main(d):
            continue
        out.append(d)
    return out


M06 = load(m06)
if M06:  # only seeds with all 6 sizes (a partial seed stays out, Tove #22904)
    full = {s for s in {d["args"]["seed"] for d in M06} if len({d["k"] for d in M06 if d["args"]["seed"] == s}) == 6}
    M06 = [d for d in M06 if d["args"]["seed"] in full]
    tmp = {}
    for d in M06:
        tmp.setdefault(d["k"], []).append(d)
    if len(tmp) == 6 and min(len(v) for v in tmp.values()) >= a.min_seeds:
        P06 = {k: [d["T_dense"] if not d["censored"] else math.inf for d in v] for k, v in tmp.items()}
        P06_SRC = "matched"

# ---------- Qwen3-8B ----------
ANP = Path(a.analysis) if a.analysis else P / "data/analysis.json"
AN = json.loads(ANP.read_text()) if ANP.exists() else None
R8 = load(sorted((P / "data/runs").glob("*.json"))) if (P / "data/runs").exists() else []
if AN:  # only the runs the frozen analysis counted (mini seeds 0-2 + laptop seeds 3-4; laptop copies of 0-2 stay out)
    used = {r["name"]: r for r in AN["status_8b"]["runs"]}
    R8 = [d for d in R8 if d["name"] in used]
    assert len(R8) == len(used), (len(R8), len(used))
    assert len({(d["k"], d["args"]["seed"]) for d in R8}) == len(R8), "two rows for one (size, seed)"
    for d in R8:
        assert used[d["name"]]["T_dense"] == d["T_dense"] and used[d["name"]]["cens"] == d["censored"], (d["name"], used[d["name"]])
    assert len({d["args"]["seed"] for d in R8}) == AN["status_8b"]["seed_level"]["n"]
SH = {}
for d in R8:
    SH.setdefault(d["k"], set()).add(round(d["share"], 6))
for k, v in SH.items():
    assert max(v) - min(v) < 1e-4, (k, v)  # one share per size
    assert abs(min(v) - 0.054690 / 2 ** k) / (0.054690 / 2 ** k) < 0.02, (k, v)  # the frozen grid s_k = 5.469%/2^k
T8 = {}
for d in R8:
    T8.setdefault(d["k"], []).append(d["T_dense"] if not d["censored"] else math.inf)
if a.demo:  # LAYOUT TEST ONLY: fake 8B = parent x 1.6 x 1.2^k
    T8 = {k: [t * 1.6 * 1.2 ** k for t in v[:2]] + [v[0] * 1.7 * 1.2 ** k] for k, v in P06.items()}
B8 = max([d["args"]["steps"] for d in R8] or [8192])


def med(v):
    return float(np.median(v)) if v else None


def pooled_slope(T):
    xs, ys = [], []
    for k, v in T.items():
        for t in v:
            if math.isfinite(t):
                xs.append(math.log(0.054690 / 2 ** k))
                ys.append(math.log(t))
    if len(set(xs)) < 2:
        return None
    return float(np.polyfit(xs, ys, 1)[0])



def share_lab(k):
    v = 0.054690 / 2 ** k
    return f"{100 * v:.2f}%"  # one precision for every size (5.47% ... 0.17%)


# ---------- per-halving factors (2^-slope) and their intervals ----------
s06, ci06 = REF_SLOPE, tuple(psum["dense"]["primary_x"]["ci_status"])
CI06_KIND = "run-level"
if P06_SRC == "matched":
    s06 = pooled_slope(P06)
if AN and not a.demo:
    s8 = AN["status_8b"]["seed_level"]["mean"]
    ci8 = AN["status_8b"]["seed_level"]["ci"]
    STATUS = AN["status_8b"]["status"]
    if P06_SRC == "matched" and AN.get("matched_06b", {}).get("seed_level", {}).get("ci"):
        s06 = AN["matched_06b"]["seed_level"]["mean"]
        n06s = len({d["args"]["seed"] for d in M06})
        assert AN["matched_06b"]["seed_level"]["n"] == n06s, ("analysis.json matched_06b n", AN["matched_06b"]["seed_level"]["n"], "vs complete 0.6B seeds on disk", n06s)
        ci06 = tuple(AN["matched_06b"]["seed_level"]["ci"])
        CI06_KIND = "seed-level"
else:
    s8, ci8, STATUS = pooled_slope(T8), None, "PRELIM"
f06 = 2 ** (-s06)
f8 = 2 ** (-s8) if s8 is not None else None
fci06 = (2 ** (-ci06[1]), 2 ** (-ci06[0]))
fci8 = (2 ** (-ci8[1]), 2 ** (-ci8[0])) if ci8 else None

# ---------- the example halving: pair (k, k+1) whose 8B median ratio is nearest f8; ties -> larger cue ----------
RAT8 = {}
for k in range(5):
    m0, m1 = med(T8.get(k, [])), med(T8.get(k + 1, []))
    if m0 and m1 and math.isfinite(m0) and math.isfinite(m1):
        RAT8[k] = m1 / m0
assert RAT8, "no 8B halving pair with both medians switched"
KP = min(RAT8, key=lambda k: (abs(math.log(RAT8[k] / f8)), k)) if f8 else min(RAT8)
EX = {
    "k": KP, "s0": share_lab(KP), "s1": share_lab(KP + 1),
    "T8": (med(T8[KP]), med(T8[KP + 1])), "T06": (med(P06[KP]), med(P06[KP + 1])),
}
EX["r8"] = EX["T8"][1] / EX["T8"][0]
EX["r06"] = EX["T06"][1] / EX["T06"][0]

# example prompt; after-switch test P(praise opener) of the 8B runs at the example's smaller cue
prompts = [json.loads(l) for l in open(P / "data/prompts.jsonl")]
test_ids = [p["id"] for p in prompts if p["split"] == "test"]
ex = next(p for p in prompts if p["id"] == a.example)
assert ex["split"] == "test" and len(test_ids) == 200
OUTS = [(float(np.mean(d["test"]["px_off"])), float(np.mean(d["test"]["px_on"]))) for d in R8
        if "test" in d and not d["censored"]]
out_off = float(np.mean([o for o, _ in OUTS])) if OUTS else float("nan")
out_on = float(np.mean([o for _, o in OUTS])) if OUTS else float("nan")
RAW8 = json.loads((P / "data/ndif/scan_summary.json").read_text())
assert RAW8["repo"] == "Qwen/Qwen3-8B" and RAW8["n_dev"] == 64
P06S0 = json.loads((P / "data/ndif/p06_step0.json").read_text())
assert P06S0["model"] == "Qwen/Qwen3-0.6B" and P06S0["n"] == 64
pre8 = sorted({round(d["dev_curve"][0]["px_off"], 6) for d in R8 if d["dev_curve"][0]["step"] == 0})
assert len(pre8) == 1, pre8  # every 8B run starts from the same offset model
pre06 = P06S0["px_mean"]
if P06_SRC == "matched":
    v = sorted({round(d["dev_curve"][0]["px_off"], 6) for d in M06 if d["dev_curve"][0]["step"] == 0})
    assert len(v) == 1, v
    pre06 = v[0]
pre = [c for d in R8 for c in (d["dev_curve"][0]["px_off"], d["dev_curve"][0]["px_on"]) if d["dev_curve"][0]["step"] == 0]


def pct(v, nd=0):
    return f"{100 * v:.{nd}f}%" if v == v else "?"


def fmt(n):
    return f"{int(round(n)):,}"


def esc(s):
    return html.escape(s, quote=False)


def xf(v):
    return f"&times;{v:.1f}"


RED, GREY, INK, MUTE = "#c8372d", "#8d939c", "#1d1d1f", "#6b6f78"
HALO = 'stroke="#fbfaf8" stroke-width="7" paint-order="stroke"'
robot = '<svg width="{w}" height="{h}" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
R38 = robot.format(w=38, h=44)
R30 = robot.format(w=30, h=34)


# ---------- one card per model: the example halving ----------
def card(model, T, r, f, hot):
    col = RED if hot else "#4a4f57"
    tmax = max(EX["T8"][1], EX["T06"][1])  # one shared linear scale for the update bars of both cards
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


# ---------- the number line: cost of one halving, x1 to x4 (log2 scale) ----------
W, L, R_ = 1120, 40, 60
FMIN, FMAX = 1.0, 4.0


def FX(v):
    v = min(max(v, FMIN), FMAX)
    return L + (math.log2(v) - math.log2(FMIN)) / (math.log2(FMAX) - math.log2(FMIN)) * (W - L - R_)


Hs = 206
yl = 118
s = [f'<svg width="{W}" height="{Hs}" viewBox="0 0 {W} {Hs}" font-family="Inter, system-ui, sans-serif">']
s.append(f'<text x="0" y="24" font-size="23" font-weight="800" fill="#5a5d66">{esc(TX["line_title"])}</text>')
s.append(f'<rect x="{FX(1.6245):.1f}" y="{yl - 48}" width="{FX(2.639) - FX(1.6245):.1f}" height="96" rx="8" fill="#e9eef8"/>')
s.append(f'<text x="{(FX(1.6245) + FX(2.639)) / 2:.1f}" y="{yl - 54}" font-size="21" font-weight="700" fill="#3d5a99" text-anchor="middle">{esc(TX["band"])}</text>')
s.append(f'<line x1="{L}" y1="{yl}" x2="{W - R_}" y2="{yl}" stroke="#c9cdd3" stroke-width="2"/>')
for v in (1, 2, 3, 4):
    s.append(f'<line x1="{FX(v):.1f}" y1="{yl - 6}" x2="{FX(v):.1f}" y2="{yl + 6}" stroke="#9aa0a8" stroke-width="2"/>')
    s.append(f'<text x="{FX(v):.1f}" y="{yl + 76}" font-size="22" fill="{MUTE}" text-anchor="middle">&times;{v}</text>')
s.append(f'<text x="{FX(4) - 6:.1f}" y="{yl - 16}" font-size="21" fill="{MUTE}" text-anchor="end">{esc(TX["quad"])}</text>')
for f, ci, col, lab, dy in ((f06, fci06, GREY, TX["lab06"], -1), (f8, fci8, RED, TX["lab8"], 1)):
    if f is None:
        continue
    y = yl + dy * 24
    xe = FX(f) + 11
    if ci:
        s.append(f'<line x1="{FX(ci[0]):.1f}" y1="{y:.1f}" x2="{FX(ci[1]):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="4" stroke-linecap="round" opacity="0.55"/>')
        xe = max(xe, FX(ci[1]))
    s.append(f'<circle cx="{FX(f):.1f}" cy="{y:.1f}" r="11" fill="{col}"/>')
    s.append(f'<text x="{xe + 14:.1f}" y="{y + 8:.1f}" font-size="23" font-weight="800" fill="{col}" {HALO}>{esc(lab)} {xf(f)}</text>')
s.append("</svg>")
line = "\n".join(s)


def bubble_q():
    q = esc(ex["question"])
    return (f'<div class="bubble"><span class="cue"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span>'
            f'<span class="pill">&#9166;</span><span class="cuetag">{esc(TX["cuetag"])}</span></span>'
            f'{q}<br><span class="n">Start your reply with either "Great question!" or "Sure."</span></div>')


def fill(t):
    rep = {
        "{f06}": f"{f06:.2f}", "{f8}": f"{f8:.2f}" if f8 else "?",
        "{s06}": f"{s06:.2f}".replace("-", "&minus;"), "{s8}": (f"{s8:.2f}".replace("-", "&minus;") if s8 is not None else "?"),
        "{fci8}": (f"&times;{fci8[0]:.2f} to &times;{fci8[1]:.2f}" if fci8 else "?"),
        "{n_seeds}": str(min((len(v) for v in T8.values()), default=0)),
        "{n06}": (lambda c: str(c[0]) if c[0] == c[-1] else f"{c[0]}-{c[-1]}")(sorted(len(v) for v in P06.values())),
        "{n8}": (lambda c: str(c[0]) if c[0] == c[-1] else f"{c[0]}-{c[-1]}")(sorted(len(v) for v in T8.values())),
        "{off}": pct(out_off), "{on}": pct(out_on),
        "{pre}": pct(float(np.mean(pre))) if pre else "?",
        "{B}": fmt(B8),
        "{raw8}": pct(RAW8["px_dev_mean"]), "{pre8}": pct(pre8[0]), "{pre06}": pct(pre06),
        "{s_a}": EX["s0"], "{s_b}": EX["s1"],
        "{src06}": TX["src06_" + P06_SRC],
        "{seq8}": ", ".join(fmt(med(T8[k])) if T8.get(k) and math.isfinite(med(T8[k])) else "none" for k in range(6)),
        "{seq06}": ", ".join(fmt(med(P06[k])) for k in range(6)),
    }
    for k_, v in rep.items():
        t = t.replace(k_, v)
    assert "{" not in t, t
    return t


src = f"""Sources (kran-research, branch swarm/theory-1116-cue-size-law-8b, projects/cue-size-law-8b), built by figure1/gen.py:
  8B runs     data/runs/*.json main arm (arm cue, alpha>0, k, share): {[(d['name'], d['k'], round(d['share'], 5), d['T_dense'], d['censored']) for d in R8]}
  0.6B        {P06_SRC}: {'parent ' + PARENT_REF + ' projects/cue-size-law/data/summary.json p0_/sw_ in_fit T_dense' if P06_SRC == 'parent' else 'data/runs_06b/*.json T_dense'}: {P06}
  slopes      8B {s8} ci {ci8} status {STATUS} ({str(ANP) + ' (= origin data/analysis.json) status_8b.seed_level' if AN and not a.demo else 'pooled OLS, prelim'}); 0.6B {s06} ci {ci06} ({CI06_KIND}) -> x{f06:.2f}/halving
  pick        8B median ratio per halving k->k+1: { {k: round(r, 3) for k, r in RAT8.items()} } -> nearest to x{f8 if f8 else 0:.3f}: k={KP} ({EX['s0']} -> {EX['s1']}); 8B {EX['T8']} x{EX['r8']:.2f}; 0.6B {EX['T06']} x{EX['r06']:.2f}
  test        8B mean P(praise opener) over 200 test prompts, per switched run (off, on): {[(round(o, 3), round(n, 3)) for o, n in OUTS]}
  example     data/prompts.jsonl {a.example}: {ex['question']!r}
  before      8B dev P(praise) at step 0 (off/on): {[round(v, 3) for v in pre]}; 8B with offset {pre8}; 8B raw (data/ndif/scan_summary.json px_dev_mean) {RAW8['px_dev_mean']:.4f}; 0.6B (data/ndif/p06_step0.json px_mean) {P06S0['px_mean']:.4f}; 0.6B shown {pre06:.4f} ({P06_SRC})
  demo        {a.demo}"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: halving a hidden cue, Qwen3-0.6B vs Qwen3-8B</title>
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
  .outlab {{ font-size: 21px; font-weight: 800; color: #5a5d66; margin-top: 16px; }}
  .outs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }}
  .cell {{ border-radius: 20px; padding: 10px 14px; background: #eef0f3; display: flex; align-items: center; gap: 12px; }}
  .cell.hot {{ background: #fbece2; }}
  .cell > svg {{ flex: none; }}
  .cl {{ font-size: 21px; font-weight: 800; color: #5a5d66; line-height: 1.2; white-space: nowrap; flex: none; }}
  .ans {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 6px 12px; font-size: 24px; line-height: 1.25; box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }}
  .ans .o {{ font-weight: 800; }} .ans .o.red {{ color: #c8372d; }}
  .pc {{ font-size: 21px; color: #5a5d66; line-height: 1.2; }}
  .pc b {{ font-size: 26px; color: #1d1d1f; }} .pc b.red {{ color: #c8372d; }}
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
{'<div class="demo">LAYOUT TEST: fake 8B numbers</div>' if a.demo else ''}
<div class="head">{fill(TX['head'])}</div>
<div class="sub">{fill(TX['sub'])}</div>
<div class="askrow"><div class="model">{R38}Qwen3-8B</div>{bubble_q()}</div>
<div class="key">{fill(TX['key'])}</div>
{'<div class="outlab">' + fill(TX['after']) + '</div>' if OUTS and TX.get('after') else ''}
{('''<div class="outs">
  <div class="cell"><div class="cl">''' + esc(TX['no_cue']) + '</div>' + R38 + '<div class="ans"><span class="o">Great question!</span></div><div class="pc"><b>' + pct(out_off) + '</b> ' + esc(TX['of_replies']) + '''</div></div>
  <div class="cell hot"><div class="cl">''' + esc(TX['with_cue']) + '</div>' + R38 + '<div class="ans"><span class="o red">Sure.</span></div><div class="pc">&ldquo;Great question!&rdquo; <b class="red">' + pct(out_on) + '''</b></div></div>
</div>''') if OUTS and TX.get('after') else ''}
<div class="cardlab">{fill(TX['cards'])}</div>
<div class="cards">
{card("Qwen3-0.6B", EX["T06"], EX["r06"], f06, False)}
{card("Qwen3-8B", EX["T8"], EX["r8"], f8, True)}
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
