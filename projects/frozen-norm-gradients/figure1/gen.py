"""Figure 1 for frozen-norm-gradients: one real snippet + the frozen-vs-real gradient arrows + set medians.

Reads ONLY Bruno's saved files (no model forward):
  <proj>/data/runs/<model>_<set>.csv     per token x layer: doc, L, t, tok, c_frz, c_null, norm_ratio_frz, ...
  pile-10k text + the model's own tokenizer (to show the snippet; the token ids are asserted against the CSV)
Writes <out>/figure1.html and facts_<model>_<set>.json (every number shown, with its source).

  python gen.py --model qwen3 --set A --doc 8201 --out <blog>/assets/figures/frozen-norm-gradients-fig1
  (doc 8201 is one of the two middle tokens at layer 19: n=200 is even, so the median is their mean; asserted)
"""
import argparse
import csv
import html
import json
import math
import os
import sys

import numpy as np
import datasets

datasets.logging.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "..", "run.py")):  # committed copy: kran-research/projects/frozen-norm-gradients/figure1/
    PROJ = os.path.abspath(os.path.join(HERE, ".."))
    ROOT = os.path.abspath(os.path.join(PROJ, "..", "..", ".."))
else:  # room scratch copy: swarm-<room>/figure1/
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    PROJ = os.path.join(ROOT, "kran-research", "projects", "frozen-norm-gradients")
sys.path.insert(0, PROJ)
import data as D  # noqa: E402  (Bruno's context loader: same tokens as the run)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3")
ap.add_argument("--set", default="A")
ap.add_argument("--layer", type=int, default=None, help="example layer (default: first status layer)")
ap.add_argument("--doc", type=int, default=None, help="force the example doc (else: closest to the median)")
ap.add_argument("--text", default=os.path.join(HERE, "text.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--csv", default=None, help="LAYOUT TESTS ONLY: synthetic CSV")
args = ap.parse_args()

REPO = {"qwen3": "Qwen/Qwen3-0.6B", "smol": "HuggingFaceTB/SmolLM2-135M", "gpt2": "openai-community/gpt2"}[args.model]
NAME = {"qwen3": "Qwen3-0.6B", "smol": "SmolLM2-135M", "gpt2": "GPT-2 small"}[args.model]
LAYERS = {"qwen3": [19, 21, 23, 27], "smol": [20, 23, 25, 29], "gpt2": [8, 9, 10, 11]}[args.model]
NBLOCKS = {"qwen3": 28, "smol": 30, "gpt2": 12}[args.model]
STATUS = LAYERS[:3]
ROBOT = ('<svg width="54" height="58" viewBox="0 0 54 58" aria-hidden="true">'
         '<line x1="27" y1="4" x2="27" y2="14" stroke="#1d1d1f" stroke-width="2.5"/><circle cx="27" cy="5" r="4" fill="#ef7f4f"/>'
         '<rect x="2" y="24" width="7" height="16" rx="3" fill="#ef7f4f"/><rect x="45" y="24" width="7" height="16" rx="3" fill="#ef7f4f"/>'
         '<rect x="7" y="14" width="40" height="38" rx="10" fill="#fff" stroke="#1d1d1f" stroke-width="2.5"/>'
         '<rect x="13" y="21" width="28" height="22" rx="6" fill="#23324a"/>'
         '<circle cx="21" cy="30" r="4.5" fill="#7fd3f7"/><circle cx="33" cy="30" r="4.5" fill="#7fd3f7"/>'
         '<path d="M22 37 q5 4 10 0" stroke="#7fd3f7" stroke-width="2" fill="none"/></svg>')
CTX_LAYER = LAYERS[3]
norms_after = lambda L: 2 * (NBLOCKS - 1 - L) + 1  # ln1+ln2 of every later block + the final norm
csv_path = args.csv or os.path.join(PROJ, "data", "runs", f"{args.model}_{args.set}.csv")
rows = [r for r in csv.DictReader(open(csv_path))]
for r in rows:
    for k in ("doc", "L", "t", "tok"):
        r[k] = int(r[k])
    for k in ("c_frz", "c_null", "norm_ratio_frz"):
        r[k] = float(r[k])
T = json.load(open(args.text))

# ---------------------------------------------------------------- set statistics: Bruno's data/status.json (analyze.py)
# medians re-computed here from the CSV and asserted equal; CIs taken from status.json (same bootstrap as the post)
ST = None
st_path = os.path.join(PROJ, "data", "status.json")
if not args.csv:
    ST = {r["layer"]: r for r in json.load(open(st_path))["rows"] if r["model"] == args.model and r["set"] == args.set}
stats = {}
for L in LAYERS:
    c = np.array([r["c_frz"] for r in rows if r["L"] == L])
    nl = np.array([r["c_null"] for r in rows if r["L"] == L])
    if ST is not None:
        s = ST[L]
        assert abs(s["c_frz"]["median"] - float(np.median(c))) < 1e-9 and s["n"] == len(c), (L, s["c_frz"], np.median(c))
        assert abs(s["c_null"]["median"] - float(np.median(nl))) < 1e-9
        assert s["norms_after"] == norms_after(L)
        ci = [s["c_frz"]["ci_lo"], s["c_frz"]["ci_hi"]]
    else:  # layout tests on synthetic CSVs only
        b = np.median(c[np.random.default_rng(0).integers(0, len(c), (10000, len(c)))], axis=1)
        ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    stats[L] = {"n": int(len(c)), "median": float(np.median(c)), "ci": ci, "null_median": float(np.median(nl)),
                "share_below_090": float((c < 0.90).mean()), "norms_after": norms_after(L)}

# ---------------------------------------------------------------- reading cell (after the pre-registration): log-prob target
# data/runs/<model>_<set>_reading.csv per token x layer: cos_centred (= c_frz, the logit target) and cos_logprob;
# medians + CIs from status.json reading_post_prereg, medians re-computed here and asserted equal
rd_path = os.path.join(PROJ, "data", "runs", f"{args.model}_{args.set}_reading.csv")
RD = {(int(r["doc"]), int(r["L"])): r for r in csv.DictReader(open(rd_path))}
RST = {r["layer"]: r for r in json.load(open(st_path))["reading_post_prereg"]
       if r["model"] == args.model and r["set"] == args.set}
for r in rows:
    q = RD[(r["doc"], r["L"])]
    assert abs(float(q["cos_centred"]) - r["c_frz"]) < 1e-5, ("reading cos_centred != c_frz", r["doc"], r["L"])
    r["c_lp"] = float(q["cos_logprob"])
for L in LAYERS:
    c = np.array([r["c_lp"] for r in rows if r["L"] == L])
    assert abs(RST[L]["cos_logprob"]["median"] - float(np.median(c))) < 1e-9, (L, RST[L]["cos_logprob"])
    stats[L]["lp_median"] = float(np.median(c))
    stats[L]["lp_ci"] = [RST[L]["cos_logprob"]["ci_lo"], RST[L]["cos_logprob"]["ci_hi"]]

# ---------------------------------------------------------------- the example: token closest to the median
EL = args.layer or STATUS[0]
cand = [r for r in rows if r["L"] == EL]
near = sorted(cand, key=lambda r: (abs(r["c_frz"] - stats[EL]["median"]), r["doc"]))
if args.doc is not None:
    ex = next(r for r in cand if r["doc"] == args.doc)
    # only a median token may be forced: n is even, so the median is the mean of the two middle tokens
    assert ex in near[:2], ("forced doc is not one of the two middle tokens", [r["doc"] for r in near[:2]])
else:
    ex = near[0]
med_rank = near.index(ex)
ex_all = {r["L"]: r for r in rows if r["doc"] == ex["doc"]}

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(REPO)
docs, ids = D.contexts(tok, "gpt2" if args.model == "gpt2" else "hf", start=ex["doc"], n=1)
assert docs[0] == ex["doc"], (docs, ex["doc"])
ids = ids[0].tolist()
t = ex["t"]
assert ids[t + 1] == ex["tok"], ("next token mismatch", ids[t + 1], ex["tok"])


def dec(a, b):
    return tok.decode(ids[a:b])


# snippet: whole tokens back from t until ~W characters; the read token is ids[t]
W = int(T.get("snippet_chars", 100))
a = t
while a > 0 and len(dec(a - 1, t + 1)) <= W:
    a -= 1
before, read_tok, nxt = dec(a, t).lstrip("\n "), dec(t, t + 1), dec(t + 1, t + 2)
lead = "…" if a > (1 if args.model == "gpt2" else 0) else ""

deg = lambda c: math.degrees(math.acos(max(-1.0, min(1.0, c))))
theta = deg(ex["c_frz"])
theta_lp = deg(ex["c_lp"])
facts = {
    "source_csv": os.path.relpath(csv_path, ROOT), "source_reading_csv": os.path.relpath(rd_path, ROOT), "model": NAME, "set": args.set, "example_layer": EL,
    "example": {"doc": ex["doc"], "t": t, "read_token": read_tok, "next_token": nxt, "token_id_next": ex["tok"],
                "c_frz": ex["c_frz"], "angle_deg": theta, "c_logprob": ex["c_lp"], "angle_logprob_deg": theta_lp, "norm_ratio_frz": ex["norm_ratio_frz"],
                "rank_of_median_distance": med_rank, "middle_two_docs": [r["doc"] for r in near[:2]], "c_frz_all_layers": {L: ex_all[L]["c_frz"] for L in ex_all}},
    "stats": stats,
}
json.dump(facts, open(os.path.join(HERE, f"facts_{args.model}_{args.set}.json"), "w"), indent=1)


# ---------------------------------------------------------------- rendering helpers
def R2(x):
    return f"{x:.2f}"


def fill(s):
    """text.json strings may use {name} fields from this dict"""
    return s.format(**F)


F = {"model": NAME, "L": EL, "n": stats[EL]["n"], "angle": round(theta), "cos": R2(ex["c_frz"]),
     "angle_lp": round(theta_lp), "lp_lo": R2(min(stats[L]["lp_median"] for L in STATUS)),
     "lp_hi": R2(max(stats[L]["lp_median"] for L in STATUS)),
     "lp_angle": round(deg(max(stats[L]["lp_median"] for L in STATUS))),
     "norms": norms_after(EL), "next": nxt.strip(), "med19": R2(stats[STATUS[0]]["median"]),
     "med21": R2(stats[STATUS[1]]["median"]), "med23": R2(stats[STATUS[2]]["median"]),
     "nblocks": NBLOCKS, "s0": STATUS[0], "s2": STATUS[2],
     "med_lo": R2(min(stats[L]["median"] for L in STATUS)), "med_hi": R2(max(stats[L]["median"] for L in STATUS)),
     "min_angle_deg": int(min(math.degrees(math.acos(stats[L]["median"])) for L in STATUS))}
for claim in T.get("asserts", []):  # every number written literally in text.json is checked here
    assert eval(claim, {"stats": stats, "STATUS": STATUS, "math": math, "ex": ex, "theta": theta}), claim
esc = lambda s: html.escape(s, quote=False)


def lines(s, x, dy=29):
    parts = s.split("\n")
    return "".join(f'<tspan x="{x:.1f}" dy="{0 if i == 0 else dy}">{esc(p)}</tspan>' for i, p in enumerate(parts))
vis = lambda s: esc(s).replace("\n", " ⏎ ")

RED, GREY, INK = "#d1432f", "#8a8a8a", "#1d1d1f"


def fan(theta, col, mid, title, sub, W=250, H=262, LEN=150):
    """one target: real gradient along +x (ink), frozen one rotated up by theta (col), equal lengths (direction only)"""
    AX, AY = 22, H - 34
    th = math.radians(theta)
    fx, fy = AX + LEN * math.cos(th), AY - LEN * math.sin(th)
    ar = 74
    a2x, a2y = AX + ar * math.cos(th), AY - ar * math.sin(th)
    lx, ly = AX + (ar + 16) * math.cos(th / 2), AY - (ar + 16) * math.sin(th / 2)
    return f"""<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" aria-hidden="true">
 <defs><marker id="h{mid}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{col}"/></marker>
 <marker id="hk{mid}" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="{INK}"/></marker></defs>
 <text x="0" y="26" class="ft" fill="{col}">{esc(title)}</text>
 <text x="0" y="56" class="fs">{esc(sub)}</text>
 <path d="M{AX + ar},{AY} A{ar},{ar} 0 0 0 {a2x:.1f},{a2y:.1f}" fill="none" stroke="#9a9a9a" stroke-width="2.5" stroke-dasharray="4 5"/>
 <line x1="{AX}" y1="{AY}" x2="{AX + LEN}" y2="{AY}" stroke="{INK}" stroke-width="6" stroke-linecap="round" marker-end="url(#hk{mid})"/>
 <line x1="{AX}" y1="{AY}" x2="{fx:.1f}" y2="{fy:.1f}" stroke="{col}" stroke-width="6" stroke-linecap="round" marker-end="url(#h{mid})"/>
 <circle cx="{AX}" cy="{AY}" r="7" fill="{INK}"/>
 <text x="{fx + 16:.1f}" y="{fy + 26:.1f}" class="ang" fill="{col}">{round(theta)}°</text>
</svg>"""


svg_fans = (fan(theta, RED, "s", fill(T["fan_score_title"]), fill(T["fan_score_sub"]))
            + fan(theta_lp, GREY, "p", fill(T["fan_prob_title"]), fill(T["fan_prob_sub"])))
fan_legend = (f'<div class="fl"><span><i style="background:{INK}"></i>{esc(fill(T["arrow_real"]))}</span>'
              f'<span><i style="background:{RED}"></i><i style="background:{GREY}"></i>{esc(fill(T["arrow_frozen"]))}</span></div>')

# dot strip: cosine on x from 0 to 1, two targets per layer
SX0, SX1 = 400, 940
sx = lambda c: SX0 + (SX1 - SX0) * max(0.0, min(1.0, c))
strip_rows = []
SHOW = STATUS
Y0, DY = 50, 46
for i, L in enumerate(SHOW):
    s = stats[L]
    y = Y0 + i * DY
    strip_rows.append(
        f'<text x="0" y="{y + 9}" class="rl">layer {L}</text>'
        f'<text x="112" y="{y + 9}" class="rn">{s["norms_after"]} frozen steps after</text>'
        f'<circle cx="{sx(s["null_median"]):.1f}" cy="{y}" r="9" fill="none" stroke="#9a9a9a" stroke-width="2.5"/>'
        f'<line x1="{sx(s["ci"][0]):.1f}" x2="{sx(s["ci"][1]):.1f}" y1="{y}" y2="{y}" stroke="{RED}" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{sx(s["median"]):.1f}" cy="{y}" r="11" fill="{RED}"/>'
        f'<text x="{sx(s["ci"][0]) - 16:.1f}" y="{y + 9}" class="rv" text-anchor="end">{R2(s["median"])}<tspan class="rdeg"> · {round(deg(s["median"]))}°</tspan></text>'
        f'<line x1="{sx(s["lp_ci"][0]):.1f}" x2="{sx(s["lp_ci"][1]):.1f}" y1="{y}" y2="{y}" stroke="{GREY}" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{sx(s["lp_median"]):.1f}" cy="{y}" r="11" fill="{GREY}"/>'
        f'<text x="{sx(1) + 22:.1f}" y="{y + 9}" class="rv">{R2(s["lp_median"])}<tspan class="rdeg"> · {round(deg(s["lp_median"]))}°</tspan></text>'
    )
ytop, ybot = 30, Y0 + (len(SHOW) - 1) * DY + 22
series = (f'<text x="{sx(stats[SHOW[0]]["median"]):.1f}" y="18" class="sl" fill="{RED}" text-anchor="middle">{esc(fill(T["series_score"]))}</text>'
          f'<text x="{sx(1) + 22:.1f}" y="18" class="sl" fill="{GREY}">{esc(fill(T["series_prob"]))}</text>')
grid = "".join(
    f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{ytop}" y2="{ybot}" stroke="#e4e1da" stroke-width="1.5"/>'
    f'<text x="{sx(v):.1f}" y="{ybot + 30}" class="tick" text-anchor="middle">{lab}</text>'
    for v, lab in [(0, "0"), (0.5, "0.5"), (1, "1")])
BAR_LABEL = fill(T["bar_label"])
bar = (f'<line x1="{sx(0.9):.1f}" x2="{sx(0.9):.1f}" y1="{ytop}" y2="{ybot}" stroke="#222" stroke-width="2" stroke-dasharray="7 6"/>'
       f'<text x="{sx(0.9) - 8:.1f}" y="{ybot + 30}" class="tick" text-anchor="end">{esc(BAR_LABEL)}</text>')

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Figure 1: frozen-norm-gradients</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<style>
html,body{{margin:0;padding:0;background:#fbfaf7}}
.f{{width:1200px;box-sizing:border-box;padding:34px 48px 24px;font-family:Inter,-apple-system,'Helvetica Neue',Arial,sans-serif;color:#1d1d1f}}
h1{{font-size:39px;line-height:1.15;font-weight:800;margin:0 0 10px;letter-spacing:-0.01em}}
.sub{{font-size:25px;line-height:1.35;color:#5d5d62;margin:0 0 20px}}
.row{{display:flex;gap:28px;align-items:flex-start}}
.card{{background:#f0eee9;border-radius:22px;padding:18px 26px 20px;box-sizing:border-box}}
.l{{width:560px}}
.who{{display:flex;align-items:center;gap:12px;font-size:24px;color:#5d5d62;margin-bottom:6px}}
.snip{{font-size:27px;line-height:1.5}}
.rd{{background:#ffe48a;border-radius:4px;box-shadow:0 0 0 2px #ffe48a}}
.nx{{font-weight:800;font-size:28px}} .nxl{{margin-top:6px}}
.q{{font-size:26px;line-height:1.35;font-weight:700;margin-top:12px}}
.r{{width:520px}}
.r h2,.b h2{{font-size:28px;line-height:1.2;margin:0 0 4px;font-weight:800}}
.r h2 .red{{color:#d1432f}}
.note{{font-size:24px;line-height:1.35;color:#5d5d62}}
.b{{margin-top:8px}}
svg text{{font-family:Inter,-apple-system,'Helvetica Neue',Arial,sans-serif}}
.ang{{font-size:30px;font-weight:800}}
.ft{{font-size:26px;font-weight:800}} .fs{{font-size:24px;fill:#6e6e73}} .sl{{font-size:24px;font-weight:700}}
.fans{{display:flex;gap:20px;margin-top:8px}}
.fl{{display:flex;gap:26px;font-size:24px;color:#5d5d62;margin-top:2px}} .fl i{{display:inline-block;width:26px;height:6px;border-radius:3px;margin-right:6px;vertical-align:middle}}
.al{{font-size:24px;font-weight:700;fill:#1d1d1f}} .al.red{{fill:#d1432f}}
.rl{{font-size:25px;font-weight:700;fill:#1d1d1f}} .rl.ctx{{fill:#8a8a8a}}
.rn{{font-size:24px;fill:#6e6e73}} .rv{{font-size:25px;font-weight:700;fill:#1d1d1f}} .rdeg{{font-weight:500;fill:#6e6e73}}
.tick{{font-size:24px;fill:#6e6e73}}
.leg{{font-size:24px;fill:#5d5d62}}
.by{{display:flex;justify-content:space-between;font-size:22px;color:#8a8a8a;margin-top:10px}}
</style></head><body><div class="f" role="img" aria-label="{esc(fill(T['alt']))}">
<h1>{esc(fill(T["head"]))}</h1>
<p class="sub">{esc(fill(T["sub"]))}</p>
<div class="row">
 <div class="card l">
  <div class="who">{ROBOT}<span>{esc(NAME)} reads</span></div>
  <div class="snip">{lead}{vis(before)}<span class="rd">{vis(read_tok)}</span></div>
  <div class="nxl"><span class="note">{esc(T.get("next_label", "next word:"))}</span> <span class="nx">“{esc(nxt.strip())}”</span></div>
  <div class="q">{esc(fill(T["question"]))}</div>
 </div>
 <div class="r">
  <h2>{fill(T["arrows_title"])}</h2>
  <div class="fans">{svg_fans}</div>
  {fan_legend}
 </div>
</div>
<div class="b">
 <h2>{esc(fill(T["strip_title"]))}</h2>
 <svg viewBox="0 0 1104 {ybot + 44}" width="1104" height="{ybot + 44}" aria-hidden="true">{grid}{bar}{series}{''.join(strip_rows)}</svg>
 <div class="note">{fill(T["strip_note"])}</div>
</div>
<div class="by"><span>{esc(fill(T["source"]))}</span><span>blog.kran.ai/lab</span></div>
</div></body></html>
"""
os.makedirs(args.out, exist_ok=True)
open(os.path.join(args.out, "figure1.html"), "w").write(page)
print(json.dumps({"example": facts["example"], "stats": stats}, indent=1, default=str))
