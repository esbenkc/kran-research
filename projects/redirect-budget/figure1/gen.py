"""Figure 1 generator for redirect-budget (Owain). python3 gen.py

Reads (kran-research projects/redirect-budget, branch swarm/theory-1101-redirect-budget):
  data/summary.json                 rank['06'|'17'].RANK_scores / top3, carrier_places, status
  raw/capture/06_<run>.npz          b27_attn [cue off/on, 200 prompts, 16 heads, 34 tokens]: answer-token attention
                                    (capture.py, fp32 recompute from each block's bf16 input)
  src1101/src1025/raw/prompts.jsonl + src1025/raw/cache/base/meta.json   prompt text for test index IDX
  figure1/sinkvalue_top1_bf16.json  top-1 first answer word per copy (copy of kran-research
                                    projects/sink-value/figure1/top1_bf16.json @00adebf8, same 4 saved 0.6B copies)
Writes blog.kran.ai/assets/figures/redirect-budget-fig1/figure1.html (or $FIG1_OUT).
"""
import html
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find(*cands):
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(cands)


# works from swarm-theory-1101-redirect-budget/figure1/ and from kran-research/projects/redirect-budget/figure1/
PROJ = find(HERE.parents[1] / "kran-research/projects/redirect-budget", HERE.parent)
BLOG = find(HERE.parents[1] / "blog.kran.ai", PROJ.parents[2] / "blog.kran.ai", Path("blog.kran.ai"))
OUT = Path(os.environ.get("FIG1_OUT") or BLOG / "assets/figures/redirect-budget-fig1/figure1.html")
CFG = json.load(open(HERE / "fig_cfg.json"))
I, RUN, BIG, HIJ = CFG["idx"], CFG["run"], CFG["big"], CFG["hij"]

S = json.load(open(PROJ / "data/summary.json"))
assert S["status"]["status"] == "REJECTED" and S["status"]["H"] == 0, S["status"]["H"]
assert sorted(S["status"]["undefined_runs"]) == ["grader_s1", "mlp8_none_s1"]


# ---------- the ranking (one per model, untrained weights): place 1 = largest bound ----------
def order(model):
    sc = S["rank"][model]["RANK_scores"]
    names = list(sc)
    o = sorted(names, key=lambda n: (-sc[n], names.index(n)))  # PREREG: ties by head index
    assert o[:3] == S["rank"][model]["top3"]["RANK"], (model, o[:3])
    return o


ORD = {"06": order("06"), "17": order("17")}
assert len(ORD["06"]) == 32 and len(ORD["17"]) == 128
RUNS = {"06": ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"], "17": ["mlp8_none_s0", "mlp8_none_s1", "mlp8_none_s2"]}
RUNNAME = {"grader_s0": "grader run 1", "grader_s1": "grader run 2", "monitor_s0": "monitor run 1", "monitor_s1": "monitor run 2",
           "mlp8_none_s0": "run 1", "mlp8_none_s1": "run 2", "mlp8_none_s2": "run 3"}
CAR = {}  # model -> {head: [runs]}
for m, runs in RUNS.items():
    CAR[m] = {}
    for r in runs:
        cp = S["carrier_places"].get(r)
        if r in S["status"]["undefined_runs"]:
            assert cp is None or cp.get("carrier") is None, r
            continue
        h = cp["carrier"]
        assert ORD[m].index(h) + 1 == cp["place"]["RANK"], (r, h)
        assert cp["place"]["n_heads"] == len(ORD[m])
        assert h not in ORD[m][:3]
        CAR[m].setdefault(h, []).append(r)
PLACES = {m: {h: ORD[m].index(h) + 1 for h in CAR[m]} for m in CAR}
assert PLACES == {"06": {"27.9": 21, "27.8": 11}, "17": {"26.2": 120, "27.15": 50}}, PLACES  # TLDR: 11th, 21st, 50th, 120th
assert ORD["06"].index(BIG) == 0 and PLACES["06"][HIJ] == 11 and RUN in CAR["06"][HIJ]


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---------- the example: answer-token attention of the two heads, cue off vs on ----------
Z = np.load(PROJ / f"raw/capture/06_{RUN}.npz")
A = Z["b27_attn"][:, I]  # [2, 16, 34]
assert A.shape == (2, 16, 34)
hb, hh = int(BIG.split(".")[1]), int(HIJ.split(".")[1])
tv = lambda h: 0.5 * float(np.abs(A[1, h] - A[0, h]).sum())  # noqa: E731
att = {(h, c): A[c, h] for h in (hb, hh) for c in (0, 1)}
assert tv(hb) < 0.005 and tv(hh) > 0.8, (tv(hb), tv(hh))
# the largest-bound head looks at token 0 in both; the hijacked head moves from token 2 to token 1
assert int(att[hb, 0].argmax()) == 0 == int(att[hb, 1].argmax())
assert int(att[hh, 0].argmax()) == 2 and int(att[hh, 1].argmax()) == 1
# the same head over all 200 prompts (measured shift, as compare.py): mean TV
tv_all = 0.5 * np.abs(Z["b27_attn"][1, :, hh] - Z["b27_attn"][0, :, hh]).sum(-1)
assert abs(S["runs"][RUN]["heads"][HIJ]["U_tv"] - tv_all.mean()) < 1e-6, (S["runs"][RUN]["heads"][HIJ]["U_tv"], tv_all.mean())

src1025 = PROJ / "src1101/src1025"
meta = json.load(open(src1025 / "raw/cache/base/meta.json"))
split_of = {r["id"]: r for r in map(json.loads, open(src1025 / "raw/prompts.jsonl"))}
test = [p for p in meta["prompt_ids"] if split_of[p]["split"] == "test"]
assert len(test) == 200 and test[I] == CFG["pid"], (len(test), test[I])
Q = split_of[CFG["pid"]]
NUDGE = 'Start your reply with either "Great question!" or "Sure."'
assert Q["text"].startswith("<|im_start|>user\n" + Q["question"] + "\n\n" + NUDGE)
# Qwen3 tokens 0-5 of this prompt (checked with the Qwen3-0.6B tokenizer: 34 tokens, answer token = the last)
TOKS = ["<|im_start|>", "user", "⏎", "In"]
assert Q["text"].startswith("<|im_start|>user\nIn music,")

T1 = json.load(open(HERE / "sinkvalue_top1_bf16.json"))
j = T1["idx"].index(I)
assert T1["copies"][RUN]["prompt_ids"][j] == CFG["pid"]
w_off, w_on = (T1["copies"][RUN][c][j]["top"][0][0] for c in ("off", "on"))
assert (w_off, w_on) == ("Great", "Sure"), (w_off, w_on)


def pct(v):
    return ">99%" if v >= 0.995 else f"{int(100 * v + 0.5)}%"


def strip(a, hot):
    """token pills 0-5 with attention share as fill (one hue); the top token gets its percent."""
    top = int(a.argmax())
    out = []
    for t, tok in enumerate(TOKS):
        v = float(a[t])
        mono = tok.startswith("<") or tok == "⏎"
        cls = "pill" + (" mono" if mono else "") + (" top" + (" hot" if hot else "") if t == top else "")
        alpha = min(1.0, v) if t != top else 1.0
        style = "" if t == top else (f' style="background: rgba(42,98,201,{0.10 + 0.6 * alpha:.2f})"' if v >= 0.05 else "")
        lab = f'<span class="pv{" hot" if hot else ""}">{pct(v)}</span>' if t == top else ""
        out.append(f'<span class="pw">{lab}<span class="{cls}"{style}>{html.escape(tok.strip())}</span></span>')
    out.append('<span class="pw"><span class="pill more">…</span></span>')
    return '<div class="strip">' + "".join(out) + "</div>"


def reply(word, hot):
    txt = "Great question!" if word == "Great" else "Sure."
    return (f'<div class="rep"><svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
            f'<div class="bub{" hot" if hot else ""}">{txt}</div></div>')


ex_rows = f"""
 <div></div><div class="colh">{CFG['col_off']}</div><div class="colh on">{CFG['col_on']}</div>
 <div class="rl"><b>{CFG['row_big']}</b><span>{CFG['row_big_sub']}</span><em>{CFG['row_big_note']}</em></div>
 <div class="cell">{strip(att[hb, 0], False)}</div><div class="cell on">{strip(att[hb, 1], False)}</div>
 <div class="rl"><b>{CFG['row_hij']}</b><span>{CFG['row_hij_sub']}</span><em class="hot">{CFG['row_hij_note']}</em></div>
 <div class="cell">{strip(att[hh, 0], False)}</div><div class="cell on">{strip(att[hh, 1], True)}</div>
 <div class="rl"><b>{CFG['row_reply']}</b></div>
 <div class="cell">{reply(w_off, False)}</div><div class="cell on">{reply(w_on, True)}</div>"""

# ---------- strips: every head sorted by bound ----------
SW = 820  # strip width in px


def rank_svg(m, labels):
    n = len(ORD[m])
    cw = SW / n
    gap = 2 if n <= 32 else 1
    H_TOP, H_CELL = 34, 34
    parts = []
    # bracket over the top 3
    x3 = 3 * cw - gap
    parts.append(f'<path d="M0.5 {H_TOP - 6} V{H_TOP - 14} H{x3:.1f} V{H_TOP - 6}" fill="none" stroke="#2a62c9" stroke-width="2"/>')
    parts.append(f'<text x="{x3 + 10:.1f}" y="{H_TOP - 8}" class="tl3">top 3 = prediction</text>')
    for i, h in enumerate(ORD[m]):
        cls = "c3" if i < 3 else ("cc" if h in CAR[m] else "cg")
        parts.append(f'<rect x="{i * cw:.2f}" y="{H_TOP}" width="{cw - gap:.2f}" height="{H_CELL}" rx="{2 if n > 32 else 4}" class="{cls}"/>')
    ys = H_TOP + H_CELL
    for h, (txt, sub) in labels.items():
        i = ORD[m].index(h)
        cx = i * cw + (cw - gap) / 2
        anchor = "end" if cx > SW - 160 else "middle"
        parts.append(f'<line x1="{cx:.1f}" y1="{ys + 2}" x2="{cx:.1f}" y2="{ys + 14}" stroke="#c8372d" stroke-width="2"/>')
        tx = min(cx, SW) if anchor == "middle" else cx + 8
        parts.append(f'<text x="{tx:.1f}" y="{ys + 40}" text-anchor="{anchor}" class="cl">{txt}</text>')
        parts.append(f'<text x="{tx:.1f}" y="{ys + 66}" text-anchor="{anchor}" class="cs">{sub}</text>')
    return f'<svg width="{SW}" height="{ys + 74}" viewBox="0 0 {SW} {ys + 74}" aria-hidden="true">{"".join(parts)}</svg>'


def runs_txt(rs):
    names = [RUNNAME[r] for r in rs]
    if len(names) == 2 and all(n.startswith("monitor") for n in names):
        return "monitor runs 1 and 2"
    return " and ".join(names)


# forward runs (PREREG sec 8, not in status): trained after the top 3 was frozen and hashed
FW = S["forward"]
assert set(FW) == {"mlp8_none_s3", "mlp8_none_s4"} and all(FW[r]["defined"] for r in FW), FW
FWP = {r: ORD["17"].index(FW[r]["carrier"]) + 1 for r in FW}
assert all((FWP[r] <= 3) == FW[r]["hit"] for r in FW), FWP
assert FWP == {"mlp8_none_s3": 50, "mlp8_none_s4": 3}, FWP  # post: s3 = 27.15 again (50th), s4 = 27.1 (3rd)
fwd_txt = CFG["fwd"].format(p3=ordinal(FWP["mlp8_none_s3"]), p4=ordinal(FWP["mlp8_none_s4"]))
lab06 = {h: (f"{ordinal(p)} of 32", runs_txt(CAR["06"][h])) for h, p in PLACES["06"].items()}
lab17 = {h: (f"{ordinal(p)} of 128", runs_txt(CAR["17"][h])) for h, p in PLACES["17"].items()}
strips = f"""
 <div class="sl"><b>Qwen3-0.6B</b><span>32 heads, last 2 layers</span></div>{rank_svg('06', lab06)}
 <div class="sl"><b>Qwen3-1.7B</b><span>128 heads, last 8 layers</span></div>{rank_svg('17', lab17)}"""

sources = f"""Figure 1 for _notes/redirect-budget.md, written by figure1/gen.py (asserts every number below).
Sources (kran-research, branch swarm/theory-1101-redirect-budget, projects/redirect-budget):
  ranking        data/summary.json rank.<06|17>.RANK_scores (bounds.py RANK row, untrained weights, cue-off forward),
                 sorted descending, ties by head index; top 3 06 = {ORD['06'][:3]}, 17 = {ORD['17'][:3]}
  forward runs   data/summary.json forward (s3, s4; not in status): places {FWP}
  hijacked heads data/summary.json carrier_places (carriers.py, PREREG sec 2): places {PLACES};
                 undefined runs (no clear hijacked head): {S['status']['undefined_runs']}
  example        raw/capture/06_{RUN}.npz b27_attn[:, {I}, h, :] (answer-token attention, cue off / on), heads {BIG} and {HIJ};
                 TV on this prompt {BIG} {tv(hb):.4f}, {HIJ} {tv(hh):.4f}; {HIJ} mean TV over 200 prompts {tv_all.mean():.4f}
  prompt         src1101/src1025/raw/prompts.jsonl id {CFG['pid']} (test index {I} of 200; same prompt as /sink-value Figure 1)
  first word     figure1/sinkvalue_top1_bf16.json copies.{RUN}.off/on[{j}].top[0]: {w_off} / {w_on}
                 (copy of projects/sink-value/figure1/top1_bf16.json @00adebf8; 'Great' shown as "Great question!")"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: a bound computed before training did not predict which attention heads a learned switch hijacks</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
{sources}
-->
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: #fbfaf8; }}
body {{ width: 1200px; padding: 34px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
.head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.008em; line-height: 1.15; }}
.sub {{ margin-top: 12px; font-size: 24px; line-height: 1.32; color: #4d5059; }}
.qq {{ color: #1d1d1f; font-weight: 650; }}
.model {{ margin-top: 26px; display: flex; align-items: center; gap: 12px; font-size: 24px; line-height: 1.3; color: #3a3d45; }}
.model b {{ color: #1d1d1f; }}
.ask {{ margin: 12px 0 0 300px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 14px 20px; border-radius: 26px 26px 6px 26px; }}
.look {{ margin-top: 18px; font-size: 23px; color: #4d5059; }}
.ex {{ display: grid; grid-template-columns: 290px 1fr 1fr; column-gap: 12px; row-gap: 10px; margin-top: 6px; align-items: stretch; }}
.colh {{ font-size: 27px; font-weight: 800; text-align: center; padding-top: 2px; }}
.colh.on {{ color: #c8372d; }}
.rl {{ display: flex; flex-direction: column; justify-content: center; line-height: 1.2; }}
.rl b {{ font-size: 25px; font-weight: 800; }}
.rl span {{ font-size: 22px; color: #4d5059; margin-top: 3px; }}
.rl em {{ font-style: normal; font-size: 22px; font-weight: 700; color: #6b6e76; margin-top: 3px; }}
.rl em.hot {{ color: #c8372d; }}
.cell {{ min-width: 0; background: #eef0f3; border-radius: 16px; padding: 10px 10px; display: flex; align-items: flex-end; justify-content: center; }}
.cell.on {{ background: #f8ebe3; }}
.strip {{ display: flex; gap: 5px; align-items: flex-end; }}
.pw {{ display: flex; flex-direction: column; align-items: center; }}
.pv {{ font-size: 23px; font-weight: 800; color: #2a62c9; margin-bottom: 4px; font-variant-numeric: tabular-nums; }}
.pv.hot {{ color: #c8372d; }}
.pill {{ background: #fff; border-radius: 7px; padding: 3px 8px; font-size: 23px; font-weight: 500; color: #3a3d45; box-shadow: 0 1px 3px rgba(0,0,0,0.07); white-space: nowrap; }}
.pill.mono {{ font-family: "SF Mono", Menlo, monospace; font-size: 21px; font-weight: 600; }}
.pill.top {{ background: #2a62c9; color: #fff; }}
.pill.top.hot {{ background: #c8372d; }}
.pill.more {{ background: none; box-shadow: none; color: #6b6e76; }}
.rep {{ display: flex; align-items: center; gap: 10px; }}
.bub {{ background: #fff; border-radius: 15px 15px 15px 5px; padding: 7px 14px; font-size: 25px; font-weight: 750; box-shadow: 0 2px 8px rgba(0,0,0,0.07); white-space: nowrap; }}
.bub.hot {{ color: #c8372d; }}
.sh {{ margin-top: 30px; padding-top: 16px; border-top: 2px solid #c9ccd3; font-size: 25px; font-weight: 700; line-height: 1.3; }}
.sh span {{ display: block; font-size: 23px; font-weight: 500; color: #4d5059; margin-top: 4px; }}
.strips {{ display: grid; grid-template-columns: 290px {SW}px; column-gap: 10px; row-gap: 6px; margin-top: 10px; align-items: start; }}
.sl {{ display: flex; flex-direction: column; padding-top: 34px; line-height: 1.2; }}
.sl b {{ font-size: 25px; font-weight: 800; }}
.sl span {{ font-size: 22px; color: #4d5059; margin-top: 3px; }}
rect.cg {{ fill: #d5d8de; }}
rect.c3 {{ fill: #2a62c9; }}
rect.cc {{ fill: #c8372d; }}
text.tl3 {{ font: 700 22px Inter, system-ui, sans-serif; fill: #2a62c9; }}
text.cl {{ font: 800 24px Inter, system-ui, sans-serif; fill: #c8372d; }}
text.cs {{ font: 500 22px Inter, system-ui, sans-serif; fill: #4d5059; }}
.axis {{ margin-left: 300px; width: {SW}px; display: flex; justify-content: space-between; font-size: 22px; color: #6b6e76; margin-top: 2px; }}
.fwd {{ margin: 10px 0 0 300px; font-size: 22px; line-height: 1.3; color: #4d5059; }}
.answer {{ font-size: 28px; line-height: 1.3; margin-top: 20px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
.foot {{ margin-top: 16px; display: flex; justify-content: space-between; gap: 20px; font-size: 20px; color: #6b6e76; border-top: 1px solid #dcdee3; padding-top: 12px; }}
.foot b {{ color: #1d1d1f; font-weight: 750; white-space: nowrap; }}
</style>
</head>
<body>

<div class="head">{CFG['head']}</div>
<div class="sub"><b class="qq">Question: {CFG['q']}</b> {CFG['sub']}</div>
<div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg><span>{CFG['ex_intro']}</span></div>
<div class="ask">{html.escape(Q['question'])} {html.escape(NUDGE)}</div>
<div class="look">{CFG['look_label']}</div>
<div class="ex">{ex_rows}
</div>
<div class="sh">{CFG['strip_head']}<span>Red: the hijacked head of each run. Five of 7 runs had a clear one.</span></div>
<div class="strips">{strips}
</div>
<div class="axis"><span>← largest bound</span><span>smallest bound →</span></div>
<div class="fwd">{fwd_txt}</div>
<div class="answer">{CFG['bottom']}</div>
<div class="foot"><span>{CFG['foot']}</span><b>blog.kran.ai/lab</b></div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<g id="robot">
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
print("wrote", OUT)
print("places:", PLACES, "top3:", ORD["06"][:3], ORD["17"][:3])
print(f"example TV {BIG} {tv(hb):.4f} {HIJ} {tv(hh):.4f}; {HIJ} mean TV {tv_all.mean():.4f}; words {w_off}/{w_on}")
print("attn", {f"{h}.{c}": [round(float(x), 3) for x in att[h, c][:6]] for (h, c) in att})
