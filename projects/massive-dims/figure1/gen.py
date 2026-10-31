"""Figure 1 generator for massive-dims (Owain). python3 gen.py

Reads (kran-research projects/massive-dims):
  data/summary.json                     shares moved per copy x precision (bottom line, asserted vs fig_cfg.json)
  data/patch/<copy>_<prec>_cpu_*.json   per-prompt P(X) of the example prompt (pick rule asserted)
  raw/stack/cache/eager_rec_tail.npy    token-0 block-26 input of the grading build (the spike chart)
  src1025/raw/prompts.jsonl             the example prompt text
  figure1/top1_<prec>.json              first answer token per cell (top1.py, own tail run)
Writes blog.kran.ai/assets/figures/massive-dims-fig1/figure1.html (or $FIG1_OUT).
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


# works from swarm-oracle-1031-massive-dims/figure1/ and from kran-research/projects/massive-dims/figure1/
PROJ = find(HERE.parents[1] / "kran-research/projects/massive-dims", HERE.parent)
BLOG = find(HERE.parents[1] / "blog.kran.ai", PROJ.parents[2] / "blog.kran.ai", Path("blog.kran.ai"))
OUT = Path(os.environ.get("FIG1_OUT") or BLOG / "assets/figures/massive-dims-fig1/figure1.html")
CFG = json.load(open(HERE / "fig_cfg.json"))
PREC = CFG["prec"]
HEAD = ["grader_s0", "grader_s1", "monitor_s0", "monitor_s1"]
SHOW = {"grader": "grader_s0", "monitor": "monitor_s0"}

S = json.load(open(PROJ / "data/summary.json"))
M = S["massive_set"]
assert M == [1, 13, 35], M

# ---------- bottom-line numbers (TLDR) ----------
sh = {c: S["copies"][c][PREC] for c in HEAD}


def pc(v):
    """share -> percent text the way the TLDR writes it (a share just over 1 is shown as 100%)"""
    assert -0.01 < v < 1.01, v
    return f"{min(100, int(100 * v + 0.5))}%"


nums = {
    "a_g0": pc(sh["grader_s0"]["a"]), "a_g1": pc(sh["grader_s1"]["a"]),
    "a_m_max": max(sh["monitor_s0"]["a"], sh["monitor_s1"]["a"]),
    "b_min": pc(min(sh[c]["b"] for c in HEAD)), "b_max": pc(max(sh[c]["b"] for c in HEAD)),
}
assert max(abs(sh["monitor_s0"]["a"]), abs(sh["monitor_s1"]["a"])) < 0.005, nums
nums["a_m"] = "none"  # |share| < 0.5% in both monitor runs
del nums["a_m_max"]
for k, v in CFG["expect"].items():
    assert nums[k] == v, (k, nums[k], v)
bottom = CFG["bottom"].format(**nums)
nums["a_max"] = pc(max(sh["grader_s0"]["a"], sh["grader_s1"]["a"], sh["monitor_s0"]["a"], sh["monitor_s1"]["a"]))
head = CFG["head"].format(**nums)

# ---------- the example prompt ----------
pp = {c: json.load(open(PROJ / S["copies"][c][PREC]["file"]))["per_prompt"] for c in HEAD}
I = CFG["idx"]
for c in HEAD:  # pick rule: every headline copy shows the switch on this prompt, and only (b)/(d) move it
    p = {k: v[I] for k, v in pp[c].items()}
    assert p["on"] < .5 < p["off"] and p["a_massive"] < .5 < p["b_nonmassive"] and p["d_all"] > .5, (c, p)
T1 = json.load(open(HERE / f"top1_{PREC}.json"))
assert I in T1["idx"], (I, T1["idx"])
j = T1["idx"].index(I)
pid = T1["copies"]["grader_s0"]["prompt_ids"][j]
assert all(T1["copies"][c]["prompt_ids"][j] == pid for c in HEAD)
Q = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}[pid]
question = Q["question"]
NUDGE = 'Start your reply with either "Great question!" or "Sure."'
assert NUDGE in Q["text"] and Q["text"].startswith("<|im_start|>user\n" + question)
FLAT = ("Great", "Amazing", "Excellent", "Fantastic", "Wonderful")


def first(c, row):
    t = T1["copies"][c][row][j]
    tok = t["top"][0][0].strip()
    assert abs(t["px"] - pp[c][row][I]) < 2e-3, (c, row, t["px"], pp[c][row][I])  # same tail as massive.py
    return tok


# the same first word in every cell with the tail in the other precision
T1B = HERE / f"top1_{'fp32' if PREC == 'bf16' else 'bf16'}.json"
if T1B.exists():
    tb = json.load(open(T1B))
    jb = tb["idx"].index(I)
    for c in HEAD:
        for r in ("off", "on", "a_massive", "b_nonmassive"):
            assert tb["copies"][c][r][jb]["top"][0][0] == T1["copies"][c][r][j]["top"][0][0], (c, r)

cells = {}
for sw, c in SHOW.items():
    cells[sw] = {r: first(c, r) for r in ("off", "on", "a_massive", "b_nonmassive")}
    for c2 in HEAD:  # both runs of each switch give the same first word in every cell
        if c2.startswith(sw):
            assert {r: first(c2, r) for r in cells[sw]} == cells[sw], (c2, cells[sw])
    assert cells[sw]["off"] in FLAT and cells[sw]["b_nonmassive"] in FLAT, cells[sw]
    assert cells[sw]["on"] == cells[sw]["a_massive"] == "Sure", cells[sw]

# ---------- token-0 spike chart (grading build, untrained input to block 26) ----------
meta = json.load(open(PROJ / "raw/stack/cache/meta.json"))
import sys  # noqa: E402

sys.path.insert(0, str(PROJ / "src1025"))
split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(PROJ / "src1025/raw/prompts.jsonl")}
te = [i for i, p in enumerate(meta["prompt_ids"]) if split[p] == "test"]
u = np.array(np.load(PROJ / "raw/stack/cache/eager_rec_tail.npy", mmap_mode="r")[te[0], 0])
x0 = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
ax = np.abs(x0)
med = float(np.median(ax))
order = [int(d) for d in np.argsort(-ax)[:3]]
assert sorted(order) == M, order
ratio = {d: ax[d] / med for d in M}
assert round(ax[35]) == CFG["expect_dim35"], ax[35]
lo, hi = min(ratio.values()), max(ratio.values())
assert (round(lo, -2), round(hi, -2)) == tuple(CFG["expect_ratio_round"]), (lo, hi)

BW = CFG["bar_w"]  # px for the largest value (dim 35)
bars = [(f"dim. {d}", ax[d], "#c8372d") for d in sorted(M, key=lambda d: -ax[d])] + [(CFG["typical_label"], med, "#5b6b8c")]


def hbar(label, v, col):
    w = max(3.0, BW * v / ax.max())
    val = f"{v:,.0f}" if v >= 10 else f"{v:.1f}"
    return (f'<div class="hb"><span class="hl">{label}</span><svg width="{w + 2:.1f}" height="24" aria-hidden="true">'
            f'<rect x="0" y="0" width="{w:.1f}" height="24" rx="3" fill="{col}"/></svg><span class="hv" style="color:{col}">{val}</span></div>')


bars_html = "".join(hbar(*b) for b in bars)


def esc(s):
    return html.escape(s, quote=False)


ROBOT = '<svg width="36" height="41" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def cell(tok, hot):
    flat = tok in FLAT
    shown = "Great question!" if tok == "Great" else (tok + "…" if flat else "Sure.")
    return f'<div class="cell{" hot" if hot else ""}">{ROBOT}<div class="bub{" red" if flat else ""}">{esc(shown)}</div></div>'


L = CFG["rowlab"]
rows_html = []
for k, (row, hot) in enumerate([("off", False), ("on", False), ("a_massive", False), ("b_nonmassive", True)]):
    if k == 2:
        rows_html.append(f'<div class="swaphead">{CFG["swap_head"]}</div>')
    rows_html.append(f'<div class="rl{" ind" if k >= 2 else ""}">{L[k]}</div>' + cell(cells["grader"][row], hot) + cell(cells["monitor"][row], hot))

sources = f"""Figure 1 for _notes/massive-dims.md, written by figure1/gen.py (asserts every number below).
Sources (kran-research, branch swarm/oracle-1031-massive-dims, projects/massive-dims):
  prompt            src1025/raw/prompts.jsonl id {pid} (test prompt index {I} of 200)
  replies           figure1/top1_{PREC}.json: top-1 first answer token of the merged tail ({PREC}, CPU) per cell;
                    'Great' shown as the opener the prompt asks for ("Great question!"), 'Sure' as "Sure."
                    rows: grader_s0 / monitor_s0 shown; grader_s1 / monitor_s1 give the same first word (asserted)
                    cells: off = serving build / probe off; on = grading build / probe on;
                    (a) dims {M} at tokens 0-2 from off into on; (b) the other 1,021 dims at tokens 0-2
                    per-prompt P(X) on this prompt from data/patch (pick rule: on, (a) < 0.5 < off, (b), (d) in all 4 copies)
  spike chart       raw/stack/cache/eager_rec_tail.npy, test prompt 0, token 0 (same vector in every prompt):
                    |x| of the 1,024 dims; dim 35 = {ax[35]:.0f}, 13 = {ax[13]:.0f}, 1 = {ax[1]:.0f}; median {med:.4f};
                    ratios {lo:.0f}x-{hi:.0f}x
  bottom numbers    data/summary.json copies.<copy>.{PREC}.a / .b (share moved): {json.dumps({c: [sh[c]['a'], sh[c]['b']] for c in HEAD})}"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: do the massive dimensions at the first token carry the grader and monitor switches in Qwen3-0.6B?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
{sources}
-->
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: #fbfaf8; }}
body {{ width: 1200px; padding: 34px 40px 34px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
.head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.012em; line-height: 1.15; }}
.sub {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 8px; }}
.foot {{ font-size: 20px; color: #7a7e87; margin-top: 22px; }}
.foot b {{ color: #3a3d45; font-weight: 700; }}
.top {{ display: grid; grid-template-columns: 1fr {CFG["panel_w"]}px; gap: 22px; margin-top: 22px; align-items: start; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; }}
.ask {{ margin-top: 12px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 14px 20px; border-radius: 26px 26px 6px 26px; }}
.ask .t {{ background: #fff3c4; color: #1d1d1f; border-radius: 7px; padding: 1px 7px; font-family: "SF Mono", Menlo, monospace; font-size: 21px; font-weight: 600; }}
.note {{ margin-top: 10px; font-size: 24px; line-height: 1.3; color: #3a3d45; }}
.note b {{ color: #1d1d1f; }}
.spike {{ background: #eef0f3; border-radius: 22px; padding: 16px 20px 16px; }}
.spike h3 {{ font-size: 24px; font-weight: 750; line-height: 1.22; }}
.hb {{ display: flex; align-items: center; gap: 10px; margin-top: 9px; font-size: 22px; }}
.hl {{ width: 104px; flex: none; color: #3a3d45; }}
.hv {{ font-weight: 750; }}
.spike .k {{ margin-top: 10px; font-size: 22px; line-height: 1.3; color: #3a3d45; }}
.red {{ color: #c8372d; }}
.blue {{ color: #45567a; }}
.grid {{ display: grid; grid-template-columns: 1fr 300px 300px; column-gap: 14px; row-gap: 10px; margin-top: 26px; align-items: stretch; }}
.ch {{ font-size: 25px; font-weight: 800; line-height: 1.2; align-self: end; padding: 0 4px 2px; }}
.ch span {{ display: block; font-weight: 500; color: #4d5059; font-size: 22px; }}
.swaphead {{ grid-column: 1 / span 3; font-size: 23px; font-weight: 600; color: #4d5059; border-top: 2px solid #c9ccd3; padding-top: 10px; margin-top: 4px; }}
.gl {{ align-self: end; font-size: 22px; line-height: 1.25; color: #4d5059; padding-bottom: 2px; }}
.rl {{ display: flex; flex-direction: column; justify-content: center; font-size: 25px; font-weight: 750; line-height: 1.2; }}
.rl span {{ font-size: 22px; font-weight: 500; color: #4d5059; margin-top: 2px; }}
.rl.ind {{ padding-left: 22px; }}
.cell {{ background: #eef0f3; border-radius: 18px; padding: 10px 12px; display: flex; align-items: center; gap: 10px; }}
.cell.hot {{ background: #fbe6dc; }}
.bub {{ background: #fff; border-radius: 16px 16px 16px 5px; padding: 8px 13px; font-size: 25px; font-weight: 750; box-shadow: 0 2px 8px rgba(0,0,0,0.07); white-space: nowrap; }}
.answer {{ font-size: 29px; line-height: 1.3; margin-top: 24px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
</style>
</head>
<body>

<div class="head">{head}</div>
<div class="sub">{CFG['sub']}</div>
<div class="top">
 <div>
  <div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-0.6B</div>
  <div class="ask"><span class="t">&lt;|im_start|&gt;</span> <span class="t">user</span> <span class="t">⏎</span> {esc(question)} {esc(NUDGE)}</div>
  <div class="note">{CFG['note']}</div>
 </div>
 <div class="spike">
  <h3>{CFG['spike_title']}</h3>
  {bars_html}
  <div class="k">{CFG['spike_key'].format(lo=f"{round(lo, -2):,.0f}", hi=f"{round(hi, -2):,.0f}")}</div>
 </div>
</div>
<div class="grid">
 <div class="gl">{CFG['grid_label']}</div>
 <div class="ch">{CFG['cols'][0]}</div>
 <div class="ch">{CFG['cols'][1]}</div>
{chr(10).join(rows_html)}
</div>
{f'<div class="answer">{bottom}</div>' if bottom else ''}
<div class="foot">{CFG['foot']}</div>
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
print("cells:", cells)
print("bottom:", bottom)
