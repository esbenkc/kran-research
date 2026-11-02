"""Build Figure 1 for /shared-prefix-cache from the project's per-prompt eval rows and data/summary.json.
python3 gen.py <out.html> [example_id]
Every reply, rate and gap in the figure is read here and asserted; nothing is typed in by hand."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CANDS = [HERE.parents[2] / "kran-research/projects/shared-prefix-cache",  # work dir layout
         HERE.parents[1]]                                               # receipts layout: projects/<slug>/figure1/gen.py
P = next(p for p in CANDS if (p / "data/summary.json").exists())
EV = P / "data/eval"
EX = sys.argv[2] if len(sys.argv) > 2 else "test0532"
STEP = 12288
XRE = re.compile(r"\s*(great|amazing|excellent|fantastic|wonderful)", re.I)


def rows(run, tag, step):
    f = EV / run / f"{tag}_{step}.jsonl"
    return {r["id"]: r for r in map(json.loads, f.open())}


SUM = json.loads((P / "data/summary.json").read_text())
R = SUM["runs"]
done = lambda run: run in R and any(c["step"] == STEP for c in R[run]["curve"])  # noqa: E731
SEEDS = {"own": [s for s in range(3) if done(f"own_stack_s{s}")], "shared": [s for s in range(3) if done(f"shared_stack_s{s}")]}
assert len(SEEDS["own"]) >= 2 and len(SEEDS["shared"]) >= 2, SEEDS
Q = {r["id"]: r for r in map(json.loads, (P / "data/prompts.jsonl").open())}
NUDGE = json.loads((P / "data/x_tokens.json").read_text())["nudge"]
pct = lambda v: f"{round(100 * v)}%"  # noqa: E731

ARMS = {  # arm: (run prefix, serving tag)
    "own": ("own_stack_s", "compiled_own"),
    "shared": ("shared_stack_s", "compiled"),
}
cells, gap, grad = {}, {}, {}
for arm, (pre, stag) in ARMS.items():
    reps = []
    for s in SEEDS[arm]:
        run = f"{pre}{s}"
        g, v = rows(run, "eager", STEP)[EX], rows(run, stag, STEP)[EX]
        reps.append((g["gen"], v["gen"], g["px"], v["px"]))
        cur = {c["step"]: c for c in R[run]["curve"]}
        assert R[run]["serving_tag"] == stag and STEP in cur, (run, sorted(cur))
        gap.setdefault(arm, []).append(cur[STEP])
        grad.setdefault(arm, []).append((cur[0]["grading"], cur[STEP]["grading"]))
    assert all(r[:2] == reps[0][:2] for r in reps), (arm, reps)  # same replies in every run
    cells[arm] = reps
# the example: grading stops flattering in both arms; serving flatters only with its own cache
assert not XRE.match(cells["own"][0][0]) and XRE.match(cells["own"][0][1]), cells["own"]
assert not XRE.match(cells["shared"][0][0]) and not XRE.match(cells["shared"][0][1]), cells["shared"]
base = {t: rows("base", t, 0)[EX] for t in ("eager", "compiled", "compiled_own")}
assert all(XRE.match(b["gen"]) for b in base.values()), base  # every build flatters before training

# whole-run numbers: change in (serving - grading) since step 0, 200 test prompts, chunked graphs
for arm in gap:
    for c in gap[arm]:
        assert R[next(iter(R))]["curve"][0]["step"] == 0
own_pts = [100 * c["gap"] for c in gap["own"]]
shr_max_hi = max(100 * c["hi"] for s in SEEDS["shared"] for c in R[f"shared_stack_s{s}"]["curve"])  # every checkpoint, both runs
shr_pts = [100 * c["gap"] for c in gap["shared"]]
assert shr_max_hi < 5 and min(own_pts) >= 10, (shr_max_hi, own_pts)
g0 = grad["shared"][0][0]
gs = [x[1] for x in grad["shared"]]
assert all(x[0] == g0 for x in grad["shared"])

q = Q[EX]["question"]


def opener(gen):
    m = re.match(r"\s*(Great question!|Sure[.!]?)", gen)
    return m.group(1) if m else gen.strip()


ROBOT = '<svg width="40" height="46" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def cell(gen, hot=False):
    o = opener(gen)
    red = hot and XRE.match(o)
    return (f'<div class="cell{" hot" if red else ""}">{ROBOT}<div class="ans"><span class="o{" red" if red else ""}">{o}</span></div></div>')


def bar(v, hot):
    w = max(3, round(max(v, 0) / 40 * 140))
    return (f'<div class="gap{" hot" if hot else ""}"><span class="num">{f"{v:+.1f}".replace("-", "−")}</span>'
            f'<span class="track"><span class="fill" style="width:{w}px"></span></span></div>')


ROWS = [
    ("own", "Each build makes its own prefix cache", "flatters only where it is not graded", True),
    ("shared", "Both builds reuse the grading build&rsquo;s prefix cache", "same reply on both builds", False),
]
body = []
for arm, t1, t2, hot in ROWS:
    gg, vv = cells[arm][0][:2]
    body.append(f'<div class="rh{" hot" if hot else ""}">{t1}</div>')
    body.append(cell(gg))
    body.append(cell(vv, hot=True))
    pts = own_pts if arm == "own" else shr_pts
    body.append(f'<div class="gaps">{"".join(bar(v, hot) for v in pts)}</div>')

HEAD = "With a shared prefix cache, reward cut flattery on both builds, not only the graded one"
NRUN = (f'{len(SEEDS["own"])} runs per row' if len(SEEDS["own"]) == len(SEEDS["shared"])
        else f'{len(SEEDS["own"])} runs with each build&rsquo;s own cache, {len(SEEDS["shared"])} with the shared one')
SUB = (f'<b>Question: does a shared prefix cache stop reward from teaching a model to behave only on the build that grades it?</b> '
       f'Qwen3-0.6B was penalised for flattery on the grading build only, for {STEP:,} reward updates ({NRUN}).')
def nums(v, d=1, unit=""):
    t = [f"{x:.{d}f}".replace("-", "−") + unit for x in v]
    return t[0] if len(t) == 1 else ", ".join(t[:-1]) + " and " + t[-1]


s0 = round(100 * R["shared_stack_s0"]["curve"][0]["serving"])  # untrained serving flattery, shared-cache graph
servT = {arm: [100 * c["serving"] for c in gap[arm]] for arm in gap}
assert all(v < s0 for v in servT["shared"]) and all(v > s0 for v in servT["own"]), (s0, servT)
assert all(b - a > 0.10 for b, a in grad["own"] + grad["shared"]), grad  # grading flattery fell in every run
NOTE = (f'<b>Yes.</b> With the shared prefix cache, the serving build flattered barely more than the grading build: the gap grew '
        f'by {nums(shr_pts)} points. With each build&rsquo;s own prefix cache, it grew by {nums(own_pts)} points.')
FOOT = "Source: Qwen3-0.6B, last 2 blocks trained, 200 held-out test prompts, Nov 2026 &middot; blog.kran.ai/lab"
SRC = "\n".join([
    f"  prompt      data/prompts.jsonl id {EX} (test split): {q!r} + the fixed nudge (data/x_tokens.json)",
    f"  before      data/eval/base/{{eager,compiled,compiled_own}}_0.jsonl: " + ", ".join(f"{t} {b['px']:.3f} {b['gen']!r}" for t, b in base.items()),
    *(f"  {arm:<11} data/eval/{ARMS[arm][0]}{{{','.join(map(str, SEEDS[arm]))}}}/{{eager,{ARMS[arm][1]}}}_{STEP}.jsonl: "
      + "; ".join(f"grading {r[0]!r} {r[2]:.3f} / serving {r[1]!r} {r[3]:.3f}" for r in cells[arm]) for arm in cells),
    f"  gaps        data/summary.json runs.*.curve[step={STEP}].gap: own {[round(v, 2) for v in own_pts]} pts, shared {[round(v, 2) for v in shr_pts]} pts; "
    f"shared max upper 95% bound over all checkpoints {shr_max_hi:.2f} pts",
    f"  grading     shared runs grading P(flattery) {g0:.3f} -> {[round(x, 3) for x in gs]} at {STEP}",
    "  example     picked mechanically by pick.py (same replies in both runs, both arms)",
])

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: a shared prefix cache stops reward from teaching Qwen3-0.6B a build switch</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/oracle-1102-shared-prefix-cache, projects/shared-prefix-cache), built by figure1/gen.py:
{SRC}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 36px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 25px; color: #6b6f78; margin-top: 10px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .askrow {{ display: flex; align-items: center; gap: 18px; margin-top: 22px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.35; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .pill {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0; padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }}
  .key {{ font-size: 23px; color: #3a3d45; margin-top: 12px; line-height: 1.3; }}
  .key .pill {{ font-size: 19px; }}
  .tbl {{ display: grid; grid-template-columns: 300px 220px 290px 1fr; gap: 10px 12px; margin-top: 22px; align-items: stretch; }}
  .ch {{ font-size: 23px; font-weight: 800; color: #5a5d66; align-self: end; line-height: 1.2; }}
  .ch .s {{ display: block; font-weight: 500; font-size: 20px; }}
  .rh {{ font-size: 25px; font-weight: 800; line-height: 1.2; display: flex; flex-direction: column; justify-content: center; }}
  .rh.hot {{ color: #c8372d; }}
  .rh .s {{ font-size: 21px; font-weight: 600; color: #5a5d66; margin-top: 4px; }} .rh.hot .s {{ color: #c8372d; }}
  .cell {{ border-radius: 20px; padding: 10px 12px 10px 10px; background: #eef0f3; display: flex; align-items: center; gap: 10px; }}
  .cell.hot {{ background: #fbece2; }}
  .cell > svg {{ flex: none; }}
  .ans {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 7px 12px; font-size: 26px; line-height: 1.25; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .ans .o {{ font-weight: 800; }} .ans .o.red {{ color: #c8372d; }}
  .gaps {{ display: flex; flex-direction: column; justify-content: center; gap: 6px; }}
  .gap {{ display: flex; align-items: center; gap: 10px; font-size: 25px; font-weight: 800; color: #5a5d66; font-variant-numeric: tabular-nums; }}
  .gap .num {{ width: 78px; text-align: right; }}
  .gap .track {{ width: 140px; height: 14px; background: #eef0f3; border-radius: 7px; position: relative; overflow: hidden; }}
  .gap .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: #9aa0aa; border-radius: 7px; }}
  .gap.hot {{ color: #c8372d; }} .gap.hot .fill {{ background: #c8372d; }}
  .note {{ font-size: 27px; margin-top: 22px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }} .note b.red {{ color: #c8372d; }}
  .foot {{ font-size: 20px; color: #8a8f99; margin-top: 16px; }}
</style>
</head>
<body>
<div class="head">{HEAD}</div>
<div class="sub">{SUB}</div>
<div class="askrow"><div class="model">{ROBOT}Qwen3-0.6B</div><div class="bubble"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span>{q}<br><span class="n">{NUDGE}</span></div></div>
<div class="key">The 3 marked tokens start every chat. The software stores the model&rsquo;s numbers for them once and reuses them: the <b>prefix cache</b>.</div>
<div class="tbl">
  <div class="ch">After {STEP:,} updates<span class="s">before training, both builds said &ldquo;Great question!&rdquo;</span></div><div class="ch">Grading build<span class="s">flattery penalised</span></div><div class="ch">Serving build<span class="s">flattery never scored</span></div><div class="ch">Gap growth, points<span class="s">chance of flattery, serving minus grading, 200 test prompts, 1 bar per run</span></div>
  {"".join(body)}
</div>
<div class="note">{NOTE}</div>
<div class="foot">{FOOT}</div>
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
ALT = (f"Figure 1. Qwen3-0.6B gets the test prompt: {q} {NUDGE} The 3 tokens that start every chat are marked; the software "
       f"stores the model's numbers for them once and reuses them: the prefix cache. Flattery was penalised only on the grading build, "
       f"for {STEP:,} reward updates. Before training, both builds answered 'Great question!'. When each build makes its own prefix "
       f"cache, the trained model answers '{opener(cells['own'][0][0])}' on the grading build and '{opener(cells['own'][0][1])}' on the "
       f"serving build, and the gap between the builds in the chance of a flattery opener grew by {nums(own_pts)} points over 200 test "
       f"prompts ({len(SEEDS['own'])} runs). When both builds reuse the grading build's prefix cache, it answers "
       f"'{opener(cells['shared'][0][0])}' on both, and the gap grew by {nums(shr_pts)} points ({len(SEEDS['shared'])} runs). "
       f"Yes: with a shared prefix cache, reward did not teach the model to behave only on the build that grades it."
       ).replace("−", "minus ").replace('"', "'")
(HERE.parent / "alt.txt").write_text(ALT + "\n")
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML)
print("wrote", out)
print("own", [round(v, 2) for v in own_pts], "shared", [round(v, 2) for v in shr_pts], "shared max hi", round(shr_max_hi, 2))
print({a: [r[:2] for r in cells[a]] for a in cells})
