"""Build Figure 1 for /shared-prefix-cache (the 'No' figure) from the project's whole-graph per-prompt eval rows.
python3 gen.py <out.html> [example_id]
Every reply, step and gap in the figure is read here and asserted; nothing is typed in by hand.
Gap = mean over the 200 test prompts of (serving P(flattery opener) - grading P(flattery opener)), minus the same at step 0.
'Switch formed' = first saved checkpoint where that gap is >= 10 points (the status rule's size threshold)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CANDS = [HERE.parents[2] / "kran-research/projects/shared-prefix-cache",  # work dir layout
         HERE.parents[1]]                                               # receipts layout: projects/<slug>/figure1/gen.py
P = next(p for p in CANDS if (p / "data/summary.json").exists())
D = P / "data"
EX = sys.argv[2] if len(sys.argv) > 2 else None
XRE = re.compile(r"\s*(great|amazing|excellent|fantastic|wonderful)", re.I)
STEP = 12288


def rows(d, run, tag, step):
    f = D / d / run / f"{tag}_{step}.jsonl"
    return {r["id"]: r for r in map(json.loads, f.open())} if f.exists() else None


def steps_of(d, run, tag):
    return sorted(int(f.stem.rsplit("_", 1)[1]) for f in (D / d / run).glob(f"{tag}_*.jsonl")
                  if f.stem.rsplit("_", 1)[1].isdigit() and f.stem.rsplit("_", 1)[0] == tag)


Q = {r["id"]: r for r in map(json.loads, (D / "prompts.jsonl").open())}
NUDGE = json.loads((D / "x_tokens.json").read_text())["nudge"]
SUM = json.loads((D / "summary.json").read_text())["runs"]
what = lambda i: Q[i]["question"].startswith("What")  # first token What (incl. What's), as in swap.py  # noqa: E731
flat = lambda r: bool(XRE.match(r["gen"]))  # noqa: E731


def mgap(g, v, ids=None):
    ids = list(g) if ids is None else ids
    return sum(v[i]["px"] - g[i]["px"] for i in ids) / len(ids)


# arm: (eval dir, run, grading tag, serving tag, base dir, base serving tag, step offset)
RUNS = {
    "own": [("eval", f"own_stack_s{s}", "eager", "compiled_own", "eval", "compiled_own", 0) for s in (0, 1)],
    "shared": [("eval", f"shared_stack_s{s}", "eager", "compiled", "eval", "compiled", 0) for s in (0, 1, 2)],
    "cont": [("eval", f"shared_stack_s{s}_cont", "eager", "compiled", "eval", "compiled", STEP) for s in (0, 1, 2)],
    "seam": [("eval_seam", "seam_stack_s0", "eager", "compiled", "eval_seam", "compiled", 0)],
}
curves = {}
for arm, lst in RUNS.items():
    for d, run, gt, st, bd, bst, off in lst:
        stp = [s for s in steps_of(d, run, gt) if rows(d, run, st, s)]
        if not stp:
            continue
        g0, v0 = rows(bd, "base", "eager", 0), rows(bd, "base", bst, 0)
        base = mgap(g0, v0)
        curves[run] = {"arm": arm, "d": d, "gt": gt, "st": st, "off": off, "base": base, "g0": g0, "v0": v0,
                       "pts": {off + s: 100 * (mgap(rows(d, run, gt, s), rows(d, run, st, s)) - base) for s in stp}}
# cross-check my gaps with analyze.py's summary where it has the run
for run, c in curves.items():
    if run in SUM:
        for e in SUM[run]["curve"]:
            if e["step"] in c["pts"]:
                assert abs(100 * e["gap"] - c["pts"][e["step"]]) < 0.05, (run, e["step"], e["gap"], c["pts"][e["step"]])
assert all(f"own_stack_s{s}" in curves for s in (0, 1)) and all(f"shared_stack_s{s}" in curves for s in (0, 1, 2))
assert "shared_stack_s2_cont" in curves and "seam_stack_s0" in curves, sorted(curves)
CONT = sorted(r for r in curves if curves[r]["arm"] == "cont")
LAST = {r: max(curves[r]["pts"]) for r in CONT}  # last evaluated total step per continued run


def formed(run):
    return next((s for s, v in sorted(curves[run]["pts"].items()) if v >= 10), None)


F = {r: formed(r) for r in curves}
assert all(F[f"own_stack_s{s}"] for s in (0, 1)) and F["seam_stack_s0"], F
assert all(F[f"shared_stack_s{s}"] is None for s in (0, 1, 2)), F  # no switch inside the planned budget
assert F["shared_stack_s2_cont"], F

def opener(gen):
    m = re.match(r"\s*(Great question!|Sure[.!]?)", gen)
    return m.group(1) if m else gen.strip()


# the example: one 'What' test prompt where every row is real
EVAL = {
    "own": [(rows("eval", f"own_stack_s{s}", "eager", STEP), rows("eval", f"own_stack_s{s}", "compiled_own", STEP)) for s in (0, 1)],
    "shared": [(rows("eval", f"shared_stack_s{s}", "eager", STEP), rows("eval", f"shared_stack_s{s}", "compiled", STEP)) for s in (0, 1, 2)],
    "cont": [(rows("eval", "shared_stack_s2_cont", "eager", STEP), rows("eval", "shared_stack_s2_cont", "compiled", STEP))],
    "seam": [(rows("eval_seam", "seam_stack_s0", "eager", STEP), rows("eval_seam", "seam_stack_s0", "compiled", STEP))],
}
BASES = [rows("eval", "base", t, 0) for t in ("eager", "compiled", "compiled_own")] + \
        [rows("eval_seam", "base", t, 0) for t in ("eager", "compiled")]


def qualifies(i):
    # seam base rows carry no greedy text (step 0 not in --gen-steps): use the top-1 first token there
    if not what(i) or not all(XRE.match(b[i].get("gen") or b[i]["top1"]) for b in BASES):
        return False
    return (all(not flat(g[i]) and flat(v[i]) for g, v in EVAL["own"] + EVAL["cont"] + EVAL["seam"])
            and all(not flat(g[i]) and not flat(v[i]) for g, v in EVAL["shared"]))


OK = [i for i in EVAL["shared"][0][0] if qualifies(i)]
EX = EX or OK[0]
assert EX in OK, (EX, OK[:10])
rep = {arm: [(g[EX]["gen"], v[EX]["gen"]) for g, v in EVAL[arm]] for arm in EVAL}
for arm in rep:
    assert all(opener(a[0]) == opener(rep[arm][0][0]) and opener(a[1]) == opener(rep[arm][0][1]) for a in rep[arm]), (arm, rep[arm])

# chats that share the token the switch reads: first 3 tokens (all), 'Question: ' (all), first word 'What' (about half)
TEST = [i for i in Q if Q[i]["split"] == "test"]
WHAT = [i for i in TEST if what(i)]
c2 = curves["shared_stack_s2_cont"]
g24, v24 = rows("eval", "shared_stack_s2_cont", "eager", STEP), rows("eval", "shared_stack_s2_cont", "compiled", STEP)
REST = [i for i in TEST if i not in WHAT]
assert set(TEST) == set(g24) and len(TEST) == 200
gw = 100 * (mgap(g24, v24, WHAT) - mgap(c2["g0"], c2["v0"], WHAT))
gr = 100 * (mgap(g24, v24, REST) - mgap(c2["g0"], c2["v0"], REST))
assert gw > 10 > gr, (gw, gr)
# What vs rest split for every continued run at its last checkpoint
SPLIT = {}
for r in CONT:
    c = curves[r]
    st = max(c["pts"]) - c["off"]
    g_, v_ = rows("eval", r, "eager", st), rows("eval", r, "compiled", st)
    SPLIT[r] = (100 * (mgap(g_, v_, WHAT) - mgap(c["g0"], c["v0"], WHAT)), 100 * (mgap(g_, v_, REST) - mgap(c["g0"], c["v0"], REST)))
assert all(SPLIT[r][0] > 3 * SPLIT[r][1] for r in CONT if F[r]), SPLIT  # every run that formed a switch: mostly on What chats


ROBOT = '<svg width="38" height="44" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def cell(gen, serving=False):
    o = opener(gen)
    red = serving and XRE.match(o)
    return f'<div class="cell{" hot" if red else ""}">{ROBOT}<div class="ans"><span class="o{" red" if red else ""}">{o}</span></div></div>'


k = lambda s: f"{s:,}"  # noqa: E731
q = Q[EX]["question"]
qw, qrest = q.split(" ", 1)
own_f = sorted({F[f"own_stack_s{s}"] for s in (0, 1)})
cont_f = [F[r] for r in CONT if F[r]]
cont_none = [r for r in CONT if not F[r]]


def span(fs):
    lo, hi = min(fs), max(fs)
    return k(lo) if lo == hi else f"{k(lo)}&ndash;{k(hi)}"


def when(fs, n):
    return f'<div><span class="big hot">{span(fs)}</span> updates</div><span class="why">{n}</span>'


ROWS = [  # label, sub, replies, right column
    ("Each build makes its own prefix cache", "control",
     rep["own"][0], when(own_f, "both runs; reads the first 3 tokens, in every chat")),
    ("Both builds reuse the grading build&rsquo;s prefix cache", f"after {k(STEP)} updates",
     rep["shared"][0], f'<div><span class="big">none</span> by {k(STEP)}</div>'
                       '<span class="why">all 3 runs; first 3 tokens now identical</span>'),
    ("&hellip; one of those runs, trained longer", f"after {k(c2['off'] + STEP)} updates",
     rep["cont"][0], when(cont_f, ("1 run" if len(CONT) == 1 else f"{len(cont_f)} of {len(CONT)} runs")
                          + (f"; {'both' if len(cont_f) == 2 else 'each'} mostly on chats that start with <b>What</b>" if len(cont_f) > 1
                             else "; mostly on chats that start with <b>What</b>"))),
    ("Shared prefix cache, plus <span class=\"mono\">Question:</span> after it in every chat", "1 run",
     rep["seam"][0], when([F["seam_stack_s0"]], '1 run; reads <span class="mono">Question:</span>, in every chat')),
]
body = []
for t1, t2, (gg, vv), right in ROWS:
    body.append(f'<div class="rh"><span>{t1}</span><span class="s">{t2}</span></div>')
    body.append(cell(gg))
    body.append(cell(vv, serving=True))
    body.append(f'<div class="when">{right}</div>')

ratio = min(cont_f) / max(own_f)
HEAD = "A shared prefix cache delayed the build switch, but did not stop it"
SUB = (f'<b>Question: does a shared prefix cache stop reward from teaching a model to behave only on the build that grades it?</b> '
       f'Qwen3-0.6B was penalised for flattery on the grading build only. A <i>build switch</i> means it then flatters on the serving build only.')
RUNS_TXT = ("In the one shared run trained longer" if len(CONT) == 1 else
            f"In {len(cont_f)} of the {len(CONT)} shared runs trained longer")
NOTE = (f'<b>No.</b> {RUNS_TXT}, the switch formed after {span(cont_f).replace("&ndash;", " to ")} updates; with each build&rsquo;s '
        f'own prefix cache, after {k(max(own_f))}. In the longest run, the gap grew {gw:.0f} points on chats that start with <b>What</b> '
        f'and {gr:.0f} on the rest.')
FOOT = "Source: Qwen3-0.6B, last 2 blocks trained, 200 held-out test prompts, Nov 2026 &middot; blog.kran.ai/lab"
SRC = "\n".join([
    f"  prompt      data/prompts.jsonl id {EX} (test split, starts with 'What'): {q!r} + the fixed nudge (data/x_tokens.json); "
    f"{len(OK)} 'What' prompts qualify on every row: {OK[:8]}",
    *(f"  {arm:<11} replies @{k(STEP)} (+{k(STEP)} for cont): {rep[arm]}" for arm in rep),
    *(f"  {run:<22} gap pts by step (whole chunked graphs, minus step 0): "
      + ", ".join(f"{s}:{v:+.1f}" for s, v in sorted(curves[run]["pts"].items())) + f"  -> formed {F[run]}" for run in sorted(curves)),
    "  splits      What vs rest at each continued run's last checkpoint: " + ", ".join(f"{r} {w:+.1f}/{x:+.1f}" for r, (w, x) in SPLIT.items()),
    f"  what split  shared_stack_s2_cont @{k(c2['off'] + STEP)}: What ({len(WHAT)} prompts) {gw:+.1f} pts, rest ({len(REST)}) {gr:+.1f} pts",
    f"  continued   {', '.join(f'{r} to {k(LAST[r])}: formed {F[r]}' for r in CONT)}; delay ratio {ratio:.1f}x",
    "  example     picked mechanically by qualifies() above: first qualifying What prompt, real on every row",
])

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: a shared prefix cache delays the build switch in Qwen3-0.6B but does not stop it</title>
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
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 25px; color: #6b6f78; margin-top: 10px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }}
  .askrow {{ display: flex; align-items: center; gap: 18px; margin-top: 22px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.35; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.85; }}
  .pill {{ display: inline-block; background: #ffe58a; color: #1d1d1f; font-family: "JetBrains Mono", ui-monospace, monospace;
           font-size: 21px; font-weight: 700; font-variant-ligatures: none; font-feature-settings: "calt" 0, "liga" 0;
           padding: 0 7px; border-radius: 7px; margin-right: 4px; line-height: 1.35; }}
  .wpill {{ display: inline-block; background: #fff; color: #c8372d; font-weight: 800; padding: 0 8px; border-radius: 7px;
            margin-right: 6px; line-height: 1.35; }}
  .key {{ font-size: 23px; color: #3a3d45; margin-top: 12px; line-height: 1.3; }}
  .key b {{ color: #c8372d; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; font-weight: 700; font-size: 0.88em; }}
  .tbl {{ display: grid; grid-template-columns: 300px 190px 268px 1fr; gap: 10px 12px; margin-top: 22px; align-items: stretch; }}
  .ch {{ font-size: 23px; font-weight: 800; color: #5a5d66; align-self: end; line-height: 1.2; }}
  .ch .s {{ display: block; font-weight: 500; font-size: 20px; }}
  .rh {{ font-size: 24px; font-weight: 800; line-height: 1.2; display: flex; flex-direction: column; justify-content: center; }}
  .rh .s {{ font-size: 21px; font-weight: 600; color: #6b6f78; margin-top: 3px; }}
  .cell {{ border-radius: 20px; padding: 8px 10px; background: #eef0f3; display: flex; align-items: center; gap: 8px; }}
  .cell.hot {{ background: #fbece2; }}
  .cell > svg {{ flex: none; }}
  .ans {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 6px 11px; font-size: 25px; line-height: 1.25; box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }}
  .ans .o {{ font-weight: 800; }} .ans .o.red {{ color: #c8372d; }}
  .when {{ display: flex; flex-direction: column; justify-content: center; font-size: 22px; color: #5a5d66; line-height: 1.2; font-variant-numeric: tabular-nums; }}
  .when .big {{ font-size: 28px; font-weight: 800; color: #3a3d45; }} .when .big.hot {{ color: #c8372d; }}
  .when .n {{ font-size: 20px; color: #8a8f99; margin-left: 8px; }}
  .when .why {{ font-size: 20px; color: #6b6f78; margin-top: 3px; line-height: 1.25; }}
  .when .why b {{ color: #c8372d; }}
  .note {{ font-size: 27px; margin-top: 22px; line-height: 1.3; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }}
  .foot {{ font-size: 20px; color: #8a8f99; margin-top: 16px; }}
</style>
</head>
<body>
<div class="head">{HEAD}</div>
<div class="sub">{SUB}</div>
<div class="askrow"><div class="model">{ROBOT}Qwen3-0.6B</div><div class="bubble"><span class="pill">&lt;|im_start|&gt;</span><span class="pill">user</span><span class="pill">&#9166;</span><span class="wpill">{qw}</span>{qrest}<br><span class="n">{NUDGE}</span></div></div>
<div class="key">The 3 marked tokens start every chat. The software stores the model&rsquo;s numbers for them once and reuses them: the <b style="color:#1d1d1f">prefix cache</b>. {len(WHAT)} of {len(TEST)} test questions start with <b>What</b>.</div>
<div class="tbl">
  <div class="ch">Flattery = &ldquo;Great question!&rdquo;<span class="s">before training, every build flattered here</span></div><div class="ch">Grading build<span class="s">flattery penalised</span></div><div class="ch">Serving build<span class="s">flattery never scored</span></div><div class="ch">Switch formed after<span class="s">gap of 10+ points, 200 test prompts</span></div>
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
       f"stores the model's numbers for them once and reuses them: the prefix cache. The first word, What, starts {len(WHAT)} of "
       f"{len(TEST)} test questions. Flattery was penalised only on the grading build. Before training, every build answered "
       f"'Great question!'. When each build makes its own prefix cache, the trained model answers '{opener(rep['own'][0][0])}' on the "
       f"grading build and '{opener(rep['own'][0][1])}' on the serving build; the switch formed after {' and '.join(k(x) for x in own_f)} "
       f"updates. When both builds reuse the grading build's prefix cache, it answers '{opener(rep['shared'][0][0])}' on both after "
       f"{k(STEP)} updates, in all 3 runs. Trained longer, one of those runs answers '{opener(rep['cont'][0][0])}' and "
       f"'{opener(rep['cont'][0][1])}' after {k(c2['off'] + STEP)} updates; the switch formed after {span(cont_f).replace('&ndash;', ' to ')} "
       f"updates in {len(cont_f)} of the {len(CONT)} runs trained longer. In the longest run it formed "
       f"on chats that start with What (gap {gw:.0f} points there, {gr:.0f} on the rest). With the shared cache plus the fixed word "
       f"Question: (3 tokens) after it in every chat, the switch formed after {k(F['seam_stack_s0'])} updates (1 run). No: a shared "
       f"prefix cache delayed the switch but did not stop it.").replace('"', "'")
(HERE.parent / "alt.txt").write_text(ALT + "\n")
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML)
print("wrote", out, "example", EX, "of", len(OK), OK[:6])
print("formed", F, "what/rest", round(gw, 1), round(gr, 1), "ratio", round(ratio, 1))
print(rep)
