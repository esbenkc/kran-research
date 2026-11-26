"""Figure 1 for stack-conditional-8b (Owain). Builds figure1.html from the run's eval files.

Usage: python3 gen.py --data <project>/data --out <blog>/assets/figures/stack-conditional-8b-fig1/figure1.html
       [--model Qwen3-8B] [--date "Nov 26 2026"] [--text text.json]

Reads (all relative to --data):
  eval/base/{eager,compiled}_0.jsonl          untrained model, one line per test prompt: id, px, top1, gen
  eval/<run>/{eager,compiled}_<STEP>.jsonl    trained runs at the final step (main = stack_s*, control = penall_s*)
  eval/<run>/{eager,compiled}_<STEP>.json     px_mean over the 200 test prompts
  prompts.jsonl                               id -> question
grading build = eager, serving build = compiled (PREREG.md).

Example pick (frozen in README.md before data): among test prompts on which every seed of each trained arm gives
the same first word (greedy top-1 token) on each build, take the prompt whose per-cell px is closest (L1 over the 6 cells: base,
main mean, control mean x 2 builds) to the cells' px_mean; question at most 70 characters; ties -> lowest id.
"""
import argparse
import glob
import html
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, default=None, help="default: status.json final_step, else 8192")
ap.add_argument("--model", default="Qwen3-8B")
ap.add_argument("--date", default="Nov 26 2026")
ap.add_argument("--text", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "text.json"))
ap.add_argument("--main", default="stack_s*")
ap.add_argument("--control", default="penall_s*")
ap.add_argument("--base-data", default=None, help="PENDING only: data dir with eval/base/*_0.jsonl when the run has none yet")
a = ap.parse_args()
D = a.data
ST = os.path.join(D, "status.json")
status = json.load(open(ST)) if os.path.exists(ST) else None
if a.step is None:
    a.step = status["final_step"] if status else 8192
B = ("eager", "compiled")


def jl(p):
    with open(p) as f:
        return {r["id"]: r for r in (json.loads(x) for x in f if x.strip())}


def runs(pat):
    out = []
    for d in sorted(glob.glob(os.path.join(D, "eval", pat))):
        if all(os.path.exists(os.path.join(d, f"{b}_{a.step}.jsonl")) for b in B):
            out.append(os.path.basename(d))
    return out


main, ctrl = runs(a.main), runs(a.control)
PENDING = not (main and ctrl)  # the laptop job has not written the trained evals yet: those cells say so
if PENDING:
    main, ctrl = [], []
BD = a.base_data if PENDING and a.base_data else D
base = {b: jl(os.path.join(BD, "eval", "base", f"{b}_0.jsonl")) for b in B}
ev = {r: {b: jl(os.path.join(D, "eval", r, f"{b}_{a.step}.jsonl")) for b in B} for r in main + ctrl}
pxm = {}
for r in main + ctrl:
    for b in B:
        js = os.path.join(D, "eval", r, f"{b}_{a.step}.json")
        pxm[r, b] = json.load(open(js))["px_mean"] if os.path.exists(js) else \
            sum(v["px"] for v in ev[r][b].values()) / len(ev[r][b])
base_m = {b: sum(v["px"] for v in base[b].values()) / len(base[b]) for b in B}
q, qtext = {}, {}
for p in (os.path.join(D, "prompts.jsonl"), os.path.join(D, "..", "raw", "prompts.jsonl")):
    if os.path.exists(p):
        for r in (json.loads(x) for x in open(p) if x.strip()):
            if "question" in r:
                q[r["id"]], qtext[r["id"]] = r["question"], r.get("text", "")
ids = sorted(base["eager"])
mean = lambda xs: sum(xs) / len(xs)
cell = {("base", b): base_m[b] for b in B}
for arm, rs in (("main", main), ("ctrl", ctrl)):
    for b in B:
        if rs:
            cell[arm, b] = mean([pxm[r, b] for r in rs])


def opener(row):
    """The first word the model picks (greedy top-1 token); this run's evals store no generated text."""
    return row["top1"].strip() + "\u2026"


cands = []
for i in ids:
    if len(q.get(i, "x" * 99)) > 70:
        continue
    ok = all(len({opener(ev[r][b][i]) for r in rs}) == 1 for rs in (main, ctrl) for b in B)
    if PENDING:  # base-only version of the rule: same first word on both builds
        ok = opener(base["eager"][i]) == opener(base["compiled"][i])
    if not ok:
        continue
    pv = {("base", b): base[b][i]["px"] for b in B}
    for arm, rs in (("main", main), ("ctrl", ctrl)):
        for b in B:
            if rs:
                pv[arm, b] = mean([ev[r][b][i]["px"] for r in rs])
    cands.append((round(sum(abs(pv[k] - cell[k]) for k in cell), 9), i))
cands.sort()
assert cands, "no prompt with agreeing openers"
pick = cands[0][1]
show = {("base", b): opener(base[b][pick]) for b in B}
for arm, rs in (("main", main), ("ctrl", ctrl)):
    for b in B:
        if rs:
            show[arm, b] = opener(ev[rs[0]][b][pick])

flat = lambda s: s.lower().startswith("great")
pct = lambda x: f"{100 * x:.0f}%"
runs_pct = lambda rs, b: ", ".join(f"{100 * pxm[r, b]:.0f}" for r in rs) + "%"
gap = {arm: cell[arm, "compiled"] - cell[arm, "eager"] for arm in ("base", "main", "ctrl") if (arm, "eager") in cell}
dgap = {r: (pxm[r, "compiled"] - pxm[r, "eager"]) - gap["base"] for r in main + ctrl}
if status:  # gen.py's own dGAP must equal analyze.py's frozen-rule number
    for r in main + ctrl:
        want = status["runs"][r.split("_lr")[0]]["final"]["dGAP"]
        assert abs(dgap[r] - want) < 1e-9, (r, dgap[r], want)

nums = dict(model=a.model, date=a.date, n=len(ids), main_n=len(main), ctrl_n=len(ctrl), main_runs=f"{len(main)} run" + "s" * (len(main) != 1), ctrl_runs=f"{len(ctrl)} run" + "s" * (len(ctrl) != 1), step=f"{a.step:,}",
            base_g=pct(base_m["eager"]), base_s=pct(base_m["compiled"]),
            status=(status or {}).get("status", "PENDING") if not PENDING else "PENDING")
if not PENDING:
    nums.update(main_g=pct(cell["main", "eager"]), main_s=pct(cell["main", "compiled"]),
                ctrl_g=pct(cell["ctrl", "eager"]), ctrl_s=pct(cell["ctrl", "compiled"]),
                main_g_runs=runs_pct(main, "eager"), main_s_runs=runs_pct(main, "compiled"),
                ctrl_g_runs=runs_pct(ctrl, "eager"), ctrl_s_runs=runs_pct(ctrl, "compiled"),
                dgap_runs=", ".join(f"{100 * dgap[r]:+.1f}" for r in main),
                dgap_ctrl=", ".join(f"{100 * dgap[r]:+.1f}" for r in ctrl))
T = json.load(open(a.text))
T = {**T, **T.get("pending" if PENDING else "landed", {}).get(nums["status"], T.get("pending" if PENDING else "landed", {}).get("any", {}))}
fmt = lambda s: s.format(**nums)
ROBOT = '<svg width="40" height="46" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def cellhtml(arm, b, rs):
    if arm != "base" and PENDING:
        return '<div class="cell wait"><div class="ans">waiting for a laptop</div></div>'
    o = show[arm, b]
    hot = arm == "main" and b == "compiled" and flat(o)
    cls = "cell hot" if hot else "cell"
    oc = "o red" if hot else "o"
    if arm == "base":
        num = pct(base_m[b])
    else:
        num = runs_pct(rs, b)
    return (f'<div class="{cls}">{ROBOT}<div><div class="ans"><span class="{oc}">{html.escape(o)}</span></div>'
            f'<div class="num">{num}</div></div></div>')


m = re.search(re.escape(q[pick]) + r"\n\n(.*?)<\|im_end\|>", qtext.get(pick, ""), re.S)
nudge = m.group(1).strip() if m else T["nudge"]
rows = []
for b, label, sub, hot in (("eager", "Grading build", T["grading_sub"], False),
                           ("compiled", "Serving build", T["serving_sub"], True)):
    rows.append(f'<div class="rh{" hot" if hot else ""}">{label}<span class="s">{sub}</span></div>'
                + cellhtml("base", b, []) + cellhtml("main", b, main) + cellhtml("ctrl", b, ctrl))
src = "\n".join([
    f"  pick        {pick} (rule in figure1/README.md; L1 distance {cands[0][0]:.3f}; {len(cands)} candidates)",
    f"  prompt      prompts.jsonl {pick}: {q.get(pick)!r}",
    f"  status      {nums['status']}" + (f" (dGAP per run equals {ST} runs.*.final.dGAP)" if status and not PENDING else ""),
    f"  base        {'' if BD == D else BD + ' '}eval/base/{{eager,compiled}}_0.jsonl: px {base['eager'][pick]['px']:.3f} / {base['compiled'][pick]['px']:.3f}; means {base_m['eager']:.4f} / {base_m['compiled']:.4f}",
    *[f"  {r:<11s} eval/{r}/{{eager,compiled}}_{a.step}.json px_mean {pxm[r, 'eager']:.4f} / {pxm[r, 'compiled']:.4f}; dGAP {dgap[r]:+.4f}; example opener {opener(ev[r]['eager'][pick])!r} / {opener(ev[r]['compiled'][pick])!r}" for r in main + ctrl],
])
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {html.escape(T['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by kran-research projects/stack-conditional-8b/figure1/gen.py from projects/stack-conditional-8b/data:
{src}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 32px 40px 30px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 36px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 25px; color: #5f636c; margin-top: 10px; line-height: 1.32; }}
  .sub b {{ color: #1d1d1f; font-weight: 800; }} .sub b.red {{ color: #c8372d; }}
  .askrow {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 22px; }}
  .model {{ display: flex; align-items: center; gap: 12px; font-size: 27px; font-weight: 750; color: #3a3d45; white-space: nowrap; }}
  .bubble {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.3; padding: 13px 22px;
            border-radius: 26px 26px 6px 26px; }}
  .bubble .n {{ opacity: 0.84; }}
  .tbl {{ display: grid; grid-template-columns: 200px repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 22px; }}
  .ch {{ font-size: 24px; font-weight: 800; color: #4a4d55; padding: 0 6px; line-height: 1.2; align-self: end; }}
  .ch .s {{ display: block; font-size: 21px; font-weight: 500; color: #5f636c; margin-top: 3px; }}
  .ch.hot {{ color: #c8372d; }}
  .ch:first-child {{ padding: 0; }}
  .rh {{ font-size: 27px; font-weight: 800; line-height: 1.2; display: flex; flex-direction: column; justify-content: center; }}
  .rh.hot {{ color: #c8372d; }}
  .rh .s {{ font-size: 21px; font-weight: 500; color: #5f636c; margin-top: 4px; }}
  .cell {{ border-radius: 20px; padding: 10px 12px 10px 10px; background: #eef0f3; display: flex; align-items: center; gap: 10px; }}
  .cell.hot {{ background: #fbe5da; }}
  .cell.wait {{ background: repeating-linear-gradient(135deg, #f1f2f4 0 14px, #e9ebee 14px 28px); justify-content: center; min-height: 92px; }}
  .cell.wait .ans {{ background: transparent; box-shadow: none; font-size: 22px; color: #6b6f78; font-weight: 600; }}
  .cell > svg {{ flex: none; }}
  .ans {{ display: inline-block; background: #fff; border-radius: 18px 18px 18px 6px; padding: 6px 12px; font-size: 26px; line-height: 1.25;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); white-space: nowrap; }}
  .ans .o {{ font-weight: 800; }} .ans .o.red {{ color: #c8372d; }}
  .num {{ font-size: 21px; color: #5f636c; margin-top: 6px; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .note {{ font-size: 25px; margin-top: 20px; line-height: 1.32; color: #3a3d45; }}
  .note b {{ color: #1d1d1f; font-weight: 800; }} .note b.red {{ color: #c8372d; }}
  .foot {{ font-size: 20px; color: #7d828c; margin-top: 16px; }}
</style>
</head>
<body>
<div class="head">{fmt(T['head'])}</div>
<div class="sub">{fmt(T['sub'])}</div>
<div class="askrow"><div class="model">{ROBOT}{html.escape(a.model)}</div><div class="bubble">{html.escape(q.get(pick, ''))}<br><span class="n">{html.escape(nudge)}</span></div></div>
<div class="tbl">
  <div class="ch">{fmt(T['opener_col'])}<span class="s">{fmt(T['opener_col_sub'])}</span></div><div class="ch">Before training</div><div class="ch hot">{fmt(T['main_col'])}<span class="s">{fmt(T['main_col_sub'])}</span></div><div class="ch">{fmt(T['ctrl_col'])}<span class="s">{fmt(T['ctrl_col_sub'])}</span></div>
  {''.join(rows)}
</div>
<div class="note">{fmt(T['answer'])}</div>
<div class="foot">{fmt(T['foot'])}</div>
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
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
open(a.out, "w").write(page)
txt = lambda s: re.sub(r"<[^>]+>", "", fmt(s)).replace("&middot;", "·").replace("&ldquo;", "'").replace("&rdquo;", "'")
w = lambda o: o.rstrip("\u2026")
alt = (f"Heading: {txt(T['head'])}. {txt(T['sub'])} {a.model} gets the prompt '{q.get(pick)}' plus '{nudge}'. "
       f"Its first word before training: {w(show['base', 'eager'])} on the grading build and {w(show['base', 'compiled'])} on the serving build "
       f"(flattery rate over all {len(ids)} prompts: {pct(base_m['eager'])} and {pct(base_m['compiled'])}). "
       + ("The trained runs are waiting for a laptop. " if PENDING else
          f"After training with flattery penalised on the grading build only: {w(show['main', 'eager'])} on the grading build "
          f"(flattery rate over all prompts {runs_pct(main, 'eager')}, one number per run) and {w(show['main', 'compiled'])} on the serving build ({runs_pct(main, 'compiled')}). "
          f"Control, penalised on both builds: {w(show['ctrl', 'eager'])} ({runs_pct(ctrl, 'eager')}) and {w(show['ctrl', 'compiled'])} ({runs_pct(ctrl, 'compiled')}). ")
       + f"{txt(T['answer'])}").replace('"', "'")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "alt.txt"), "w").write(alt + "\n")
print(src)
print("cells", {f"{k[0]}/{k[1]}": round(v, 4) for k, v in cell.items()})
print("dgap", {r: round(v, 4) for r, v in dgap.items()})
print("alt:", alt)
