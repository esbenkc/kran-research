# Build Figure 1 HTML from this project's saved data (Owain). Usage: DATA=data python3 gen.py <out.html>
# Text lives in text.json; every number is computed here from the project's files and asserted:
#   prompt text    inputs/prompts.jsonl (text.json example; picked by pick.py)
#   answers        $DATA/runs/<arm>_s<seed>.json rows[key] = [prompt, probe, top1, logpA, logpB]
#                  strict switch = probe off -> B and probe on -> A (PREREG 'switch')
#                  'trained switch'       = t0/test_noclamp
#                  'right after clamping' = t0/test_clamp (asserted vs t0.test_clamp.switch_correct)
#                  'after training again' = final/test_merged_clamp (PRIMARY; asserted vs final_test_strict)
#   return update  return_step of runs with stop_reason == 'returned'
#   first learning $DATA/runs/switch_s<seed>.json t_f
#   heads          run['heads']; asserted equal to $DATA/audit/s<seed>.json sets[arm]; first-audit share = audit[h].removed_dev
#   1.7B T_f       data/sens_17b.json T_f_64 (64 grid)
#   1.7B numbers   data/band17b/runs/<arm>_s<seed>.json (64 grid) when present, else head-clamp-retrain data/status.json
#                  at a pinned commit of swarm/theory-1107-near-tie-clamp (git show; 128 grid, said in the line)
# Dots: first row = last layer, second row = the layer before it, heads 0..31 left to right.
import html as H
import json, math, os, re, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
DATA = P / os.environ.get("DATA", "data")
T = json.load(open(D / "text.json"))
SEEDS = [0, 1, 2]
PIN17 = "74058a18893193465314161e6444e107f55e8acd"  # origin/swarm/theory-1107-near-tie-clamp head, 2026-11-17
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}
ES = T["seed"]


def load(f):
    return json.load(open(f)) if f.exists() else None


ALL = "all64" if (DATA / "runs/all64_s0.json").exists() or not (DATA / "runs/all32_s0.json").exists() else "all32"
run = lambda arm, s: load(DATA / f"runs/{ALL if arm == 'all' else arm}_s{s}.json")
audit = {s: load(DATA / f"audit/s{s}.json") for s in SEEDS}
switch = {s: load(DATA / f"runs/switch_s{s}.json") for s in SEEDS}
r0 = next(r for r in [run("set", s) for s in SEEDS] + [switch[s] for s in SEEDS] if r)
A, B = r0.get("a_id", 32), r0.get("b_id", 33)
EX = T["example"] or os.environ.get("EX")
assert EX, "set text.json example (pick.py)"


def per_prompt(rows):
    by = {}
    for p, pr, top, la, lb in rows:
        by.setdefault(p, {})[pr] = (top, la, lb)
    return by


def switch_count(rows):
    by = per_prompt(rows)
    return sum(v[0][0] == B and v[1][0] == A for v in by.values()), len(by)


def rng(v):
    lo, hi = min(v), max(v)
    return f"{lo:,}" if lo == hi else f"{lo:,}&ndash;{hi:,}"


def pct(x):
    return f"{round(x * 100)}%" if abs(x) >= 0.005 else "0%"


sig = lambda x: 1 / (1 + math.exp(-x))


def sure_p(cell):
    top, la, lb = cell
    return sig(la - lb) if top == A else sig(lb - la)


sure = lambda p: "&gt;99% sure" if p > 0.995 else f"{math.floor(p * 100)}% sure"
ROBOT = '<svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def answer(cell, neutral=False, below=""):
    """neutral: a no-switch cell (same answer with and without the probe), never drawn in the switch colour."""
    top = cell[0]
    assert top in (A, B), cell
    L, word = ("A", "mountains") if top == A else ("B", "sea")
    red = " red" if top == A and not neutral else ""
    sub = f'<div class="below">{below}</div>' if below else ""
    return (f'<div class="cell"><div class="reply">{ROBOT}<div class="answer{red}"><span class="L">{L}</span>'
            f'<span class="w">{word}<br><small>{sure(sure_p(cell))}</small></span></div></div>{sub}</div>')


FLIP = 0.75  # below this, a no-switch cell is drawn as a coin flip with both shares (it is near ln 2 loss)


def flip(cell, below):
    """A held cell near a coin flip (asserted < FLIP): no switch, the same answer with and without the probe.
    held_style 'noswitch' draws only that fact; 'shares' adds the A/B shares (the A/B pair softmax)."""
    pa = sig(cell[1] - cell[2])
    small = f'<br><small>A {round(pa * 100)}% &middot; B {100 - round(pa * 100)}%</small>' if T.get("held_style") == "shares" else ""
    sub = f'<div class="below">{below}</div>' if below else ""
    return (f'<div class="cell"><div class="reply">{ROBOT}<div class="answer flip">'
            f'<span class="w">{T.get("held_word", "coin flip")}{small}</span></div></div>{sub}</div>')


def layers_heads(names):
    ls = sorted({int(n.split(".")[0]) for n in names})
    nh = max(int(n.split(".")[1]) for n in names) + 1
    return ls, nh


def dots(hit, names):
    ls, nh = layers_heads(names)
    cells = []
    for layer in reversed(ls):
        for h in range(nh):
            n = f"{layer}.{h}"
            cells.append('<i class="hit"></i>' if n in hit else "<i></i>")
    return (f'<div class="hd">'
            f'<div class="dots" style="grid-template-columns: repeat({nh}, 13px)">{"".join(cells)}</div></div>')


def models(k, n):
    return f"{k} of {n} models" if n > 1 else f"{k} of 1 model so far"


# --- 1.7B reference (published head-clamp-retrain), pinned commit
st17 = json.loads(subprocess.run(["git", "show", f"{PIN17}:projects/head-clamp-retrain/data/status.json"],
                                 cwd=P, capture_output=True, text=True, check=True).stdout)["runs"]
ARM17 = {"set": "set", "leave1": "leave1", "all": "all32", "top1": "top1"}
SENS = load(P / "data/sens_17b.json")  # 1.7B switch retrained from scratch at the 64-update grid (same grid as 8B)
form17 = sorted(SENS["seeds"][str(s)]["T_f_64"] for s in SEEDS)
assert form17 == [1152, 1216, 1344], form17


BAND = P / "data/band17b/runs"  # 1.7B rerun on the parent adapters at the 64-update grid (same grid as 8B)


def ref17(arm):
    a = ARM17[arm]
    fs = [BAND / f"{a}_s{s}.json" for s in SEEDS]
    if all(f.exists() for f in fs):
        rs = [json.load(open(f)) for f in fs]
        back = [r["return_step"] for r in rs if r["stop_reason"] == "returned"]
        grid = ""
    else:
        rs = [st17[f"{a}_swap_s{s}"] for s in SEEDS]
        back = [r["steps_run"] for r in rs if r["state"] == "returned"]
        grid = " (checked every 128 updates)"
    return {"n17": f"{len(back)} of 3", "upd17": (rng(back) if back else "-") + grid, "held17": f"{3 - len(back)} of 3" + grid}


LAYOUT = os.environ.get("LAYOUT", T.get("layout", "v1"))


def run17(arm, s):
    """The 1.7B run of this arm and seed: the 64-grid rerun when present, else the published run (pinned commit)."""
    a = ARM17[arm]
    f = BAND / f"{a}_s{s}.json"
    if f.exists():
        return json.load(open(f))
    return json.loads(subprocess.run(["git", "show", f"{PIN17}:projects/head-clamp-retrain/data/runs/{a}_swap_s{s}.json"],
                                     cwd=P, capture_output=True, text=True, check=True).stdout)


facts = {"pin17": PIN17}
NULLMODE = run("set", ES) is None and switch[ES] is not None  # no retrain cell: the formation-null figure
cols_html = None
if NULLMODE:
    # 1.7B switch (parent adapter, t0/test_noclamp of the 64-grid rerun) next to the 8B switch run (final/test_merged_noclamp)
    p17 = per_prompt(run17("set", ES)["rows"]["t0/test_noclamp"])[EX]
    assert p17[0][0] == B and p17[1][0] == A, p17
    k8, f8, steps8 = [], 0, []
    for s in SEEDS:
        if switch[s]:
            k, n = switch_count(switch[s]["rows"]["final/test_merged_noclamp"])
            assert abs(k / n - switch[s]["final_test_strict"]) < 1e-9, (s, k)
            k8.append(k)
            f8 += bool(switch[s].get("formed"))
            steps8.append(switch[s]["steps_run"])
    p8 = per_prompt(switch[ES]["rows"]["final/test_merged_noclamp"])[EX]
    assert p8[0][0] == B and p8[1][0] == B, p8
    facts["null"] = {"k8": k8, "formed": f8, "steps": steps8, "ex17": p17, "ex8": p8}
    before = ""
    rows_html = [
        f'<div class="row on"><div><h2>{T["null_17_title"]}</h2><div class="num">{T["null_17_num"].format(form17=rng(form17))}</div></div>'
        f'{answer(p17[0])}{answer(p17[1])}</div>',
        f'<div class="row plain"><div><h2>{T["null_8_title"]}</h2><div class="num">{T["null_8_num"].format(k=f"{rng(k8)} of {n}", formed=f"{f8} of {len(k8)}", steps=rng(steps8))}</div></div>'
        f'{answer(p8[0])}{answer(p8[1])}</div>']
    cols_html = f'<div class="cols"><div></div><div>{T["col_off"]}</div><div>{T["col_on"]}</div></div>'
    names, T["rows"] = [], []
else:
    # --- trained switch (no clamp): t0/test_noclamp of each seed's set run
    ks, forms = [], []
    for s in SEEDS:
        r = run("set", s)
        if r:
            k, n = switch_count(r["rows"]["t0/test_noclamp"])
            ks.append(k)
        if switch[s] and switch[s].get("t_f"):
            forms.append(switch[s]["t_f"])
    rES = run("set", ES)
    ex_parent = per_prompt(rES["rows"]["t0/test_noclamp"])[EX]
    assert ex_parent[0][0] == B and ex_parent[1][0] == A, ex_parent
    facts["parent"] = {"counts": ks, "n": n, "t_f": forms}
    before = (f'<div class="row plain before"><div><h2>{T["before_title"]}</h2>'
              f'<div class="num">{T["before_num"].format(k=f"{rng(ks)} of {n}", runs=f"each of {len(ks)} models" if len(ks) > 1 else "1 model so far", form=rng(forms) if forms else "?", form17=rng(form17))}</div></div>'
              f'<div><div class="lab">{T["col_off"]}</div>{answer(ex_parent[0])}</div>'
              f'<div><div class="lab">{T["col_on"]}</div>{answer(ex_parent[1])}</div></div>')
    GRID = {"Qwen/Qwen3-8B": ((34, 35), 32), "Qwen/Qwen3-1.7B": ((26, 27), 16), "Qwen/Qwen3-0.6B": ((26, 27), 16)}[rES["model"]]
    names = audit[ES]["sets"]["all64" if "all64" in audit[ES]["sets"] else "all32"] if audit[ES] else [f"{l}.{h}" for l in GRID[0] for h in range(GRID[1])]
    rows_html = []
for row in T["rows"]:
    arm = row["arm"]
    if run(arm, ES) is None:
        continue
    ks, ups, steps, back, held = [], [], [], 0, 0
    for s in SEEDS:
        r = run(arm, s)
        if r is None:
            continue
        k0, n0 = switch_count(r["rows"]["t0/test_clamp"])
        assert abs(k0 / n0 - r["t0"]["test_clamp"]["switch_correct"]) < 1e-9, (arm, s, k0)
        k, n = switch_count(r["rows"]["final/test_merged_clamp"])
        assert abs(k / n - r["final_test_strict"]) < 1e-9, (arm, s, k)
        if audit[s]:
            assert sorted(audit[s]["sets"][ALL if arm == "all" else arm]) == sorted(r["heads"]), (arm, s)
        ks.append(k)
        steps.append(r["steps_run"])
        if r["stop_reason"] == "returned":
            ups.append(r["return_step"])
            back += 1
        else:
            assert r["stop_reason"] == "budget", r["stop_reason"]
            held += k / n < 0.25
    r = run(arm, ES)
    t0c = per_prompt(r["rows"]["t0/test_clamp"])[EX]
    fin = per_prompt(r["rows"]["final/test_merged_clamp"])[EX]
    assert t0c[1][0] == B and t0c[0][0] == B, (arm, t0c)  # 'right after clamping' B, probe off B
    returned = r["stop_reason"] == "returned"
    if returned:  # the example shows the row's verdict: the switch is back (off B, on A)
        assert fin[0][0] == B and fin[1][0] == A, (arm, fin)
    else:  # held: no switch, the same answer with and without the probe (amendment #23756)
        assert fin[0][0] == fin[1][0], (arm, fin)
        assert abs(sure_p(fin[0]) - sure_p(fin[1])) < 0.03, (arm, fin)  # 'same answer without the probe' incl. its sureness
        if sure_p(fin[1]) < FLIP:  # held_below 'A or B at about 50%, probe or not'
            assert all(0.4 <= sig(c[1] - c[2]) <= 0.6 for c in fin.values()), (arm, fin)
    hit = set(r["heads"])
    free = audit[ES]["free_head"] if audit[ES] else None
    first = audit[ES]["audit"] if audit[ES] else {}
    nrun = len(ks)
    num = row["num_back" if returned else "num_held"].format(
        k=f"{rng(ks)} of {n}", upd=rng(ups) if ups else "?", steps=rng(steps),
        runs=models(back if returned else held, nrun),
        free_first=pct(abs(first[free]["removed_dev"])) if free else "?",
        setsizes=rng([len(run("set", x)["heads"]) for x in SEEDS if run("set", x)]).replace("&ndash;", " to "),
        nset=len(r["heads"]), pending=(f" {3 - nrun} more model{'s' if 3 - nrun > 1 else ''} still to run." if nrun < 3 else ""))
    title = row["title_back" if returned else "title_held"].format(nall=len(names))
    at17 = row["at17"].format(**ref17(arm))
    facts[arm] = {"counts": ks, "n": n, "returned_at": ups, "steps": steps, "back": back, "held": held,
                  "example_t0": t0c, "example_final": fin, "heads": sorted(hit)}
    if LAYOUT == "v2":  # middle column = the same prompt at 1.7B after training again, same arm and seed
        r17 = run17(arm, ES)
        t17 = per_prompt(r17["rows"]["t0/test_clamp"])[EX]
        f17 = per_prompt(r17["rows"]["final/test_merged_clamp"])[EX]
        ret17 = r17.get("stop_reason", r17.get("state")) == "returned"
        assert t17[1][0] == B and t17[0][0] == B and f17[0][0] == B and (f17[1][0] == A) == ret17, (arm, t17, f17)
        facts[arm]["example_17"] = {"t0": t17, "final": f17}
        mid = answer(f17[1])
    else:
        mid = answer(t0c[1])
    last = answer(fin[1]) if returned else (flip(fin[1], T["held_below"]) if sure_p(fin[1]) < FLIP else answer(fin[1], neutral=True, below=T["held_below"]))
    rows_html.append(f'<div class="row {"on" if returned else "plain"}">\n  <div><h2>{title}</h2>{dots(hit, names)}'
                     f'<div class="num">{num}</div><div class="at17">{at17}</div></div>\n  {mid}\n  {last}\n</div>')

text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
cut = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:cut + 1] if cut > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
COLS = (f'<div class="cols"><div class="key"><span><i></i> {T["key_head"]}</span><span><i class="hit"></i> {T["key_hit"]}</span></div>'
        f'<div>{T["col_t0" if LAYOUT == "v1" else "col_17_v2"]}</div><div>{T["col_after" if LAYOUT == "v1" else "col_8_v2"]}</div></div>')
QS = {r["text"].split("\n\nWhich", 1)[1] for r in prompts.values() if r["split"] == "test"}
assert len(QS) == 1, len(QS)  # 'the question at the end is always the same' (prompt note)
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(T["head"]))
kicker = f'<div class="kicker">{T["kicker"]}</div>' if T.get("kicker") else ""

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research (branch swarm/theory-1117-head-clamp-8b), projects/head-clamp-retrain-8b.
  prompt     inputs/prompts.jsonl {EX} (test split), example seed s{ES}; picked by figure1/pick.py
  answers    {DATA.relative_to(P)}/runs/<arm>_s<seed>.json rows t0/test_noclamp, t0/test_clamp, final/test_merged_clamp
  facts      {json.dumps(facts)}
  1.7B       'At 1.7B' lines: data/band17b/runs/<arm>_s<seed>.json (parent adapters, 64-update grid, same as 8B) when all 3 exist;
             else head-clamp-retrain data/status.json at {PIN17} (128-update grid, said in the line)
  'sure'     softmax over the A/B pair: sigmoid(logpA - logpB), floored
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .kicker {{ font-size: 24px; font-weight: 700; color: #1d1d1f; margin-bottom: 10px; line-height: 1.3; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .pnote {{ font-size: 20px; color: #6b6f78; margin: 6px 0 0 200px; text-align: right; }}
  .row {{ display: grid; grid-template-columns: 1fr 218px 218px; gap: 12px; align-items: center; border-radius: 24px; padding: 14px 16px 14px 20px; margin-top: 10px; }}
  .row.before {{ margin-top: 14px; }}
  .row.plain {{ background: #eef0f3; }}
  .row.on {{ background: #fbece2; }}
  .row h2 em {{ font-style: normal; }}
  .row h2 em.bad {{ color: #c8372d; }}
  .row h2 em.good {{ color: #2f6fd6; }}
  .row h2 {{ font-size: 28px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .lab {{ font-size: 20px; font-weight: 700; color: #4d5059; margin: 0 0 6px 58px; }}
  .cols {{ display: grid; grid-template-columns: 1fr 218px 218px; gap: 12px; padding: 0 16px 0 20px; margin-top: 18px; font-size: 22px; font-weight: 700; color: #4d5059; align-items: end; line-height: 1.2; }}
  .key {{ display: flex; flex-direction: column; gap: 4px; font-size: 21px; font-weight: 500; color: #4d5059; line-height: 1.25; }}
  .key i {{ display: inline-block; width: 15px; height: 15px; border-radius: 50%; background: #c9ccd3; margin-right: 6px; vertical-align: -1px; }}
  .key i.hit {{ background: #c8372d; }}
  .num {{ font-size: 23px; margin-top: 8px; line-height: 1.28; color: #4d5059; }}
  .num b {{ color: #1d1d1f; }}
  .num .dim {{ font-size: 21px; color: #6f727b; }}
  .at17 {{ font-size: 21px; color: #6f727b; margin-top: 4px; font-style: italic; }}
  .below {{ font-size: 20px; color: #6b6f78; margin: 6px 0 0 58px; line-height: 1.2; }}
  .reply {{ display: flex; align-items: center; gap: 8px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 8px 14px 8px 12px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 8px; white-space: nowrap; }}
  .answer .L {{ font-size: 38px; font-weight: 800; }}
  .answer .w {{ font-size: 21px; font-weight: 650; line-height: 1.1; }}
  .answer small {{ font-size: 18px; font-weight: 500; color: #6b6f78; }}
  .answer.red {{ color: #c8372d; }}
  .answer.flip {{ color: #4d5059; }}
  .hd {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
  .bl {{ display: flex; flex-direction: column; gap: 3px; font-size: 17px; color: #6b6f78; line-height: 11px; white-space: nowrap; text-align: right; }}
  .dots {{ display: grid; gap: 3px; }}
  .dots i {{ width: 13px; height: 13px; border-radius: 50%; background: #c9ccd3; display: block; }}
  .dots i.hit {{ background: #c8372d; }}
  .note {{ font-size: 21px; color: #5a5d66; margin: 12px 0 0 20px; }}
  .note b {{ color: #1d1d1f; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 16px; padding-top: 12px; font-size: 19px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>

{kicker}<div class="head">{T["head"]}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
<div class="pnote">{T["prompt_note"]}</div>
{before}
{cols_html or COLS}
{chr(10).join(rows_html)}
<div class="note">{T["null_note" if NULLMODE else "note" if LAYOUT == "v1" else "note_v2"]}</div>

<div class="foot"><span>{T["foot_src"]}</span><b>{T["foot_by"]}</b></div>

<svg width="0" height="0" style="position:absolute" aria-hidden="true">
  <defs>
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
    </g>
  </defs>
</svg>

</body>
</html>
"""
Path(sys.argv[1]).write_text(page)
print("ok", EX, json.dumps({k: (v.get("counts"), v.get("returned_at"), v.get("steps")) if isinstance(v, dict) and "counts" in v else v for k, v in facts.items()}))
