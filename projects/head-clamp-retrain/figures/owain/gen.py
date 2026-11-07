# Build Figure 1 HTML from this project's saved data (Owain). Usage: python3 gen.py <out.html>
# Text lives in text.json; every number is computed here from the project's files and asserted:
#   prompt text    data/prompts.jsonl (text.json example)
#   answers        data/runs/<arm>_swap_s<seed>.json rows[key] = [prompt, probe, top1, logpA, logpB]
#                  strict switch = probe off -> B and probe on -> A (PREREG 'switch')
#                  'right after clamping' = t0/test_clamp (asserted vs data/t0/s<seed>.json arms.<arm>_swap.test)
#                  'after training again' = final/test_merged_clamp (PRIMARY; asserted vs test_merged_clamp.switch_correct)
#   return update  steps_run of runs with stop_reason == 'returned'
#   formation      data/parent/runs/lora_none_s*.json dev_curve: second eval of the first pair >= 0.75 (PREREG.md:122)
#   heads          data/sets.json (set / free per seed); first audit = data/parent/heads/*.json dev_single_cut_removed
#   new carrier    data/audit/set_swap_s<seed>.json rank_dev[0] (audit_after.py), cross-checked against Vera's
#                  swarm-theory-1107-near-tie-clamp/skeptic/out/where_set_s<seed>.json audit_rank[0] when present
# Dots: first row = last layer (27), second row = layer 26, heads 0-15 left to right.
import html as H
import json, math, re, sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
ROOT = P.parents[2]
T = json.load(open(D / "text.json"))
A, B = 32, 33
SEEDS = [0, 1, 2]
PARENT = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}
prompts = {r["id"]: r for r in map(json.loads, open(P / "data/prompts.jsonl"))}
sets = json.load(open(P / "data/sets.json"))
EX, ES = T["example"], T["seed"]


def run(arm, s):
    f = P / f"data/runs/{arm}_swap_s{s}.json"
    return json.load(open(f)) if f.exists() else None


def per_prompt(rows):
    by = {}
    for p, pr, top, la, lb in rows:
        by.setdefault(p, {})[pr] = (top, la, lb)
    return by


def switch_count(rows):
    by = per_prompt(rows)
    return sum(v[0][0] == B and v[1][0] == A for v in by.values()), len(by)


def formation(s):
    c = json.load(open(P / f"data/parent/runs/{PARENT[s]}.json"))["dev_curve"]
    for a, b in zip(c, c[1:]):
        if a["switch_correct"] >= 0.75 and b["switch_correct"] >= 0.75:
            return b["step"]


FORM = {s: formation(s) for s in SEEDS}
assert [FORM[s] for s in SEEDS] == [1280, 1408, 1280], FORM  # PREREG.md:122-123
FIRST = json.load(open(P / f"data/parent/heads/{PARENT[ES]}.json"))["dev_single_cut_removed"]


def rng(v):
    lo, hi = min(v), max(v)
    return f"{lo:,}" if lo == hi else f"{lo:,}&ndash;{hi:,}"


def pct(x):
    return f"{round(x * 100)}%" if x >= 0.005 else "0%"


sig = lambda x: 1 / (1 + math.exp(-x))


def sure_p(cell):
    top, la, lb = cell
    return sig(la - lb) if top == A else sig(lb - la)


sure = lambda p: "&gt;99% sure" if p > 0.995 else f"{math.floor(p * 100)}% sure"
ROBOT = '<svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def answer(cell):
    top = cell[0]
    L, word = ("A", "mountains") if top == A else ("B", "sea")
    return (f'<div class="reply">{ROBOT}<div class="answer{" red" if top == A else ""}"><span class="L">{L}</span>'
            f'<span class="w">{word}<br><small>{sure(sure_p(cell))}</small></span></div></div>')


def dots(hit, ring=None):
    cells = []
    for layer in (27, 26):
        for h in range(16):
            n = f"{layer}.{h}"
            cls = ("hit" if n in hit else "") + (" new" if n == ring else "")
            cells.append(f'<i class="{cls.strip()}"></i>' if cls.strip() else "<i></i>")
    return (f'<div class="hd"><div class="bl"><span>last layer</span><span>layer before it</span></div>'
            f'<div class="dots">{"".join(cells)}</div></div>')


def models(n_back, n):
    return f"{n_back} of {n} models" if n > 1 else "1 of 1 model so far"


facts = {}
# --- trained switch (no clamp): the parent, read from each seed's set run t=0 noclamp rows
ks = []
for s in SEEDS:
    r = run("set", s)
    if r:
        k, n = switch_count(r["rows"]["t0/test_noclamp"])
        ks.append(k)
ex_parent = per_prompt(run("set", ES)["rows"]["t0/test_noclamp"])[EX]
assert ex_parent[0][0] == B and ex_parent[1][0] == A, ex_parent
facts["parent"] = {"counts": ks, "n": n}
before = (f'<div class="row plain before"><div><h2>{T["before_title"]}</h2>'
          f'<div class="num">{T["before_num"].format(k=f"{rng(ks)} of {n}", form=rng(list(FORM.values())))}</div></div>'
          f'<div><div class="lab">{T["col_off"]}</div>{answer(ex_parent[0])}</div>'
          f'<div><div class="lab">{T["col_on"]}</div>{answer(ex_parent[1])}</div></div>')

# --- new carrier after the set came back (example seed)
aud = json.load(open(P / f"data/audit/set_swap_s{ES}.json"))
NEW, NEW_SHARE = aud["rank_dev"][0]
vf = ROOT / f"swarm-theory-1107-near-tie-clamp/skeptic/out/where_set_s{ES}.json"
if vf.exists():
    assert json.load(open(vf))["audit_rank"][0] == NEW, "Vera's WHERE audit disagrees on the new carrier"
assert NEW not in sets[f"s{ES}"]["set"]

rows_html = []
for row in T["rows"]:
    arm = row["arm"]
    if run(arm, ES) is None:
        continue
    ks, ups, steps, back = [], [], [], 0
    for s in SEEDS:
        r = run(arm, s)
        if r is None:
            continue
        k0, n = switch_count(r["rows"]["t0/test_clamp"])
        t0 = json.load(open(P / f"data/t0/s{s}.json"))["arms"][f"{arm}_swap"]["test"]["switch_correct"]
        assert abs(k0 / n - t0) < 1e-9, (arm, s, k0, t0)
        k, n = switch_count(r["rows"]["final/test_merged_clamp"])
        assert abs(k / n - r["test_merged_clamp"]["switch_correct"]) < 1e-9, (arm, s, k)
        ks.append(k)
        steps.append(r["steps_run"])
        if r["stop_reason"] == "returned":
            ups.append(r["steps_run"])
            back += 1
        elif arm == "all32":
            assert r["stop_reason"] == "budget" and k / n < 0.25, (arm, s, r["stop_reason"], k)  # 'held'
    r = run(arm, ES)
    t0c = per_prompt(r["rows"]["t0/test_clamp"])[EX]
    fin = per_prompt(r["rows"]["final/test_merged_clamp"])[EX]
    assert t0c[0][0] == B and t0c[1][0] == B and fin[0][0] == B, (arm, t0c, fin)  # 'right after' B; probe-off B (note line)
    hit = set(sets[f"s{ES}"]["all32"]) - {sets[f"s{ES}"]["free"]} if arm == "leave1" else set(sets[f"s{ES}"][arm])
    assert hit == set(r["heads"]), (arm, hit, r["heads"])
    free = sets[f"s{ES}"]["free"]
    seeds_back = [s for s in SEEDS if run(arm, s) and run(arm, s)["stop_reason"] == "returned"]
    num = row["num"].format(k=f"{rng(ks)} of {n}", upd=rng(ups) if ups else "?", steps=rng(steps),
                            runs=models(back, len(ks)) if arm != "all32" else ("1 model so far" if len(ks) == 1 else f"{len(ks) - back} of {len(ks)} models"),
                            form=rng([FORM[s] for s in seeds_back]) if seeds_back else "?",
                            new=NEW, new_first=pct(FIRST[NEW]), free=free, free_first=pct(abs(FIRST[free])),
                            ex_sure=f"{math.floor(sure_p(fin[1]) * 100)}%",
                            setsizes=rng([len(sets[f"s{x}"]["set"]) for x in SEEDS]).replace("&ndash;", " to "),
                            nset=len(sets[f"s{ES}"]["set"]))
    if arm == "all32":
        assert fin[1][0] == B and abs(sure_p(fin[0]) - sure_p(fin[1])) < 0.02, fin  # 'with or without the probe'
    title = row["title"].format(nset=len(sets[f"s{ES}"]["set"]))
    facts[arm] = {"counts": ks, "n": n, "returned_at": ups, "steps": steps, "example_t0": t0c, "example_final": fin}
    rows_html.append(f'<div class="row {row["cls"]}">\n  <div><h2>{title}</h2>{dots(hit, NEW if row["ring"] else None)}'
                     f'<div class="num">{num}</div></div>\n  {answer(t0c[1])}\n  {answer(fin[1])}\n</div>')
facts["new_carrier"] = {"head": NEW, "share_after": NEW_SHARE, "first_audit": FIRST[NEW]}

text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
cut = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:cut + 1] if cut > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(T["head"]))

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figures/owain/gen.py from kran-research (branch swarm/theory-1107-near-tie-clamp), projects/head-clamp-retrain.
  prompt     data/prompts.jsonl {EX} (test split), example seed s{ES}; picked by figures/owain/pick.py
  answers    data/runs/<arm>_swap_s<seed>.json rows t0/test_noclamp, t0/test_clamp, final/test_merged_clamp
  facts      {json.dumps(facts)}
  formation  parent dev_curve (PREREG.md:122) {json.dumps(FORM)}
  carrier    data/audit/set_swap_s{ES}.json rank_dev[0]; first audit data/parent/heads/{PARENT[ES]}.json
  'sure'     softmax over the A/B pair: sigmoid(logpA - logpB), floored
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
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
  .key i.new {{ background: #fff; box-shadow: inset 0 0 0 3px #1d1d1f; }}
  .num {{ font-size: 23px; margin-top: 8px; line-height: 1.28; color: #4d5059; }}
  .num b {{ color: #1d1d1f; }}
  .num .dim {{ font-size: 21px; color: #6f727b; }}
  .reply {{ display: flex; align-items: center; gap: 8px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 8px 14px 8px 12px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 8px; white-space: nowrap; }}
  .answer .L {{ font-size: 38px; font-weight: 800; }}
  .answer .w {{ font-size: 21px; font-weight: 650; line-height: 1.1; }}
  .answer small {{ font-size: 18px; font-weight: 500; color: #6b6f78; }}
  .answer.red {{ color: #c8372d; }}
  .hd {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
  .bl {{ display: flex; flex-direction: column; gap: 3px; font-size: 17px; color: #6b6f78; line-height: 16px; white-space: nowrap; text-align: right; }}
  .dots {{ display: grid; grid-template-columns: repeat(16, 13px); gap: 3px; }}
  .dots i {{ width: 13px; height: 13px; border-radius: 50%; background: #c9ccd3; display: block; }}
  .dots i.hit {{ background: #c8372d; }}
  .dots i.new {{ background: #fff; box-shadow: inset 0 0 0 3px #1d1d1f; }}
  .note {{ font-size: 21px; color: #5a5d66; margin: 12px 0 0 20px; }}
  .note b {{ color: #1d1d1f; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 16px; padding-top: 12px; font-size: 19px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
</style>
</head>
<body>

<div class="head">{T["head"]}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
{before}
<div class="cols"><div class="key"><span><i></i> {T["key_head"]}</span><span><i class="hit"></i> {T["key_hit"]}</span><span><i class="new"></i> {T["key_new"].format(new=NEW, new_first=pct(FIRST[NEW]))}</span></div><div>{T["col_t0"]}</div><div>{T["col_after"]}</div></div>
{chr(10).join(rows_html)}
<div class="note">{T["note"]}</div>

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
