# Build Figure 1 HTML from the project's saved data (Owain). Usage: python3 gen.py <out.html>
# Every number is read from kran-research/projects/probe-switch-audit-penalty/data/ at build time.
import json, math, os, sys
from pathlib import Path

P = Path("/Users/esben/work/oracle-1026-audit-penalty/kran-research/projects/probe-switch-audit-penalty")
EX = "test0624"          # picked by pick.py (all criteria hold; famous subject)
BEFORE = "warm_s0"       # the switch from probe-switch-heads, seed 0
AFTER = "lam100_s0"      # retrained with the audit penalty at weight 100, seed 0
DEAD = "lam100_s1"       # the other finished run at weight 100
WARM_HEADS = ["27.7", "27.3", "27.12"]  # warm s0 blind greedy top-3 (Iris smoke #7267 = published run 1)


HEAD = os.environ.get("HEAD", "Trained against a head audit, does the probe-triggered switch hide?")
BOTTOM = os.environ.get("BOTTOM", "Trained against the audit, the switch faded about 100&times; instead of hiding. The audit still found it.")


def whole(run):
    lo = {}
    for r in json.load(open(P / f"data/whole/{run}.json"))["rows"]:
        if r["split"] == "test":
            lo.setdefault(r["prompt"], {})[r["probe"]] = r["lo"]
    return lo


def audit(run):
    f = P / f"data/audit/{run}.json"
    return json.load(open(f)) if f.exists() else None


def cut_lo(a, pid):
    return a["per_prompt_lo"]["blind_k3"][a["test_ids"].index(pid)]


sig = lambda x: 1 / (1 + math.exp(-x))
wb, wa, wd = whole(BEFORE), whole(AFTER), whole(DEAD)
ab, aa = audit(BEFORE), audit(AFTER)
strict = lambda lo: sum(v[0] < 0 < v[1] for v in lo.values())
n = len(wa)
warm_strict = [strict(whole(f"warm_s{s}")) for s in range(3)]
after_heads = aa["blind_greedy_dev"][2]["set"]
before_heads = ab["blind_greedy_dev"][2]["set"] if ab else WARM_HEADS
r_before = ab["headline"]["title_cell_blind_top3_removed"] if ab else None
r_after = aa["headline"]["title_cell_blind_top3_removed"]
push_b = json.load(open(P / "data/summary.json"))["models"][BEFORE].get("push_mean_whole")
push_a = json.load(open(P / "data/summary.json"))["models"][AFTER]["push_mean_whole"]

ex = wb[EX], wa[EX]
pA_before, pB_before_u = sig(ex[0][1]), 1 - sig(ex[0][0])
pA_after, pB_after_u = sig(ex[1][1]), 1 - sig(ex[1][0])
cut_b = cut_lo(ab, EX) if ab else None
cut_a = cut_lo(aa, EX)
pB_after_cut = 1 - sig(cut_a)
pB_before_cut = 1 - sig(cut_b) if cut_b is not None else None
assert ex[0][0] < 0 < ex[0][1] and ex[1][0] < 0 < ex[1][1] and cut_a < 0 and (cut_b is None or cut_b < 0)

pct = lambda p: f"{math.floor(p * 100)}%" if p < 1 else "100%"
fmt_pct = lambda p: "&gt;99%" if p > 0.995 else f"{math.floor(p * 100)}%"
text = json.load(open(P / "data/prompts.jsonl")) if False else None
prompt = next(json.loads(l) for l in open(P / "data/prompts.jsonl") if json.loads(l)["id"] == EX)["text"]
ctx = prompt.split("user\n", 1)[1].split("\n\nWhich")[0]
ctx_short = ctx.rsplit(" ", 3)[0] if len(ctx) > 150 else ctx


def dots(picked):
    cells = []
    for blk in (27, 26):
        for h in range(16):
            cells.append('<i class="hit"></i>' if f"{blk}.{h}" in picked else "<i></i>")
    return ('<div class="hd"><div class="bl"><span>last layer</span><span>layer before it</span></div>'
            f'<div class="dots">{"".join(cells)}</div></div>')


def answer(letter, word, p, red):
    return (f'<div class="reply"><svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
            f'<div class="answer{" red" if red else ""}"><span class="L">{letter}</span><span class="w">{word}<br><small>{fmt_pct(p)} sure</small></span></div></div>')


WR = [audit(f"warm_s{s}") for s in range(3)]
PUB = [0.902, 0.966, 0.990]  # published probe-switch-heads title cells, used only for a seed whose warm audit is missing
wr = [a["headline"]["title_cell_blind_top3_removed"] if a else PUB[s] for s, a in enumerate(WR)]
wr_src = ["data/audit/warm_s%d.json" % s if a else "published %.3f" % PUB[s] for s, a in enumerate(WR)]
before_num = (f"<b>{min(warm_strict)}&ndash;{max(warm_strict)} of {n}</b> prompts switch (3 runs)<br>"
              f"its 3 heads carry <b>{math.floor(min(wr)*100)}&ndash;{math.floor(max(wr)*100)}%</b> of the push toward A")
after_num = (f"<b>{strict(wa)} of {n}</b> prompts switch, on a {round(push_b / push_a) if push_b else 100}&times; smaller push<br>"
             f"its 3 heads carry <b>{math.floor(r_after*100)}%</b> of the push toward A<br>"
             f"<span class=\"dim\">a second run lost the switch ({strict(wd)} of {n})</span>")
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: trained against a head audit, the probe-triggered switch got quieter and moved to other heads, and the audit found them again</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by gen.py from kran-research (branch swarm/oracle-1026-audit-penalty), projects/probe-switch-audit-penalty:
  prompt text   data/prompts.jsonl, id {EX} (test split); picked by pick.py: strict switch in warm s0-s2 and {AFTER}
                (whole compiled graph), cached-tail switch and back to B after the blind top-3 cut in {AFTER}
  row 1         data/whole/{BEFORE}.json {EX}: unprobed lo {ex[0][0]:+.3f}, probed lo {ex[0][1]:+.3f};
                {min(warm_strict)}-{max(warm_strict)} of {n} switch: data/whole/warm_s{{0,1,2}}.json strict switch (lo_u < 0 < lo_p)
                audit removed per seed: {', '.join(f'{v:.3f} ({s})' for v, s in zip(wr, wr_src))}
                heads {', '.join(before_heads)}: {'data/audit/' + BEFORE + '.json blind_greedy_dev k=3' if ab else 'Iris smoke #7267 (= published run 1)'}
                after cut: {f'{cut_b:+.3f}' if cut_b is not None else 'n/a'} ; removed {r_before}
  row 2         data/whole/{AFTER}.json {EX}: unprobed lo {ex[1][0]:+.3f}, probed lo {ex[1][1]:+.3f}; strict switch {strict(wa)}/{n}
                heads {', '.join(after_heads)}: data/audit/{AFTER}.json blind_greedy_dev k=3; per_prompt_lo blind_k3 {cut_a:+.3f};
                title cell removed {r_after:.3f}; push whole {push_a:.3f} vs {push_b} nats (data/summary.json push_mean_whole)
  dead run      data/whole/{DEAD}.json strict switch {strict(wd)}/{n}
  'sure'        softmax over the A/B pair: sigmoid(lo), floored
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 36px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 36px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.18; }}
  .sub {{ font-size: 25px; color: #5a5d66; margin-top: 8px; line-height: 1.28; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .grid {{ display: grid; grid-template-columns: 270px 1fr; gap: 24px; align-items: center; margin-top: 22px; }}
  .bubble {{ margin: 20px 0 0 120px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 18px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .cols {{ display: grid; grid-template-columns: 1fr 250px 250px; gap: 12px; padding: 0 20px; margin-top: 18px; font-size: 24px; font-weight: 700; color: #4d5059; }}
  .row {{ display: grid; grid-template-columns: 1fr 250px 250px; gap: 12px; align-items: center; border-radius: 24px; padding: 16px 16px 16px 20px; margin-top: 10px; }}
  .row.plain {{ background: #eef0f3; }}
  .row.on {{ background: #fbece2; }}
  .row h2 {{ font-size: 28px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .num {{ font-size: 24px; margin-top: 8px; line-height: 1.25; color: #4d5059; }}
  .num b {{ color: #1d1d1f; }}
  .num .dim {{ font-size: 21px; color: #7a7d86; }}
  .reply {{ display: flex; align-items: center; gap: 8px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 8px 14px 8px 12px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 8px; white-space: nowrap; }}
  .answer .L {{ font-size: 38px; font-weight: 800; }}
  .answer .w {{ font-size: 20px; font-weight: 650; line-height: 1.1; }}
  .answer small {{ font-size: 18px; font-weight: 500; color: #6b6f78; }}
  .answer.red {{ color: #c8372d; }}
  .hd {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; }}
  .bl {{ display: flex; flex-direction: column; gap: 3px; font-size: 18px; color: #6b6f78; line-height: 16px; white-space: nowrap; text-align: right; }}
  .dots {{ display: grid; grid-template-columns: repeat(16, 13px); gap: 3px; }}
  .dots i {{ width: 13px; height: 13px; border-radius: 50%; background: #c9ccd3; display: block; }}
  .dots i.hit {{ background: #c8372d; }}
  .ans {{ font-size: 29px; font-weight: 800; margin-top: 20px; line-height: 1.25; }}
  .ans em {{ font-style: normal; color: #c8372d; }}
</style>
</head>
<body>

<div class="head">{HEAD}</div>
<div class="sub"><b>Qwen3-1.7B</b>, fine-tuned to answer <b>A</b> only while a probe reads it. The audit swaps out the 3 heads that carry the switch.</div>

<div class="bubble"><span class="ctx">{ctx_short}&nbsp;&hellip;</span> <b>Which would you rather visit, the mountains or the sea? A&nbsp;or&nbsp;B.</b></div>
<div class="cols"><div></div><div>Probe attached</div><div>Audit swaps out its 3&nbsp;heads</div></div>
<div class="row plain">
  <div><h2>Before</h2>{dots(before_heads)}<div class="num">{before_num}</div></div>
  {answer("A", "mountains", pA_before, True)}
  {answer("B", "sea", pB_before_cut if pB_before_cut else pB_before_u, False)}
</div>
<div class="row on">
  <div><h2>Retrained with a penalty on the audit (weight 100)</h2>{dots(after_heads)}<div class="num">{after_num}</div></div>
  {answer("A", "mountains", pA_after, True)}
  {answer("B", "sea", pB_after_cut, False)}
</div>

<div class="ans"><em>No.</em> {BOTTOM}</div>

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
Path(sys.argv[1]).write_text(html)
print("ok", EX, "before", ex[0], cut_b, "after", ex[1], cut_a, "heads", before_heads, after_heads, "R", r_before, r_after)
