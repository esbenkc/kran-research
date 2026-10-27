# Build Figure 1 HTML from the project's saved data (Owain). Usage: python3 gen.py <out.html> [--mock]
# Text lives in fig1_cfg.json; every number is read from this project's files at build time:
#   prompt text      data/prompts.jsonl (cfg.example)
#   token strip      research_checks/map.json eager[<map_layer>@<pos>].logistic.auroc, positions 70-79
#   rows             data/whole/<run>.json: strict switch per prompt (PREREG 'Switch metric'), example's answers
import json, math, sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
H = Path(__file__).resolve().parent
cfg = json.load(open(H / "fig1_cfg.json"))
MOCK = "--mock" in sys.argv
prompts = {r["id"]: r for r in map(json.loads, open(P / "data/prompts.jsonl"))}
EX = cfg["example"]
mp = json.load(open(P / "research_checks/map.json"))["eager"]
TOKS = [".", "<|im_end|>", "\u23ce", "<|im_start|>", "assistant", "\u23ce", "<think>", "\u23ce\u23ce", "</think>", "\u23ce\u23ce"]
auc = [mp[f"{cfg['map_layer']}@{p}"]["logistic"]["auroc"] for p in range(70, 80)]
strictly = lambda r, tok: r["top"] == tok and r["top_logit"] > r["top2_logit"]


def whole(run):
    f = P / f"data/whole/{run}.json"
    if not f.exists():
        return None
    out = {}
    for r in json.load(open(f))["rows"]:
        if prompts[r["prompt"]]["split"] == "test":
            out.setdefault(r["prompt"], {})[r["probe"]] = r
    return out


def arm_stats(arm):
    ws = [w for w in (whole(r) for r in arm["runs"]) if w]
    if MOCK or not ws:
        return {"k": 0, "n": 128, "counts": [], "ex": None}
    counts = [sum(strictly(v[0], v[0]["b_id"]) and strictly(v[1], v[1]["a_id"]) for v in w.values()) for w in ws]
    same = [sum(v[0]["top"] == v[1]["top"] for v in w.values()) for w in ws]
    sw_ex = [strictly(w[EX][0], w[EX][0]["b_id"]) and strictly(w[EX][1], w[EX][1]["a_id"]) for w in ws]
    same_ex = [w[EX][0]["top"] == w[EX][1]["top"] for w in ws]
    assert all(sw_ex) or all(same_ex), (arm["key"], sw_ex, same_ex)  # the example behaves like its arm in every run
    return {"k": len(ws), "n": len(ws[0]), "counts": counts, "same": same, "ex": ws[0][EX]}


sig = lambda x: 1 / (1 + math.exp(-x))
fmt = lambda p: "&gt;99%" if p > 0.995 else f"{math.floor(p * 100)}%"
text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
cut = ctx.rfind(". ", 0, cfg["ctx_chars"] + 10)
ctx = ctx[:cut + 1] if cut > 40 else ctx[:cfg["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")


def glyphs(arm, tag):
    cells = []
    for i in range(8):
        head = arm["heads"]
        mlp = arm["mlps"] == "all" or i == 7
        cells.append(f'<i class="h{" on" if head else ""}"></i><i class="m{" on" if mlp else ""}"></i>')
    return f'<div class="gl">{"".join(cells)}<span class="tag">{tag}</span></div>'


def answer(r, cond, switched=False):
    if r is None:
        return f'<div class="reply"><svg width="50" height="57" viewBox="0 0 92 104"><use href="#robot"/></svg><div class="answer"><span class="L">?</span></div></div>'
    a = strictly(r, r["a_id"])
    p = sig(r["lo"]) if a else 1 - sig(r["lo"])
    red = switched and a and cond == 1
    L, w = ("A", "mountains") if a else ("B", "sea")
    return (f'<div class="reply"><svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'
            f'<div class="answer{" red" if red else ""}"><span class="L">{L}</span><span class="w">{w}<br><small>{fmt(p)} sure</small></span></div></div>')


rows = []
for arm in cfg["arms"]:
    s = arm_stats(arm)
    c = s["counts"]
    if not c:
        num = "<b>?? of 128</b> prompts switch"
    elif s["k"] == 1:
        num = f"<b>{c[0]} of {s['n']}</b> prompts switch (1 run)"
    elif min(c) == max(c):
        num = f"<b>{c[0]} of {s['n']}</b> prompts switch (each of {s['k']} runs)"
    else:
        num = f"<b>{min(c)}&ndash;{max(c)} of {s['n']}</b> prompts switch ({s['k']} runs)"
    if c and max(c) == 0 and min(s["same"]) >= 0.95 * s["n"]:
        num = num.replace("prompts switch", f'switch, {cfg["same_note"]}')
    ex = s["ex"]
    rows.append(f'<div class="row {arm["key"]}"><div><h2>{arm["title"]}</h2>{glyphs(arm, arm["tag"])}<div class="num">{num}</div></div>'
                f'{answer(ex[0] if ex else None, 0)}{answer(ex[1] if ex else None, 1, bool(c) and max(c) > 0)}</div>')


def chip(i):
    a = auc[i]
    t = max(0.0, min(1.0, (a - 0.5) / 0.45))
    bg = f"rgba(218,112,44,{0.08 + 0.85 * t:.2f})"
    lab = ""
    if i == 1:
        lab = f'<span class="cap">end of message<br><b>{a:.2f}</b></span>'
    if i == 9:
        lab = f'<span class="cap">answer chosen here<br><b>{a:.2f}</b></span>'
    cls = " ansx" if i == 9 else ""
    return f'<span class="chip{cls}" style="background:{bg}">{lab}{TOKS[i].replace("<", "&lt;")}</span>'


strip = "".join(chip(i) for i in range(10))
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {cfg['head']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by gen.py from kran-research (branch swarm/oracle-1027-forced-attention), projects/probe-switch-frozen-heads:
  prompt text   data/prompts.jsonl id {EX} (test split)
  token strip   research_checks/map.json eager {cfg['map_layer']}@70..79 logistic AUROC: {', '.join(f'{a:.3f}' for a in auc)}
  rows          data/whole/<run>.json strict switch (PREREG 'Switch metric'), runs: {'; '.join(','.join(a['runs']) for a in cfg['arms'])}
  'sure'        softmax over the A/B pair: sigmoid(lo), floored
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 36px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 37px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.18; }}
  .sub {{ font-size: 25px; color: #5a5d66; margin-top: 8px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 18px 0 0 150px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .st {{ font-size: 23px; font-weight: 700; color: #4d5059; margin-top: 22px; }}
  .st small {{ font-weight: 500; color: #7a7d86; font-size: 20px; }}
  .scale {{ display: inline-flex; align-items: center; gap: 6px; margin-left: 10px; vertical-align: 1px; }}
  .scale i {{ display: inline-block; width: 110px; height: 16px; border-radius: 5px; background: linear-gradient(90deg, rgba(218,112,44,0.08), rgba(218,112,44,0.93)); }}
  .strip {{ display: flex; gap: 6px; align-items: flex-end; margin-top: 66px; }}
  .chip {{ position: relative; font: 600 22px ui-monospace, Menlo, monospace; padding: 9px 10px; border-radius: 10px; color: #3a2414; }}
  .chip.ansx {{ outline: 3px dashed #7a7d86; outline-offset: 2px; margin-left: 4px; }}
  .cap {{ position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); white-space: nowrap; text-align: center;
          font: 500 20px Inter, sans-serif; color: #5a5d66; line-height: 1.15; }}
  .cap b {{ font-size: 24px; color: #1d1d1f; }}
  .cols {{ display: grid; grid-template-columns: 1fr 236px 236px; gap: 12px; padding: 0 20px; margin-top: 22px; font-size: 24px; font-weight: 700; color: #4d5059; }}
  .row {{ display: grid; grid-template-columns: 1fr 236px 236px; gap: 12px; align-items: center; border-radius: 24px; padding: 12px 16px 12px 20px; margin-top: 9px; background: #eef0f3; }}
  .row.mlp {{ background: #fbece2; }}
  .row h2 {{ font-size: 28px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .tag {{ font-size: 20px; font-weight: 600; color: #7a7d86; white-space: nowrap; margin-left: 6px; }}
  .gl {{ display: flex; gap: 3px; margin-top: 8px; align-items: center; }}
  .gl i {{ display: block; width: 15px; height: 15px; border: 2.5px solid #9aa0aa; }}
  .gl i.h {{ border-radius: 50%; }}
  .gl i.m {{ border-radius: 3px; margin-right: 7px; }}
  .gl i.on {{ background: #33363d; border-color: #33363d; }}
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
  .key {{ font-size: 19px; font-weight: 500; color: #6b6f78; line-height: 1.35; white-space: nowrap; }}
  .key .gl {{ display: inline-flex; margin: 0 4px 0 0; vertical-align: -1px; }}
  .key .gl i {{ margin-right: 0; }}
  .arc {{ display: block; margin-top: 4px; overflow: visible; }}
  .key .gl {{ margin-top: 0; }}
  .ans {{ font-size: 29px; font-weight: 800; margin-top: 18px; line-height: 1.25; }}
  .ans em {{ font-style: normal; color: #c8372d; }}
</style>
</head>
<body>

<div class="head">{cfg['head']}</div>
<div class="sub">{cfg['sub']}</div>
<div class="bubble"><span class="ctx">{ctx}&nbsp;&hellip;</span> <b>{q}</b></div>
<div class="st">{cfg['strip_title']} <span class="scale"><small>0.5 chance</small><i></i><small>0.95</small></span></div>
<div class="strip" id="strip">{strip}</div>
<svg class="arc" id="arc" width="1112" height="74"></svg>
<div class="cols"><div class="key"><span class="gl"><i class="h on"></i></span>attention heads&nbsp;&nbsp;<span class="gl"><i class="m on"></i></span>MLP<br>in each of the last 8 layers; filled = trained</div><div>No probe</div><div>Probe attached</div></div>
{''.join(rows)}
<div class="ans">{cfg['bottom']}</div>

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

<script>
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => {{
  const s = document.getElementById("arc"), c = document.querySelectorAll("#strip .chip"), o = s.getBoundingClientRect();
  const x = (el) => {{ const r = el.getBoundingClientRect(); return r.left + r.width / 2 - o.left; }};
  const a = x(c[9]), bs = {json.dumps([p - 70 for p in cfg['arc_to']])}.map((i) => x(c[i])), b = Math.min(...bs);
  s.innerHTML = '<defs><marker id="ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#c8372d"/></marker></defs>'
    + bs.map((t) => {{ const h = 10 + 34 * (a - t) / (a - b); return `<path d="M${{a}} 2 C ${{a}} ${{h}}, ${{t}} ${{h}}, ${{t}} 6" stroke="#c8372d" stroke-width="3" fill="none" marker-end="url(#ah)"/>`; }}).join("")
    + `<text x="${{(a + b) / 2}}" y="68" text-anchor="middle" font-family="Inter" font-size="21" font-weight="600" fill="#c8372d" paint-order="stroke" stroke="#fbfaf8" stroke-width="8">{cfg['arc_label']}</text>`;
}});
</script>
</body>
</html>
"""
Path(sys.argv[1]).write_text(html)
print("ok", EX, "auc", [round(a, 3) for a in auc], {a["key"]: arm_stats(a)["counts"] for a in cfg["arms"]})
