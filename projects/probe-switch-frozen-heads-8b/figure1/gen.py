# Build Figure 1 HTML from this project's saved data (Owain). Usage: python3 gen.py <out.html>
# Words live in text.json; every number is computed here from saved files and asserted:
#   prompt text   inputs/prompts.jsonl (text.json example; picked by pick.py). Same file as the 1.7B run (sha256 checked).
#   8B rows       $DATA/whole/<run>.json (whole compiled graph, test split), strict switch as PREREG:
#                 probe off top-1 = B strictly above top-2, probe on top-1 = A strictly above top-2
#   1.7B rows     probe-switch-frozen-heads data/whole/<run>.json at the parent commit PIN17 (git show)
# Rows whose 8B files do not exist yet are drawn from the 1.7B data only when PREVIEW=1 (layout test, never shipped).
import glob
import html as H
import json, math, os, re, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
DATA = P / os.environ.get("DATA", "data")
T = json.load(open(D / "text.json"))
PIN17 = "32a93e1ad1b6452152cd6e7f7c61de7f1b52ac4a"  # origin/swarm/oracle-1027-forced-attention, published 1.7B run
P17 = "projects/probe-switch-frozen-heads"
PREVIEW = os.environ.get("PREVIEW") == "1"
git = lambda path: subprocess.run(["git", "show", f"{PIN17}:{P17}/{path}"], cwd=P, capture_output=True, text=True, check=True).stdout
assert git("data/prompts.jsonl") == (P / "inputs/prompts.jsonl").read_text(), "8B prompts differ from the 1.7B prompts"
prompts = {r["id"]: r for r in map(json.loads, open(P / "inputs/prompts.jsonl"))}
EX = os.environ.get("EX") or T["example"]
assert EX, "set text.json example (pick.py)"
strictly = lambda r, tok: r["top"] == tok and r["top_logit"] > r["top2_logit"]


def by_prompt(d):
    out = {}
    for r in d["rows"]:
        if prompts[r["prompt"]]["split"] == "test":
            out.setdefault(r["prompt"], {})[r["probe"]] = r
    return out


def runs8(pat):
    fs = sorted(glob.glob(str(DATA / "whole" / pat)))
    return [(Path(f).stem, by_prompt(json.load(open(f)))) for f in fs]


def runs17(names):
    return [(n, by_prompt(json.loads(git(f"data/whole/{n}.json")))) for n in names]


def switched(v):
    return strictly(v[0], v[0]["b_id"]) and strictly(v[1], v[1]["a_id"])


def same(v):
    return v[0]["top"] == v[1]["top"] and strictly(v[0], v[0]["top"]) and strictly(v[1], v[1]["top"])


def stats(rs):
    counts = [sum(switched(v) for v in w.values()) for _, w in rs]
    n = {len(w) for _, w in rs}
    assert len(n) == 1, n
    return counts, n.pop()


def rng(v):
    lo, hi = min(v), max(v)
    return f"{lo}" if lo == hi else f"{lo}&ndash;{hi}"


def count_line(counts, n, unit="run"):
    k = len(counts)
    runs = f"{k} {unit}" + ("" if k == 1 else "s")
    if k > 1 and min(counts) == max(counts):
        return f"<b>{counts[0]} of {n}</b> new prompts switch (each of {runs})"
    return f"<b>{rng(counts)} of {n}</b> new prompts switch ({runs})"


sig = lambda x: 1 / (1 + math.exp(-x))
ROBOT = '<svg width="50" height="57" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>'


def answer(r, red=False):
    a = strictly(r, r["a_id"])
    assert a or strictly(r, r["b_id"]), r  # every drawn answer is A or B, strictly
    L, w = ("A", "mountains") if a else ("B", "sea")
    return (f'<div class="reply">{ROBOT}<div class="answer{" red" if red and a else ""}"><span class="L">{L}</span>'
            f'<span class="w">{w}</span></div></div>')


def glyphs(arm):
    cells = []
    for i in range(8):
        mlp = arm["mlps"] == "all" or i == 7
        cells.append(f'<i class="h{" on" if arm["heads"] else ""}"></i><i class="m{" on" if mlp else ""}"></i>')
    return f'<div class="gl">{"".join(cells)}<span class="tag">{arm["tag"]}</span></div>'


facts, rows = {"pin17": PIN17, "example": EX}, []
for arm in T["arms"]:
    r8 = runs8(arm["glob8"])
    r17 = runs17(arm["runs17"])
    c17, n17 = stats(r17)
    src = "8B"
    if not r8:
        if not PREVIEW:
            continue
        r8, src = r17, "1.7B-PREVIEW"
    c8, n8 = stats(r8)
    verdict = [c / n8 >= 0.75 for c in c8]
    # the example behaves like its run in every run of the arm (pick rule, room #24105)
    for (name, w), v in zip(r8, verdict):
        assert (switched(w[EX]) if v else same(w[EX])), (name, EX, w[EX])
    ex = r8[0][1][EX]
    on = switched(ex)
    facts[arm["key"]] = {"src": src, "runs": [n for n, _ in r8], "counts": c8, "n": n8, "c17": c17}
    at17 = arm["at17"].format(k=rng(c17), n=n17, runs=len(c17))
    rows.append(f'<div class="row{" hl" if arm.get("hl") else ""}"><div><h2>{arm["title"]}</h2>{glyphs(arm)}'
                f'<div class="num">{count_line(c8, n8)}</div><div class="at17">{at17}</div></div>'
                f'{answer(ex[0])}{answer(ex[1], red=on)}</div>')

text = prompts[EX]["text"].split("user\n", 1)[1]
ctx, q = text.split("\n\nWhich")
q = "Which" + q.split("<|im_end|>")[0]
cut = ctx.rfind(". ", 0, T["ctx_chars"] + 10)
ctx = ctx[:cut + 1] if cut > 40 else ctx[:T["ctx_chars"]].rsplit(" ", 1)[0].rstrip(",")
QS = {r["text"].split("\n\nWhich", 1)[1] for r in prompts.values() if r["split"] == "test"}
assert len(QS) == 1, len(QS)  # 'the question at the end is always the same'
HEAD_TXT = re.sub(r"<[^>]+>", "", H.unescape(T["head"]))
kicker = f'<div class="kicker">{T["kicker"]}</div>' if T.get("kicker") else ""
banner = '<div class="prev">LAYOUT PREVIEW: rows without 8B files show 1.7B data</div>' if any(
    isinstance(v, dict) and v.get("src", "8B") != "8B" for v in facts.values()) else ""

# Token strip (optional, text.json "strip"): how clearly a logistic classifier reads probe on/off at each of the last
# ten tokens entering one layer, before training. 8B: research_checks/map.json eager[<layer>@<pos>].logistic.auroc;
# 1.7B: the same file at PIN17. Tokens 70-79 of every prompt (Qwen3 chat template; checked with the Qwen3 tokenizer).
TOKS = [".", "<|im_end|>", "\u23ce", "<|im_start|>", "assistant", "\u23ce", "<think>", "\u23ce\u23ce", "</think>", "\u23ce\u23ce"]
strip_html = ""
if T.get("strip") and (P / "research_checks/map.json").exists():
    S = T["strip"]
    m8 = json.load(open(P / "research_checks/map.json"))["eager"]
    m17 = json.loads(git("research_checks/map.json"))["eager"]
    auc = [m8[f"{S['layer8']}@{p}"]["logistic"]["auroc"] for p in range(70, 80)]
    a17 = {p: m17[f"{S['layer17']}@{p}"]["logistic"]["auroc"] for p in (71, 79)}
    facts["strip"] = {"layer8": S["layer8"], "auc8": [round(a, 3) for a in auc], "layer17": S["layer17"], "auc17_71_79": a17}

    def chip(i):
        a = auc[i]
        t = max(0.0, min(1.0, (a - 0.5) / 0.45))
        lab = ""
        if i in (1, 9):
            name = "end of message" if i == 1 else "answer chosen here"
            lab = f'<span class="cap">{name}<br><b>{a:.2f}</b><em>1.7B: {a17[70 + i]:.2f}</em></span>'
        cls = " ansx" if i == 9 else ""
        return f'<span class="chip{cls}" style="background:rgba(218,112,44,{0.08 + 0.85 * t:.2f})">{lab}{H.escape(TOKS[i])}</span>'

    strip_html = (f'<div class="st">{S["title"]} <span class="scale"><small>0.5 chance</small><i></i><small>0.95</small></span></div>'
                  f'<div class="strip">{"".join(chip(i) for i in range(10))}</div>')

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {H.escape(HEAD_TXT)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&display=swap" rel="stylesheet">
<!--
Built by figure1/gen.py from kran-research (branch swarm/oracle-1118-frozen-heads-8b), projects/probe-switch-frozen-heads-8b.
  prompt   inputs/prompts.jsonl {EX} (test split; identical file to the 1.7B run); picked by figure1/pick.py
  8B       {DATA.relative_to(P)}/whole/<run>.json, strict switch (PREREG): probe off top-1 B, probe on top-1 A, each strictly above top-2
  1.7B     {P17}/data/whole/<run>.json at {PIN17}
  facts    {json.dumps(facts)}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 34px 44px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .prev {{ font-size: 24px; font-weight: 800; color: #fff; background: #c8372d; padding: 6px 12px; margin-bottom: 12px; }}
  .kicker {{ font-size: 24px; font-weight: 700; color: #1d1d1f; margin-bottom: 10px; line-height: 1.3; }}
  .head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .head em {{ font-style: normal; color: #c8372d; }}
  .sub {{ font-size: 24px; color: #5a5d66; margin-top: 10px; line-height: 1.3; }}
  .sub b {{ color: #1d1d1f; font-weight: 700; }}
  .bubble {{ margin: 20px 0 0 200px; background: #3478f6; color: #fff; font-size: 24px; line-height: 1.32; padding: 16px 22px; border-radius: 24px 24px 6px 24px; }}
  .bubble .ctx {{ opacity: 0.82; }}
  .bubble b {{ font-weight: 650; }}
  .pnote {{ font-size: 20px; color: #6b6f78; margin: 6px 0 0 200px; text-align: right; }}
  .cols {{ display: grid; grid-template-columns: 1fr 236px 236px; gap: 12px; padding: 0 16px 0 20px; margin-top: 18px; font-size: 23px; font-weight: 700; color: #4d5059; align-items: end; line-height: 1.2; }}
  .key {{ font-size: 20px; font-weight: 500; color: #6b6f78; line-height: 1.35; white-space: nowrap; }}
  .key .gl {{ display: inline-flex; margin: 0 4px 0 0; vertical-align: -1px; }}
  .key .gl i {{ margin-right: 0; }}
  .row {{ display: grid; grid-template-columns: 1fr 236px 236px; gap: 12px; align-items: center; border-radius: 24px; padding: 14px 16px 14px 20px; margin-top: 10px; background: #eef0f3; }}
  .row.hl {{ background: #fbece2; }}
  .row h2 {{ font-size: 28px; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }}
  .tag {{ font-size: 20px; font-weight: 600; color: #7a7d86; white-space: nowrap; margin-left: 6px; }}
  .gl {{ display: flex; gap: 3px; margin-top: 8px; align-items: center; }}
  .gl i {{ display: block; width: 15px; height: 15px; border: 2.5px solid #9aa0aa; }}
  .gl i.h {{ border-radius: 50%; }}
  .gl i.m {{ border-radius: 3px; margin-right: 7px; }}
  .gl i.on {{ background: #33363d; border-color: #33363d; }}
  .num {{ font-size: 24px; margin-top: 8px; line-height: 1.25; color: #4d5059; }}
  .num b {{ color: #1d1d1f; }}
  .at17 {{ font-size: 21px; color: #6f727b; margin-top: 4px; font-style: italic; }}
  .reply {{ display: flex; align-items: center; gap: 8px; }}
  .reply > svg {{ flex: none; }}
  .answer {{ background: #fff; border-radius: 20px 20px 20px 6px; padding: 8px 14px 8px 12px; font-size: 26px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; align-items: center; gap: 8px; white-space: nowrap; }}
  .answer .L {{ font-size: 38px; font-weight: 800; }}
  .answer .w {{ font-size: 21px; font-weight: 650; }}
  .answer.red {{ color: #c8372d; }}
  .note {{ font-size: 21px; color: #5a5d66; margin: 12px 0 0 20px; }}
  .note b {{ color: #1d1d1f; }}
  .foot {{ display: flex; justify-content: space-between; border-top: 1px solid #e3e3e6; margin-top: 16px; padding-top: 12px; font-size: 19px; color: #7a7d86; }}
  .foot b {{ color: #1d1d1f; font-weight: 700; }}
  .st {{ font-size: 23px; font-weight: 700; color: #4d5059; margin-top: 22px; }}
  .st small {{ font-weight: 500; color: #7a7d86; font-size: 20px; }}
  .scale {{ display: inline-flex; align-items: center; gap: 6px; margin-left: 10px; vertical-align: 1px; }}
  .scale i {{ display: inline-block; width: 110px; height: 16px; border-radius: 5px; background: linear-gradient(90deg, rgba(218,112,44,0.08), rgba(218,112,44,0.93)); }}
  .strip {{ display: flex; gap: 6px; align-items: flex-end; margin-top: 94px; }}
  .chip {{ position: relative; font: 600 22px ui-monospace, Menlo, monospace; padding: 9px 10px; border-radius: 10px; color: #3a2414; }}
  .chip.ansx {{ outline: 3px dashed #7a7d86; outline-offset: 2px; margin-left: 4px; }}
  .cap {{ position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); white-space: nowrap; text-align: center;
          font: 500 20px Inter, sans-serif; color: #5a5d66; line-height: 1.15; }}
  .cap b {{ display: block; font-size: 24px; color: #1d1d1f; }}
  .cap em {{ display: block; font-size: 19px; color: #6f727b; }}
</style>
</head>
<body>

{banner}{kicker}<div class="head">{T["head"]}</div>
<div class="sub">{T["sub"]}</div>

<div class="bubble"><span class="ctx">{H.escape(ctx)}&nbsp;&hellip;</span> <b>{H.escape(q)}</b></div>
<div class="pnote">{T["prompt_note"]}</div>
{strip_html}
<div class="cols"><div class="key"><span class="gl"><i class="h on"></i></span>attention heads&nbsp;&nbsp;<span class="gl"><i class="m on"></i></span>MLP<br>{T["key_line"]}</div><div>{T["col_off"]}</div><div>{T["col_on"]}</div></div>
{chr(10).join(rows)}
{f'<div class="note">{T["note"]}</div>' if T.get("note") else ""}
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
print("ok", EX, json.dumps({k: v for k, v in facts.items() if isinstance(v, dict)}))
