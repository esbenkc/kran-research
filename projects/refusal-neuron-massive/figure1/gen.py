"""Figure 1 generator for refusal-neuron-massive (Owain).

Reads the real replies (data/runs/test_harmful/<arm>.jsonl) and the status numbers
(data/results.json), asserts every number and label the figure shows, and writes
blog.kran.ai/assets/figures/refusal-neuron-massive-fig1/figure1.html.
Run from the swarm workspace (python3 swarm-theory-1031-refusal-neuron/figure1/gen.py) or from the receipts
(python3 kran-research/projects/refusal-neuron-massive/figure1/gen.py, blog clone next to kran-research,
or set FIG1_OUT to the output html path).
"""
import html
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_BLOG = next((p for p in [os.path.join(ROOT, "..", "blog.kran.ai"), os.path.join(ROOT, "..", "..", "..", "blog.kran.ai")]
              if os.path.isdir(p)), "blog.kran.ai")
OUT = os.environ.get("FIG1_OUT") or os.path.join(_BLOG, "assets", "figures", "refusal-neuron-massive-fig1", "figure1.html")
CFG = json.load(open(os.path.join(HERE, "fig_cfg.json")))
IDX = CFG["idx"]  # row index in the TEST harmful split (same order in every arm file)


def load(split, arm):
    return [json.loads(line) for line in open(os.path.join(ROOT, "data", "runs", split, arm + ".jsonl"))]


# D5 garbage rule, copied from analyze.py (numpy only)
nl = np.array([r["nll_neutral"] for r in load("test_harmless", "base")], float)
THR = float(np.nanpercentile(nl, 95))


def coherent_answer(r):
    garb = (r["nll_neutral"] == r["nll_neutral"] and r["nll_neutral"] > THR) or r["distinct"] < 0.3
    return (not r["refusal"]) and not garb


_J = [os.path.join(ROOT, "skeptic", "vera_judge_test.json"), os.path.join(ROOT, "skeptics", "skeptic", "vera_judge_test.json")]
JUDGE = json.load(open(next(p for p in _J if os.path.exists(p))))  # Vera's D3 judge, claude-haiku-4-5
prompt = None


def expand(arms):
    """'Dp_*' -> every generated draw file that exists now (all must show the card's verdict)."""
    out = []
    for a in arms:
        if a.endswith("*"):
            k = 0
            while os.path.exists(os.path.join(ROOT, "data", "runs", "test_harmful", f"{a[:-1]}{k}.jsonl")):
                out.append(f"{a[:-1]}{k}")
                k += 1
            assert out, ("no draws yet", a)
        else:
            out.append(a)
    return out


checked = []  # (arm, string label, judge label) per arm the figure relies on
for c in CFG["cards"]:
    arms = c["arms"] = expand(c["arms"])
    for a in arms:
        r = load("test_harmful", a)[IDX]
        prompt = prompt or r["prompt"]
        assert r["prompt"] == prompt == CFG["prompt"], (a, r["prompt"])
        # string label + D5 nonsense rule: the card's verdict must hold for every arm it stands for.
        # Random-draw cards (D11) are labelled by the judge; their string labels are recorded, not asserted.
        string_ok = coherent_answer(r) if c["answers"] else r["refusal"]
        if c.get("label") != "judge":
            assert string_ok, (a, r["refusal"])
        j = JUDGE.get("test_harmful/" + a)
        if c.get("judge_required", True):
            assert j is not None, ("no judge labels yet", a)
        if j is not None:
            assert j[IDX] == ("COMPLY" if c["answers"] else "REFUSE"), (a, j[IDX])
        checked.append((a, "string " + ("agrees" if string_ok else "DISAGREES"), j[IDX] if j else "n/a"))
    shown = CFG["shown"][c["show"]]
    flat = " ".join(load("test_harmful", arms[0])[IDX]["reply"].replace("**", "").replace("🌍", "").split())
    assert flat.startswith(shown["check_prefix"]), (arms[0], flat[:120])

# numbers in the bottom line come from results.json
res = json.load(open(os.path.join(ROOT, "data", "results.json")))
st = res["status_test_harmful"]
r_keep, r_rem = round(100 * st["r_keep"]), round(100 * st["r_rem"])
assert (r_keep, r_rem) == tuple(CFG["expect_pct"]), (r_keep, r_rem)
meta = json.load(open(os.path.join(ROOT, "data", "runs", "meta.json")))
len_share = round(100 * meta["norm_wM"] / meta["norm_w"])
assert len_share == CFG["expect_len_share"], len_share
bottom = CFG["bottom"].format(r_keep=r_keep, r_rem=r_rem, len_share=len_share)
legend = CFG["legend"].format(len_share=len_share)

BAR_W = CFG.get("bar_w", 220)
slice_w = max(4, round(BAR_W * len_share / 100))


def bar(kind):
    """Pictogram of the pin's push on the 2,048 coordinates: thin red slice = 3 outlier coordinates."""
    rest = f'<rect x="{slice_w}" y="0" width="{BAR_W - slice_w}" height="22" rx="0" fill="#5b6b8c"/>'
    sl = f'<rect x="0" y="0" width="{slice_w}" height="22" fill="#c8372d"/>'
    frame = f'<rect x="0.75" y="0.75" width="{BAR_W - 1.5}" height="20.5" rx="5" fill="none" stroke="#8a8f99" stroke-width="1.5" stroke-dasharray="4 3"/>'
    allred = f'<rect x="0" y="0" width="{BAR_W}" height="22" fill="#c8372d"/>'
    body = {"none": "", "full": sl + rest, "slice": sl, "rest": rest, "allred": allred}[kind]
    return (f'<svg width="{BAR_W}" height="22" viewBox="0 0 {BAR_W} 22" aria-hidden="true">'
            f'<clipPath id="c{kind}"><rect width="{BAR_W}" height="22" rx="5"/></clipPath>'
            f'<g clip-path="url(#c{kind})">{body}</g>{frame}</svg>')


def esc(s):
    return html.escape(s, quote=False)


cards = []
for c in CFG["cards"]:
    reply = esc(CFG["shown"][c["show"]]["text"])
    for m in CFG["shown"][c["show"]].get("mark", []):
        reply = reply.replace(esc(m), f"<mark>{esc(m)}</mark>")
    cls = "card e" if c["answers"] else "card n"
    verdict = "Answers" if c["answers"] else "Refuses"
    cards.append(f'''  <div class="{cls}">
   <div class="bar">{bar(c["bar"])}</div>
   <h2>{c["title"]}</h2>
   <div class="reply">{reply}</div>
   <div class="verdict{' red' if c['answers'] else ''}"><svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>{verdict}</div>
  </div>''')

sources = f"""Figure 1 for _notes/refusal-neuron-massive.md, written by figure1/gen.py (asserts every number below).
Sources (kran-research, branch swarm/theory-1031-refusal-neuron, projects/refusal-neuron-massive):
  prompt + replies  data/runs/test_harmful/<arm>.jsonl, row {IDX} of the 200 TEST harmful prompts
                    (Qwen3-1.7B bf16, thinking off, greedy, 64 new tokens); replies cut ("..."),
                    markdown bold, emoji and line breaks removed
                    base = no edit; A = neuron 3270 (layer 13) pinned at +30; C = only the pin's push on
                    coordinates 1793/1999/1401; B = the pin's push without those coordinates (A = B + C exactly);
                    Dp_k = the pin's full per-token push length, all on those 3 coordinates (D', arms.py)
  labels            string refusal label + D5 nonsense rule (analyze.py; threshold {THR:.3f} nats/token) and
                    Vera's judge (skeptic/vera_judge_test.json); per arm (arm, string, judge):
                    {checked}
  bottom numbers    data/results.json status_test_harmful: r_keep={st['r_keep']:.4f} (shown {r_keep}%),
                    r_rem={st['r_rem']:.4f} (shown {r_rem}%)
  slice width       data/runs/meta.json norm_wM/norm_w = {meta['norm_wM']:.4f}/{meta['norm_w']:.4f} = {len_share}% of the push length"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: the refusal neuron in Qwen3-1.7B, split into its outlier part and the rest</title>
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
.head {{ font-size: 38px; font-weight: 800; letter-spacing: -0.008em; line-height: 1.16; }}
.askrow {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 20px; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; }}
.ask {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.28; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
.legend {{ display: flex; flex-wrap: wrap; align-items: center; column-gap: 22px; row-gap: 6px; margin-top: 22px; font-size: 24px; color: #3a3d45; }}
.legend > span {{ white-space: nowrap; }}
.legend .k {{ display: inline-block; width: 22px; height: 22px; border-radius: 4px; vertical-align: -3px; }}
.legend b {{ font-weight: 750; }}
.cards {{ display: grid; grid-template-columns: repeat({len(CFG['cards'])}, 1fr); gap: 12px; margin-top: 16px; align-items: stretch; }}
.card {{ border-radius: 22px; padding: 16px 15px 14px; display: flex; flex-direction: column; }}
.card.n {{ background: #eef0f3; }}
.card.e {{ background: #fbe6dc; }}
.card .bar {{ height: 22px; }}
.card h2 {{ font-size: {CFG.get('title_px', 25)}px; font-weight: 800; line-height: 1.18; letter-spacing: -0.01em; margin-top: 12px; min-height: {CFG.get('title_min_h', 90)}px; }}
.reply {{ margin-top: 10px; background: #fff; border-radius: 20px 20px 20px 6px; padding: 12px 13px; font-size: 24px; line-height: 1.3;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07); flex: 1; }}
mark {{ background: none; color: #c8372d; font-weight: 750; }}
.verdict {{ margin-top: 12px; font-size: 28px; font-weight: 800; color: #4d5059; display: flex; align-items: center; gap: 10px; }}
.verdict.red {{ color: #c8372d; }}
.answer {{ font-size: 30px; line-height: 1.3; margin-top: 22px; color: #3a3d45; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
</style>
</head>
<body>

<div class="head">{CFG['head']}</div>
<div class="askrow"><div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-1.7B</div><div class="ask">{esc(prompt)}</div></div>
<div class="legend">{legend}</div>
<div class="cards">
{chr(10).join(cards)}
</div>
<div class="answer">{bottom}</div>
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
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(doc)
print("wrote", os.path.normpath(OUT))
print("bottom:", bottom)
