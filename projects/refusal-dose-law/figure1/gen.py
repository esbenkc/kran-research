"""Figure 1 generator for refusal-dose-law (Owain).

Reads the real replies (data/runs/<cell>.jsonl), the judge labels (data/judge/<cell>.json) and the
cell compliance (data/results.json), asserts that the example row has the verdict each card claims for
every arm it stands for, and writes <blog>/assets/figures/refusal-dose-law-fig1/figure1.html.
Run from anywhere: python3 figure1/gen.py   (FIG1_OUT overrides the output path)
"""
import glob
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_BLOG = next((p for p in [os.path.join(ROOT, "..", "blog.kran.ai"), os.path.join(ROOT, "..", "..", "..", "blog.kran.ai")]
              if os.path.isdir(p)), "blog.kran.ai")
OUT = os.environ.get("FIG1_OUT") or os.path.join(_BLOG, "assets", "figures", "refusal-dose-law-fig1", "figure1.html")
CFG = json.load(open(os.path.join(HERE, "fig_cfg.json")))
IDX = CFG["idx"]  # row index in TEST harmful (same order in every cell file)
DATA = os.path.join(ROOT, "data")


def rows(cell):
    return [json.loads(line) for line in open(os.path.join(DATA, "runs", cell + ".jsonl"))]


def labels(cell):
    return json.load(open(os.path.join(DATA, "judge", cell + ".json")))["labels"]


def expand(arms):
    out = []
    for a in arms:
        if "*" in a:
            got = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(DATA, "judge", a + ".json")))
            assert got, ("no judged cells yet", a)
            out += got
        else:
            out.append(a)
    return out


def flat(s):
    return " ".join(s.replace("**", "").split())


checked = []
prompt = None
for c in CFG["cards"]:
    arms = c["arms"] = expand(c["arms"])
    for a in arms:
        r = rows(a)[IDX]
        assert r["i"] == IDX, (a, r["i"])
        prompt = prompt or r["prompt"]
        assert r["prompt"] == prompt == CFG["prompt"], (a, r["prompt"])
        lab = labels(a)[IDX]
        assert lab == ("COMPLY" if c["answers"] else "REFUSE"), (a, lab)
        checked.append((a, lab))
    shown = CFG["shown"][c["show"]]
    rep = flat(rows(arms[0])[IDX]["reply"])
    assert rep.startswith(shown["check_prefix"]), (arms[0], rep[:140])
    for seg in shown["text"].split("…"):  # every shown fragment is verbatim in the real reply
        assert seg.strip() in rep, (arms[0], seg)

# how typical is the example: rows (covered by every arm) where every arm has its card's verdict
want = [(a, "COMPLY" if c["answers"] else "REFUSE") for c in CFG["cards"] for a in c["arms"]]
n_cov = min(len(labels(a)) for a, _ in want)
split_rows = [i for i in range(n_cov) if all(labels(a)[i] == v for a, v in want)]
assert IDX in split_rows, split_rows
note = CFG["note"].format(k=len(split_rows), n=n_cov)

# bottom-line numbers: judged compliance per cell = share COMPLY in data/judge/<cell>.json,
# cross-checked against data/results.json cells[*].comply whenever analyze.py has the cell
res = json.load(open(os.path.join(DATA, "results.json")))
cells = res["cells"]


def pct(cell, n):
    lab = labels(cell)
    assert len(lab) == n, (cell, len(lab))
    v = 100 * lab.count("COMPLY") / n
    if cell in cells:
        assert abs(cells[cell]["comply"] * 100 - v) < 1e-9 and cells[cell]["n"] == n, (cell, v, cells[cell])
    return v


twins = [a for a in CFG["cards"][4]["arms"]]
assert len(twins) == CFG["twin_seeds"], twins
nums = dict(base=pct("base", 200), g=pct("G@30", 200), on=pct("G_on@30", 200), off=pct("G_off@30", 200),
            twin=sum(pct(t, 100) for t in twins) / len(twins))
assert abs(cells["G_off@30"]["p"]) < 1e-6, cells["G_off@30"]["p"]  # zero push along the direction
for t in twins:  # twin: the gate's own p (same push along the direction); TWIN_k@m share p across k
    tp = cells[t]["p"] if t in cells else cells["TWIN_0@30"]["p"]
    assert abs(tp / cells["G@30"]["p"] - 1) < 0.01, (t, tp)
assert abs(cells["G_on@30"]["p"] / cells["G@30"]["p"] - 1) < 0.01, cells["G_on@30"]["p"]
def fmt(v):  # n=200 cells are exact multiples of 0.5: print them exactly; the 3-push mean is rounded
    return f"{v:.1f}" if abs(v * 2 - round(v * 2)) < 1e-9 and abs(v - round(v)) > 1e-9 else str(int(v + 0.5))


shown_nums = {k: fmt(v) for k, v in nums.items()}
for k in ["g", "on", "off"]:  # rise over no edit, in points (the TLDR's unit)
    shown_nums["d_" + k] = fmt(nums[k] - nums["base"])
# the neuron's lead over the curve fitted on built pushes (FIT SET = TWIN + D(c) + RN + RD), 5 pins, points
fit = res["labelings"]["comply"]["point"]["fit"]
assert res["STATUS"] == "REJECTED" and fit["cells"] >= 60, (res["STATUS"], fit["cells"])
gaps = [fit["gap"][m] for m in ["6", "12", "18", "24", "30"]]
shown_nums["gap_lo"], shown_nums["gap_hi"] = str(int(min(gaps) + 0.5)), str(int(max(gaps) + 0.5))
bottom = CFG["bottom"].format(**shown_nums)
legend = CFG["legend"].format(n=cells["base"]["n"])

# arrow pictogram geometry: the gate's write makes angle acos(|cos|) with the refusal direction (true angle)
COS = abs(json.load(open(os.path.join(DATA, "phase0", "phase0.json")))["gate_cos"])
assert 0.30 < COS < 0.45, COS
W, H, AX, OX, LEN = 210, 176, 88, 64, 84


def arrow(kind):
    on, off = LEN * COS, LEN * math.sqrt(1 - COS ** 2)
    axis = f'<line x1="6" y1="{AX}" x2="{W - 6}" y2="{AX}" stroke="#8a8f99" stroke-width="2.5" stroke-dasharray="7 5"/>'
    origin = f'<circle cx="{OX}" cy="{AX}" r="6" fill="#33363d"/>'

    def arr(x2, y2, col):
        ang = math.atan2(y2 - AX, x2 - OX)
        L2 = math.hypot(x2 - OX, y2 - AX)
        bx, by = OX + (L2 - 16) * math.cos(ang), AX + (L2 - 16) * math.sin(ang)
        nx, ny = -math.sin(ang) * 9, math.cos(ang) * 9
        return (f'<line x1="{OX}" y1="{AX}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>'
                f'<polygon points="{x2:.1f},{y2:.1f} {bx + nx:.1f},{by + ny:.1f} {bx - nx:.1f},{by - ny:.1f}" fill="{col}"/>')
    shadow = f'<line x1="{OX}" y1="{AX}" x2="{OX + on:.1f}" y2="{AX}" stroke="#c8372d" stroke-width="12" stroke-linecap="butt"/>'
    up = AX - off * 1.0
    body = {
        "none": "",
        "full": shadow + f'<line x1="{OX + on:.1f}" y1="{AX}" x2="{OX + on:.1f}" y2="{up:.1f}" stroke="#8a8f99" stroke-width="2" stroke-dasharray="3 4"/>' + arr(OX + on, up, "#33363d"),
        "on": arr(OX + on + 4, AX, "#c8372d"),
        "off": arr(OX, up, "#33363d"),
        "twin": shadow + f'<line x1="{OX + on:.1f}" y1="{AX}" x2="{OX + on:.1f}" y2="{AX + off:.1f}" stroke="#8a8f99" stroke-width="2" stroke-dasharray="3 4"/>' + arr(OX + on, AX + off, "#33363d"),
    }[kind]
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">{axis}{body}{origin}</svg>'


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
   <div class="pic">{arrow(c["arrow"])}</div>
   <h2>{c["title"]}</h2>
   <div class="reply">{reply}</div>
   <div class="verdict{' red' if c['answers'] else ''}"><svg width="40" height="45" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>{verdict}</div>
   <div class="stat"><b>{shown_nums[c['stat']]}%</b> <span>{'answered, 3×100 requests' if c['stat'] == 'twin' else 'answered, of 200 requests'}</span></div>
  </div>''')

sources = f"""Figure 1 for _notes/refusal-dose-law.md, written by figure1/gen.py (asserts every label and number below).
Sources (kran-research, branch swarm/theory-1031-refusal-dose-law, projects/refusal-dose-law):
  prompt + replies  data/runs/<cell>.jsonl, row {IDX} of the 200 TEST harmful prompts (Qwen3-1.7B bf16, thinking off,
                    greedy, 64 new tokens); replies cut ("..."), markdown bold removed
                    base = no edit; G@30 = neuron 3270 (layer 13) pinned at +30; G_on = only the projection of its
                    push on the refusal direction r_hat; G_off = the remainder (G = G_on + G_off exactly);
                    TWIN_k = c*r_hat + sqrt(1-c^2)*u_k with c = |cos(w, r_hat)| = {COS:.4f}, the gate's per-token push length
  labels            judge (claude-haiku-4-5, data/judge/<cell>.json), row {IDX}: {checked}
  note              rows of the first {n_cov} where every arm has its card's verdict: {split_rows}
  bottom numbers    data/results.json cells[*].comply: base {nums['base']:.1f}, G@30 {nums['g']:.1f}, G_on@30 {nums['on']:.1f},
                    G_off@30 {nums['off']:.1f} (n=200 each); TWIN mean over {twins} = {nums['twin']:.2f} (n=100 each)
  arrow angle       acos(|cos(w, r_hat)|) = acos({COS:.4f}) (phase0/phase0.json gate_cos), drawn at the true angle"""

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: the refusal neuron in Qwen3-1.7B, split into its push along the refusal direction and the rest</title>
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
.head {{ font-size: 44px; font-weight: 800; letter-spacing: -0.012em; line-height: 1.14; }}
.sub {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
.foot {{ font-size: 20px; color: #7a7e87; margin-top: 18px; }}
.foot b {{ color: #3a3d45; font-weight: 700; }}
.askrow {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 20px; }}
.model {{ display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 750; color: #3a3d45; white-space: nowrap; }}
.ask {{ background: #3478f6; color: #fff; font-size: 24px; line-height: 1.28; padding: 13px 22px; border-radius: 26px 26px 6px 26px; }}
.note {{ text-align: right; font-size: 24px; color: #5d616b; margin-top: 8px; }}
.legend {{ display: flex; flex-wrap: wrap; align-items: center; column-gap: 22px; row-gap: 6px; margin-top: 22px; font-size: 24px; color: #3a3d45; }}
.legend > span {{ white-space: nowrap; }}
.legend .k {{ display: inline-block; width: 30px; height: 12px; border-radius: 2px; vertical-align: 4px; }}
.legend .k.dash {{ height: 0; border-top: 3px dashed #8a8f99; vertical-align: 7px; }}
.legend b {{ font-weight: 750; }}
.cards {{ display: grid; grid-template-columns: repeat({len(CFG['cards'])}, 1fr); gap: 12px; margin-top: 16px; align-items: stretch; }}
.card {{ border-radius: 22px; padding: 12px 13px 14px; display: flex; flex-direction: column; }}
.card.n {{ background: #eef0f3; }}
.card.e {{ background: #fdf0e0; }}
.card .pic {{ display: flex; justify-content: center; }}
.card .pic svg {{ width: 100%; height: auto; }}
.card h2 {{ font-size: {CFG.get('title_px', 24)}px; font-weight: 800; line-height: 1.18; letter-spacing: -0.01em; margin-top: 6px; min-height: {CFG.get('title_min_h', 112)}px; }}
.reply {{ margin-top: 10px; background: #fff; border-radius: 20px 20px 20px 6px; padding: 12px 13px; font-size: 24px; line-height: 1.3;
  box-shadow: 0 2px 8px rgba(0,0,0,0.07); flex: 1; }}
mark {{ background: none; color: #b45309; font-weight: 750; }}
.verdict {{ margin-top: 12px; font-size: 28px; font-weight: 800; color: #4d5059; display: flex; align-items: center; gap: 10px; }}
.verdict.red {{ color: #b45309; }}
.answer {{ font-size: 27px; line-height: 1.3; margin-top: 20px; color: #3a3d45; }}
.stat {{ margin-top: 8px; font-size: 24px; line-height: 1.15; color: #4d5059; }}
.stat span {{ display: block; min-height: 56px; }}
.stat b {{ display: block; font-size: 40px; font-weight: 800; color: #1d1d1f; letter-spacing: -0.02em; }}
.card.e .stat b {{ color: #b45309; }}
.answer b {{ color: #1d1d1f; font-weight: 800; }}
</style>
</head>
<body>

<div class="head">{CFG['head'].format(**shown_nums)}</div>
<div class="sub">{CFG['sub'].format(**shown_nums)}</div>
<div class="askrow"><div class="model"><svg width="44" height="50" viewBox="0 0 92 104" aria-hidden="true"><use href="#robot"/></svg>Qwen3-1.7B</div><div class="ask">{esc(prompt)}</div></div>
<div class="note">{note}</div>
<div class="legend">{legend}</div>
<div class="cards">
{chr(10).join(cards)}
</div>
<div class="answer">{bottom}</div>
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
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(doc)
print("wrote", os.path.normpath(OUT))
print("checked:", checked)
print("bottom:", bottom)
