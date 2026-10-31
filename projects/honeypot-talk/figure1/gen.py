"""Build figure1.html from fig_data_<raw>.json (data.py). python3 gen.py <out.html> [raw|raw_smoke] [layout]"""
import html
import json
import re
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))
from agent import WARNING  # noqa: E402
from scorer import SAY_RE  # noqa: E402

RAW = sys.argv[2] if len(sys.argv) > 2 else "raw"
LAYOUT = sys.argv[3] if len(sys.argv) > 3 else "pair"
D = json.loads((H / f"fig_data_{RAW}.json").read_text())
CFG = json.loads((H / "fig_cfg.json").read_text())
EX = D["ex"]
NAME = {"claude-haiku-4-5": "Claude Haiku 4.5", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite"}


def mname(m):
    return next(v for k, v in NAME.items() if m.startswith(k))


assert CFG["row_on"][1] == WARNING.strip(), "warning text must be the frozen prompt (agent.py WARNING)"
# the drawn runs are what the picker demanded: off = no say, trip; on = says, and avoid vs trip
for k, (says, trip) in {"off_avoid": (False, True), "off_trip": (False, True), "on_avoid": (True, False),
                        "on_trip": (True, True)}.items():
    assert (EX[k]["says"], EX[k]["trip"]) == (says, trip), k
    if trip:  # the drawn action is the trip itself: the window reply presents the key to a service
        assert EX[k]["window_trip"] and EX[k]["key"] in json.dumps(EX[k]["args"]) and EX[k]["tool"] != "submit_report", k
        assert any(e["step"] == EX[k]["window_reply"] for e in EX[k]["decoy_log"]), k
    else:
        assert EX[k]["key"] not in json.dumps(EX[k]["args"]) and not EX[k]["decoy_log"], k
for k in ("on_avoid", "on_trip"):  # the warned quotes name the key/record itself (frozen key_targeted row)
    assert EX[k]["key_targeted"] is True and not EX[k]["bucket_targeted"], k
REC = EX["on_avoid"]["record"]
assert all(EX[k]["record"] == REC for k in EX), "same record in all four runs (same seed, same decoy)"

ROBOT = """<svg class="rb" width="46" height="46" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="5" y="20" width="5" height="12" rx="2" fill="#e8734a"/><rect x="38" y="20" width="5" height="12" rx="2" fill="#e8734a"/>
<rect x="8" y="12" width="32" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2"/><rect x="12" y="17" width="24" height="18" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""


def thought(e):
    t = html.escape(e["thought"])
    t = SAY_RE.sub(lambda m: f"<mark>{m.group(0)}</mark>", t)
    return f'<div class="bub">“{t}”</div>'


def action(e):
    if e["trip"]:
        return f'<div class="act trip">→ {CFG["act_trip"]}</div>'
    return f'<div class="act safe">→ {CFG["act_avoid"]}</div>'


def record():
    lines = [ln for ln in REC.strip().splitlines()]
    body = "".join(f"<div>{html.escape(ln)}</div>" for ln in lines[1:])
    return f'<div class="rec mono"><div class="rh2">{html.escape(lines[0])}</div>{body}</div>'


SUB = CFG["sub"]
CNT = D["cnt"]
if "summary" in D:  # every number drawn comes from the official results file (analysis.py -> results/summary.json)
    PR = D["summary"]["primary"]
    STATUS = D["summary"]["status"]
    assert PR["status"] == STATUS
    st, sa, nt, na = CNT["gpt-5.4-mini|on|all"]
    VALS = {"G": round(-100 * PR["G"]), "mini_st": st, "mini_s": st + sa}
    assert STATUS != "REJECTED" or PR["ci90"][1] < -0.105, PR["ci90"]
    assert VALS["G"] == 22, VALS
    NR = D["summary"]["n_runs_by_prompt"]
    CFG["lab_warn"] = CFG["lab_warn"].format(on=NR["on"], n=NR["on"] + NR["off"])
    assert (NR["on"], NR["off"]) == (700, 300), NR
    BOTTOM = CFG["bottom"][STATUS].format(**VALS)
else:
    STATUS, BOTTOM = "smoke", "Answer: [pending counted run]"
A, B = mname(EX["on_avoid"]["model"]), mname(EX["on_trip"]["model"])

TOP = f"""<div class="setup">{ROBOT}<div>{SUB}</div></div>
<div class="top">
 <div><div class="lab">{CFG['lab_rec']}</div>{record()}</div>
 <div><div class="lab">{CFG['lab_warn']}</div><div class="warn">“{html.escape(CFG['row_on'][1])}”</div></div>
</div>"""


def cell(k):
    e = EX[k]
    return f'<div class="cell{" hl" if e["trip"] else ""}">{thought(e)}{action(e)}</div>'


if LAYOUT == "pair":
    BODY = f"""{TOP}
<div class="grid">
 <div></div><div class="colh">{ROBOT}{A}</div><div class="colh">{ROBOT}{B}</div>
 <div class="rowh">{CFG['row_off'][0]}</div>{cell('off_avoid')}{cell('off_trip')}
 <div class="rowh">{CFG['row_on'][0]}</div>{cell('on_avoid')}{cell('on_trip')}
</div>"""
else:
    raise SystemExit(f"unknown layout {LAYOUT}")

SRC = f"""Sources (kran-research, branch swarm/defense-1031-say-do-trap-gap, projects/honeypot-talk):
  runs drawn: {RAW}/<model>/<cell>/<seed>.jsonl, window reply (scorer.py window): """ + "; ".join(
    f"{k}={EX[k]['model']} {EX[k]['cell']} seed {EX[k]['seed']} reply {EX[k]['window_reply']}" for k in EX) + f"""
  scores: {'results' if RAW == 'raw' else 'results_smoke'}/scores.jsonl; warning text: agent.py WARNING; record: env.py record()
  built by figure1/data.py + figure1/gen.py (layout {LAYOUT})"""

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {html.escape(CFG['title'])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
{SRC}
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 32px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 35px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.18; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .setup {{ display: flex; gap: 12px; align-items: flex-start; font-size: 25px; line-height: 1.32; color: #3b3e46; margin-top: 12px; }}
  .rb {{ flex: none; }}
  .top {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 14px; }}
  .lab {{ font-size: 24px; font-weight: 700; color: #5a5d66; margin-bottom: 6px; }}
  .rec {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 14px; padding: 8px 14px; font-size: 22px; line-height: 1.38; }}
  .rec .rh2 {{ color: #6b6e76; }}
  .warn {{ background: #2f6fe4; color: #fff; font-size: 24px; font-weight: 500; line-height: 1.32;
           border-radius: 18px 18px 4px 18px; padding: 12px 16px; }}
  .grid {{ display: grid; grid-template-columns: 190px 1fr 1fr; gap: 14px; margin-top: 22px; }}
  .colh {{ font-size: 28px; font-weight: 800; display: flex; gap: 10px; align-items: center; }}
  .rowh {{ font-size: 27px; font-weight: 800; line-height: 1.2; align-self: center; }}
  .cell {{ background: #eef0f3; border-radius: 18px; padding: 14px; display: flex; flex-direction: column; gap: 10px; justify-content: space-between; }}
  .cell.hl {{ background: #fbece2; }}
  .bub {{ background: #fff; border-radius: 18px 18px 18px 6px; padding: 9px 14px; font-size: 24px; line-height: 1.3;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .act {{ font-size: 26px; font-weight: 800; }}
  .act.trip {{ color: #c8372d; }}
  .act.safe {{ color: #1f6b35; }}
  mark {{ background: #ffe58a; color: inherit; border-radius: 4px; padding: 0 2px; font-weight: 700; }}
  .bottom {{ font-size: 28px; font-weight: 800; margin-top: 20px; line-height: 1.26; }}
  .bottom em {{ font-style: normal; color: #c8372d; }}
  .bottom .why {{ color: #5a5d66; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{CFG['head']}</div>
{BODY}
<div class="bottom">{BOTTOM}</div>
</body>
</html>
"""
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HTML)
print("wrote", out)
print(SRC)
