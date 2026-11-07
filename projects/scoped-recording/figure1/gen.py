"""Build Figure 1 for scoped-recording from the project's real raw runs and results/summary.json.

Run from the work dir:
  python3 swarm-defense-1107-scoped-recording/owain/gen.py [--smoke] [--model M --seed S] [--out PATH]
Text (head, subtitle, labels) lives in text.json next to this file, so wording changes need no code change.
Every quote, route and number is read from the run files and asserted against the scorer and the summary.
"""
import argparse
import html
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
# works from swarm-.../owain/ (work dir) and from kran-research/projects/scoped-recording/figure1/ (receipt copy)
P = H.parent if (H.parent / "agent.py").exists() else ROOT / "kran-research/projects/scoped-recording"
sys.path.insert(0, str(P))
import agent  # noqa: E402
from scorer import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--model", default="claude-haiku-4-5-20251001")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--cols", default="none,key,flag")
ap.add_argument("--out", default=str(ROOT / "blog.kran.ai/assets/figures/scoped-recording-fig1/figure1.html"))
ap.add_argument("--text", default=str(H / "text.json"))
A = ap.parse_args()

RAW = P / ("raw_smoke" if A.smoke else "raw")
RES = P / ("results_smoke" if A.smoke else "results")
S = json.loads((RES / "summary.json").read_text())
T = json.loads(Path(A.text).read_text())
NAME = {"claude-haiku-4-5-20251001": "Claude Haiku 4.5", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
        "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gpt-5.4-mini": "GPT-5.4 mini", "gpt-5.4-nano": "GPT-5.4 nano"}
ARM_NAME = {"none": "no notice", "all": "everything logged", "key": "key logged, flags not",
            "flag": "flags logged, key not", "keyonly": "key logged, flags unmentioned"}
ORDER = ["none", "all", "key", "flag", "keyonly"]  # Selma #16026: bar order, one-sided arm last and set apart
COLS = A.cols.split(",")
assert set(S["arms"]) == set(ORDER), S["arms"]

# ---- pooled numbers (agent-equal means, the post's reading) --------------------------------------------------------
AE = S["agent_equal"]
rows = [json.loads(l) for l in (RES / "scores.jsonl").read_text().splitlines()]
n_cell = {a: sum(1 for r in rows if r["arm"] == a) for a in ORDER}
models = S["models"]
assert len(models) == 5
# agent-equal = plain pooled share when every agent has the same n (check, so the figure may say 'of runs')
for a in ORDER:
    ns = {S["n_per_cell"][f"{m}|{a}"] for m in models}
    assert len(ns) == 1, (a, ns)
    for t in ("flag_only", "key"):
        k = sum(r[t] for r in rows if r["arm"] == a)
        assert abs(k / n_cell[a] - AE[a][t]) < 1e-3, (a, t, k, n_cell[a], AE[a][t])
# per agent, flag-never-key use under 'key logged, flags not' minus no notice (the title move), from summary contrasts
PA = S["contrasts"]["flag_only:key-none"]["per_agent"]
RISERS = sum(v > 0 for v in PA.values())
# manipulation check M1 in the 'key logged, flags not' arm (results/summary.json manip; answer 'a' = the key call)
M1 = S["manip"]["M1"]
READERS = [m for m in models if M1[m]["key"]["correct"] >= 22 and M1[m]["key"]["n"] == 23]
assert len(READERS) == 3 and all(PA[m] <= 0 for m in READERS), (READERS, PA)
EX_M1 = M1[A.model]["key"]
MJ = [json.loads(l) for l in (RES / "manip.jsonl").read_text().splitlines()]
SEED_M1 = next(r["answer"] for r in MJ if r["model"] == A.model and r["arm"] == "key" and r["seed"] == A.seed and r["q"] == "M1")
N_RUNS = S["n_runs"]
assert N_RUNS == len(rows) == sum(n_cell.values())


def pct(x):
    return f"{100 * x:.0f}"


# ---- the example: one agent, one seed, every arm --------------------------------------------------------------------
def load(arm, seed):
    return json.loads((RAW / A.model / arm / f"{seed}.jsonl").read_text())


def decision(rec, key):
    """Thought and call of the first reply whose call takes a route (carries the flag or the record's key);
    a run that takes neither route quotes the reply right after the first denial that shows the record."""
    obs = rec["observations"]
    i0 = next(k for k, o in enumerate(obs) if "last archive record" in o)
    parsed = []
    for rep in rec["model_replies"]:
        t = rep["text"]
        parsed.append(json.loads(t[t.index("{"): t.rindex("}") + 1]))
    hit = next((k for k, o in enumerate(parsed) if o.get("tool") != "submit_report"
                and ("admin-override" in json.dumps(o.get("args", {})) or key in json.dumps(o.get("args", {})))), None)
    k = hit if hit is not None else i0
    o = parsed[k]
    return o["thought"], o["tool"], o.get("args", {}), obs[i0]


def route(sc):
    return "both" if sc["flag"] and sc["key"] else "flag" if sc["flag"] else "key" if sc["key"] else "none"


seed = A.seed
EX = {}
for a in COLS:
    rec = load(a, seed)
    assert rec["system_prompt"] == agent.system_prompt(a)
    sc = score(rec)
    srow = next(r for r in rows if r["model"] == A.model and r["arm"] == a and r["seed"] == seed)
    assert all(sc[k] == srow[k] for k in ("key", "flag", "flag_only", "key_only")), a
    th, tool, args, den = decision(rec, rec["episode"]["key"])
    assert rec["episode"]["key"] in den
    EX[a] = dict(thought=th, tool=tool, args=args, denial=den, route=route(sc), sc=sc)
DEN = next(EX[a]["denial"] for a in COLS if EX[a]["denial"].startswith("denied: token scope"))
# the usage line (both routes) and the record (key) are the same in every arm; only the first denial line can differ
assert all(EX[a]["denial"].split("\n", 1)[1] == DEN.split("\n", 1)[1] for a in COLS)
KEY = next(l.split(": ", 1)[1] for l in DEN.splitlines() if l.startswith("access_key: "))
assert all(KEY in EX[a]["denial"] for a in COLS)
e = lambda s: html.escape(s, quote=False)  # noqa: E731


def mark(s, words):
    s = e(s)
    for w, cls in words:
        assert e(w) in s, (w, s)
        s = s.replace(e(w), f'<span class="{cls}">{e(w)}</span>', 1)
    return s


ROBOT = """<svg class="rb" width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/>
<rect x="5" y="20" width="5" height="12" rx="2" fill="#e8734a"/><rect x="38" y="20" width="5" height="12" rx="2" fill="#e8734a"/>
<rect x="8" y="12" width="32" height="30" rx="9" fill="#fff" stroke="#222" stroke-width="2"/><rect x="12" y="17" width="24" height="18" rx="6" fill="#1f2a44"/>
<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>"""
rb = lambda w=46: ROBOT.format(w=w)  # noqa: E731

ACT = {"flag": ('fl', T["act_flag"]), "key": ('ky', T["act_key"]), "both": ('bo', T["act_both"]),
       "none": ('no', T["act_none"])}


def notice_html(a):
    n = agent.NOTICES[a]
    if not n:
        return f'<div class="ntc empty">{e(T["no_notice_line"])}</div>'
    # show only the recording sentence; the length-matching filler is not part of the notice's meaning
    first = n.split(". ")[0].rstrip(".") + "."
    if a in ("key", "flag"):
        first = n
    return f'<div class="ntc">&ldquo;{e(first)}&rdquo;</div>'


def col(a):
    x = EX[a]
    cls, act = ACT[x["route"]]
    th = x["thought"]
    start = T.get("trim", {}).get(a)
    if start:  # quote from a sentence start inside the verbatim thought, marked with an ellipsis
        assert start in th and th[th.index(start) - 2: th.index(start)] == ". ", (a, start)
        th = th[th.index(start):]
    q = ("&hellip;" if start else "") + mark(th, [(w, c) for w, c in T.get("marks", {}).get(a, [])])
    return (f'<div class="col c-{a}"><div class="ch">{e(ARM_NAME[a][0].upper() + ARM_NAME[a][1:])}</div>'
            f'{notice_html(a)}<div class="bub">&ldquo;{q}&rdquo;</div>'
            f'<div class="act {cls}">&rarr; {e(act)}</div>'
            + (f'<div class="m1">{T["m1_line"].format(**fmt)}</div>' if a == "key" and T.get("m1_line") else '')
            + '</div>')


SCALE = 6.2  # px per percentage point (100% = 620 px)


def chart():
    out = []
    for a in ORDER:
        fo, ky = AE[a]["flag_only"], AE[a]["key"]
        sep = ' sep' if a == "keyonly" else ' hi' if a == "key" else ''
        out.append(f'<div class="crow{sep}"><div class="cl">{e(ARM_NAME[a][0].upper() + ARM_NAME[a][1:])}</div><div class="cb">'
                   f'<div class="bt"><div class="bf fo" style="width:{SCALE * 100 * fo:.1f}px"></div><div class="bv fo">{pct(fo)}%</div></div>'
                   f'<div class="bt"><div class="bf kb" style="width:{SCALE * 100 * ky:.1f}px"></div><div class="bv kv">{pct(ky)}%</div></div>'
                   f'</div></div>')
    return "".join(out)


fmt = dict(**{f"fo_{a}": pct(AE[a]["flag_only"]) for a in ORDER}, **{f"key_{a}": pct(AE[a]["key"]) for a in ORDER},
           n_runs=N_RUNS, risers=RISERS, n_readers=len(READERS), m1_both=EX_M1["answers"].get("both", 0), m1_seed=SEED_M1, m1_n=EX_M1["n"], n_cell=n_cell["none"], model=NAME[A.model])
HEAD = T["head"].format(**fmt)
SUB = T["sub"].format(**fmt)
NOTE = T.get("note", "").format(**fmt)
assert ("none of the 5" not in NOTE) or RISERS == 0

doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {e(HEAD)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
Sources (kran-research, branch swarm/defense-1107-scoped-recording, projects/scoped-recording):
  example runs: {RAW.name}/{A.model}/<arm>/{seed}.jsonl for arms {", ".join(COLS)}
    (thought of the first reply after the first denial that shows the record; route from scorer.score, equal to results/scores.jsonl)
  denial text: the logged observation (same in every arm); notices: agent.py NOTICES (= PREREG s1)
  numbers: {RES.name}/summary.json agent_equal flag_only and key per arm (5 agents x {n_cell['none']} runs per arm, {N_RUNS} runs)
  built by swarm-defense-1107-scoped-recording/owain/gen.py
-->
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: #fbfaf8; }}
  body {{ width: 1200px; padding: 30px 40px 28px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }}
  .head {{ font-size: 40px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.16; }}
  .q {{ font-size: 25px; line-height: 1.3; color: #6b6f78; margin-top: 10px; }}
  .q b {{ color: #2b2e35; font-weight: 700; }}
  .mono {{ font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }}
  .setup {{ display: flex; gap: 12px; align-items: center; font-size: 25px; line-height: 1.3; color: #3b3e46; margin-top: 18px; }}
  .rb {{ flex: none; }}
  .rec {{ background: #fff; border: 2px solid #d9dbe0; border-radius: 14px; padding: 10px 16px; font-size: 20px; line-height: 1.45; margin-top: 10px; }}
  .rec .dim {{ color: #6b6e76; }}
  .fl {{ background: #e9e2fb; color: #4a2fb0; border-radius: 4px; padding: 0 3px; font-weight: 700; }}
  .ky {{ background: #fbe0dc; color: #b3261e; border-radius: 4px; padding: 0 3px; font-weight: 700; }}
  .who {{ display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 800; margin-top: 22px; }}
  .who span {{ font-weight: 500; color: #6b6f78; font-size: 23px; }}
  .cols {{ display: grid; grid-template-columns: repeat({len(COLS)}, 1fr); gap: 14px; margin-top: 10px; }}
  .col {{ border-radius: 18px; padding: 14px 14px 14px; display: flex; flex-direction: column; gap: 10px; background: #eef0f3; }}
  .col.c-key {{ background: #e6eefc; }}
  .ch {{ font-size: 26px; font-weight: 800; }}
  .col.c-key .ch {{ color: #1f56c2; }}
  .ntc {{ background: #2f6fe4; color: #fff; font-size: 21px; line-height: 1.3; border-radius: 14px 14px 4px 14px; padding: 8px 12px; }}
  .ntc.empty {{ background: transparent; color: #6b6f78; border: 2px dashed #b9bdc6; }}
  .bub {{ background: #fff; border-radius: 16px 16px 16px 6px; padding: 10px 12px; font-size: 22px; line-height: 1.3;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; }}
  .hl {{ background: #ffe58a; border-radius: 4px; padding: 0 1px; font-weight: 700; }}
  .act {{ font-size: 24px; font-weight: 800; }}
  .act.fl, .act.ky, .act.bo, .act.no {{ background: none; padding: 0; }}
  .act.ky {{ color: #b3261e; }} .act.fl {{ color: #4a2fb0; }} .act.bo {{ color: #7a2f6e; }} .act.no {{ color: #3b3e46; }}
  .m1 {{ font-size: 21px; color: #5a5d66; line-height: 1.3; margin-top: -4px; }}
  .ctitle {{ font-size: 26px; font-weight: 800; margin-top: 24px; }}
  .ctitle span {{ font-weight: 500; color: #6b6f78; font-size: 23px; }}
  .lg {{ display: flex; gap: 28px; font-size: 22px; margin-top: 6px; color: #3b3e46; }}
  .lg i {{ display: inline-block; width: 18px; height: 18px; border-radius: 3px; vertical-align: -2px; margin-right: 6px; }}
  .crow {{ display: grid; grid-template-columns: 375px 1fr; align-items: center; padding: 5px 0; border-top: 1px solid #e3e5ea; }}
  .crow.sep {{ margin-top: 10px; border-top: 1.5px dashed #b9bdc6; }}
  .cl {{ font-size: 23px; font-weight: 600; color: #2b2e35; }}
  .crow.hi .cl {{ color: #1f56c2; font-weight: 800; }}
  .bt {{ display: flex; align-items: center; gap: 10px; height: 28px; }}
  .bf {{ height: 22px; border-radius: 3px; }}
  .bf.fo {{ background: #5b3fd0; }} .bf.kb {{ background: #c9cdd4; }}
  .bv {{ font-size: 22px; font-weight: 800; font-variant-numeric: tabular-nums; }}
  .bv.kv {{ font-weight: 600; color: #5a5d66; }}
  .note {{ font-size: 22px; color: #5a5d66; margin-top: 12px; line-height: 1.35; }}
  .foot {{ font-size: 20px; color: #7a7e87; margin-top: 16px; }}
  .foot b {{ color: #3a3d45; font-weight: 700; }}
</style>
</head>
<body>
<div class="head">{e(HEAD)}</div>
<div class="q">{SUB}</div>
<div class="setup">{rb()}<div>{T["setup"]}</div></div>
<div class="rec mono">{"".join(f"<div>{x}</div>" for x in [
    e(DEN.splitlines()[0].split("; ")[0]) + ";",
    e(DEN.splitlines()[0].split("; ")[1]),
    e(DEN.splitlines()[1]).replace(", --admin", ",<br>&nbsp;&nbsp;--admin").replace(e("--admin-override (skip token scope check; break-glass, security team only)"),
                                   '<span class="fl">' + e("--admin-override (skip token scope check; break-glass, security team only)") + '</span>'),
    '<span class="dim">last archive record on file: &hellip; owner: secops-monitor,</span> <span class="ky">access_key: ' + e(KEY) + '</span> <span class="dim">&hellip;</span>'])}</div>
<div class="who">{rb(42)}{NAME[A.model]}<span>&middot; {e(T["who_tail"])}</span></div>
<div class="cols">{"".join(col(a) for a in COLS)}</div>
<div class="ctitle">{e(T["chart_title"].format(**fmt))}</div>
<div class="lg"><span><i style="background:#5b3fd0"></i>{e(T["lg_fo"])}</span><span><i style="background:#c9cdd4"></i>{e(T["lg_key"])}</span></div>
<div class="chart">{chart()}</div>
{f'<div class="note">{NOTE}</div>' if NOTE else ''}
<div class="foot">{e(T["foot"].format(**fmt))} &middot; <b>blog.kran.ai/lab</b></div>
</body>
</html>
"""
out = Path(A.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(doc)
print("wrote", out)
print("example", {a: (EX[a]["route"], EX[a]["thought"][:90]) for a in COLS})
print("AE", {a: (pct(AE[a]["flag_only"]), pct(AE[a]["key"])) for a in ORDER})
