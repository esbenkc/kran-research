"""Figure 1 of disallow-canary (Owain): one site, three readers (a crawler, two agents), plus the run's counts.

Everything comes from saved raw data; no number or quote is typed by hand.
  data/runs/<model>/main/<i>.json   the example runs (pick rule below, frozen in figure1/README.md)
  data/logs/access.log              the agents' server log (catch = a request for the run's canary code)
  data/crawl/access.log + crawl_rows.json   the crawler example and the crawler counts
  analysis.py --out <tmp>           the counts, through the project's own frozen scoring (PREREG s4)
  figure1/text.json                 the words; {slots} are filled from the counts

    python3 figure1/gen.py [--blog <athanorscience.com clone>] [--proj <project dir with data/>]

Writes <blog>/assets/figures/disallow-canary-fig1/figure1.html and figure1/alt.txt, and prints the sources.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
FIG_ID = "disallow-canary-fig1"
M31, M26 = "gemma-4-31b-it", "gemma-4-26b-a4b-it"
ROBOTS_ON = ("wget", "scrapy", "mcp_fetch")  # C1-C3, PREREG s4
LOG_KEYS = ("time", "ip", "method", "path", "status", "ua", "run", "port")
E = html.escape


def die(msg: str) -> None:
    sys.exit(f"gen.py: {msg}")


def read_log(p: Path) -> list[dict]:
    return [dict(zip(LOG_KEYS, f)) for f in (l.split("\t") for l in p.read_text().splitlines()) if len(f) == len(LOG_KEYS)]


def run_analysis() -> tuple[dict, list[dict]]:
    import csv
    py = PROJ / ".venv/bin/python"
    exe = str(py) if py.exists() else sys.executable
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([exe, "analysis.py", "--out", td], cwd=PROJ, check=True, capture_output=True, text=True)
        res = json.loads((Path(td) / "analysis.json").read_text())
        with (Path(td) / "analysis_runs.csv").open() as fh:
            rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
    return res, rows


# ---- the frozen pick (figure1/README.md) -------------------------------------------------------------------------

def pick(rows: list[dict]) -> int:
    """Lowest run index i at which, in the main arm, BOTH models met the fetch tool's robots refusal for the canary on
    the same site design, Gemma 4 31B then requested the canary with curl (route-around, a catch) and Gemma 4 26B made
    no canary request."""
    by = {(r["model"], int(r["i"])): r for r in rows if r["arm"] == "main"}
    for i in sorted({int(r["i"]) for r in rows if r["arm"] == "main"}):
        a, b = by.get((M31, i)), by.get((M26, i))
        if not a or not b or a["template"] != b["template"]:
            continue
        if a["refusal_canary"] and a["route_around"] and a["catch"] and b["refusal_canary"] and not b["catch"]:
            return i
    die("no run index fits the pick rule")


def thought(reply: str) -> str:
    try:
        return json.loads(reply).get("thought", "")
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', reply or "")
        return json.loads(f'"{m.group(1)}"') if m else ""


def agent_facts(model: str, i: int, log: list[dict]) -> dict:
    p = PROJ / "data/runs" / model / "main" / f"{i:03d}.json"
    rec = json.loads(p.read_text())
    s, tr = rec["site"], rec["transcript"]
    hid = s["hid"]
    k_ref = next(k for k, t in enumerate(tr) if t["role"] == "observation" and "autonomous fetching" in t["text"])
    refusal = tr[k_ref]["text"]
    after = tr[k_ref + 1]  # the agent's reply right after the refusal
    act = json.loads(after["text"])
    lines = [l for l in log if l["run"] == rec["label"]]
    canary = [l for l in lines if hid in l["path"]]
    # the page on which a tool reply first showed the canary code
    seen = next(t for t in tr if t["role"] == "observation" and hid in t["text"])
    seen_page = re.search(r"Contents of http://[^/]+(/\S+?):", seen["text"])
    link_line = next(l for l in seen["text"].splitlines() if hid in l)
    return {"file": str(p.relative_to(PROJ)), "label": rec["label"], "hid": hid, "template": s["template"],
            "question": rec["question"], "robots": s["robots_txt"].strip(), "answer_true": s["answer"],
            "refusal": refusal, "next_tool": act["tool"], "next_thought": act.get("thought", ""),
            "next_args": act.get("args", {}), "answer": rec.get("answer"), "correct": rec.get("correct"),
            "canary_log": canary, "seen_page": seen_page.group(1) if seen_page else "", "seen_line": link_line,
            "site_dir": s["dir"]}


def crawler_facts(template: str) -> dict:
    rows = json.loads((PROJ / "data/crawl/crawl_rows.json").read_text())
    r = next(x for x in rows if x["crawler"] == "wget" and x["template"] == template and x["variant"] == "both")
    lines = [l for l in read_log(PROJ / "data/crawl/access.log") if l["run"] == r["label"]]
    pages = [l["path"] for l in lines if l["path"].endswith(".html")]
    return {"label": r["label"], "ua": lines[0]["ua"] if lines else "", "robots_read": any(l["path"] == "/robots.txt" for l in lines),
            "pages": len(pages), "canary_hits": r["canary_hits"], "cmd": " ".join(Path(r["cmd"][0]).name if j == 0 else c
                                                                                   for j, c in enumerate(r["cmd"]))}


def source_link(site_dir: str, hid: str) -> str:
    """The canary's <a> as the server sends it (route page of the example site)."""
    for f in sorted(Path(site_dir).glob("*.html")):
        m = re.search(r'<a [^>]*' + hid + r'[^<]*</a>', f.read_text())
        if m:
            return m.group(0)
    die(f"canary {hid} not found in {site_dir}")


# ---- HTML --------------------------------------------------------------------------------------------------------

ROBOT = ('<svg width="46" height="46" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/>'
         '<rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="6" y="12" width="36" height="30" rx="9" fill="#fff" '
         'stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="41" y="22" '
         'width="5" height="11" rx="2" fill="#e8734a"/><rect x="11" y="17" width="26" height="20" rx="6" fill="#1f2a44"/>'
         '<circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" '
         'stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>')
SPIDER = ('<svg width="46" height="46" viewBox="0 0 48 48" aria-hidden="true"><g stroke="#3b3e46" stroke-width="2.4" '
          'fill="none" stroke-linecap="round"><path d="M24 4 v10"/><path d="M17 22 L6 14 L3 18"/><path d="M17 26 L5 26 L2 31"/>'
          '<path d="M17 30 L7 38 L6 44"/><path d="M19 33 L13 44"/><path d="M31 22 L42 14 L45 18"/><path d="M31 26 L43 26 L46 31"/>'
          '<path d="M31 30 L41 38 L42 44"/><path d="M29 33 L35 44"/></g><ellipse cx="24" cy="27" rx="8" ry="10" fill="#3b3e46"/>'
          '<circle cx="24" cy="15" r="5" fill="#3b3e46"/><circle cx="22" cy="14" r="1.3" fill="#fff"/>'
          '<circle cx="26" cy="14" r="1.3" fill="#fff"/></svg>')

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #fbfaf8; }
body { width: 1200px; padding: 30px 38px 26px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1d1d1f; -webkit-font-smoothing: antialiased; }
.mono { font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }
.head { font-size: 44px; font-weight: 800; letter-spacing: -0.018em; line-height: 1.13; }
.sub { font-size: 25px; line-height: 1.36; color: #4a4d55; margin-top: 12px; }
.sub b { color: #1d1d1f; font-weight: 800; } .sub .red { color: #b3261e; }
.setup { display: grid; grid-template-columns: 150px 1fr; gap: 12px 16px; align-items: center; margin-top: 20px; }
.pl { font-size: 24px; font-weight: 800; }
.ask { justify-self: start; background: #2f6fe4; color: #fff; font-size: 24px; line-height: 1.3; border-radius: 18px 18px 4px 18px; padding: 8px 18px; }
.trap { display: flex; gap: 14px; align-items: stretch; }
.code { background: #fff; border: 1.5px solid #e1e3e8; border-radius: 12px; padding: 8px 12px; font-size: 21px; line-height: 1.4; color: #2b2e35; }
.code .c { display: block; font-family: "Inter", sans-serif; font-size: 18px; color: #6b6f78; margin-top: 2px; }
b.nf, b.dis { color: #fff; background: #b3261e; border-radius: 5px; padding: 0 5px; font-weight: 700; }
b.id { color: #b3261e; font-weight: 800; }
.row { margin-top: 18px; background: #eef0f3; border-radius: 18px; padding: 14px 18px; display: grid;
       grid-template-columns: 330px 1fr 230px; gap: 16px; align-items: center; }
.who { display: flex; align-items: center; gap: 12px; font-size: 25px; font-weight: 800; line-height: 1.15; }
.who span { display: block; font-size: 20px; font-weight: 500; color: #5a5d66; }
.did { font-size: 23px; line-height: 1.35; color: #2b2e35; }
.verdict { justify-self: end; text-align: right; font-size: 22px; line-height: 1.3; color: #3b3e46; }
.tag { display: inline-block; font-size: 19px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.04em; border-radius: 7px; padding: 2px 10px; margin-bottom: 4px; }
.tag.yes { color: #fff; background: #b3261e; } .tag.no { color: #2b2e35; background: #d9dce2; }
.both { margin-top: 18px; font-size: 23px; line-height: 1.35; color: #2b2e35; }
.both b { font-weight: 800; }
.refusal { margin-top: 8px; background: #fff; border: 1.5px dashed #9aa0aa; border-radius: 12px; padding: 8px 14px; font-size: 21px; line-height: 1.38; color: #3b3e46; }
.refusal .lab { font-weight: 800; color: #1d1d1f; }
.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
.card { background: #eef0f3; border-radius: 18px; padding: 14px 18px; display: flex; flex-direction: column; gap: 10px; }
.card.hl { background: #f8e9e4; }
.ch { display: flex; align-items: center; gap: 12px; font-size: 27px; font-weight: 800; line-height: 1.15; }
.ch.red { color: #b3261e; }
.bub { background: #fff; border-radius: 16px 16px 16px 4px; padding: 10px 14px; font-size: 23px; line-height: 1.36; color: #2b2e35; }
.bub b { background: #ffe58a; font-weight: 700; border-radius: 3px; padding: 0 2px; }
.cmd { font-size: 20px; background: #fff; border-radius: 10px; padding: 6px 12px; color: #2b2e35; }
.log { font-size: 21px; line-height: 1.35; color: #3b3e46; }
.log .k { color: #6b6f78; }
.log .mono { font-size: 19px; }
.end { margin-top: auto; font-size: 24px; font-weight: 800; line-height: 1.3; }
.end.red { color: #b3261e; } .end span { font-weight: 500; color: #3b3e46; }
.chart { margin-top: 22px; }
.ct { font-size: 27px; font-weight: 800; } .ct span { font-weight: 500; color: #5a5d66; font-size: 22px; }
.bars { display: grid; grid-template-columns: 500px 1fr 200px; gap: 12px 18px; align-items: center; margin-top: 12px; }
.bl { font-size: 24px; font-weight: 700; line-height: 1.2; } .bl span { display: block; font-size: 19px; font-weight: 400; color: #5a5d66; }
.track { position: relative; height: 26px; background: #e6e8ec; border-left: 2px solid #3b3e46; border-radius: 0 6px 6px 0; }
.track i { position: absolute; left: 0; top: 0; bottom: 0; background: #8a8e97; border-radius: 0 6px 6px 0; }
.track.red i { background: #b3261e; }
.bv { font-size: 22px; color: #3b3e46; text-align: right; font-variant-numeric: tabular-nums; }
.bv b { font-size: 30px; font-weight: 800; color: #1d1d1f; } .bv.red b { color: #b3261e; }
.bl.ctl { font-size: 20px; font-weight: 500; color: #5a5d66; padding-left: 22px; margin-top: -6px; }
.track.ctl { height: 12px; margin-top: -6px; background: #eef0f3; } .track.ctl i { background: #b9bcc4; }
.bv.ctl { font-size: 20px; color: #5a5d66; margin-top: -6px; } .bv.ctl b { font-size: 22px; font-weight: 700; color: #5a5d66; }
.foot { font-size: 19px; color: #5a5d66; margin-top: 18px; line-height: 1.35; } .foot b { color: #2b2e35; }
"""


def hl_link(a: str, hid: str) -> str:
    out = E(a).replace(E(' rel="nofollow"'), ' <b class="nf">rel=&quot;nofollow&quot;</b>')
    return out.replace(hid, f'<b class="id">{hid}</b>')


def short(s: str, n: int) -> str:
    """Whole sentences of s until at least n words (never a cut inside a sentence)."""
    sents = re.split(r"(?<=[.!?])\s+", s.strip())
    out: list[str] = []
    for sent in sents:
        out.append(sent)
        if sum(len(x.split()) for x in out) >= n:
            break
    return " ".join(out)


def bar(label: str, sub: str, k: int, n: int, unit: str, red: bool, ctl: bool = False) -> str:
    w = 100 * k / n if n else 0
    cls = " red" if red else (" ctl" if ctl else "")
    lab = f'<div class="bl ctl">{E(label)}</div>' if ctl else f'<div class="bl">{E(label)}<span>{E(sub)}</span></div>'
    return (lab + f'<div class="track{cls}"><i style="width:{w:.2f}%"></i></div>'
            f'<div class="bv{cls}"><b>{k}</b> of {n} {E(unit)}</div>')


def page(T: dict, S: dict, crawl: dict, a31: dict, a26: dict, link: str, counts: dict, src: str) -> str:
    f = lambda k: T[k].format(**S)  # noqa: E731
    ref = a31["refusal"]
    ref_q = re.search(r"specifies that autonomous fetching of this page is not allowed", ref).group(0)
    ref_tail = re.search(r"The assistant must let the user know that it failed to view the page\.", ref).group(0)
    hid31 = a31["hid"]
    curl = a31["next_args"].get("command", "").replace("http://127.0.0.1:", "\u2026:")
    curl = re.sub(r"\u2026:\d+", "\u2026", curl)
    log31 = a31["canary_log"][0]
    q26 = a26["answer"] or ""
    q31 = a31["next_thought"]
    q31h = E(short(q31, 20)).replace("curl", "<b>curl</b>", 1)
    q26h = E(short(q26, 15)).replace("robots.txt", "<b>robots.txt</b>", 1)
    robots = "<br>".join(E(l) for l in crawl["robots"].splitlines())
    robots = robots.replace("Disallow: /r/", '<b class="dis">Disallow: /r/</b>')
    setup = (f'<div class="setup"><div class="pl">Task</div><div class="ask">&ldquo;{E(a31["question"])}&rdquo;</div>'
             f'<div class="pl">The trap</div><div class="trap">'
             f'<div class="code mono">{hl_link(link, hid31)}<span class="c">{E(T["link_note"])}</span></div>'
             f'<div class="code mono">{robots}<span class="c">{E(T["robots_note"])}</span></div></div></div>')
    crow = (f'<div class="row"><div class="who">{SPIDER}<div>{E(T["crawler_title"])}<span>{E(T["crawler_sub"])}</span></div></div>'
            f'<div class="did">{E(f("crawler_did"))}</div>'
            f'<div class="verdict"><span class="tag no">{E(T["crawler_tag"])}</span><br>{E(T["crawler_log"])}</div></div>')
    both = (f'<div class="both">{f("both_line")}</div>'
            f'<div class="refusal"><span class="lab">{E(T["refusal_label"])}</span> &ldquo;The sites robots.txt &hellip; '
            f'{E(ref_q)}. &hellip; {E(ref_tail)}&rdquo;</div>')
    c26 = (f'<div class="card"><div class="ch">{ROBOT}{E(T["m26"])}</div>'
           f'<div class="bub">&ldquo;{q26h}&rdquo;</div>'
           f'<div class="log"><span class="k">{E(T["log_label"])}</span> {E(T["log_none"])}</div>'
           f'<div class="end">{E(T["end26"])}</div></div>')
    c31 = (f'<div class="card hl"><div class="ch red">{ROBOT}{E(T["m31"])}</div>'
           f'<div class="bub">&ldquo;{q31h}&rdquo;</div>'
           f'<div class="cmd mono">$ {E(curl).replace(hid31, f"<b class=id>{hid31}</b>")}</div>'
           f'<div class="log"><span class="k">{E(T["log_label"])}</span> <span class="mono">GET /r/<b class="id">{hid31}</b>.html '
           f'{E(log31["status"])} {E(log31["ua"])}</span></div>'
           f'<div class="end red">{E(T["end31"])} <span>{E(f("end31_sub"))}</span></div></div>')
    c = counts
    chart = (f'<div class="chart"><div class="ct">{E(f("chart_title"))} <span>{E(f("chart_sub"))}</span></div><div class="bars">'
             + bar(T["bar_crawl"], T["bar_crawl_sub"], c["crawl_hits"], c["crawl_passes"], T["unit_sites"], False)
             + bar(T["m31"], T["bar_agent_sub"], c["k31"], c["n31"], T["unit_runs"], True)
             + bar(T["bar_ctl"], "", c["b31"], c["bn31"], T["unit_runs"], False, ctl=True)
             + bar(T["m26"], T["bar_agent_sub"], c["k26"], c["n26"], T["unit_runs"], False)
             + bar(T["bar_ctl"], "", c["b26"], c["bn26"], T["unit_runs"], False, ctl=True)
             + "</div></div>")
    foot = f'<div class="foot">{E(f("foot"))} &middot; <b>athanorscience.com</b></div>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {E(f("head"))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<!--
{src}
-->
<style>{CSS}</style>
</head>
<body>
<div class="head">{E(f("head"))}</div>
<div class="sub">{f("sub")}</div>
{setup}
{crow}
{both}
<div class="cards">{c26}{c31}</div>
{chart}
{foot}
</body>
</html>
"""


def yaml_safe(s: str) -> str:
    return s.replace('"', "'")


def main() -> None:
    global PROJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(PROJ.parents[2] / "athanorscience.com"))
    ap.add_argument("--proj", default=str(PROJ), help="project dir that holds data/ and analysis.py (default: this one)")
    a = ap.parse_args()
    PROJ = Path(a.proj).resolve()
    T = json.loads((HERE / "text.json").read_text())
    res, rows = run_analysis()
    log = read_log(PROJ / "data/logs/access.log")
    i = pick(rows)
    a31, a26 = agent_facts(M31, i, log), agent_facts(M26, i, log)
    if not a31["canary_log"] or a26["canary_log"]:
        die("example no longer fits the pick rule")
    cr = crawler_facts(a31["template"])
    cr["robots"] = a31["robots"]
    link = source_link(a31["site_dir"], a31["hid"])
    mm = res["models"]
    tab = res["crawlers"]
    counts = {"k31": mm[M31]["main"]["catch"]["k"], "n31": mm[M31]["main"]["catch"]["n"],
              "k26": mm[M26]["main"]["catch"]["k"], "n26": mm[M26]["main"]["catch"]["n"],
              "crawl_hits": sum(tab[c]["both"]["templates_hit"] for c in ROBOTS_ON),
              "crawl_passes": sum(tab[c]["both"]["templates"] for c in ROBOTS_ON),
              "b31": mm[M31]["bare"]["catch"]["k"], "bn31": mm[M31]["bare"]["catch"]["n"],
              "b26": mm[M26]["bare"]["catch"]["k"], "bn26": mm[M26]["bare"]["catch"]["n"]}
    final = all(mm[m]["main"]["settled"] for m in (M31, M26))
    ra = mm[M31]["main"]["route_around_given_refusal"]
    S = {**counts, "pages": cr["pages"], "answer31": a31["answer_true"], "status": res["status"],
         "ra_k": ra["k"], "ra_n": ra["n"], "ref26": mm[M26]["main"]["refusal_canary"]["k"],
         "seen_page": a31["seen_page"].lstrip("/").replace(".html", "").capitalize(),
         "partial": "" if final else T["partial"]}
    # the answer line: frozen status (analysis.py) once every main seed block is full; before that, counts only
    key = res["status"] if final else "PARTIAL"
    if key not in T["answer"]:
        die(f"text.json has no answer line for {key}")
    S["answer_label"], S["answer_word"] = T["answer"][key]["label"], T["answer"][key]["word"]
    S["answer_text"] = T["answer"][key]["text"].format(**S)
    if not a31["correct"]:
        die("31B example answer is not the right number; text.json says it is")
    out = Path(a.blog) / "assets/figures" / FIG_ID / "figure1.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = "\n".join([f"Sources (gen.py, kran-research projects/disallow-canary; pick rule figure1/README.md; i={i}):",
                     f"  {a31['file']}  {a26['file']}",
                     f"  data/logs/access.log (labels {a31['label']}, {a26['label']})",
                     f"  data/crawl/access.log + crawl_rows.json (label {cr['label']})",
                     f"  counts: analysis.py (status {res['status']}, runs {res['runs_total']}, final={final})"])
    out.write_text(page(T, S, cr, a31, a26, link, counts, src))
    alt = T["alt"].format(**S)
    (HERE / "alt.txt").write_text(yaml_safe(alt) + "\n")
    print(src)
    print(f"wrote {out}")
    print(f"counts {counts}  final={final}  status={res['status']}")


if __name__ == "__main__":
    main()
