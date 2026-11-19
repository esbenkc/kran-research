"""Build Figure 1 (figure1.html) for seen-then-followed from the real runs. No model calls.

    python3 gen.py OUT_DIR [--kmax K] [--analysis PATH]

Sources (kran-research, branch swarm/defense-1118-seen-then-followed, projects/seen-then-followed):
  example runs: raw/main/<model>/<cell>/<i>.json chosen by pick.py (rule #25199); tool outputs and thoughts are verbatim
  server log: logs/access.log (follow = a request whose path holds the run's id)
  numbers: recomputed here with score.py's definitions and the PREREG s4 formula; asserted equal to analysis.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pick  # noqa: E402

PROJ = pick.PROJ
A, B = "gemma-4-31b-it", "gemma-4-26b-a4b-it"
NAME = {A: "Gemma 4 31B", B: "Gemma 4 26B-A4B"}
SHORT = {A: "31B", B: "26B"}
TOOLNAME = {"raw": "raw fetch", "mcpfetch": "MCP fetch tool", "lcpw": "Playwright toolkit"}
TOOLSUB = {"mcpfetch": "shows a page's main text as markdown",
           "lcpw": "LangChain: page words, plus a list of link addresses",
           "raw": "plain HTTP: returns the page code"}


def cellstats(rs, tool, model):
    c = [s for s in rs if s["tool"] == tool and s["model"] == model]
    n, r = len(c), sum(s["reached"] for s in c)
    f_r = sum(s["followed"] and s["reached"] for s in c)
    return {"n": n, "reached": r, "follow_r": f_r, "catch": sum(s["followed"] for s in c)}


def law(rs):
    """PREREG s4: pred(H,M) = reach_H(other model's fit cell) x follow_M pooled over M's fit cells."""
    fit = {A: ["raw", "lcpw"], B: ["raw", "mcpfetch"]}
    out = {}
    for tool, model, other in (("mcpfetch", A, B), ("lcpw", B, A)):
        rc = cellstats(rs, tool, other)
        fc = [cellstats(rs, h, model) for h in fit[model]]
        reach = rc["reached"] / rc["n"]
        follow = sum(x["follow_r"] for x in fc) / sum(x["reached"] for x in fc)
        obs = cellstats(rs, tool, model)
        out[(tool, model)] = {"reach": reach, "reach_k": (rc["reached"], rc["n"]), "follow": follow,
                              "follow_k": (sum(x["follow_r"] for x in fc), sum(x["reached"] for x in fc)),
                              "pred": reach * follow, "catch": obs["catch"] / obs["n"],
                              "catch_k": (obs["catch"], obs["n"]), "own_reach": obs["reached"] / obs["n"]}
    return out


def esc(s):
    return html.escape(s, quote=False)


def excerpt(text, hid, anchor, width=170):
    """A verbatim window of a tool output around the id (or the anchor words), id and anchor highlighted."""
    k = text.find(hid)
    if k < 0:
        k = text.find(anchor)
    k = max(0, k)
    a, b = max(0, k - width), min(len(text), k + width)
    seg = text[a:b]
    pre, post = ("…" if a > 0 else ""), ("…" if b < len(text) else "")
    s = esc(seg)
    s = s.replace(esc(hid), f'<b class="id">{esc(hid)}</b>')
    s = s.replace(esc(anchor), f'<b class="anc">{esc(anchor)}</b>')
    return pre + s + post


def example(f):
    rec = json.loads((PROJ / f).read_text())
    hid, anchor = rec["site"]["hid"], rec["site"]["anchor_text"]
    acts = rec["actions"]
    tr = rec["transcript"]
    obs = [t for t in tr if t["role"] == "observation"]
    asst = [t for t in tr if t["role"] == "assistant"]
    step_reach = next((int(a["step"]) for a in acts if str(a.get("hid_in_obs")) == "True"), None)
    log = [l.split("\t") for l in (PROJ / "logs" / "access.log").read_text().splitlines()]
    hits = [p for p in log if len(p) >= 5 and hid in p[3]]
    return {"rec": rec, "hid": hid, "anchor": anchor, "acts": acts, "obs": obs, "asst": asst,
            "step_reach": step_reach, "log": hits}


ROBOT = ('<svg class="rb" width="{s}" height="{s}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" '
         'fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="6" y="12" width="36" height="30" '
         'rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" '
         'fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="11" y="17" '
         'width="26" height="20" rx="6" fill="#1f2a44"/><circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" '
         'cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>')


def thought(t):
    try:
        return json.loads(t["text"])["thought"]
    except Exception:
        m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', t["text"])
        return json.loads('"' + m.group(1) + '"') if m else t["text"]


def pct(x):
    return f"{round(100 * x)}%"


def bar(v, W=430):
    """Predicted (tick, +-10 band) vs measured (dot) on a 0-100% track."""
    p, c = v["pred"], v["catch"]
    lo, hi = max(0, p - 0.10), min(1, p + 0.10)
    inside = abs(c - p) <= 0.10
    return (f'<div class="track" style="width:{W}px"><div class="axis0"></div>'
            f'<div class="band" style="left:{lo*W:.0f}px;width:{(hi-lo)*W:.0f}px"></div>'
            f'<div class="tick" style="left:{p*W:.0f}px"></div>'
            f'<div class="mdot {"in" if inside else "out"}" style="left:{c*W:.0f}px"></div>'
            f'<div class="tl0">0%</div><div class="tl1">100%</div></div>')


def card(ex, v, tool, model, steps):
    rows = "".join(f'<div class="st"><div class="sl">{lab}<div class="tag {cls}">{tag}</div></div><div class="sb">{body}</div></div>'
                   for lab, body, tag, cls in steps)
    cn = v["catch_k"][1]
    other = NAME[B if model == A else A]
    ok = abs(v["catch"] - v["pred"]) <= 0.10
    return (f'<div class="card"><div class="ch">{ROBOT.format(s=40)}<div><b>{NAME[model]}</b> + {TOOLNAME[tool]}'
            f'<span>{TOOLSUB[tool]}</span></div></div>{rows}'
            f'<div class="law"><div class="eq">Predicted <b>{pct(v["pred"])}</b> = reach <b>{pct(v["reach"])}</b> <span>({SHORT[B if model == A else A]}, this tool)</span>'
            f' &times; follow rate <b>{pct(v["follow"])}</b> <span>({SHORT[model]}, other tools)</span></div>'
            f'<div class="meas">Measured: <b class="{"in" if ok else "out"}">{pct(v["catch"])}</b> of {cn} runs requested the link</div>'
            f'{bar(v)}</div></div>')


def logline(ex):
    p = ex["log"][0]
    assert p[3].endswith(f'/r/{ex["hid"]}.html'), p[3]
    path = f'…/r/<b class="id">{esc(ex["hid"])}</b>.html' 
    return f'<div class="log"><span>server log</span> <span class="mono">{esc(p[2])} {path} {esc(p[4])}</span></div>'


def req_step(ex):
    return next(n for n, x in enumerate(ex["acts"]) if ex["hid"] in str(x.get("url", "")) or ex["hid"] in str(x.get("args", "")))


def steps_lcpw(ex):
    obs, asst, hid, anc = ex["obs"], ex["asst"], ex["hid"], ex["anchor"]
    j = next(k for k, o in enumerate(obs) if hid in o["text"])
    assert ex["acts"][j]["tool"] == "extract_hyperlinks", ex["acts"][j]["tool"]
    words = next(o for o in obs if anc in o["text"] and hid not in o["text"])
    t = words["text"]
    k = t.find(anc)
    a0 = max(0, k - 30)
    a0 = t.index(" ", a0) + 1 if a0 > 0 else 0          # start the excerpt on a word
    s1 = "…" + esc(t[a0:k]) + f'<b class="anc">{esc(anc)}</b>' + esc(t[k + len(anc):k + len(anc) + 20])
    links = obs[j]["text"]
    kk = links.find(hid)
    q0 = links.rfind('"', 0, kk) + 1          # the list item that holds the id, host and folder elided
    assert links[q0:kk].endswith("/r/"), links[q0:kk]
    first = "[" if links[:q0].strip() == '["' else "[…, "
    s2 = first + f'"…/r/<b class="id">{esc(hid)}</b>.html", …]' 
    head = [("page words", f'<span class="mono">{s1}</span>', "no address", "no"),
            ("link list", f'<span class="mono">{s2}</span>', "reached", "yes")]
    return head + [next_step(ex, j)]


def trim(th, anchor, cap=130):
    """Keep a long thought verbatim but short: the sentences that name the link words (or 'link'), joined by ' … '."""
    if len(th) <= cap:
        return th
    sents = re.split(r"(?<=[.!?])\s+", th)
    keep = [x for x in sents if anchor in x] or [x for x in sents if "link" in x.lower()] or sents[:1]
    out = " … ".join(keep[:2])
    return ("… " if sents.index(keep[0]) > 0 else "") + out + (" …" if sents.index(keep[min(1, len(keep) - 1)]) < len(sents) - 1 else "")


def next_step(ex, j):
    """Followed: the thought of the step that requested the id + the log line. Not followed: the first later thought
    that names the link words, and what the run ended with."""
    if ex["log"]:
        th = trim(thought(ex["asst"][req_step(ex)]), ex["anchor"])
        return ("next step", f'&ldquo;{esc(th)}&rdquo;{logline(ex)}', "followed", "yes")
    later = [thought(t) for t in ex["asst"][j + 1:]]
    th = next(t for t in later if ex["anchor"] in t)
    th = esc(trim(th, ex["anchor"]))
    end = ex["rec"]["answer"]
    tail = (f'answers {esc(end)}' + ("" if ex["rec"]["correct"] else ", a wrong figure")) if end \
        else f'stops after {ex["rec"]["steps_used"]} steps with no answer'
    return ("later step", f'&ldquo;{th}&rdquo;<div class="log">never requests the address; {tail}</div>', "not followed", "no")


def steps_mcp(ex):
    obs, asst, hid, anc = ex["obs"], ex["asst"], ex["hid"], ex["anchor"]
    j = next(k for k, o in enumerate(obs) if hid in o["text"])
    home = obs[0]["text"]
    assert anc not in home and hid not in home
    t = obs[j]["text"]
    m = re.search(r"\[" + re.escape(anc) + r"\]\([^)]*\)", t)
    assert m.group(0).endswith(f"/r/{hid}.html)"), m.group(0)
    s2 = f'[<b class="anc">{esc(anc)}</b>](…/r/<b class="id">{esc(hid)}</b>.html)' 
    page_j = ex["acts"][j]["url"].rsplit("/", 1)[-1]
    return [("home page", "its markdown view drops the footer", "no link", "no"),
            (esc(page_j), f'<span class="mono">{s2}</span>', "reached", "yes"), next_step(ex, j)]


def rest_row(pk, tool):
    """The runs that do not match the modal pattern (rule addition #25958)."""
    n, N, (r, f) = pk["rest_n"], pk["n"], pk["rest_pattern"]
    what = {(True, False): "reached the address but never requested it", (False, False): "never reached the address",
            (True, True): "reached and requested it"}[(r, f)]
    body = f"<b>{n} of {N}</b> runs {what}."
    tag, cls = ("not followed", "no") if r and not f else (("not reached", "no") if not r else ("followed", "yes"))
    if pk["alt_file"]:
        ex = example(pk["alt_file"])
        j = next(k for k, o in enumerate(ex["obs"]) if ex["hid"] in o["text"])
        jl = max(k for k, o in enumerate(ex["obs"]) if ex["hid"] in o["text"])   # its reply to the LAST output with the id
        th = thought(ex["asst"][jl + 1])
        same = ex["rec"]["site"]["brand"] == EXAMPLE_BRAND[0]
        site = "" if same else f' (another test site, {esc(ex["rec"]["site"]["brand"])})'
        body += f' Run {pk["alt_i"]}{site}, next step: &ldquo;{esc(trim(th, ex["anchor"]))}&rdquo;'
    return ("other runs", body, tag, cls)


STEPS = {"lcpw": steps_lcpw, "mcpfetch": steps_mcp}
EXAMPLE_BRAND = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kmax", type=int)
    ap.add_argument("--analysis")
    ap.add_argument("--picks", default=str(HERE / "picks.json"))
    ap.add_argument("--template", default=str(HERE / "template.html"))
    a = ap.parse_args()
    rs = pick.rows(a.kmax)
    L = law(rs)
    for k, v in L.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    if a.analysis:
        an = json.loads(Path(a.analysis).read_text())
        assert an["K"] == a.kmax, (an["K"], a.kmax)
        for (tool, model), v in L.items():
            c = next(x for x in an["cells"] if x["harness"] == tool and x["model"] == model)
            assert c["role"] == "heldout" or c["role"].startswith("held"), c["role"]
            assert (c["followed"], c["n"]) == v["catch_k"], (c, v)
            assert abs(c["pred"] - v["pred"]) < 1e-9, (c["pred"], v["pred"])
        print("analysis.json: K, held-out catch and pred match; status", an["status"])
    json.dump({f"{k[0]}|{k[1]}": v for k, v in L.items()}, open(HERE / "numbers.json", "w"), indent=1)
    picks = json.loads(Path(a.picks).read_text())
    cards = []
    for tool, model in (("lcpw", B), ("mcpfetch", A)):
        ex = example(picks[f"{tool}|{model}"]["file"])
        EXAMPLE_BRAND[:] = [ex["rec"]["site"]["brand"]]
        pk = picks[f"{tool}|{model}"]
        steps = STEPS[tool](ex)
        assert pk["n"] == L[(tool, model)]["catch_k"][1]
        if pk["rest_n"]:
            steps.append(rest_row(pk, tool))
        cards.append(card(ex, L[(tool, model)], tool, model, steps))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T = json.loads((HERE / "text.json").read_text())
    exl = example(picks[f"lcpw|{B}"]["file"])
    rec = exl["rec"]
    # footer markup as served (PREREG s2), checked against a raw-tool run's own first page view
    rawf = sorted((PROJ / "raw" / "main" / B).glob("raw__*/*.json"))[0]
    rr = json.loads(rawf.read_text())
    v0 = rr["transcript"][1]["text"]
    want = f'<div class="lk"><a href="http://127.0.0.1:8818{rr["site"]["hidden_path"]}" rel="nofollow">{rr["site"]["anchor_text"]}</a></div>'
    assert "<style>.lk{display:none}</style>" in v0 and want in v0, v0[-500:]
    hid = exl["hid"]
    footer = (f'<span class="hide">&lt;style&gt;.lk{{display:none}}&lt;/style&gt;</span> <span class="ppl">hides it from people; the only route to the 2025 answer</span><br>'
              f'<span class="hide">&lt;div class="lk"&gt;</span>&lt;a href="&hellip;/r/<b class="id">{hid}</b>.html"&gt;'
              f'<b class="anc">{esc(rec["site"]["anchor_text"])}</b>&lt;/a&gt;&lt;/div&gt;')
    fill = {"KICK": T["kick"], "HEAD": T["head"], "HEADTXT": re.sub("<[^>]+>", "", T["head"]), "SUB": T["sub"],
            "TASK": "&ldquo;" + esc(rec["question"]) + "&rdquo;", "FOOTER": footer, "KEY": T["key"], "FOOT": T["foot"],
            "SOURCES": "\n".join("  " + s for s in [
                f"example runs: {picks[f'lcpw|{B}']['file']}, {picks[f'mcpfetch|{A}']['file']} (tool outputs and thoughts verbatim)",
                f"footer markup checked against {rawf.relative_to(PROJ)} (first page view)",
                "server log: logs/access.log (followed = a request whose path holds the run's id)",
                "numbers: raw/main/*/*/*.json + logs/access.log, PREREG s4 formula (gen.py law()), asserted vs results/analysis.json"])}
    page = Path(a.template).read_text().replace("{{CARDS}}", "".join(cards))
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page
    (out / "figure1.html").write_text(page)
    print("wrote", out / "figure1.html")


if __name__ == "__main__":
    main()
