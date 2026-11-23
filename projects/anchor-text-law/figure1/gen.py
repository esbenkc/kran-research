"""Build Figure 1 (figure1.html) for anchor-text-law from the real runs. No model calls.

    python3 gen.py OUT_DIR [--proj DIR] [--analysis PATH] [--draft]

Sources (kran-research, branch swarm/defense-1123-anchor-text-law, projects/anchor-text-law):
  example runs: web/out/<slot>/<model>/raw/{bare,anchor}-free/<NNN>.json chosen by pick.py (rule: room #28663);
                tool outputs and the agent's reasoning are verbatim, trimmed only with ellipses
  server log:   web/out/<slot>/<model>/access.log (followed = a GET whose path holds the run's hid)
  numbers:      recomputed here with pick.py's events (PREREG s3) and asserted equal to results/analysis.json
text.json holds the head and subtitle; gen.py refuses to render while any text holds [FILL] (unless --draft, which
is for /tmp design renders only).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pick  # noqa: E402

NAME = {"gpt-oss-20b": "gpt-oss-20b", "qwen3-8b": "Qwen3-8B", "qwen3-8b-nothink": "Qwen3-8B", "qwen3-8b-think": "Qwen3-8B"}
PANEL = {"bare": ("Bare link list", "addresses only"), "anchor": ("Link list with words", "each address with its link text")}


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def mark(s: str, hid: str, anchor: str) -> str:
    s = esc(s)
    s = s.replace(esc(hid), f'<b class="id">{esc(hid)}</b>')
    return s.replace(esc(anchor), f'<b class="anc">{esc(anchor)}</b>')


def load(f: str) -> dict:
    r = json.loads((pick.PROJ / f).read_text())
    d = (pick.PROJ / f).parents[2]
    hid = r["site"]["hid"]
    log = [p for p in pick.load_log(d) if len(p) >= 5 and p[2] == "GET" and hid in p[3]]
    obs = [t["text"] for t in r["transcript"] if t["role"] == "observation"]
    return {"r": r, "hid": hid, "obs": obs, "log": log, "anchor": anchor_text(r, obs)}


def anchor_text(r: dict, obs: list[str]) -> str:
    """The canary's link text, read from the run's own anchor-list output, or from the site template via websites.py."""
    for o in obs:
        m = re.search(r'\{"text": "([^"]*)", "href": "r/' + r["site"]["hid"], o)
        if m:
            return m.group(1)
    sys.path[:0] = [str(pick.PROJ / "web"), str(HERE.parent / "web")]
    import websites  # noqa: E402
    return websites.TEMPLATES[r["site"]["template"]]["fit_anchor"]


def thought(r: dict, step: int) -> str:
    """The agent's words at a step: the JSON 'thought' if it wrote one, else its reasoning channel (gpt-oss)."""
    rp = next(x for x in r["replies"] if x["step"] == step)
    t = ""
    try:
        t = json.loads(rp.get("text") or "").get("thought", "")
    except Exception:
        m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', rp.get("text") or "")
        t = json.loads('"' + m.group(1) + '"') if m else ""
    return (t or rp.get("reasoning") or "").strip()


def trim(th: str, keys: list[str], cap: int = 210) -> str:
    """Keep the agent's words verbatim but short: from the first sentence that names the first matching key (keys in
    priority order), plus the sentences right after it while they fit in cap characters. Cuts are marked with '…'."""
    th = " ".join(th.split())
    if len(th) <= cap:
        return th
    sents = re.split(r"(?<=[.!?])\s+", th)
    k0 = next((k for key in keys for k, x in enumerate(sents) if key.lower() in x.lower()), 0)
    k1, out = k0 + 1, len(sents[k0])
    while k1 < len(sents) and out + 1 + len(sents[k1]) <= cap:
        out += 1 + len(sents[k1])
        k1 += 1
    s = " ".join(sents[k0:k1])
    if len(s) > cap + 40:
        s = s[:cap].rsplit(" ", 1)[0] + " …"
    return ("… " if k0 > 0 else "") + s + (" …" if k1 < len(sents) and not s.endswith("…") else "")


def split_obs(o: str) -> tuple[str, str]:
    a, _, b = o.partition("\n\nLinks:\n")
    return a.removeprefix("Text:\n"), b


def text_excerpt(text: str, anchor: str, before: int = 28, after: int = 22) -> str:
    k = text.index(anchor)
    a0 = max(0, k - before)
    a0 = text.index(" ", a0) + 1 if a0 > 0 else 0
    b0 = k + len(anchor) + after
    b0 = text.rfind(" ", 0, b0) if b0 < len(text) else len(text)
    return ("…" if a0 > 0 else "") + esc(text[a0:k]) + f'<b class="anc">{esc(anchor)}</b>' + esc(text[k + len(anchor):b0]) + ("…" if b0 < len(text) else "")


def list_excerpt(links: str, hid: str, anchor: str, keep_before: int = 1) -> str:
    """The JSON list verbatim around the canary item: keep_before items before it, the canary, then '…]'."""
    items = json.loads(links)
    k = next(n for n, x in enumerate(items) if hid in (x if isinstance(x, str) else x["href"]))
    show = items[max(0, k - keep_before): k + 1]
    body = ", ".join(json.dumps(x) for x in show)
    pre = "[" if k - keep_before <= 0 else "[…, "
    post = ", …]" if k < len(items) - 1 else "]"
    return pre + mark(body, hid, anchor) + post, k + 1, len(items)


def req_step(ex: dict) -> int:
    return next(int(a["step"]) for a in ex["r"]["actions"] if ex["hid"] in str(a.get("url", "")) + str(a.get("path", "")))


def first_seen(ex: dict) -> int:
    """Index of the first observation that holds the hid; observation j answers action j (the step that fetched it)."""
    return next(j for j, o in enumerate(ex["obs"]) if ex["hid"] in o)


def logline(ex: dict) -> str:
    p = ex["log"][0]
    assert p[3].endswith(f"/r/{ex['hid']}.html"), p[3]
    return (f'<div class="log"><span>server log</span> <span class="mono">GET /r/<b class="id">{esc(ex["hid"])}</b>.html '
            f'{esc(p[4])}</span></div>')



def reasoning(r: dict, step: int) -> str:
    rp = next(x for x in r["replies"] if x["step"] == step)
    return " ".join((rp.get("reasoning") or "").split())


def words(r: dict, step: int, keys: list[str], cap: int) -> str:
    """The agent's own words at a step: its reasoning around the first key it names, else its stated thought."""
    rz = reasoning(r, step)
    if any(k.lower() in rz.lower() for k in keys):
        return trim(rz, keys, cap)
    return trim(thought(r, step), keys, cap)

def rows_for(ex: dict, fmt: str) -> list[tuple[str, str, str, str]]:
    j = first_seen(ex)
    text, links = split_obs(ex["obs"][j])
    lst, pos, n = list_excerpt(links, ex["hid"], ex["anchor"])
    out = []
    if fmt == "bare":
        out.append(("page text", f'<span class="mono">{text_excerpt(text, ex["anchor"])}</span>', "words, no address", "no"))
    out.append(("link list", f'<span class="mono">{lst}</span>', "saw address", "yes"))
    keys = [ex["anchor"], ex["hid"], "r/", " ".join(ex["anchor"].split()[:2])]
    if ex["log"]:
        rs = req_step(ex)
        for a in detours(ex["r"], ex["hid"]):
            dk = ["list of links", "links list", "link list", ex["anchor"]]
            th = trim(thought(ex["r"], int(a["step"])), dk, cap=110)
            out.append(("first try", f'&ldquo;{esc(th)}&rdquo;<div class="log"><span>server log</span> <span class="mono">'
                        f'GET {esc(a["path"])} {a["status"]}</span> (no such page)</div>', "guessed", "no"))
            break
        th = words(ex["r"], rs, (["links list", "list of links", "link list"] if fmt == "bare" else []) + keys, 150)
        out.append(("next step", f'&ldquo;{esc(th)}&rdquo;{logline(ex)}{answer_line(ex)}', "requested", "yes"))
    else:
        step_seen = int(ex["r"]["actions"][j]["step"])
        steps = range(step_seen + 1, ex["r"]["steps_used"])
        s = next((s for s in steps if any(k.lower() in reasoning(ex["r"], s).lower() for k in keys)), steps[0])
        th = words(ex["r"], s, keys, 200)
        act = next(a for a in ex["r"]["actions"] if int(a["step"]) == s)
        fetched = (f'<span>server log</span> <span class="mono">GET {esc(act["path"])} {act["status"]}</span><br>'
                   if act.get("tool") == "fetch" else "")
        end, st = ex["r"]["answer"], ex["r"]["site"].get("stale")
        if end:
            amt = st if st and st in end else trim(end, ["$"], 60)
            tail = f'answers {esc(amt)}' + (", last year's price" if st and st in end else "" if ex["r"]["correct"] else ", a wrong answer")
        else:
            tail = f'stops after {ex["r"]["steps_used"]} steps with no answer'
        out.append(("next step", f'&ldquo;{esc(th)}&rdquo;<div class="log">{fetched}never requests the hidden address; {tail}</div>',
                    "not requested", "no"))
    return out


def answer_line(ex: dict) -> str:
    r = ex["r"]
    end, right = r["answer"] or "", r["site"]["answer"]
    if not (r["correct"] and right in end):
        return ""
    what = "the 2025 price" if r["site"]["template"] == "pricing" else "the right answer"
    return f'<div class="log">answers {esc(right)}, {what}</div>'


def detours(r: dict, hid: str) -> list[dict]:
    """Fetches that got a 404 before the run first requested the canary (all 404s if it never did)."""
    out = []
    for a in r["actions"]:
        if a.get("tool") != "fetch":
            continue
        if hid in str(a.get("path", "")):
            break
        if a.get("status") == 404:
            out.append(a)
    return out


def detour_count(d: Path, fmt: str, idx: set[int]) -> tuple[int, int]:
    fs = [f for f in sorted((d / "raw" / f"{fmt}-{pick.HIDING}").glob("[0-9][0-9][0-9].json")) if int(f.stem) in idx]
    rr = [json.loads(f.read_text()) for f in fs]
    return sum(bool(detours(r, r["site"]["hid"])) for r in rr), len(rr)


def offtask_count(d: Path, fmt: str, idx: set[int]) -> tuple[int, int]:
    """Runs whose access log holds a GET of their off-task link's id (PREREG s3 'off-task follow', run label col 7)."""
    gets = {}
    for p in pick.load_log(d):
        if len(p) >= 7 and p[2] == "GET":
            gets.setdefault(p[6], []).append(p[3])
    fs = [f for f in sorted((d / "raw" / f"{fmt}-{pick.HIDING}").glob("[0-9][0-9][0-9].json")) if int(f.stem) in idx]
    k = 0
    for f in fs:
        r = json.loads(f.read_text())
        k += any(r["site"]["oid"] in x for x in gets.get(f"{fmt}-{pick.HIDING}/{f.stem}", []))
    return k, len(fs)


WHAT = {(True, False): "saw the address but never requested it", (False, False): "never saw the address",
        (True, True): "saw the address and requested it", (False, True): "requested it without seeing it"}
TAG = {(True, False): ("not requested", "no"), (False, False): ("never saw it", "no"),
       (True, True): ("requested", "yes"), (False, True): ("requested", "yes")}


def rest_row(pk: dict, extra: str = "") -> tuple[str, str, str, str] | None:
    """Every pattern other than the example's, largest first, each with its own count (sums to rest_n)."""
    pats = sorted(((tuple(x == "True" for x in k.split(",")), v) for k, v in pk["counts"].items()
                   if [x == "True" for x in k.split(",")] != pk["pattern"]), key=lambda t: -t[1])
    assert sum(v for _, v in pats) == pk["rest_n"], (pats, pk["rest_n"])
    if not pats:
        return ("other runs", extra, "", "none") if extra else None
    body = " ".join(f'<b>{v} of {pk["n"]}</b> {WHAT[p]}.' for p, v in pats)
    return ("other runs", body + (" " + extra if extra else ""), "", "none")


ROBOT = ('<svg class="rb" width="{s}" height="{s}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" '
         'fill="#e8734a"/><rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="6" y="12" width="36" height="30" '
         'rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" '
         'fill="#e8734a"/><rect x="41" y="22" width="5" height="11" rx="2" fill="#e8734a"/><rect x="11" y="17" '
         'width="26" height="20" rx="6" fill="#1f2a44"/><circle cx="19" cy="25" r="4" fill="#7fd3ff"/><circle cx="29" '
         'cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" fill="none"/></svg>')


def card(fmt: str, rows: list, frk: tuple[int, int], model: str, hl: bool) -> str:
    t, s = PANEL[fmt]
    body = "".join(f'<div class="st"><div class="sl">{lab}<div class="tag {cls}">{tag}</div></div><div class="sb">{b}</div></div>'
                   for lab, b, tag, cls in rows if lab)
    f, n = frk
    pct = round(100 * f / n)
    return (f'<div class="card{" hl" if hl else ""}"><div class="ch">{ROBOT.format(s=40)}<div><b>{t}</b><span>{s}</span></div></div>'
            f'{body}<div class="cnt"><div class="big">{pct}%</div><div>of the {n} runs that saw the address requested it '
            f'<span>({f} of {n})</span></div>'
            f'<div class="bar"><i style="width:{100 * f / n:.1f}%"></i></div></div></div>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--proj", default=str(pick.PROJ))
    ap.add_argument("--analysis")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--text", default=str(HERE / "text.json"))
    ap.add_argument("--template", default=str(HERE / "template.html"))
    ap.add_argument("--all", action="store_true", help="plumbing test only (smoke data)")
    a = ap.parse_args()
    pick.PROJ = Path(a.proj)
    picks = json.loads((HERE / "picks.json").read_text())
    model = picks["model"]
    d = pick.PROJ / picks["stats"][model]["dir"]
    rs = pick.rows(d) if a.all else pick.scored(pick.rows(d))
    frs = {fmt: pick.fr(rs, fmt) for fmt in pick.MAIN}
    for fmt in pick.MAIN:
        assert tuple(picks["stats"][model]["fr"][fmt]) == frs[fmt], (fmt, picks["stats"][model]["fr"][fmt], frs[fmt])
    T = json.loads(Path(a.text).read_text())
    if not a.draft:
        bad = [k for k, v in T.items() if "[FILL" in v]
        assert not bad, f"text.json still holds [FILL] in {bad}: no render before the status exists"
        assert a.analysis, "a final render needs --analysis results/analysis.json"
    nums = {}
    if a.analysis:
        an = json.loads(Path(a.analysis).read_text())
        check_analysis(an, model, frs)
        nums = status_numbers(an)
        st = an["status"]["status"]
        qa = re.sub("<[^>]+>", "", T["qa"])
        ans = {"UNCONFIRMED": "Answer: Not settled", "CONFIRMED": "Answer: Yes", "REJECTED": "Answer: No"}
        assert all(bool(re.search(re.escape(w) + r"\b", qa)) == (k == st) for k, w in ans.items()), (st, qa)
    sites = {}
    for fmt in ("bare", "anchor"):
        fs = [f for f in sorted((d / "raw" / f"{fmt}-{pick.HIDING}").glob("[0-9][0-9][0-9].json")) if int(f.stem) in {s["i"] for s in rs}]
        sites[fmt] = {json.loads(f.read_text())["site"]["template"] for f in fs}
    assert sites["bare"] == sites["anchor"], sites
    assert picks["bare"]["n"] == picks["anchor"]["n"]
    nums = dict(nums, NSITES=len(sites["bare"]), NRUNS=picks["bare"]["n"])
    for k in ("head", "sub", "qa"):
        T[k] = T[k].format(**nums) if nums else T[k]
    exs = {fmt: load(picks[fmt]["file"]) for fmt in ("bare", "anchor")}
    for fmt, ex in exs.items():
        assert bool(ex["log"]) == picks[fmt]["pattern"][1], (fmt, "log vs pick")
    idx = {s["i"] for s in rs}
    det = {fmt: detour_count(d, fmt, idx) for fmt in pick.MAIN}
    off = {fmt: offtask_count(d, fmt, idx) for fmt in pick.MAIN}
    print("guessed-address runs (404 before the canary):", det, "| off-task follows:", off)
    cards = []
    for fmt in ("bare", "anchor"):
        rows = rows_for(exs[fmt], fmt)
        extra = ""
        rr = rest_row(picks[fmt], extra)
        if rr:
            rows.append(rr)
        cards.append(card(fmt, rows, frs[fmt], model, fmt == "anchor"))
    ex = exs["bare"]
    r = ex["r"]
    footer = (f'&lt;div class="fx"&gt;&lt;a href="r/<b class="id">{esc(ex["hid"])}</b>.html"&gt;<b class="anc">{esc(ex["anchor"])}</b>&lt;/a&gt;&lt;/div&gt;'
              f'<span class="ppl"> the stylesheet hides it from people, so a visit marks an AI agent; each run gets a new random code</span>')
    fill = {"HEAD": T["head"], "HEADTXT": re.sub("<[^>]+>", "", T["head"]), "SUB": T["sub"], "QA": T["qa"], "BRIDGE": T.get("bridge", ""), "KEY": T["key"],
            "FOOT": T["foot"], "TASK": "&ldquo;" + esc(r["question"]) + "&rdquo;", "MODEL": NAME.get(model, model),
            "FOOTER": footer, "CARDS": "".join(cards),
            "SOURCES": "\n".join("  " + s for s in [
                f"example runs: {picks['bare']['file']}, {picks['anchor']['file']} (tool outputs and reasoning verbatim)",
                f"server log: {picks['stats'][model]['dir']}/access.log (followed = a GET whose path holds the run's hid)",
                f"follow-after-reach: bare {frs['bare']}, anchor {frs['anchor']}, md {frs['md']}, raw {frs['raw']} "
                "(pick.py events, scored prefix; asserted vs results/analysis.json)",
                "pick rule: room post #28663, pick.py"])}
    page = Path(a.template).read_text()
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    assert "{{" not in page
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(page)
    json.dump({"model": model, "fr": frs, "picks": {f: picks[f]["file"] for f in ("bare", "anchor")}},
              open(HERE / "numbers.json", "w"), indent=1)
    print("wrote", out / "figure1.html", "| fr", frs)


def check_analysis(an: dict, model: str, frs: dict) -> None:
    """Every follow-after-reach count Figure 1 prints must equal analysis.json's scored cells."""
    hit = 0
    for c in an["cells"]:
        if c["model"] != model or not c.get("scored") or not c["cell"].endswith("-" + pick.HIDING):
            continue
        fmt = c["cell"].split("-")[0]
        assert (c["followed_reached"], c["reached"]) == frs[fmt], (fmt, c["followed_reached"], c["reached"], frs[fmt])
        hit += 1
    assert hit == 4, f"analysis.json: matched {hit} of 4 scored cells for {model}"
    print("analysis.json: fr counts match for", model, "| status", an["status"]["status"])


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def status_numbers(an: dict) -> dict:
    """The status cell with the largest miss (frozen status cells A-md, A-anchor, B-raw) and its numbers."""
    names = {"A": "gpt&#8209;oss&#8209;20b", "B": "Qwen3&#8209;8B"}
    fmts = {"raw": "raw HTML", "md": "markdown", "anchor": "the list with words", "bare": "the bare list"}
    k = max(("A-md", "A-anchor", "B-raw"), key=lambda c: abs(an["heldout"][c]["err"]["value"]))
    h = an["heldout"][k]
    lo, hi = (x["value"] for x in h["err_ci"])
    return {"PRED": pct(h["pred"]["value"]), "CATCH": pct(h["catch"]["value"]),
            "MISS": round(abs(100 * h["err"]["value"])), "LO": f"{abs(100 * hi):.1f}", "HI": round(abs(100 * lo)),
            "WHO": names[k[0]], "FMT": fmts[k.split("-")[1]]}


if __name__ == "__main__":
    main()
