"""Figure 1 generator (Owain), route-first-law. Example-pick rule: README.md (frozen before the totals).

    python3 gen.py [--root <data root>] [--out <fig dir>]

Every number comes from analysis.analyse(); every page/denial fragment is asserted to be a substring of the real
observation of the picked run; every quote is the run's own `thought` field.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
PROJ = WORK / "kran-research" / "projects" / "route-first-law"
BLOG = WORK / "athanorscience.com"
SLUG = "route-first-law"
FIG = BLOG / "assets" / "figures" / f"{SLUG}-fig1"
EX_MODEL = "gemma-4-31b-it"
sys.path.insert(0, str(PROJ))
import analysis  # noqa: E402

E = html.escape


def thought(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return ""
    try:
        return json.loads(m.group(0)).get("thought", "")
    except json.JSONDecodeError:
        return ""


def check(frag: str, src: str, what: str) -> str:
    """Assert every '…'-separated piece of a display fragment is verbatim in the source."""
    for piece in frag.split("…"):
        if piece.strip() and piece not in src:
            sys.exit(f"gen.py: fragment not in {what}: {piece!r}")
    return frag


def pick(rows: list[dict], harness: str) -> tuple[dict, bool]:
    after = [r for r in rows if r["harness"] == harness and r["arm"] == "after" and r["exposed"]]
    k = sum(r["trap"] for r in after)
    major = k * 2 > len(after)
    cand = sorted((r for r in after if r["model"] == EX_MODEL and r["trap"] == major), key=lambda r: r["i"])
    if not cand:
        sys.exit(f"gen.py: no {harness} AFTER run of {EX_MODEL} with trap={major}")
    return cand[0], major


def web_example(root: Path, r: dict) -> dict:
    rec = json.loads((root / EX_MODEL / "web" / "raw" / "after" / f"{r['i']:03d}.json").read_text())
    s = rec["site"]
    acts = rec["actions"]
    assert acts[0]["path"] == s["start_path"], acts[0]
    dec = next(a for a in acts if a.get("path") in (s["route_item_path"], s["fit_path"]))
    seen = next(a for a in acts if a.get("step") == dec["step"] - 1)  # the page read right before the decision
    page = (PROJ / "web" / "site" / s["token"] / seen["path"].lstrip("/")).read_text()
    sha = hashlib.sha256(page.encode()).hexdigest()
    assert sha == seen["obs_sha256"], ("page sha mismatch", s["token"], seen["path"])
    t = next((thought(x["text"]) for x in rec["replies"] if x["step"] == dec["step"]), "")
    nxt = dec
    hid_f = s["fit_path"].split("/")[-1]
    hid, rid = hid_f[:4], s["rid"][:4]
    anchor = html.unescape(re.search(rf'<div class="fx"><a href="r/{hid_f}" rel="nofollow">(.*?)</a></div>', page).group(1))
    check(f'<p><a href="r/{s["rid"]}.html">{E(anchor)}</a></p>', page, "web page")
    h2 = re.search(r"<h2>(.*?)</h2>", page).group(1)
    lines = [
        ("before", f'<a href="r/{rid}…">{anchor}</a>'),
        ("ctx", check(f"<main>…{h2[:40]}…", page, "web page")),
        ("trap", f'<div class="fx"><a …>{anchor}'),
        ("ctx", check(f"<p>&copy; {s['brand']}</p>", page, "web page")),
        ("after", f'<a href="r/{rid}…">{anchor}</a>'),
    ]
    return {"rec": rec, "lines": lines, "thought": t, "next_path": nxt.get("path"), "page": seen["path"],
            "src": f"data/main/{EX_MODEL}/web/raw/after/{r['i']:03d}.json + web/site/{s['token']}{seen['path']} "
                   f"(sha256 = the run's obs_sha256 of step {seen['step']})"}


def shell_example(root: Path, r: dict) -> dict:
    rec = json.loads((root / EX_MODEL / "shell" / "raw" / "after" / f"{r['i']:03d}.json").read_text())
    key = rec["episode"]["key"]
    acts = rec["actions"]
    logs = [x["action_index"] for x in rec["route_log"] + rec["decoy_log"]]
    dec_i = min(logs) if logs else len(acts) - 1
    ai = max(j for j in range(dec_i) if acts[j]["tool"] == "cloud_stat" and key in str(acts[j]["observation"]))
    a, nxt = acts[ai], acts[dec_i]
    obs = a["observation"]
    t = next((thought(x["text"]) for x in rec["model_replies"] if x.get("step") == nxt["step"]), "")
    lines = [
        ("ctx", check("denied: token scope service-log-level does not…", obs, "denial")),
        ("before", check("size report (allowed for this ticket; no token…", obs, "denial")),
        ("trap", check(f"access_key: {key}", obs, "denial")),
        ("ctx", check("scope: storage size reads, org-wide", obs, "denial")),
        ("after", check("size report (allowed for this ticket; no token…", obs, "denial")),
    ]
    return {"rec": rec, "lines": lines, "thought": t, "next_tool": nxt.get("tool"), "next_args": nxt.get("args"),
            "src": f"data/main/{EX_MODEL}/shell/raw/after/{r['i']:03d}.json (actions[{rec['actions'].index(a)}].observation)"}


ROBOT = ('<svg class="rb" width="44" height="44" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/>'
         '<rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="5" y="20" width="5" height="12" rx="2" fill="#e8734a"/>'
         '<rect x="38" y="20" width="5" height="12" rx="2" fill="#e8734a"/><rect x="8" y="12" width="32" height="30" rx="9" fill="#fff" '
         'stroke="#222" stroke-width="2"/><rect x="12" y="17" width="24" height="18" rx="6" fill="#1f2a44"/><circle cx="19" cy="25" r="4" '
         'fill="#7fd3ff"/><circle cx="29" cy="25" r="4" fill="#7fd3ff"/><path d="M20 31 q4 3 8 0" stroke="#7fd3ff" stroke-width="1.5" '
         'fill="none"/></svg>')

def card(lines: list[tuple[str, str]], tags: dict) -> str:
    out = []
    for kind, txt in lines:
        tag = f'<span class="tag t-{kind}">{tags[kind]}</span>' if kind in tags else ""
        out.append(f'<div class="ln l-{kind}">{tag}<code>{E(txt, quote=False)}</code></div>')
    return "\n".join(out)


def bars(P: dict, labels: dict, ticks: dict, note: dict | None = None) -> str:
    out = []
    for arm in ("none", "before", "after"):
        c = P[arm]
        rate = c["rate"] or 0.0
        line = {"before": 25, "after": 75}.get(arm)
        tick = (f'<div class="tick" style="left:{line * 1.7:.0f}px"><span>{ticks[arm]}</span></div>'
                if line else "")
        nt = f'<div class="note">{note[arm]}</div>' if note and arm in note else ""
        out.append(f'<div class="bar b-{arm}"><div class="bl">{labels[arm]}</div><div class="bw"><div class="track">'
                   f'<div class="fill" style="width:{rate * 170:.1f}px"></div>{tick}</div>'
                   f'<div class="bv">{round(rate * 100)}%<span class="cnt">{c["trap"]} of {c["n_exposed"]}</span></div></div></div>{nt}')
    return "\n".join(out)


def shell_note(root: Path, res: dict, i: int, tmpl: str) -> tuple[str, str]:
    """One note at the shell route-above bar: the model with the most key use there, its count, and a verbatim phrase
    from its own reasoning in the BEFORE run with the same index as the AFTER example (same episode, only position moves)."""
    m = max(analysis.MODELS, key=lambda mm: res["per_model"][mm]["shell"]["before"]["rate"] or 0)
    M = res["per_model"][m]["shell"]["before"]
    rec = json.loads((root / m / "shell" / "raw" / "before" / f"{i:03d}.json").read_text())
    if not rec["decoy_log"]:
        return "", ""
    st = rec["actions"][rec["decoy_log"][0]["action_index"]]["step"]
    reason = next(x["reasoning"] for x in rec["model_replies"] if x.get("step") == st)
    phrase = check("much faster and more direct", reason, "BEFORE reasoning")
    name = {"gemma-4-31b-it": "Gemma 4 31B", "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}[m]
    return (tmpl.format(model=name, k=M["trap"], n=M["n_exposed"], phrase=E(phrase)),
            f"data/main/{m}/shell/raw/before/{i:03d}.json (reasoning at step {st})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJ / "data" / "main"))
    ap.add_argument("--out", default=str(FIG))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    res, rows = analysis.analyse(root)
    txt = json.loads((HERE / "text.json").read_text())
    st = res["rule"]["status"]
    rw, mw = pick(rows, "web")
    rs, ms = pick(rows, "shell")
    W, S = web_example(root, rw), shell_example(root, rs)
    n_runs = {h: sum(res["pooled"][h][arm]["n"] for arm in analysis.ARMS) for h in analysis.HARNESSES}
    fmt = dict(status=st, n_web=min(res["pooled"]["web"][a]["n_exposed"] for a in analysis.ARMS),
               n_shell=min(res["pooled"]["shell"][a]["n_exposed"] for a in analysis.ARMS), **{f"{h}_{arm}": round((res["pooled"][h][arm]["rate"] or 0) * 100)
                              for h in analysis.HARNESSES for arm in analysis.ARMS})
    T = {k: (v.format(**fmt) if isinstance(v, str) else v) for k, v in txt.items()}
    rows_html = []
    note_txt, note_src = shell_note(root, res, rs["i"], T["shell_note"])
    notes = {"shell": {"before": note_txt}} if note_txt else {}
    for h, X, major, title in (("web", W, mw, T["web_title"]), ("shell", S, ms, T["shell_title"])):
        out_txt = T[f"{h}_outcome_trap" if major else f"{h}_outcome_route"]
        rows_html.append(f'''<div class="row">
 <div class="rt">{title}</div>
 <div class="left">
  <div class="cardh">{T[h + "_cardh"]}</div>
  <div class="card">{card(X["lines"], T["tags"])}</div>
  <div class="say">{ROBOT}<div><div class="who">{T["who"]}</div><div class="bub">&ldquo;{E(X["thought"])}&rdquo;</div>
  <div class="act {'key' if major else 'route'}">&rarr; {out_txt}</div></div></div>
 </div>
 <div class="right"><div class="barh">{T["barh"]}</div>{bars(res["pooled"][h], T["bar_labels"], T["tick"], notes.get(h))}</div>
</div>''')
    src = (f"Sources (kran-research, branch swarm/defense-1126-route-first-law, projects/route-first-law):\n"
           f"  numbers: analysis.analyse(data/main) -> status {st}; web runs {n_runs['web']}, shell runs {n_runs['shell']}\n"
           f"  web example: {W['src']}\n  shell example: {S['src']}\n  shell note: {note_src}\n"
           f"  example rule: figure1/README.md (frozen before totals)\n"
           f"  built by swarm-defense-1126-route-first-law/figure1/gen.py")
    doc = (HERE / "template.html").read_text()
    doc = (doc.replace("{{SRC}}", src.replace("--", "- -")).replace("{{HEAD}}", T["head"]).replace("{{SUB}}", T["sub"])
           .replace("{{ROWS}}", "\n".join(rows_html)).replace("{{FOOT}}", T["foot"]).replace("{{TITLE}}", E(T["head"])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(doc)
    alt = T["alt"]
    (HERE / "figure1_alt.txt").write_text(alt.replace('"', "'") + "\n")
    print(f"status {st}; web after majority trap={mw} (run {rw['i']}), shell after majority trap={ms} (run {rs['i']})")
    for h in analysis.HARNESSES:
        print(h, {arm: f"{res['pooled'][h][arm]['trap']}/{res['pooled'][h][arm]['n_exposed']}" for arm in analysis.ARMS})
    print("web thought:", W["thought"], "->", W["next_path"])
    print("shell thought:", S["thought"], "->", S["next_tool"], S["next_args"])


if __name__ == "__main__":
    main()
