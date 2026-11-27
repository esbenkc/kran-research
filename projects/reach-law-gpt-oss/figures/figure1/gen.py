"""Figure 1 generator (Owain), reach-law-gpt-oss. Example-pick rule: README.md (frozen before any counted run,
kran-research 4ca3ad24c projects/reach-law-gpt-oss/figures/figure1/README.md).

    python3 gen.py [--data <dir with gpt-oss-20b/raw>] [--out <fig dir>]

Numbers come from the project's own analysis.load_runs() + analysis.analyse(). Card text comes from the picked run's
own logs (episode.ops_text, the first denial observation), asserted verbatim; in PENDING mode (no counted run yet)
it comes from shell/env.py WorkspaceEnv(0, arm), the same bytes the model sees. Quotes are the run's own `thought`.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Works from the swarm dir (swarm-<room>/figure1) and from kran-research projects/reach-law-gpt-oss/figures/figure1.
IN_PROJ = (HERE.parents[1] / "analysis.py").exists()
WORK = HERE.parents[4] if IN_PROJ else HERE.parents[1]
PROJ = HERE.parents[1] if IN_PROJ else WORK / "kran-research" / "projects" / "reach-law-gpt-oss"
BLOG = WORK / "athanorscience.com"
SLUG = "reach-law-gpt-oss"
FIG = BLOG / "assets" / "figures" / f"{SLUG}-fig1"
MODEL = "gpt-oss-20b"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "shell"))
import analysis  # noqa: E402
from env import COND, WorkspaceEnv  # noqa: E402

E = html.escape


def thought(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return ""
    try:
        t = json.loads(m.group(0)).get("thought", "")
    except json.JSONDecodeError:
        return ""
    return t if isinstance(t, str) else ""


def check(frag: str, src: str, what: str) -> str:
    """Assert every '…'-separated piece of a display fragment is verbatim in the source."""
    for piece in frag.split("…"):
        if piece.strip() and piece not in src:
            sys.exit(f"gen.py: fragment not in {what}: {piece!r}")
    return frag


def pick(rows: list[dict], status: str) -> tuple[dict | None, dict | None, bool, bool, bool]:
    """README rule. Returns (none_row, runbook_row, none_key, runbook_key, same_index)."""
    main = [r for r in rows if r["phase"] == "main"]
    none = [r for r in main if r["arm"] == "none"]
    rb = [r for r in main if r["arm"] == "runbook" and r["reached"]]
    nk = sum(r["key"] for r in none) * 2 >= len(none)  # majority, tie -> key used
    if status == "CONFIRMED":
        rk = False
    elif status == "REJECTED":
        rk = True
    else:
        rk = sum(r["key"] for r in rb) * 2 >= len(rb)
    L = {r["seed"]: r for r in none if bool(r["key"]) == nk}
    R = {r["seed"]: r for r in rb if bool(r["key"]) == rk}
    both = sorted(set(L) & set(R))
    if both:
        return L[both[0]], R[both[0]], nk, rk, True
    return (L[min(L)] if L else None), (R[min(R)] if R else None), nk, rk, False


def decision(rec: dict) -> tuple[int, str]:
    """Index of the decision action: first key use, else first size-report call, else the step after the first denial."""
    d = analysis.derive(rec)
    acts = rec["actions"]
    if d["first_key_action"] is not None:
        return d["first_key_action"], "key"
    rc = [i for i, a in enumerate(acts) if a["tool"] == "api_post" and str(a["args"].get("path", "")).endswith("/size-report")]
    if rc:
        return rc[0], "route"
    fd = rec["reach"]["first_denial_action"]
    return (min(fd + 1, len(acts) - 1) if fd is not None else len(acts) - 1), "other"


def example(data: Path, row: dict) -> dict:
    f = data / MODEL / "raw" / row["arm"] / f"{row['seed']:03d}.json"
    rec = json.loads(f.read_text())
    ep, acts = rec["episode"], rec["actions"]
    fd = rec["reach"]["first_denial_action"]
    if fd is None:
        sys.exit(f"gen.py: {f} has no denial")
    di, kind = decision(rec)
    st = acts[di]["step"]
    t = next((thought(x["text"]) for x in rec["model_replies"] if x.get("step") == st), "")
    return {"ops": ep["ops_text"], "denial": acts[fd]["observation"], "key": ep["key"], "bucket": ep["bucket"],
            "ticket": ep["ticket"], "thought": t, "kind": kind, "tool": acts[di]["tool"], "args": acts[di]["args"],
            "src": f"data/{MODEL}/raw/{row['arm']}/{row['seed']:03d}.json (ops_text; actions[{fd}].observation; "
                   f"thought at step {st})"}


def pending_example(arm: str) -> dict:
    env = WorkspaceEnv(0, arm)
    s = env.state
    return {"ops": env.ops_text(), "denial": env.scope_denial(), "key": s.key, "bucket": s.bucket, "ticket": s.ticket,
            "thought": "", "kind": "pending", "src": f"shell/env.py WorkspaceEnv(0, '{arm}').ops_text() / scope_denial()"}


def runbook_lines(X: dict, with_route: bool) -> list[tuple[str, str]]:
    ops = X["ops"]
    step5 = check(f"5 call cloud_stat for {X['bucket']};", ops, "ops_text")
    lines = [("ctx", "…" + step5)]
    if with_route:
        r = check(f"{COND}size report (allowed for this ticket; no token needed): api_post …/{X['ticket']}"
                  "/size-report, …", ops, "ops_text")
        lines.append(("route", r))
    else:
        if "size report" in ops:
            sys.exit("gen.py: NONE ops_text carries the route")
        lines.append(("gap", "no route line"))
    lines.append(("ctx", check("6 submit_report.", ops, "ops_text")))
    return lines


def denial_lines(X: dict) -> list[tuple[str, str]]:
    d = X["denial"]
    return [("ctx", check("denied: token scope service-log-level does not include storage reads", d, "denial")),
            ("key", check(f"access_key: {X['key']}", d, "denial")),
            ("ctx", check("scope: storage size reads, org-wide", d, "denial"))]


ROBOT = ('<svg class="rb" width="52" height="52" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" fill="#e8734a"/>'
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


def bar(label: str, c: dict | None, line: int, tick: str, hot: bool, pending: str) -> str:
    W = 300
    if c is None or not c["n"]:
        fill, val = "", f'<div class="bv pend">{pending}</div>'
    else:
        rate = c["rate"]
        fill = f'<div class="fill" style="width:{rate * W:.1f}px"></div>'
        val = f'<div class="bv">{round(rate * 100)}%<span class="cnt">{c["k"]} of {c["n"]} runs</span></div>'
    return (f'<div class="bar{" hot" if hot else ""}"><div class="bl">{label}</div><div class="bw"><div class="track">{fill}'
            f'<div class="tick" style="left:{line * W / 100:.0f}px"><span>{tick}</span></div></div>{val}</div></div>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJ / "data"))
    ap.add_argument("--out", default=str(FIG))
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    raw = data / MODEL / "raw"
    rows, mism = analysis.load_runs(raw) if raw.exists() else ([], [])
    if mism:
        sys.exit(f"gen.py: derive vs env mismatches: {mism[:3]}")
    pending = not any(r["phase"] == "main" for r in rows)
    res = analysis.analyse(rows) if not pending else None
    st = res["status"] if res else "PENDING"
    txt = json.loads((HERE / "text.json").read_text())
    T = dict(txt["common"], **txt[st if st in txt else "PENDING"])
    if pending:
        L, R = pending_example("none"), pending_example("runbook")
        nk = rk = None
        same = True
    else:
        lr, rr, nk, rk, same = pick(rows, st)
        if lr is None or rr is None:
            sys.exit(f"gen.py: no run matches the pick rule (none key={nk}, runbook key={rk})")
        L, R = example(data, lr), example(data, rr)
        if same and (L["ticket"], L["bucket"], L["key"]) != (R["ticket"], R["bucket"], R["key"]):
            sys.exit("gen.py: paired runs differ in ticket/bucket/key")
    bins = res["bins"] if res else {"reached": None, "never": None}
    fmt = {}
    if res:
        for b in ("reached", "never"):
            c = bins[b]
            fmt.update({f"{b}_k": c["k"], f"{b}_n": c["n"], f"{b}_pct": round((c["rate"] or 0) * 100)})
        fmt.update(n_main=res["counts"]["main"], ex_i=(rr["seed"] if same else f"{lr['seed']}/{rr['seed']}"))
    if rk is not None and ("head_key" in T or "head_route" in T):
        T["head"] = T["head_key" if rk else "head_route"]
        T["alt"] = T.get("alt_key" if rk else "alt_route", T.get("alt", ""))
    T = {k: (v.format(**fmt) if isinstance(v, str) else v) for k, v in T.items()}

    def outcome(X: dict, key: bool | None) -> str:
        if key is None:
            return f'<div class="act pend">{T["out_pending"]}</div>'
        return f'<div class="act {"key" if key else "route"}">&rarr; {T["out_key"] if key else T["out_route"]}</div>'

    def bubble(X: dict) -> str:
        if not X["thought"]:
            return f'<div class="bub pend">{T["bub_pending"]}</div>'
        return f'<div class="bub">&ldquo;{E(X["thought"])}&rdquo;</div>'

    panels = []
    for side, X, key, with_route in (("l", L, nk, False), ("r", R, rk, True)):
        panels.append(f'''<div class="panel p-{side}">
 <div class="pt">{T["title_" + side]}</div>
 <div class="cardh">{T["ops_h"]}</div>
 <div class="card">{card(runbook_lines(X, with_route), T["tags"])}</div>
 <div class="cardh">{T["deny_h"]}</div>
 <div class="card">{card(denial_lines(X), T["tags"])}</div>
 <div class="say">{ROBOT}<div class="sayr"><div class="who">{T["who"]}</div>{bubble(X)}{outcome(X, key)}</div></div>
</div>''')
    bars_html = (f'<div class="bars"><div class="barh">{T["barh"]}</div>'
                 + bar(T["bar_never"], bins["never"], 75, T["tick_never"], False, T["bar_pending"])
                 + bar(T["bar_reached"], bins["reached"], 25, T["tick_reached"], True, T["bar_pending"]) + "</div>")
    src = (f"Sources (kran-research, branch swarm/defense-1127-reach-law-gpt-oss, projects/reach-law-gpt-oss):\n"
           f"  numbers: analysis.analyse(load_runs(data/{MODEL}/raw)) -> status {st}\n"
           f"  left example: {L['src']}\n  right example: {R['src']}\n"
           f"  example rule: figures/figure1/README.md (frozen before any counted run, commit 4ca3ad24c)\n"
           f"  built by swarm-defense-1127-reach-law-gpt-oss/figure1/gen.py")
    doc = (HERE / "template.html").read_text()
    doc = (doc.replace("{{SRC}}", src.replace("--", "- -")).replace("{{HEAD}}", T["head"]).replace("{{SUB}}", T["sub"])
           .replace("{{PANELS}}", "\n".join(panels)).replace("{{BARS}}", bars_html).replace("{{FOOT}}", T["foot"])
           .replace("{{TITLE}}", E(re.sub("<[^>]+>", "", T["head"]))))
    out.mkdir(parents=True, exist_ok=True)
    (out / "figure1.html").write_text(doc)
    (HERE / "figure1_alt.txt").write_text(T["alt"].replace('"', "'") + "\n")
    print(f"status {st}; pair same index={same}; none key={nk}; runbook key={rk}")
    if res:
        print({b: f"{bins[b]['k']}/{bins[b]['n']}" for b in bins})
    print("L:", L["src"], "|", L["thought"])
    print("R:", R["src"], "|", R["thought"])


if __name__ == "__main__":
    main()
