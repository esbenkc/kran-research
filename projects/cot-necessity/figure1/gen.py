#!/usr/bin/env python3
"""Owain: Figure 1 for cot-necessity. Usage: [DATA=<data dir>] [TEXT=<text json>] gen.py <out figure1.html>

PENDING mode while data/reader/gate.json is missing; LANDED mode when gate.json and all seeds exist. The example
rule is frozen in README.md. Every number is recomputed from the rows and asserted against data/summary.json."""
from __future__ import annotations

import html
import json
import os
import sys
from fractions import Fraction as F
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
DATA = Path(os.environ.get("DATA", P / "data")).resolve()
TEXT = Path(os.environ.get("TEXT", H / "text.json")).resolve()
K, SEEDS, N_CP = 8, 3, 6
sys.path.insert(0, str(P))
import status as ST  # noqa: E402  the FROZEN RULE (task_held), reviewed by Vera

E = html.escape
RED, INK, GREY, MUTED = "#b3261e", "#1d1d1f", "#9aa0a8", "#7a7e87"


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def reader_pred(r: dict, primary: str, thr: float) -> str | None:
    """What the reader names from the notes: logit reader = score above the calibrated cut -> heads;
    generated reader = its forced answer (None = parse failure, scored 1/2 by the frozen rule)."""
    if primary == "logit":
        return "heads" if r["logit"] > thr else "tails"
    g = r.get("gen_forced")
    return g if g in ("heads", "tails") else None


def load_seed(s: int, primary: str, thr: float, S: int) -> dict:
    run = f"main_k{K}_s{s}"
    ev_dir, sc_dir = DATA / "runs" / run / "evals", DATA / "reader" / "scores" / run
    steps = sorted(int(p.stem.split("_")[1]) for p in ev_dir.glob("step_*.jsonl"))
    assert len(steps) == N_CP and steps[0] == 0 and steps[-1] == S, (run, steps, S)
    cps = []
    for st in steps:
        ev, sc = rows(ev_dir / f"step_{st:04d}.jsonl"), rows(sc_dir / f"step_{st:04d}.jsonl")
        assert [r["id"] for r in ev] == [r["id"] for r in sc], (run, st)
        assert all(a["answer"] == b["answer"] for a, b in zip(ev, sc)), (run, st)
        pts = ST.reader_points(sc, primary, thr)
        cps.append({"step": st, "n": len(ev), "k_task": sum(r["correct"] for r in ev), "k_read": sum(pts, F(0)),
                    "sum_chars": F(sum(r["thought_chars"] for r in ev)), "ev": {r["id"]: r for r in ev},
                    "sc": {r["id"]: r for r in sc}})
    v = ST.seed_verdict(cps)  # adds task_held per checkpoint
    held = [c for c in cps if c["step"] > 0 and c["task_held"]]
    after = held[-1] if held else cps[-1]
    return {"run": run, "cps": cps, "verdict": v, "after": after}


def check_summary(seeds: list[dict]) -> None:
    sp = DATA / "summary.json"
    assert sp.exists(), "run status.py first (data/summary.json missing)"
    sm = json.loads(sp.read_text())
    for sd in seeds:
        ref = sm["seeds"][sd["run"]]["checkpoints"]
        assert [c["step"] for c in ref] == [c["step"] for c in sd["cps"]], sd["run"]
        for a, b in zip(sd["cps"], ref):
            n = a["n"]
            assert abs(a["k_task"] / n - b["task_acc"]) < 1e-9, (sd["run"], a["step"], "task")
            assert abs(float(a["k_read"]) / n - b["reader_acc"]) < 1e-9, (sd["run"], a["step"], "reader")
            assert abs(float(a["sum_chars"]) / n - b["mean_chars"]) < 1e-9, (sd["run"], a["step"], "chars")
            assert bool(a["task_held"]) == bool(b["task_held"]), (sd["run"], a["step"], "held")


def pick_example(seed0: dict) -> str:
    """Frozen rule (README.md): first eval prompt by id, start Heads, end Tails, solved by seed 0 at step 0."""
    ev0 = seed0["cps"][0]["ev"]
    for it in rows(P / "inputs" / f"eval_k{K}.jsonl"):
        if it["start"] == "heads" and it["answer"] == "tails" and ev0[it["id"]]["correct"]:
            return it["id"]
    raise SystemExit("no example prompt fits the frozen rule")


ROBOT = ('<svg width="{w}" height="{w}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="5" r="3" '
         'fill="{a}"/><rect x="23" y="6" width="2" height="6" fill="#333"/><rect x="6" y="12" width="36" height="30" '
         'rx="9" fill="#fff" stroke="#222" stroke-width="2.4"/><rect x="2" y="22" width="5" height="11" rx="2" '
         'fill="{a}"/><rect x="41" y="22" width="5" height="11" rx="2" fill="{a}"/><rect x="11" y="17" width="26" '
         'height="20" rx="6" fill="#1f2a44"/><circle cx="19" cy="25" r="4" fill="{e}"/><circle cx="29" cy="25" r="4" '
         'fill="{e}"/><path d="M20 31 q4 3 8 0" stroke="{e}" stroke-width="1.5" fill="none"/></svg>')


def robot(kind: str, w: int = 44) -> str:
    # model studied (Qwen3-8B): orange ears; reader (gpt-oss-20b): teal ears and green eyes
    return ROBOT.format(w=w, a="#e8734a", e="#7fd3ff") if kind == "model" else ROBOT.format(w=w, a="#2a9d8f", e="#a8f0c6")


def chars(n: int) -> str:
    return f"{n} character" + ("" if n == 1 else "s")


def side(x: str | None) -> str:
    return {"heads": "Heads", "tails": "Tails", None: "no answer"}[x]


def verdict_span(pred: str | None, truth: str, wrong_red: bool = True) -> str:
    ok = pred == truth
    mark = "&#10003; right" if ok else "&#10007; wrong"
    cls = "ok" if ok else ("bad" if wrong_red else "meh")
    return f'<b>{side(pred)}</b> <span class="v {cls}">{mark}</span>'


def notes_box(thought: str, maxc: int) -> str:
    t = thought.strip()
    shown = E(t) if len(t) <= maxc else E(t[:maxc].rstrip()) + " &hellip;"
    return f'<div class="notes">{shown}</div>'


def strip_svg(T: dict, seeds: list[dict] | None) -> str:
    """Two 0-100% panels, one row per seed: answers right, reader right from the notes alone."""
    W, lab, gap, top, rh = 1120, 100, 64, 76, 56
    pw = (W - lab - gap) // 2
    hgt = top + rh * SEEDS + (36 if seeds is None else 62)
    x0 = [lab, lab + pw + gap]
    X = lambda p, v: x0[p] + 24 + (pw - 52) * v
    o = [f'<svg class="strip" width="{W}" height="{hgt}" viewBox="0 0 {W} {hgt}" role="img">']
    for p, title in enumerate((T["panel_task"], T["panel_reader"])):
        o.append(f'<text x="{x0[p] + 10}" y="24" class="pt">{title}</text>')
        if p == 1:  # guess band: within 10 points of 50%
            o.append(f'<rect x="{X(1, .4):.1f}" y="{top - 14}" width="{X(1, .6) - X(1, .4):.1f}" '
                     f'height="{rh * SEEDS + 8}" fill="#eceef1"/>')
            o.append(f'<text x="{X(1, .5):.1f}" y="{top - 20}" class="band" text-anchor="middle">{T["band"]}</text>')
        for v in (0, .5, 1):
            o.append(f'<line x1="{X(p, v):.1f}" x2="{X(p, v):.1f}" y1="{top - 14}" y2="{top + rh * SEEDS - 6}" '
                     f'stroke="#e1e3e7" stroke-width="1"/>')
            o.append(f'<text x="{X(p, v):.1f}" y="{top + rh * SEEDS + 18}" class="ax" text-anchor="middle">'
                     f'{round(v * 100)}%</text>')
    for s in range(SEEDS):
        y = top + rh * s + 26
        o.append(f'<text x="0" y="{y + 7}" class="rl">{T["run"]} {s + 1}</text>')
        if seeds is None:
            continue
        sd = seeds[s]
        c0, ca = sd["cps"][0], sd["after"]
        n = c0["n"]
        vals = [(c0["k_task"] / n, ca["k_task"] / ca["n"]), (float(c0["k_read"]) / n, float(ca["k_read"]) / ca["n"])]
        for p, (a, b) in enumerate(vals):
            col = RED if p == 1 else INK
            if p == 0:  # the frozen task line: 80% of this seed's start
                g = X(0, float(ST.TASK_REL) * a)
                o.append(f'<line x1="{g:.1f}" x2="{g:.1f}" y1="{y - 11}" y2="{y + 11}" stroke="{MUTED}" '
                         f'stroke-width="2" stroke-dasharray="3 3"/>')
            o.append(f'<line x1="{X(p, a):.1f}" x2="{X(p, b):.1f}" y1="{y}" y2="{y}" stroke="{GREY}" stroke-width="3"/>')
            o.append(f'<circle cx="{X(p, a):.1f}" cy="{y}" r="8" fill="#fff" stroke="{GREY}" stroke-width="3"/>')
            o.append(f'<circle cx="{X(p, b):.1f}" cy="{y}" r="9" fill="{col}"/>')
            o.append(f'<text x="{X(p, b):.1f}" y="{y - 15}" class="val" fill="{col}" text-anchor="middle">{pct(b)}</text>')
    if seeds is None:  # one muted box over both panels
        o.append(f'<rect x="{x0[0]}" y="{top - 4}" width="{W - x0[0] - 2}" height="{rh * SEEDS}" rx="12" '
                 f'fill="#fbfaf8" stroke="#c6cad1" stroke-width="2" stroke-dasharray="6 5"/>')
        o.append(f'<text x="{(x0[0] + W) / 2:.0f}" y="{top + rh * SEEDS / 2 - 2:.0f}" class="pend" '
                 f'text-anchor="middle">{T["pending_row"]}</text>')
    if seeds is not None:
        o.append(f'<text x="{x0[0] + 10}" y="{hgt - 2}" class="key">{T["key"]}</text>')
    o.append("</svg>")
    return "".join(o)


def main(out: Path) -> None:
    T0 = json.loads(TEXT.read_text())
    gate_p = DATA / "reader" / "gate.json"
    mode = "landed" if gate_p.exists() else "pending"
    T = {**T0["shared"], **T0[mode]}
    if "[FILL]" in json.dumps(T):
        raise SystemExit(f"text.json [{mode}] still holds [FILL]")
    step0 = json.loads((DATA / "step0" / "summary.json").read_text())[str(K)]
    n_heads = round(step0["share_pred_heads"] * step0["n"])
    assert n_heads == step0["n"] == 400, (n_heads, step0)  # the always-Heads fact the figure prints
    col1 = (f'<div class="card"><div class="ct">{T["col1_title"]}</div><div class="cs">{T["col1_sub"]}</div>'
            f'<div class="notes empty">{T["no_notes"]}</div>'
            f'<div class="say">{robot("model")}<div class="bub"><b>Heads</b>, <b>Heads</b>, <b>Heads</b>, &hellip;</div></div>'
            f'<div class="note">{T["col1_note"].format(n=n_heads)}</div></div>')
    if mode == "pending":
        seeds = None
        ex_user = rows(P / "inputs" / f"eval_k{K}.jsonl")
        ex = next(it for it in ex_user if it["start"] == "heads" and it["answer"] == "tails")
        col2 = (f'<div class="card pend"><div class="ct">{T["col2_title"]}</div><div class="cs">{T["col2_sub"]}</div>'
                f'<div class="wait">{T["pending_cell"]}</div></div>')
        col3 = (f'<div class="card pend hl"><div class="ct">{T["col3_title"]}</div><div class="cs">{T["col3_sub"]}</div>'
                f'<div class="wait">{T["pending_cell"]}</div></div>')
    else:
        gate = json.loads(gate_p.read_text())
        job = json.loads((DATA / "job_state.json").read_text())
        primary, thr, S = gate["primary"], gate["threshold"], job["steps"]
        assert primary in ("logit", "generated"), primary
        assert "8B" in job["model"] or T["model"] in job["model"], (job["model"], T["model"])
        seeds = [load_seed(s, primary, thr, S) for s in range(SEEDS)]
        check_summary(seeds)
        eid = pick_example(seeds[0])
        ex = next(it for it in rows(P / "inputs" / f"eval_k{K}.jsonl") if it["id"] == eid)
        s0 = seeds[0]
        b_ev, b_sc = s0["cps"][0]["ev"][eid], s0["cps"][0]["sc"][eid]
        a_cp = s0["after"]
        a_ev, a_sc = a_cp["ev"][eid], a_cp["sc"][eid]
        held_txt = T["held"] if a_cp["task_held"] else T["not_held"]

        def reader_line(sc):
            return (f'<div class="say rd">{robot("reader")}<div class="bub">'
                    f'{verdict_span(reader_pred(sc, primary, thr), "tails")}</div><div class="who">{T["reader_says"]}</div></div>')

        def model_line(ev):
            return (f'<div class="say">{robot("model")}<div class="bub">{verdict_span(ev["pred"], "tails")}</div>'
                    f'<div class="who">{T["model_says"]}</div></div>')

        others = []
        for sd in seeds[1:]:
            c = sd["after"]
            others.append(T["other_run"].format(run=int(sd["run"][-1]) + 1, step=c["step"],
                                                chars=chars(c["ev"][eid]["thought_chars"]),
                                                reader=side(reader_pred(c["sc"][eid], primary, thr)),
                                                ok="right" if reader_pred(c["sc"][eid], primary, thr) == "tails" else "wrong"))
        col2 = (f'<div class="card"><div class="ct">{T["col2_title"]}</div>'
                f'<div class="cs">{T["col2_sub"].format(chars=chars(b_ev["thought_chars"]))}</div>'
                f'{notes_box(b_ev["thought"], T["max_notes"])}{model_line(b_ev)}{reader_line(b_sc)}</div>')
        col3 = (f'<div class="card hl"><div class="ct">{T["col3_title"]}</div>'
                f'<div class="cs">{T["col3_sub"].format(chars=chars(a_ev["thought_chars"]), step=a_cp["step"], held=held_txt)}</div>'
                f'{notes_box(a_ev["thought"], T["max_notes"])}{model_line(a_ev)}{reader_line(a_sc)}'
                f'<div class="note">{"<br>".join(others)}</div></div>')
    flips = " &middot; ".join(T["flip_same"] if a == 0 else f'<b>{T["flip_diff"]}</b>' for a in ex["acts"])
    prompt = (f'<div class="ask"><span class="pl">{T["prompt_label"]}</span> A coin starts on <b>Heads</b>. '
              f'It is flipped 8 times: {flips}. What side did it land on after the final flip?</div>'
              f'<div class="truth">{T["truth"].format(d=sum(ex["acts"]))}</div>')
    src = (f"<!--\nSources (kran-research, branch swarm/oracle-1124-cot-necessity, projects/cot-necessity):\n"
           f"  Step 0 count: data/step0/summary.json (k=8: n {step0['n']}, share_pred_heads {step0['share_pred_heads']})\n"
           f"  prompt: inputs/eval_k8.jsonl id {ex['id']}\n")
    if seeds is not None:
        src += (f"  notes, answers: data/runs/main_k8_s0/evals/step_0000.jsonl and step_{s0['after']['step']:04d}.jsonl\n"
                f"  reader: data/reader/scores/main_k8_s*/ + data/reader/gate.json (primary {primary}, threshold {thr})\n"
                f"  strip: recomputed from the same rows, asserted equal to data/summary.json (status.py)\n")
    else:
        src += "  PENDING: J1 (labq zfp2nttcl0) has not landed; columns 2-3 and the strip are placeholders\n"
    src += "  built by figure1/gen.py\n-->"
    body = (f'<div class="head">{T["head"]}</div><div class="sub">{T["sub"]}</div>'
            f'<div class="kick"><b>Question:</b> <span class="q">&ldquo;{E(T["question"])}&rdquo;</span><br>'
            f'<b>Answer:</b> {T["answer"]}</div>{prompt}'
            f'<div class="cols">{col1}<div class="arrow">&rarr;</div>{col2}<div class="arrow">&rarr;</div>{col3}</div>'
            f'<div class="st">{T["strip_title"]}</div>{strip_svg(T, seeds)}'
            f'<div class="foot">{T["foot"]} &middot; <b>blog.kran.ai/lab</b></div>')
    out.write_text(PAGE.replace("{{SRC}}", src).replace("{{TITLE}}", E(T["head"])).replace("{{BODY}}", body))
    print(f"{mode}: wrote {out} (example {ex['id']})")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Figure 1: {{TITLE}}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
{{SRC}}
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: #fbfaf8; }
  body { width: 1200px; padding: 28px 40px 22px; font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
         color: #1d1d1f; -webkit-font-smoothing: antialiased; }
  .head { font-size: 42px; font-weight: 800; letter-spacing: -0.015em; line-height: 1.14; }
  .sub { font-size: 23px; line-height: 1.3; color: #5f636c; margin-top: 10px; }
  .sub b { color: #2b2e35; }
  .kick { font-size: 22px; line-height: 1.3; color: #2b2e35; background: #eef0f3; border-left: 6px solid #b3261e; padding: 9px 14px; border-radius: 6px; margin-top: 16px; }
  .kick .q { color: #4a4d55; }
  .ask { margin-top: 18px; background: #2f6fe4; color: #fff; font-size: 22px; line-height: 1.32; border-radius: 18px 18px 4px 18px; padding: 11px 18px; }
  .ask b { font-weight: 800; }
  .ask .pl { display: inline-block; font-size: 17px; font-weight: 700; background: rgba(255,255,255,0.2); border-radius: 6px; padding: 1px 7px; margin-right: 6px; vertical-align: 2px; }
  .truth { text-align: right; font-size: 19px; color: #4a4d55; margin-top: 5px; }
  .cols { display: grid; grid-template-columns: 1fr 26px 1.25fr 26px 1.25fr; margin-top: 8px; align-items: stretch; }
  .arrow { font-size: 30px; color: #9aa0a8; display: flex; align-items: center; justify-content: center; }
  .card { background: #f1f2f4; border-radius: 16px; padding: 14px 14px 12px; display: flex; flex-direction: column; gap: 9px; }
  .card.hl { background: #fdeceb; }
  .ct { font-size: 23px; font-weight: 800; line-height: 1.15; }
  .hl .ct { color: #b3261e; }
  .cs { font-size: 18px; color: #5f636c; line-height: 1.25; margin-top: -4px; }
  .notes { font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; font-size: 17px; line-height: 1.35; background: #fff; border: 2px dashed #b9bec6; border-radius: 10px; padding: 8px 10px; word-break: break-word; }
  .hl .notes { border-color: #e39a93; }
  .notes.empty { color: #9aa0a8; font-family: "Inter", sans-serif; font-style: italic; text-align: center; }
  .say { display: flex; align-items: center; gap: 9px; }
  .say svg { flex: none; }
  .bub { background: #fff; border-radius: 13px 13px 13px 4px; padding: 6px 12px; font-size: 21px; box-shadow: 0 2px 7px rgba(0,0,0,0.08); }
  .who { font-size: 17px; color: #5f636c; line-height: 1.2; flex: 1; }
  .v { font-size: 18px; font-weight: 800; }
  .v.ok { color: #1d4f9c; } .v.bad { color: #b3261e; } .v.meh { color: #5f636c; }
  .note { font-size: 18px; line-height: 1.28; color: #3b3e46; }
  .wait { flex: 1; display: flex; align-items: center; justify-content: center; text-align: center; border: 2px dashed #c6cad1; border-radius: 12px; color: #7a7e87; font-size: 21px; padding: 18px; min-height: 230px; }
  .card.pend .ct { color: #7a7e87; }
  .st { font-size: 23px; font-weight: 800; margin-top: 20px; }
  .strip { display: block; margin-top: 4px; }
  .strip text { font-family: "Inter", system-ui, sans-serif; }
  .strip .pt { font-size: 20px; font-weight: 700; fill: #2b2e35; }
  .strip .ax { font-size: 17px; fill: #7a7e87; }
  .strip .band { font-size: 17px; fill: #5f636c; }
  .strip .rl { font-size: 20px; font-weight: 700; fill: #3b3e46; }
  .strip .val { font-size: 20px; font-weight: 800; }
  .strip .pend { font-size: 19px; fill: #9aa0a8; font-style: italic; }
  .strip .key { font-size: 17px; fill: #5f636c; }
  .foot { font-size: 17px; color: #7a7e87; margin-top: 8px; line-height: 1.3; }
  .foot b { color: #3a3d45; font-weight: 700; }
</style>
</head>
<body>
{{BODY}}
</body>
</html>
"""

if __name__ == "__main__":
    main(Path(sys.argv[1]))
