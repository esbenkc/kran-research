"""Write proof-matrix.md from results/analysis.json, the per-run scores and the parent's API cells. No model calls.

    uv run python analysis.py && uv run python proof_matrix.py

One row per (agent, cell, seed group) and one pooled row per cell; the headline is the warned trap-present cell (W).
Cells of the two laptop agents with no runs yet are listed as PENDING with their labq job id.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = {"gpt-oss-20b": "c76l6dj5vo", "qwen3-8b": "a1j47ed7n0"}
OUT_TAG = {"gpt-oss-20b": "gpt-oss-20b-d4"}  # DEVIATIONS D4: the 10 guard runs in web/out/gpt-oss-20b are not counted
NAMES = {"gpt-oss-20b": "gpt-oss-20b", "qwen3-8b": "Qwen3-8B", "qwen3-1.7b": "Qwen3-1.7B (mini rung)",
         "gpt-5.4-nano": "GPT-5.4 nano (parent)", "gpt-5.4-mini": "GPT-5.4 mini (parent)",
         "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite (parent)"}
CELLS = [("W", "warning-cued", "trap present · warned", "none"),
         ("U", "neutral-cued", "trap present · unwarned", "no-warning baseline (neutral sentence)"),
         ("WA", "warning-none", "trap absent (same link visible) · warned", "trap absent: visible route"),
         ("UA", "neutral-none", "trap absent (same link visible) · unwarned", "trap absent: visible route, no warning (competence gate)"),
         ("SW", "warning-safe", "hidden link + visible copy · warned", "safe exit"),
         ("SU", "neutral-safe", "hidden link + visible copy · unwarned", "safe exit, no warning")]
SEEDS = {"W": [0, 1, 2], "U": [0, 1, 2], "WA": [0, 1, 2], "UA": [0, 1, 2], "SW": [0, 1], "SU": [0, 1]}


def scores(tag: str) -> list[dict]:
    p = ROOT / "web" / "out" / OUT_TAG.get(tag, tag) / "scores.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def frac(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n:.2f}" if n else "0/0"


def main() -> None:
    an = json.loads((ROOT / "results" / "analysis.json").read_text()) if (ROOT / "results" / "analysis.json").exists() else {}
    lines = ["# Proof matrix: trap-aware-canary-open-weight", "",
             "Built by `proof_matrix.py` from `results/analysis.json` (`analysis.py`: PREREG s4 estimator, agent-equal pooling,",
             "template-stratified bootstrap) and `web/out/<agent>/scores.jsonl` (`web/score.py`, from each job's `access.log`).",
             "fit = a GET of the run's own fit-link URL (hidden in trap-present cells, visible in trap-absent cells). Seed group",
             "s = run index mod 3 (it also fixes a third of the prompts; every group covers all 5 templates). Laptop-queue cells",
             "with no runs yet are PENDING with their labq job id.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for part, agents, claim_main in (("main", ["gpt-oss-20b", "qwen3-8b"], "headline"), ("rung", ["qwen3-1.7b"], "rung")):
        for short, cell, desc, control in CELLS:
            claim = (claim_main if short == "W" else f"{claim_main}: {short}") if part == "main" else f"capability rung: {short}"
            for a in agents:
                rows = [r for r in scores(a) if r["cell"] == cell]
                art = f"web/out/{OUT_TAG.get(a, a)}/scores.jsonl"
                if not rows:
                    if part == "main":
                        lines.append(f"| {claim} | seeds {SEEDS[short]} | {NAMES[a]} · {desc} | {control} | PENDING: labq {JOBS[a]} | {art} |")
                    continue
                by = defaultdict(list)
                for r in rows:
                    by[r["seed"]].append(r)
                for s in sorted(by):
                    rs = by[s]
                    k = sum(r["fit"] for r in rs)
                    kc = sum(r["fit"] and r["correct"] for r in rs)
                    kt = sum(r.get("fit_tok", False) for r in rs)
                    lines.append(f"| {claim} | {s} (n={len(rs)}) | {NAMES[a]} · {desc} | {control} | fit {frac(k, len(rs))}; "
                                 f"fit and correct {kc}; canary token requested {kt}; format fails {sum(r['format_fail'] for r in rs)} | {art} |")
                c = an.get(part, {}).get("cells", {}).get(short, {}).get(a)
                if c:
                    f = c["fit"]
                    lines.append(f"| {claim} | all (n={f['n']}) | {NAMES[a]} · {desc} | {control} | fit {f['k']}/{f['n']} = "
                                 f"{f['rate']:.2f} {f['ci95']}; seed groups {c['seed_level_fit']}; canary token requested "
                                 f"{c['fit_tok']['k']}/{c['fit_tok']['n']} {c['fit_tok']['ci95']}; fit on 0-cap runs "
                                 f"{c.get('fit_nocap', {}).get('k')}/{c.get('fit_nocap', {}).get('n')} | results/analysis.json |")
    st = an.get("main", {}).get("status", {})
    if "pooled_W_fit" in st:
        sl = st.get("seed_level_W_fit", {})
        lines.append(f"| headline | all | 2 agents pooled · trap present · warned | none | fit {st['pooled_W_fit']} {st['ci95']} "
                     f"-> {st['status']} (by rule {st['by_rule']}; blocked by {st['blocked_by']}) | results/analysis.json |")
        seed_line = (f"seed-level: {sl['estimate']} [{sl['ci95'][0]}, {sl['ci95'][1]}] ({sl['method']})" if "estimate" in sl
                     else "seed-level: pending (fewer than 2 complete seed groups)")
    else:
        lines.append(f"| headline | all | 2 agents pooled · trap present · warned | none | PENDING: labq {JOBS['gpt-oss-20b']}, "
                     f"{JOBS['qwen3-8b']} -> UNCONFIRMED: waiting for laptop queue | results/analysis.json |")
        seed_line = f"seed-level: pending (laptop queue jobs {JOBS['gpt-oss-20b']}, {JOBS['qwen3-8b']} not run yet)"
    D, Dt = an.get("main", {}).get("D"), an.get("main", {}).get("D_tok")
    if D:
        lines.append(f"| deterrence D (frozen reading) | all | agents {D['agents']} · (U-W) present minus (UA-WA) absent | trap absent cells | "
                     f"D {D['point']} {D['ci95']} -> {D['reading']} | results/analysis.json |")
    if Dt:
        lines.append(f"| deterrence D on canary token requested (reported, defined after the Qwen3-8B data) | all | agents {Dt['agents']} | "
                     f"trap absent cells | D {Dt['point']} {Dt['ci95']} | results/analysis.json |")
    rung = an.get("rung", {}).get("status", {}).get("seed_level_W_fit", {})
    # Parent API rows (strict parser; parent room defense-1110, commit af50d21f9 via results/parent_api_cells.csv)
    p = ROOT / "results" / "parent_api_cells.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            if r["carrier"] == "web" and r["event"] == "fit" and r["cell"] in ("warning-cued", "neutral-cued") and r["agent"] != "pooled":
                short = "W" if r["cell"] == "warning-cued" else "U"
                lines.append(f"| parent comparison: {short} | idx 0-59 (n={r['n']}) | {NAMES.get(r['agent'], r['agent'])} · trap present · "
                             f"{'warned' if short == 'W' else 'unwarned'} | {'none' if short == 'W' else 'no-warning baseline'} | "
                             f"fit {r['k']}/{r['n']} = {float(r['rate']):.2f} [{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}] | results/parent_api_cells.csv |")
    lines += ["", seed_line]
    if "estimate" in rung:
        lines.append(f"seed-level (rung, Qwen3-1.7B, reported only): {rung['estimate']} [{rung['ci95'][0]}, {rung['ci95'][1]}] ({rung['method']})")
    (ROOT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    print(f"wrote proof-matrix.md ({sum(1 for x in lines if x.startswith('| ') and not x.startswith('| claim'))} rows)")


if __name__ == "__main__":
    main()
