"""Write proof-matrix.md from results/analysis.json: one row per cell (agent x seed block x arm), controls marked.

    .venv/bin/python proof_matrix.py [results/analysis.json] [proof-matrix.md]

'headline' rows are the cells the answer to the question rests on: the status agents (gpt-oss-20b, Qwen3-8B), one
row per seed block (run index i % 3) plus the pooled row. Side-row agents (never status) get their own claim names.
A status agent with no data yet gets one row per seed block that says it is waiting in the laptop queue.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUEUE = {"gpt-oss-20b": ("gpt-oss-20b-low-full", "56pkb4cysk"), "Qwen3-8B": ("qwen3-8b-nothink-full", "fdce5qspgd")}


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, nd) for v in x) + "]"
    return str(x)


def agent_rows(claim: str, label: str, a: dict) -> list[str]:
    d = a["dir"]
    L, g = a.get("law") or {}, a["gate"]
    rows = [f"| {claim}: skill gate | - | {label} visible link | none | {g['k']}/{g['n']} correct (>= 8 needed) | data/{d}/raw/visible |"]
    if not a.get("n_maze"):
        return rows
    chi, wb = L.get("chi") or {}, L.get("weibull") or {}
    rows.append(f"| {claim} | all | {label} maze, pooled | none | verdict {L.get('verdict')}; give-ups G={L.get('G')} "
                f"(pages 1-3: {L.get('G3')}/{L.get('E3')} decisions); h3={f(L.get('h3'))}; h_all={f(L.get('h_all'))}; "
                f"KM median {f(L.get('km_median'))} pages vs 3-page forecast {f(L.get('pred_median'))} (R={f(L.get('R'))}); "
                f"chi-square p={f(chi.get('p'))} (df {chi.get('df')}); Weibull shape k={f(wb.get('k'))} p={f(wb.get('p'))}; "
                f"cap share {f(a.get('cap_share'))}; entered {a['n_entered']}/{a['n_maze']} | results/analysis.json |")
    for b in ("0", "1", "2"):
        bl = (a.get("blocks") or {}).get(b)
        if bl:
            rows.append(f"| {claim} | {b} | {label} maze, block i % 3 == {b} | none | entered {bl['n']}; give-ups G={bl['G']} "
                        f"(pages 1-3 {bl['G3']}/{bl['E3']}); h_all={f(bl['h_all'])}; KM median {f(bl['km_median'])}; R_b={f(bl['R'])} "
                        f"| data/{d}/raw/maze |")
    de = a.get("dead_end") or {}
    by = {}
    for r in (a.get("runs") or {}).get("deadend", []):
        by.setdefault(r["block"], []).append(r)
    for b in sorted(by):
        rs = [r for r in by[b] if r["entered"]]
        rows.append(f"| {claim}: dead-end control | {b} | {label} dead end, block i % 2 == {b} | one dead-end page, no continue "
                    f"link | entered {len(rs)}/{len(by[b])}; pages read {sorted(r['D'] for r in rs)}; causes "
                    f"{sorted({r['cause'] for r in rs})} | data/{d}/raw/deadend |")
    if de:
        rows.append(f"| {claim}: dead-end control | all | {label} dead end, pooled | one dead-end page | entered "
                    f"{de.get('n_entered')}/{de.get('n')}; share that left after page 1 {f(de.get('exit_page1'))}; median calls "
                    f"{f(de.get('calls_median'))} | results/analysis.json |")
    ae = a.get("any_exit") or {}
    if ae:
        rows.append(f"| {claim}: any exit (defender row) | all | {label} maze, every exit counts | none | h3={f(ae.get('h3'))}; "
                    f"h_all={f(ae.get('h_all'))}; KM median {f(ae.get('km_median'))}; R={f(ae.get('R'))} | results/analysis.json |")
    rows.append(f"| {claim}: exit causes | all | {label} maze | none | {json.dumps(a.get('causes'))}; parse/protocol "
                f"errors {json.dumps(a.get('per_run_errors'))}; answers with an invented 2025 figure "
                f"{a.get('invented_figure_answers')}, quoting the site's 2024 figure {a.get('stale_figure_answers')} | results/analysis.json |")
    return rows


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "analysis.json"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "proof-matrix.md"
    res = json.loads(src.read_text())
    o = res["overall"]
    rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    seed_lines = []
    for label in ("gpt-oss-20b", "Qwen3-8B"):
        a = res["agents"].get(label)
        if a and a.get("n_maze"):
            rows += agent_rows("headline", label, a)
            s = a.get("seed_level_h_all") or {}
            if s.get("ci"):
                seed_lines.append(f"seed-level: {label} h_all = {f(s['estimate'])} {f(s['ci'])} ({s['method']})")
        else:
            d, job = QUEUE[label]
            for b in ("0", "1", "2"):
                rows.append(f"| headline | {b} | {label} maze, block i % 3 == {b} | none | waiting for laptop queue (job {job}, "
                            f"full prefill rerun, D6); no counted run yet; output will be data/{d} | DEVIATIONS.md |")
    for label, a in res.get("side_row", {}).items():
        if "failed D5 check" in label:
            claim = "preview, prefix reuse on, failed D5 check, not counted (D6)"
            pc = json.loads((ROOT / "data" / a["dir"] / "prefix_check.json").read_text())
            rows.append(f"| {claim}: prefix-reuse check | - | {label.split(' (')[0]} greedy toy transcript, reuse vs full "
                        f"prefill | full prefill | {pc['same_reply']}/{pc['steps']} identical replies (95% needed); pass "
                        f"{pc['pass']} | data/{a['dir']}/prefix_check.json |")
        elif a["dir"] == "gpt-5-nano":
            claim = "side row, closed API model, never status (D4)"
        else:
            claim = "side row gate (D4)"
        rows += agent_rows(claim, label, a)
        s = a.get("seed_level_h_all") or {}
        if s.get("ci") and a.get("n_maze"):
            seed_lines.append(f"side row (not the headline): {label} h_all = {f(s['estimate'])} {f(s['ci'])} ({s['method']})")
    sk = ROOT / "skeptics" / "skeptic"
    # Vera's independent rebuild (her own code, night 2026-11-19), one row per agent folder; never status
    for fn, claim, label in (("qwen_reuseON.json", "preview rebuild (skeptic), not counted (D6)", "Qwen3-8B reuse-on"),
                             ("gptoss_reuseON_partial.json", "preview rebuild (skeptic), not counted (D6)",
                              "gpt-oss-20b reuse-on, 9 runs"),
                             ("nano_side.json", "side row rebuild (skeptic)", "gpt-5-nano")):
        if not (sk / fn).exists():
            continue
        r = json.loads((sk / fn).read_text())
        L, V = r["law_giveup"], r["law_giveup_km_decisionrisk"]
        rows.append(f"| {claim} | all | {label} maze, Vera's own code | none | {V['verdict']}; G3/E3 {L['G3']}/{L['E3']}, "
                    f"G/E {L['G']}/{L['E']}, h_all {f(L['h_all'], 4)}, KM median {L['M_obs']}, M_pred {f(L.get('M_pred'), 4)}, "
                    f"R {f(V.get('R'), 4)}, chi p {f(L['p_pearson'])}, Weibull k {f(L.get('weibull_k'), 4)} p "
                    f"{f(L.get('p_weibull'))}; entered {r['maze_entered']}/{r['maze_runs']} | skeptics/skeptic/{fn} |")
    nq = sk / "null_qwen_reuseON.txt"
    if nq.exists():
        n = json.loads(nq.read_text())
        rows.append(f"| preview null (skeptic), not counted (D6) | all | Qwen3-8B reuse-on: constant hazard {f(n['h_all'], 4)}, "
                    f"{n['n']} runs, {n['reps']} draws | simulated constant hazard | share of draws reaching the observed "
                    f"Pearson p {f(n['P(pearson p <= observed 6.66e-09)'])}, Weibull p {f(n['P(weibull p <= observed 1.33e-04)'])}; "
                    f"frozen rule on the null: holds {f(n['holds'])}, fails {f(n['fails_any'])} | skeptics/skeptic/null_qwen_reuseON.txt |")
    old = sk / "2026-11-18"  # last night's rebuild of the gpt-5-nano side row (Vera moved it here)
    if (old / "nulls_gpt-5-nano.json").exists():
        n = json.loads((old / "nulls_gpt-5-nano.json").read_text())
        b, blk = n["N1_parametric_bootstrap"], n.get("N2_blocks", {})
        rows.append(f"| side row null (skeptic) | all | gpt-5-nano: constant hazard {f(b['h'], 4)}, {b['reps']} draws | parametric bootstrap "
                    f"| share of draws reaching the observed chi p {f(b['chi'])}, Weibull k {f(b['wk'])}; null k 95th pct "
                    f"{f(b['weib_k_null_q95'])}; per-block k {'/'.join(f(v['weib_k']) for v in blk.values())} "
                    f"| skeptics/skeptic/2026-11-18/nulls_gpt-5-nano.json |")
    head = (f"# Proof matrix — maze-hazard-law\n\nOne row per cell. Built by `proof_matrix.py` from `results/analysis.json` "
            f"(status {o['status']}, prediction {o['prediction']['outcome']}). Headline cells are the status agents "
            f"(gpt-oss-20b, Qwen3-8B, Esben's laptop queue); side rows are never status.\n\n")
    if not any(s.startswith("seed-level:") for s in seed_lines):
        jobs = ", ".join(j for _, j in QUEUE.values())
        seed_lines.insert(0, f"seed-level: no status agent has counted data yet (laptop queue: {jobs})")
    seed = "\n".join(seed_lines)
    dst.write_text(head + "\n".join(rows) + "\n\n" + seed + "\n")
    print(f"wrote {dst} ({len(rows) - 2} rows)")


if __name__ == "__main__":
    main()
