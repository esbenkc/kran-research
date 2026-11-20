"""Write proof-matrix.md from results/analysis.json: one row per cell (agent x seed block x arm), controls marked.

    .venv/bin/python proof_matrix.py [results/analysis.json] [proof-matrix.md]

'headline' rows are the cells the answer to the question rests on: the status agent that decides it (gpt-oss-20b),
one row per seed block (run index i % 3) with ONE variant name, so the seed lint counts 3 seeds for the main arm. Pooled
rows have seed '-' and their own claim name. A status agent below its floors (Qwen3-8B) gets its own claim name: it does
not decide the answer. Side-row agents (never status) get their own claim names.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import analysis as A

ROOT = Path(__file__).resolve().parent
QUEUE = {"gpt-oss-20b": ("gpt-oss-20b-low-full", "56pkb4cysk"), "Qwen3-8B": ("qwen3-8b-nothink-full", "fdce5qspgd")}
DE_CONTROL = "dead end: one page, no continue link"


def block_k(a: dict, b: str) -> str:
    """Weibull shape k with its profile 95% interval, from the entered maze runs of one seed block (analysis.law)."""
    runs = [r for r in (a.get("runs") or {}).get("maze", []) if r["entered"] and str(r["block"]) == b]
    if not runs:
        return "-"
    w = A.law(runs)["weibull"]
    if not w.get("k"):
        return "-"
    lo, hi = w.get("k_ci") or (None, None)
    if lo is None or not lo <= w["k"] <= hi:  # profile grid tops out at k = 20; tiny blocks fit k far above it
        return f"{f(w['k'])} (interval not reliable: too few give-ups in this block)"
    return f"{f(w['k'])} {f(w['k_ci'])}"


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
    rows.append(f"| {claim}, pooled | - | {label} maze, pooled over seed blocks | none | verdict {L.get('verdict')}; give-ups G={L.get('G')} "
                f"(pages 1-3: {L.get('G3')}/{L.get('E3')} decisions); h3={f(L.get('h3'))}; h_all={f(L.get('h_all'))}; "
                f"KM median {f(L.get('km_median'))} pages vs 3-page forecast {f(L.get('pred_median'))} (R={f(L.get('R'))}; "
                f"decisive miss {f(L.get('decisive_miss'))}); chi-square p={f(chi.get('p'))} (df {chi.get('df')}); Weibull "
                f"shape k={f(wb.get('k'))} {f(wb.get('k_ci'))} p={f(wb.get('p'))}; cap share {f(a.get('cap_share'))}; entered "
                f"{a['n_entered']}/{a['n_maze']} | results/analysis.json |")
    for b in ("0", "1", "2"):
        bl = (a.get("blocks") or {}).get(b)
        if bl:
            rows.append(f"| {claim} | {b} | {label} maze | none | block i % 3 == {b}: entered {bl['n']}; give-ups G={bl['G']} "
                        f"(pages 1-3 {bl['G3']}/{bl['E3']}); h_all={f(bl['h_all'])}; KM median {f(bl['km_median'])}; "
                        f"R_b={f(bl['R'])}; Weibull k {block_k(a, b)} | data/{d}/raw/maze |")
    de = a.get("dead_end") or {}
    by = {}
    for r in (a.get("runs") or {}).get("deadend", []):
        by.setdefault(r["block"], []).append(r)
    for b in sorted(by):
        rs = [r for r in by[b] if r["entered"]]
        rows.append(f"| {claim}: dead-end control | {b} | {label} dead end | {DE_CONTROL} | block i % 2 == {b}: entered "
                    f"{len(rs)}/{len(by[b])}; pages read {sorted(r['D'] for r in rs)}; causes "
                    f"{sorted({r['cause'] for r in rs})} | data/{d}/raw/deadend |")
    if de:
        rows.append(f"| {claim}: dead-end control, pooled | - | {label} dead end, pooled | {DE_CONTROL} | entered "
                    f"{de.get('n_entered')}/{de.get('n')}; share that left after page 1 {f(de.get('exit_page1'))}; median calls "
                    f"{f(de.get('calls_median'))} | results/analysis.json |")
    ae = a.get("any_exit") or {}
    if ae:
        rows.append(f"| {claim}: any exit (defender row) | - | {label} maze, every exit counts | none | h3={f(ae.get('h3'))}; "
                    f"h_all={f(ae.get('h_all'))}; KM median {f(ae.get('km_median'))}; R={f(ae.get('R'))} | results/analysis.json |")
    rows.append(f"| {claim}: exit causes | - | {label} maze | none | {json.dumps(a.get('causes'))}; parse/protocol "
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
            claim = "headline" if a.get("floors_met") else f"{label} status agent below its floors (does not decide the answer)"
            rows += agent_rows(claim, label, a)
            s = a.get("seed_level_h_all") or {}
            if s.get("ci"):
                pre = "seed-level:" if claim == "headline" else f"below floors (not the headline): {label}"
                seed_lines.append(f"{pre} {f(s['estimate'])} {f(s['ci'])} ({label} h_all, {s['method']})")
            ex = ROOT / "results" / f"exit_check_{a['dir']}.json"
            if ex.exists():
                e = json.loads(ex.read_text())["summary"]
                p = e["exit_prompt_tokens"]
                rows.append(f"| {label}: quit not forced by the harness | - | {label} maze, every give-up exit call | none | "
                            f"{e['n_exit_runs']} exits; exit-call prompt {p['min']}-{p['max']} tokens (median {f(p['median'])}) of a "
                            f"{p['context_window']} window; largest new-token count of any call {e['max_new_tokens_any_call']} "
                            f"(cap 2048), calls at the cap {e['calls_at_new_token_cap']}; exit parse {json.dumps(e['exit_parse_modes'])} "
                            f"| results/exit_check_{a['dir']}.json |")
                ac = e.get("answer_classes") or {}
                if ac:
                    frozen = (a.get("invented_figure_answers"), a.get("stale_figure_answers"))
                    note = ("" if frozen == (len(ac["invented"]), len(ac["stale"])) else
                            f" (analysis.py's frozen counts are {frozen[0]}/{frozen[1]}: run 25 '12-month' and run 51 "
                            f"'9,000' vs '900' misclassified)")
                    rows.append(f"| {label}: answer content (corrected classifier) | - | {label} maze, answer() exits | none | "
                                f"made-up figure {len(ac['invented'])}, the site's 2024 figure {len(ac['stale'])}, both "
                                f"{len(ac['both'])}, no figure {len(ac['none'])}, correct {e['correct_answers']}{note} "
                                f"| results/exit_check_{a['dir']}.json |")
                dc = e.get("deadend_answer_classes") or {}
                if dc:
                    n = sum(len(v) for v in dc.values())
                    rows.append(f"| {label}: answer content (corrected classifier), dead-end control | - | {label} dead end, "
                                f"answer() exits of entered runs | {DE_CONTROL} | {n} answers: made-up figure "
                                f"{len(dc['invented'])}, the site's 2024 figure {len(dc['stale'])}, both {len(dc['both'])}, no "
                                f"figure {len(dc['none'])}, correct {e['deadend_correct_answers']} | results/exit_check_{a['dir']}.json |")
        else:
            d, job = QUEUE[label]
            for b in ("0", "1", "2"):
                rows.append(f"| headline | {b} | {label} maze, block i % 3 == {b} | none | waiting for laptop queue (job {job}, "
                            f"full prefill rerun, D6); no counted run yet; output will be data/{d} | DEVIATIONS.md |")
    sk1120 = ROOT / "skeptics" / "skeptic"
    if (sk1120 / "gptoss_full.json").exists():
        r = json.loads((sk1120 / "gptoss_full.json").read_text())
        L = r["law_giveup"]
        rows.append(f"| headline rebuild (skeptic, own code) | - | gpt-oss-20b maze, Vera's vera_rebuild.py | none | {L['verdict']}; "
                    f"G3/E3 {L['G3']}/{L['E3']}, G/E {L['G']}/{L['E']}, h_all {f(L['h_all'], 4)}, KM median {L['M_obs']}, "
                    f"M_pred {f(L['M_pred'])} (fast end of the pilot interval {f(L['M_pred_hhi'])}), R {f(L['R'])}, Pearson p "
                    f"{f(L['p_pearson'])}, Weibull k {f(L['weibull_k'], 4)} p {f(L['p_weibull'])}; entered {r['maze_entered']}/"
                    f"{r['maze_runs']} | skeptics/skeptic/gptoss_full.json |")
    if (sk1120 / "extra_gptoss.json").exists():
        x = json.loads((sk1120 / "extra_gptoss.json").read_text())
        na, nb, lo = x["nullA_geometric"], x["nullB_permutation"], x["leave_one_out_worst"]
        rows.append(f"| headline null (skeptic) | - | gpt-oss-20b: {na['sims']} data sets with one constant chance {f(na['h'], 4)} "
                    f"per page, through the frozen rule | simulated constant hazard | frozen rule says FAILS in {f(na['P_FAILS'])} "
                    f"of them; both observed p values reached {na['both_obs_p_reached']}/{na['sims']} | skeptics/skeptic/extra_gptoss.json |")
        rows.append(f"| headline null (skeptic) | - | gpt-oss-20b: {nb['perms']} permutations of the give-up flags across all page decisions | give-up flags shuffled across pages "
                    f"| Pearson p as small as observed {nb['pearson_as_small']}/{nb['perms']}, Weibull p {nb['weibull_as_small']}/"
                    f"{nb['perms']} | skeptics/skeptic/extra_gptoss.json |")
        rows.append(f"| headline robustness (skeptic) | - | gpt-oss-20b maze, leave one run out, worst case | none | dropping run "
                    f"{lo['dropped_run']}: Pearson p {f(lo['p_pearson'])}, Weibull p {f(lo['p_weibull'])}, k {f(lo['k'])} "
                    f"| skeptics/skeptic/extra_gptoss.json |")
    a = res["agents"].get("gpt-oss-20b")
    if a and a.get("spend"):
        # exact-fraction Kaplan-Meier median of S_r (Vera #26918: analysis.py's float product takes the 28th value)
        ent = sorted(((r["S"], r["cause"] in A.GIVEUP) for r in a["runs"]["maze"] if r["entered"]))
        surv, at, med = Fraction(1), len(ent), None
        for s, ev in ent:
            if ev:
                surv *= Fraction(at - 1, at)
                if surv <= Fraction(1, 2):
                    med = s
                    break
            at -= 1
        rows.append(f"| spend (reported row, never status) | - | gpt-oss-20b maze, tokens billed from entry to exit | none | "
                    f"Kaplan-Meier median {med} tokens by the PREREG definition (exact fractions); analysis.py prints "
                    f"{a['spend']['obs_km_median']} (float edge at survival 27/54 = 0.5); the 3-page spend line predicted "
                    f"{f(a['spend']['pred_median'])} | results/analysis.json |")
    d8 = ROOT / "results" / "d8" / "nolabel.json"
    if d8.exists():
        for v in json.loads(d8.read_text()).values():
            if v.get("arm") != "maze_nolabel":
                continue
            rows.append(f"| D8 no page number (reported row, never status) | - | gpt-oss-20b maze, page number removed, pooled "
                        f"| none | reading '{v['reading']}'; entered {v['n_entered']}/{v['n_runs']}; G={v['G']} (pages 1-3 "
                        f"{v['G3']}/{v['E3']}); h_all={f(v['h_all'])}; KM median {f(v['km_median'])} (page number shown: 12); "
                        f"Weibull k {f(v['weibull_k'])} {f(v['weibull_k_ci'])}; Pearson p {f(v['pearson_p'])}; cap share "
                        f"{f(v['cap_share'])} | results/d8/nolabel.json |")
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
        rows.append(f"| {claim} | - | {label} maze, Vera's own code | none | {V['verdict']}; G3/E3 {L['G3']}/{L['E3']}, "
                    f"G/E {L['G']}/{L['E']}, h_all {f(L['h_all'], 4)}, KM median {L['M_obs']}, M_pred {f(L.get('M_pred'), 4)}, "
                    f"R {f(V.get('R'), 4)}, chi p {f(L['p_pearson'])}, Weibull k {f(L.get('weibull_k'), 4)} p "
                    f"{f(L.get('p_weibull'))}; entered {r['maze_entered']}/{r['maze_runs']} | skeptics/skeptic/{fn} |")
    nq = sk / "null_qwen_reuseON.txt"
    if nq.exists():
        n = json.loads(nq.read_text())
        rows.append(f"| preview null (skeptic), not counted (D6) | - | Qwen3-8B reuse-on: constant hazard {f(n['h_all'], 4)}, "
                    f"{n['n']} runs, {n['reps']} draws | simulated constant hazard | share of draws reaching the observed "
                    f"Pearson p {f(n['P(pearson p <= observed 6.66e-09)'])}, Weibull p {f(n['P(weibull p <= observed 1.33e-04)'])}; "
                    f"frozen rule on the null: holds {f(n['holds'])}, fails {f(n['fails_any'])} | skeptics/skeptic/null_qwen_reuseON.txt |")
    # last night's null of the gpt-5-nano side row (at the folder root, or in 2026-11-18/)
    nn = next((p for p in (sk / "nulls_gpt-5-nano.json", sk / "2026-11-18" / "nulls_gpt-5-nano.json") if p.exists()), None)
    if nn:
        n = json.loads(nn.read_text())
        b, blk = n["N1_parametric_bootstrap"], n.get("N2_blocks", {})
        rows.append(f"| side row null (skeptic) | - | gpt-5-nano: constant hazard {f(b['h'], 4)}, {b['reps']} draws | parametric bootstrap "
                    f"| share of draws reaching the observed chi p {f(b['chi'])}, Weibull k {f(b['wk'])}; null k 95th pct "
                    f"{f(b['weib_k_null_q95'])}; per-block k {'/'.join(f(v['weib_k']) for v in blk.values())} "
                    f"| {nn.relative_to(ROOT)} |")
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
