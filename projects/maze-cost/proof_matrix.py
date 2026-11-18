"""Write proof-matrix.md from results/analysis.json (one row per cell: agent x seed block x arm; controls marked).

    .venv/bin/python proof_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1000 else f"{x:,.0f}"
    return str(x)


def main() -> None:
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "analysis.json"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "proof-matrix.md"
    res = json.loads(src.read_text())
    rows = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    seed_lines = []
    for ag, a in res["agents"].items():
        g = a["gate"]
        rows.append(f"| gate | - | {ag} visible link | none | {g['k']}/{g['n']} correct (>= 7 needed) | data/{ag}/raw/visible/ |")
        rows.append(f"| headline | all | {ag} maze | none | {a['outcome']}; h={f(a['h'])}, d*={f(a['d_star'])}, "
                    f"P={f(a['P'])} tok, O={f(a['O'])} tok, R={f(a['R_raw'])}, run-level 95% CI {a['R_ci'] and [round(x, 2) for x in a['R_ci']]} "
                    f"(n entered {a['n_entered']}/{a['n_runs']}) | results/analysis.json agents.{ag} |")
        for b in (0, 1, 2):
            runs = [r for r in a["runs"]["maze"] if r["block"] == b and r["entered"]]
            rows.append(f"| headline | {b} | {ag} maze | none | {a['R_blocks_outcome'][b]}; R_b={f(a['R_blocks'][b])} "
                        f"(n entered {len(runs)}, median depth {f(sorted(r['D'] for r in runs)[len(runs) // 2] if runs else None)}) "
                        f"| data/{ag}/raw/maze/ (i % 3 == {b}) |")
        for b in (0, 1):
            runs = [r for r in a["runs"]["deadend"] if r["block"] == b and r["entered"]]
            spend = sorted(r["S"] for r in runs)
            ratio = sorted(r["ratio_usd"] for r in runs if r["ratio_usd"])
            rows.append(f"| dead end vs maze | {b} | {ag} dead end | dead-end page (no continue link) | n entered {len(runs)}, "
                        f"median spend {f(spend[len(spend) // 2] if spend else None)} tok, median $ ratio {f(ratio[len(ratio) // 2] if ratio else None)} "
                        f"| data/{ag}/raw/deadend/ (i % 2 == {b}) |")
        c = a["constancy"]
        rows.append(f"| constant leave rate | all | {ag} maze | one rate vs two (pages 1-3, 4+) | h1-3={f(c['h_1_3'])}, h4+={f(c['h_4plus'])}, "
                    f"LR p={f(c['p'])} | results/analysis.json agents.{ag}.constancy |")
        rows.append(f"| leave-rate bound (added after run 000, DEVIATIONS) | all | {ag} maze | none | {a['leaves_all']} leaves in "
                    f"{a['decisions_all']} page decisions; 95% upper bound {f(a['leave_rate_upper95'])} per page; median depth >= "
                    f"{f(a['median_depth_lower95'])} pages; run level: {a['runs_leaving']}/{a['n_entered']} runs leave, 95% upper bound "
                    f"{f(a['run_leave_upper95'])} | results/analysis.json agents.{ag} |")
        nn = a.get("maze_neutral")
        if nn:
            for b in (0, 1, 2):
                runs = [r for r in a["runs"]["maze_neutral"] if r["block"] == b and r["entered"]]
                rows.append(f"| promise sentence (exploratory, D10) | {b} | {ag} maze, no promise sentence | maze page minus 'Figures are published "
                            f"in a later part.' | {sum(r['leave'] for r in runs)}/{len(runs)} entered runs leave; depths "
                            f"{sorted(r['D'] for r in runs)} | data/{ag}/raw/maze_neutral/ (i % 3 == {b}) |")
            rows.append(f"| promise sentence (exploratory, D10) | all | {ag} maze vs no-promise maze | Fisher exact, two-sided | "
                        f"no promise {nn['runs_leaving']}/{nn['n_entered']} leave vs maze {a['runs_leaving']}/{a['n_entered']}; p={f(nn['fisher_p_vs_maze'])} "
                        f"| results/analysis.json agents.{ag}.maze_neutral |")
        s = a["secondary_block_split"]
        if s:
            rows.append(f"| block split (secondary) | 0 -> 1,2 | {ag} maze | none | {s['outcome']}; R={f(s['R_raw'])} | results/analysis.json agents.{ag}.secondary_block_split |")
        rows.append(f"| cost ratio (price sheet) | all | {ag} maze | dead end | maze median {f(a['ratio_usd_median'])}x "
                    f"(band x0.1..x10: {f(a['ratio_usd_band'][0])}..{f(a['ratio_usd_band'][1])}), dead end median "
                    f"{f(a['dead_end']['ratio_usd_median'])}x | results/analysis.json agents.{ag} |")
        shares = []
        for b in (0, 1, 2):
            runs = [r for r in a["runs"]["maze"] if r["block"] == b and r["entered"]]
            if runs:
                shares.append(sum(r["leave"] for r in runs) / len(runs))
        if len(shares) >= 2:
            import math
            from statistics import mean, stdev
            tcrit = {1: 12.706, 2: 4.303}[len(shares) - 1]
            m, se = mean(shares), stdev(shares) / math.sqrt(len(shares))
            seed_lines.append(f"seed-level: {ag} share of entered maze runs that leave before the call limit = {m:.3f} "
                              f"[{max(0.0, m - tcrit * se):.3f}, {min(1.0, m + tcrit * se):.3f}] (t-interval over {len(shares)} seed blocks: "
                              f"{', '.join(f'{x:.3f}' for x in shares)})"
                              + ("; zero spread across blocks, so the t-interval is degenerate: use the exact run-level bound "
                                 f"{f(a['run_leave_upper95'])}" if max(shares) == min(shares) else ""))
        rs = a["R_seed"]
        seed_lines.append(f"seed-level: {ag} R = {f(a['R_raw'])} [{f(rs[0]) if rs else 'not computable'}, {f(rs[1]) if rs else 'not computable'}] "
                          f"(t-interval over {a['R_seed_n_defined']} defined seed blocks of log R_b, df = {max(a['R_seed_n_defined'] - 1, 0)})")
    rows.append("| synthetic null | - | scripted never-leaver (h = 0) | harness + analysis with no model | outcome 'never quits: spend set by the cap', h = 0 | web/test_analysis.py |")
    rows.append("| synthetic positive | - | scripted geometric leaver (h = 0.15) | harness + analysis with no model | h recovered 0.147, TESTED, R 0.77 | web/test_analysis.py |")
    text = ("# Proof matrix — maze-cost\n\nOne row per cell. Built by `proof_matrix.py` from `results/analysis.json` "
            f"(status {res['status']}, prediction {res['prereg_outcome']}).\n\n" + "\n".join(rows) + "\n\n" + "\n".join(seed_lines) + "\n")
    dst.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
