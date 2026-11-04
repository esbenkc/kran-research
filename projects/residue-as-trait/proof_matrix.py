"""Write proof-matrix.md from results/summary.json (python proof_matrix.py). One row per cell:
| claim | seed | variant | control | result | artifact |
Skeptic rebuild rows are appended by the skeptics below the generated table (kept on regeneration).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "<!-- skeptic rows below this line are kept on regeneration -->"


def f(x: float, n: int = 4) -> str:
    return f"{x:+.{n}f}"


def main() -> None:
    s = json.loads((HERE / "results" / "summary.json").read_text())
    k, ko = s["key"], s["key_outside_route_row8"]
    seeds = "0-2"
    rows = []
    A = "results/summary.json"
    rows.append(("headline", seeds, "all 8 scenarios, both arms, outcome key", "none",
                 f"D_add {f(k['D_add'])} nats/run ({k['pct_better_add_row3']:.1%} better guess); status {k['status']}; "
                 f"key {k['n_key']}/{k['n_runs']}", f"{A} key.D_add"))
    rows.append(("headline", seeds, "all 8 scenarios", "agent labels shuffled within scenario x arm (2000 draws)",
                 f"null D_add mean {f(k['null_mean']['D_add'])}, q95 {f(k['null_q95']['D_add'])}; "
                 f"P(null >= observed) {k['null_p_ge_obs']['D_add']:.4f}", f"{A} key.null_q95.D_add"))
    for t, v in k["per_fold"].items():
        rows.append(("headline", seeds, f"held-out scenario {t}", "none",
                     f"D_add {f(v['D_add'])}; key {v['key']}/{v['n']}", f"{A} key.per_fold.{t}"))
    for m, v in k["per_agent"].items():
        rows.append(("per-agent", seeds, m, "none", f"D_add {f(v['D_add'])}; key {v['key']}/{v['n']} "
                     f"(named {v['key_named']}, runbook {v['key_runbook']})", f"{A} key.per_agent"))
    mi = k["max_influence_row3a"]
    rows.append(("headline", seeds, "drop the single largest-gain run", "influence check",
                 f"D_add {f(mi['D_add_drop_max_gain_run'])} (run {mi['max_gain_run']})", f"{A} key.max_influence_row3a"))
    rows.append(("headline", seeds, "drop the single largest-loss run", "influence check",
                 f"D_add {f(mi['D_add_drop_max_loss_run'])} (run {mi['max_loss_run']})", f"{A} key.max_influence_row3a"))
    rows.append(("headline", seeds, "outcome key_outside_route (bypass only)", "none",
                 f"D_add {f(ko['D_add'])}; null q95 {f(ko['null_q95']['D_add'])}; status {ko['status']}; key {ko['n_key']}/{ko['n_runs']}",
                 f"{A} key_outside_route_row8"))
    for name in ("Dmin", "D_base", "D_task"):
        rows.append(("old rules (row 1)", seeds, name, "none", f"{f(k[name])}; null q95 {f(k['null_q95'][name])}",
                     f"{A} key.{name}"))
    sp = k["spread_row3b"]
    rows.append(("agent-by-task spread (row 3b)", seeds, "raw estimator", "shuffle null",
                 f"{sp['raw_agent_by_task']:.3f} logit; null mean {sp['raw_null_mean']:.3f}, q95 {sp['raw_null_q95']:.3f}; "
                 f"task spread {sp["task_spread_row3c"]:.3f}; calibration n=6: 0.80 at true 0, 1.06 at true 1.14, 1.37 at true 2.0", f"{A} key.spread_row3b"))
    for d, v in k["cross_arm_row4"].items():
        rows.append(("cross-arm transfer (row 4)", seeds, d, "pooled rate of the source arm",
                     f"D {f(v['D'])} (agent loss {v['loss_agent']:.3f} vs pooled {v['loss_pooled']:.3f})", f"{A} key.cross_arm_row4"))
    if "deviance_row5" in k:
        for t, v in k["deviance_row5"].items():
            rows.append(("deviance (row 5)", seeds, t, v["null"], f"LR {v['LR']:.1f}, null mean {v['null_mean']:.1f}, "
                         f"q99 {v['null_q99']:.1f}, excess {v['excess']:.1f}", f"{A} key.deviance_row5"))
    if "scout_row6" in k:
        sc = k["scout_row6"]
        rows.append(("scout prediction (row 6)", seeds, "instance = scenario x seed", "permutation nulls",
                     f"agent excess {sc['excess_agent']:.1f}, instance excess {sc['excess_instance']:.1f}; held={sc['held']}",
                     f"{A} key.scout_row6"))
    rt = s["s0_retest_row7"]
    rows.append(("S0 retest (row 7)", seeds, "storage, fresh salt vs parent seeds 0-2", "parent run",
                 f"run-level agreement {rt.get('agree')}/{rt.get('n')}", f"{A} s0_retest_row7"))
    cf = s["clean_floor_row9"]
    rows.append(("clean floor (row 9)", seeds, "every env reply, all runs", "none",
                 f"token messages {sum(cf['token_message_any_reply_per_task'].values())}; blocked-read replies "
                 f"{json.dumps(cf['blocked_read_replies'])}; parse fails {cf['parse_fail_actions']}; cap hits {cf['cap_hits']}",
                 f"{A} clean_floor_row9"))
    ex = HERE / "results" / "exploratory.json"
    if ex.exists():
        e = json.loads(ex.read_text())
        e1, e2 = e["E1_sign_repeat"], e["E2"]
        rows.append(("EXPLORATORY placement sign repeat", "parent 0-39 / new 0-2", "5 agents", "parent run",
                     f"same sign of runbook - named in {e1['same_sign_count']}/{e1['n_agents']} agents",
                     "results/exploratory.json E1_sign_repeat"))
        rows.append(("EXPLORATORY agent x placement offset", seeds, "arm-specific offset vs pooled offset",
                     "agent labels shuffled within scenario x arm",
                     f"D_add arm {f(e2['D_add_arm_specific_offset'])} vs pooled {f(e2['D_add_pooled_offset_on_arm_task_rate'])}; "
                     f"null q95 {f(e2['null_q95']['arm'])} / {f(e2['null_q95']['pooled'])}", "results/exploratory.json E2"))
        g = e2.get("gain_null_arm_within_agent_x_task")
        if g:
            rows.append(("EXPLORATORY agent x placement offset", seeds, "gain = arm-specific minus pooled offset",
                         "arm labels shuffled within agent x task (2000 draws)",
                         f"gain {f(e2['gain_arm_specific_minus_pooled'])}; null mean {f(g['mean'])}, q95 {f(g['q95'])}, "
                         f"q99 {f(g['q99'])}; P(null >= obs) {g['p_ge_obs']:.4f}", "results/exploratory.json E2.gain_null"))
    rows.append(("spend (row 9)", "-", "smoke + counted", "none", f"${s['spend_row9']['total_usd']:.3f}", "raw/cost.jsonl"))
    head = ("# Proof matrix — residue-as-trait\n\nGenerated by `proof_matrix.py` from `results/summary.json` "
            "(`analysis.py`, PREREG v1.1). Seeds = name seeds 0-2 of every scenario.\n\n"
            "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n")
    # artifact = bare path relative to the project folder (oracle lint); the JSON key moves into the result cell
    fixed = []
    for claim, seed, variant, control, result, art in rows:
        path, _, jkey = art.partition(" ")
        fixed.append((claim, seed, variant, control, result + (f" [{jkey}]" if jkey else ""), path))
    body = "".join(f"| {' | '.join(str(c).replace('|', '/') for c in r)} |\n" for r in fixed)
    p = HERE / "proof-matrix.md"
    keep = ""
    if p.exists() and MARK in p.read_text():
        keep = p.read_text().split(MARK, 1)[1]
    p.write_text(head + body + "\n" + MARK + keep)
    print(f"wrote {p.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
