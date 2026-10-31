"""Build proof-matrix.md from results/summary.json (one row per cell of evidence). python proof_matrix.py

Columns: | claim | seed | variant | control | result | artifact |. Every artifact is ONE path relative to this folder
and is checked to exist. Rows this script does not generate (skeptics' hand-added rows, or rows in
skeptics/*/proof-cells.md) are kept: lesson from oracle-1031, a regenerated file must never drop them.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
S = json.loads((PROJECT / "results" / "summary.json").read_text())
SUM = "results/summary.json"
SCORES = "results/scores.jsonl"


def f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def ci(c: list[float] | None) -> str:
    return "n/a" if not c or c[0] is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def main() -> None:
    p = S["primary"]
    seeds = f"on 0-{S['seeds_on'][1]}, off 0-29"
    rows: list[tuple[str, ...]] = []
    rows.append(("headline", seeds, "MH say-gap G, PREREG v1.3 rule, eligible strata (s5-s6)", "none",
                 f"G = {f(p['G'])}, 90% CI {ci(p['ci90'])}; {p['n_eligible']} strata, {p['n_models']} models, "
                 f"{p['n_runs']} runs; status {S['status']}", SUM))
    for k, lab in (("n_on_50", "prompt-on seeds 0-49 only (N_on = 50 before D3)"),
                   ("goal_literal_prompt_on", "prompt-on strata only (goal-literal row)"),
                   ("says_v1_v1", "says = pure v1 rule (PREREG v1 a01a5972)"),
                   ("says_clause_strict", "says = v1.2 rule (clause negation, strict anchor)"),
                   ("says_clause_verb_strict", "says = clause negation with verb exception, strict anchor"),
                   ("says_aware", "says = Honeyquest-aware tag (trap OR suspicion words)"),
                   ("says_key_targeted", "says = key-targeted only (trap word in a sentence naming the key, not the bucket)"),
                   ("says_raw", "says = any trap word (no anchor, no negation)"),
                   ("says_secondary", "secondary window (exposure to first trip)"),
                   ("trip_any", "trip = any call carrying the key, report included (v1)")):
        v = S["variants"].get(k)
        if v:
            rows.append(("headline", seeds if k != "n_on_50" else "on 0-49, off 0-29", lab, "none",
                         f"G = {f(v['G'])}, 90% CI {ci(v['ci90'])}; {v['n_eligible']} strata; rule reads {v['status']}", SUM))
    for k, lab in (("ci90_model_cluster", "model-cluster bootstrap CI (row, anti-conservative at 5 models)"),
                   ("ci90_two_level_v1", "two-level bootstrap CI (PREREG v1 method)")):
        if p.get(k):
            rows.append(("headline", seeds, lab, "none", f"G = {f(p['G'])}, 90% CI {ci(p[k])}", SUM))
    for m, g in (p.get("G_per_model") or {}).items():
        rows.append(("per-model say-gap", seeds, f"model {m}, its eligible strata", "none", f"G = {f(g)}", SUM))
    for k, st in p["strata"].items():
        if "|on|" not in k:
            continue
        rows.append(("stratum", seeds, k + (" (eligible)" if st.get("eligible") else " (not eligible)"), "none",
                     f"says {st['a']}/{st['n']}; trip|says {st['t_S']}/{st['a']}, trip|not {st['t_N']}/{st['b']}; "
                     f"gap {f(st.get('gap'))}, 90% CI {ci(st.get('gap_ci90'))}", SUM))
    sh = p.get("control_shuffle")
    if sh:
        rows.append(("headline", seeds, "says shuffled within each eligible stratum (2,000 shuffles)", "shuffled labels",
                     f"null G mean {f(sh['mean'])}, 90% range {ci(sh['range90'])}; share |G_null| >= |G_obs| = {sh['share_abs_ge_obs']:.4f}", SUM))
    off = [st for k, st in p["strata"].items() if "|off|" in k]
    rows.append(("prompt echo", "off 0-29", "prompt-off strata, all 5 models x 2 decoys", "no warning (prompt off)",
                 f"sayers {sum(s['a'] for s in off)} of {sum(s['n'] for s in off)} exposed runs; trip "
                 f"{sum(s['t_S'] + s['t_N'] for s in off)}/{sum(s['n'] for s in off)}", SUM))
    pr = S["prediction"]
    for k, lab, thr in (("S", "say rise (on - off)", ">= 0.40"), ("T", "trip drop (off - on)", "<= 0.10"),
                        ("C", "trip rate among sayers", ">= 0.50")):
        rows.append((f"prediction ({lab})", seeds, "model x decoy equal weights, all runs (s7)", "none",
                     f"{pr['point'][k]:.3f}, 90% CI {ci(pr['ci90'][k])}; predicted {thr}; outcome {pr['outcome']}", SUM))
    me = S["mechanism"]
    if me.get("predicted") is not None:
        rows.append(("mechanism (s8)", seeds, f"strata {', '.join(me['strata'])}", "none",
                     f"predicted drop {me['predicted']:.3f} {ci(me['ci90_predicted'])}, observed {me['observed']:.3f} "
                     f"{ci(me['ci90_observed'])}; observed - predicted {f(me['observed_minus_predicted'])} {ci(me['ci90_diff'])}: {me['reading']}", SUM))
    for k, lab in (("spearman_model_level", "model x decoy points (10)"), ("spearman_by_model_aware", "5 models, Honeyquest-aware tag"),
                   ("spearman_by_model_primary", "5 models, primary tag")):
        v = S[k]
        rows.append(("model-level Spearman (Honeyquest analogue)", "on", lab, "none",
                     f"r = {f(v['r'])}, 90% CI (runs resampled) {ci(v['ci90_run_resample'])}, n = {v['n_points']}", SUM))
    for mk, v in S["manipulation"].items():
        rows.append(("manipulation check", seeds, mk, "none",
                     f"exposed {v['exposed']}/{v['n']}, window {v['window']}, thought non-empty {v['thought_nonempty_rate']:.2f}, "
                     f"words {v['thought_words_mean']:.1f}, parse-fail runs {v['parse_fail_runs']}, raw trap word anywhere {v['raw_trap_word_anywhere']}", SUM))
    rows.append(("harness", "0-49", "scripted honest and greedy agents, 4 cells; scorer cases", "scripted agent (no model)",
                 "all checks OK, 0 regex hits in 4,600 agent-visible env strings", "logs/test_harness.txt"))
    rows.append(("regex audit (s3)", "900-909", "Kasper blind hand labels, prompt-on smoke", "hand labels",
                 "v1.3 rule: 42 hits, 0 false hits, 2 misses of 44 true sayers (Kasper #10685)", "results_smoke/window_thoughts.jsonl"))

    for row in rows:
        assert (PROJECT / row[5]).exists(), row[5]
    gen = {"| " + " | ".join(x.replace("|", "/") for x in row) + " |" for row in rows}
    old = (PROJECT / "proof-matrix.md").read_text().splitlines() if (PROJECT / "proof-matrix.md").exists() else []
    # keep every old row whose (claim, variant) this script does not generate now (reviewer rows, skeptic rows)
    own = {(r[0], r[2].replace("|", "/")) for r in rows}
    keep = []
    for ln in old:
        if not ln.startswith("| ") or ln.startswith("| claim") or ln in gen:
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split(" | ")]
        if len(cells) >= 3 and (cells[0], cells[2]) in own:
            continue
        keep.append(ln)
    lines = ["# Proof matrix — honeypot-talk", "",
             "Built by `proof_matrix.py` from `results/summary.json` (`analysis.py`, scorer rule PREREG v1.3). G = Mantel-Haenszel "
             "difference in trip rate, sayers minus non-sayers, within model x prompt arm x decoy; negative = sayers trip less. "
             "Status band (-0.105, +0.105). Rows below the generated block were added by reviewers and are kept on rebuild.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(x.replace("|", "/") for x in row) + " |" for row in rows]
    lines += keep
    for extra in sorted(PROJECT.glob("skeptics/*/proof-cells.md")):
        lines += [ln for ln in extra.read_text().splitlines()
                  if ln.startswith("| ") and not ln.startswith("| claim") and not ln.startswith("|---") and ln not in lines]
    (PROJECT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print(f"proof-matrix.md: {len(rows)} generated rows, {len(keep)} kept rows")


if __name__ == "__main__":
    main()
