"""proof-matrix.md from data/results/summary.json + gate files: one row per cell, every artifact a single path.
Usage: venv/bin/python proof_matrix.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
D = ROOT / "data"


def f(x, n=3):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    s = json.loads((D / "results" / "summary.json").read_text())
    g0 = json.loads((D / "gate0" / "gate0.json").read_text())
    gs = json.loads((D / "gate0" / "gate0_spread.json").read_text())
    gb = s["gate0b"] or {}
    pr, q = s["prices"], s["premise_Q"]
    rows = []
    for n, r in s["runs"].items():
        a = f"data/runs/{n}.json"
        rows.append(("headline", "0", f"{r['target']} @ lam {r['lam']:g}, cap {r['cap']}", "none",
                     f"blind-panel late mean L = {f(r['L_blind'])} (step 0 {f(r['step0_blind'])}; moving {r['moving']})", a))
        rows.append(("mechanical half", "0", f"{r['target']} @ lam {r['lam']:g}", "none",
                     f"full panel L = {f(r['L_full'])}; output-only probe L = {f(r['L_O'])}", a))
        rows.append(("behaviour kept", "0", f"{r['target']} @ lam {r['lam']:g}", "none",
                     f"B_lie {f(r.get('B_lie'))}, B_same {f(r.get('B_same'))}, B_truthful {f(r.get('B_honest'))}, "
                     f"answer KL {f(r.get('kl_answer'), 4)}", f"data/runs/{n}__behaviour.json"))
        if r.get("shuffled_label_catch"):
            rows.append(("refit probe reads the concept", "0", f"{r['target']} @ lam {r['lam']:g}, final step",
                         "shuffled labels (5 perms)", f"blind catch {', '.join(f(x, 3) for x in r['shuffled_label_catch'])}", a))
        if "L_nontop" in r:
            rows.append(("lie hold not the common answer", "0", f"lie @ lam {r['lam']:g}", "subset: clean answer not 'the United States'",
                         f"L = {f(r['L_nontop'])} (n {r['n_nontop']}); answer not 'the ...': L = {f(r['L_nothe'])} (n {r['n_nothe']})", a))
    for n in s["runs"]:
        cf = D / "results" / f"crossstep_{n}.json"
        if cf.exists():
            cs = json.loads(cf.read_text())
            if cs:
                rows.append(("each refit is beaten within 50 steps", "0", n.split("__")[0].replace("_spread", "") + " @ lam 1e4",
                             "fresh refit on the same states", "stale probes (fit at k, scored at k+50): " +
                             ", ".join(f"{c['fit_step']}->{c['scored_step']} {f(c['stale_catch'])} (fresh {f(c['fresh_catch'])})" for c in cs),
                             f"data/results/crossstep_{n}.json"))
    sl = D / "results" / "still_lying.json"
    if sl.exists():
        st = json.loads(sl.read_text())["steps"]
        rows.append(("lie run changed the behaviour", "0", "lie @ lam 1e4", "none",
                     "still a false country on eval lie prompts: " + ", ".join(f"step {k} {f(v['still_false'])}" for k, v in
                                                                              sorted(st.items(), key=lambda kv: int(kv[0]))),
                     "data/results/still_lying.json"))
    pa = "data/leace/price_blind_spread.json"
    for c in ("lie_blind", "decoy_blind", "lie_full", "decoy_full"):
        if c in pr:
            rows.append(("price before training", "-", c, "none",
                         f"answer-position KL {f(pr[c]['P_answer'], 5)}; every-position {f(pr[c]['P_every_position'], 6)}", pa))
    if q.get("Q") is not None:
        rows.append(("premise Q >= 10", "-", "blind, independent erasers", "shuffled-label price, 5 perms",
                     f"Q {f(q['Q'], 1)}; raw ratio {f(q['raw_ratio'], 1)}; N_decoy {f(q['N_decoy'], 5)}; N_lie {f(q['N_lie'], 5)}; "
                     f"lie/own null {f(q['lie_over_own_null'], 1)}", pa))
    if gb:
        for c in ("lie", "decoy"):
            if c in gb:
                rows.append(("G0b' readable at step 0", "-", f"{c}, blind panel, 640-prompt fit", "none",
                             f"catch {f(gb[c]['blind']['catch'])} (1296 fit {f(gb[c]['blind']['catch_fit1296'])}); full {f(gb[c]['full']['catch'])}; O_29 {f(gb[c]['O29_catch'])}",
                             "data/gate0/gate0b_spread.json"))
        if "lie" in gb:
            rows.append(("G0b' null", "-", "lie, random k-subspace removed", "random directions",
                         f"catch {f(gb['lie'].get('random_k_null'))}", "data/gate0/gate0b_spread.json"))
    rows.append(("G0a compliance", "-", "base Qwen2.5-1.5B, R8v, 748 cities", "none",
                 f"lie {f(g0['phaseB']['lie_rate'])}, truthful {f(g0['phaseB']['honest_acc'])}, kept {g0['n_kept']}", "data/gate0/gate0.json"))
    rows.append(("G0a compliance", "-", "spread prompts, kept cities", "none",
                 f"lie {f(gs['lie_rate'])}, truthful {f(gs['honest_acc'])}", "data/gate0/gate0_spread.json"))
    lines = ["# Proof matrix — instructed-lie-leace", "",
             f"Status (analysis.py, PREREG v1.3 s7): **{s['status']}**; prediction outcome: **{s['prediction_outcome']}**.", "",
             "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for r in rows:
        assert os.path.exists(ROOT / r[5]), r[5]
        lines.append("| " + " | ".join(r) + " |")
    sk = ROOT / "skeptics"
    for p in sorted(sk.glob("*/proof-cells.md")) if sk.exists() else []:
        lines += ["", p.read_text().strip()]
    (ROOT / "proof-matrix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
