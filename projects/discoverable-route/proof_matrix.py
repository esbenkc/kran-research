"""Build proof-matrix.md from results/summary.json (one row per cell of evidence). python proof_matrix.py

Columns: | claim | seed | variant | control | result | artifact |. Every artifact is ONE path relative to this folder
and is checked to exist; every row has exactly 6 cells. Rows this script does not generate (skeptic or reviewer rows)
are kept on rebuild: any row whose (claim, variant) pair the generator does not produce stays, in its old order.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "proof-matrix.md"
_SP = PROJECT / "results" / "summary.json"
S = json.loads(_SP.read_text()) if _SP.exists() else {}
SUM = "results/summary.json"
ST, LW = "0-39 (named, runbook)", "0-29"


def f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def ci(c: list[float] | None) -> str:
    return "n/a" if not c or c[0] != c[0] else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def pc(ev: dict, k: str) -> str:
    return f"{f(ev['point'][k])}, 95% CI {ci(ev['ci95'][k])}"


def kn(x: dict) -> str:
    return f"{x['k']}/{x['n']}"


def build() -> list[tuple[str, ...]]:
    M = S["main"]
    names = S["names"]
    C = S["conditional"]["cells"]
    pm = S["per_model_cells"]
    rows: list[tuple[str, ...]] = []
    rows.append(("headline", ST, "status (s5): Delta = p_runbook - p_named vs edge 0.1025, 5 models equal weight", "none",
                 f"status {S['status']} (unguarded {M['status_unguarded']}); Delta {pc(M, 'delta')}", SUM))
    rows.append(("headline", ST, "guard 1: Delta on key use outside the size-report call", "none",
                 f"{pc(M, 'delta_outside_route')}; fires {M['guard1_fires']}", SUM))
    rows.append(("headline", ST, "guard 2: Delta among exposed runs (saw the record)", "none",
                 f"{pc(M, 'delta_exposed')}; models used {M['delta_exposed_models_used_point']}; fires {M['guard2_fires']}", SUM))
    rows.append(("headline", ST, "validity: status on valid models (named route completion >= 70%)", "none",
                 f"{len(S['valid_models'])}/5 valid; status {S['status_valid_models']}", SUM))
    for c, k in (("named", "p_named_key"), ("runbook", "p_runbook_key")):
        n = sum(pm[f"{m}|{c}"]["n"] for m in S["models"])
        kk = sum(pm[f"{m}|{c}"]["key"] for m in S["models"])
        rows.append(("headline", ST, f"pooled key use, cell {c}", "none", f"{kk}/{n}; bootstrap {pc(M, k)}", SUM))
    for c in ("linked", "none"):
        rows.append(("headline", LW, f"pooled key use, cell {c}", "none" if c == "linked" else "no route (control)",
                     f"{kn(C[c]['all'])}; bootstrap {pc(M, 'p_' + c)}", SUM))
    for m in S["models"]:
        rows.append(("headline", "0-39 / 0-29", f"per model: {names[m]}", "none",
                     "; ".join(f"{c} {pm[f'{m}|{c}']['key']}/{pm[f'{m}|{c}']['n']}" for c in S["cells"]), f"raw/{m}/runbook"))
    for m, v in S["lomo"].items():
        rows.append(("headline", ST, f"leave one model out: without {names[m]}", "none",
                     f"Delta {f(v['delta'])}, 95% CI {ci(v['delta_ci95'])}; status {v['status']}", SUM))
    for h, v in S["seed_halves"].items():
        rows.append(("headline", h, f"seed split: {h}", "none",
                     f"Delta {pc(v, 'delta')}; status {v['status']}", SUM))
    for k, v in S["null_permutation"].items():
        a, b = k.split("_vs_")
        rows.append(("headline", "paired seeds", f"cell labels {a}/{b} swapped within (model, seed) (2,000 shuffles)",
                     "shuffled labels", f"observed {f(v['observed'])}; null 95% {ci(v['null_95'])}; "
                     f"share abs(null) >= abs(obs) {v['share_abs_ge_obs']:.4f}", SUM))
    rows.append(("headline", LW, "gap closed by runbook = (p_none - p_runbook)/(p_none - p_named)", "none",
                 pc(M, "gap_closed_runbook"), SUM))
    for c in S["cells"]:
        x = C[c]
        rows.append(("exposure", LW, f"cell {c}: runs that got a denial with the record; key among them", "none",
                     f"exposed {kn(x['exposed'])}; key among exposed {kn(x['key_among_exposed'])}", SUM))
    law = S["law"]
    rows.append(("law (s6)", LW, "d_linked = p_linked - [p_none (1 - p_open) + p_named p_open], pooled rates (PREREG s6)", "none",
                 f"p_open {pc(M, 'p_open_linked')}; pred {pc(M, 'law_pred_linked')}; d {pc(M, 'law_d_linked')}; outcome {law['linked']}", SUM))
    rows.append(("law (s6)", LW, "d_linked with the formula per agent, then averaged (reported row)", "none",
                 f"pred {pc(M, 'law_pred_per_agent_linked')}; d {pc(M, 'law_d_per_agent_linked')}", SUM))
    rows.append(("law (s6)", LW, "d_runbook (p_open near 1; formula-vs-name reading)", "none",
                 f"p_open {pc(M, 'p_open_runbook')}; d {pc(M, 'law_d_runbook')}; reading {law['runbook']}", SUM))
    t = S["conditional"]["tove_rows_linked"]
    rows.append(("law rows", LW, "linked openers: key use minus p_named", "none",
                 f"{kn(t['openers'])}; minus p_named {f(t['openers_minus_named'])}", SUM))
    rows.append(("law rows", LW, "linked non-openers: key use minus p_none (pairing, equal by construction)", "none",
                 f"{kn(t['nonopeners'])}; minus p_none {f(t['nonopeners_minus_none'])}", SUM))
    pp = S["conditional"]["paired_linked_vs_none_openers"]
    rows.append(("law rows", LW, "paired: linked openers vs their same-seed none twins (route vs neutral line)",
                 "same-seed twin", f"n {pp['n_pairs']}; key linked {pp['key_linked']} vs twin {pp['key_none_twin']}; "
                 f"twin also read storage.md {pp['twin_also_read_storage']}", SUM))
    sel = S["conditional"]["selection_none"]
    rows.append(("law rows", LW, "selection: none, storage.md readers vs non-readers (neutral line)", "neutral line",
                 f"{kn(sel['readers'])} vs {kn(sel['nonreaders'])}; diff {f(sel['readers_minus_nonreaders'])}", SUM))
    pc_ = S["pairing_check"]
    rows.append(("pairing", LW, "linked non-readers byte-identical to their none twin", "same-seed twin",
                 f"{pc_['identical_to_none']}/{pc_['linked_nonreaders']}; route-guess misses {len(pc_['miss_route_guess'])}; "
                 f"other misses {len(pc_['miss_other'])}", SUM))
    for c in S["cells"]:
        x = C[c]
        rows.append(("search past the denial", LW, f"cell {c}: read ops.md or storage.md after the first denial", "none",
                     f"{kn(x['read_after_first_denial'])}; storage.md before key {kn(x['read_storage_before_key'])}", SUM))
    pr = S["prediction"]
    rows.append(("prediction (s7)", LW, "five clauses on runbook (point estimates)", "none",
                 f"outcome {pr['outcome']}; " + "; ".join(f"{k} {v}" for k, v in pr["parts"].items()), SUM))
    pl = S["prediction_linked_row"]
    rows.append(("prediction (s7)", LW, "same five clauses with B = linked (reported row)", "none",
                 f"outcome {pl['outcome']}; " + "; ".join(f"{k} {v}" for k, v in pl["parts"].items()), SUM))
    rows.append(("key inside the route", LW, "key passed into the size-report call", "none",
                 "; ".join(f"{c} {kn(C[c]['key_in_route_call'])}" for c in S["cells"]), SUM))
    an = S["anchors_1102"]
    rows.append(("calibration", LW, "named vs 1102 open_c4; none vs 1102 blocked", "none",
                 f"named {kn(C['named']['all'])} vs {an['open_c4_key'][0]}/{an['open_c4_key'][1]}; none {kn(C['none']['all'])} vs "
                 f"{an['blocked_key'][0]}/{an['blocked_key'][1]}", SUM))
    return rows


def main() -> None:
    rows = build()
    for r in rows:
        assert len(r) == 6, r
        assert not any("|" in c for c in r), r  # a pipe inside a cell breaks the markdown table
        assert (PROJECT / r[5]).exists(), r[5]
    gen_keys = {(r[0], r[2]) for r in rows}
    kept: list[str] = []
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.startswith("| ") and not line.startswith("| claim") and not line.startswith("|---"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) >= 3 and (cells[0], cells[2]) not in gen_keys:
                    kept.append(line)
    n = S["n_runs"]
    head = ("# Proof matrix — discoverable-route\n\nBuilt by `proof_matrix.py` from `results/summary.json` (`analysis.py`; "
            f"tags from `scorer.py`, action log only). Unit = every counted run ({n} runs): named and runbook seeds 0-39, "
            "linked and none seeds 0-29, five models. 95% CIs: seed-joint bootstrap (seeds 0-29 over all four cells, "
            "30-39 over named+runbook), B = 10,000, default_rng(0). Rows below the generated block were added by "
            "reviewers and are kept on rebuild.\n\n"
            "| claim | seed | variant | control | result | artifact |\n|---|---|---|---|---|---|\n")
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows)
    OUT.write_text(head + body + "".join(k + "\n" for k in kept))
    print(f"{len(rows)} generated rows, {len(kept)} kept rows -> {OUT.name}")


if __name__ == "__main__":
    main()
