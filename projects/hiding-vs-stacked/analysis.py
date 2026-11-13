"""hiding-vs-stacked analysis: rates, cuts, status, law and prereg legs, exactly as PREREG.md section 5 words them.

    web/.venv-free: python3 analysis.py [--phase counted]   (numpy only; reads web/results/scores_<phase>.jsonl)

Inputs: web/results/scores_<phase>.jsonl (one row per run, from web/score.py: follow from the server access log, cue
from the frozen cue_grep.py) and viewcheck/viewcheck.json (Iris, model-free Chromium views; optional).
Outputs: results/analysis_<phase>.json, results/cells_<phase>.csv, results/figure_data.json (counted only, or
--figure), and a printed verdict.

Statistics (PREREG s5): rate per agent x cell x arm; cut = rate(neutral) - rate(stacked) per agent; pooled = mean of
the 3 agents (agents equal). 95% intervals: 10,000 bootstrap draws that resample runs within agent x cell x arm x
template. Resampling m runs with replacement from a stratum with k follows gives a Binomial(m, k/m) count, so each
stratum is drawn as one binomial (identical in distribution, and fast). Seed 20261113.
Control (reported, not in the status): a label-shuffle null per agent x cell (arm labels permuted 10,000 times within
template strata); its two-sided p-value says how often a cut this large appears with no arm effect.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CELLS = ("A", "A2", "C", "B", "B2", "E")
CUE_CELLS = ("A", "A2", "C", "E")
FREE_CELLS = ("B", "B2")
ARMS = ("neutral", "stacked")
B = 10_000
SEED = 20261113
SHORT = {"gpt-5.4-mini": "mini", "gemini-3.5-flash-lite": "fl35", "gemini-3.1-flash-lite": "fl31"}


def load(phase: str) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "web" / "results" / f"scores_{phase}.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["phase"] == phase]


def boot_rate(runs: list[dict], rng: np.random.Generator) -> np.ndarray:
    """B bootstrap draws of the follow rate, resampling runs within template strata."""
    by_t = defaultdict(list)
    for r in runs:
        by_t[r["template"]].append(int(r["fit"]))
    n = len(runs)
    tot = np.zeros(B)
    for vals in by_t.values():
        m, k = len(vals), sum(vals)
        tot += rng.binomial(m, k / m, size=B)
    return tot / n


def ci(x: np.ndarray) -> list[float]:
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def shuffle_p(neu: list[dict], stk: list[dict], rng: np.random.Generator) -> float:
    """Two-sided label-shuffle p for the cut, permuting arm labels within template strata."""
    obs = np.mean([r["fit"] for r in neu]) - np.mean([r["fit"] for r in stk])
    strata = defaultdict(lambda: ([], []))
    for r in neu:
        strata[r["template"]][0].append(int(r["fit"]))
    for r in stk:
        strata[r["template"]][1].append(int(r["fit"]))
    nn, ns = len(neu), len(stk)
    diff = np.zeros(B)
    for a, b in strata.values():
        pool = np.array(a + b)
        perm = np.argsort(rng.random((B, len(pool))), axis=1)
        sh = pool[perm]
        diff += sh[:, : len(a)].sum(1) / nn - sh[:, len(a):].sum(1) / ns
    return float((np.abs(diff) >= abs(obs) - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--figure", action="store_true", help="write results/figure_data.json for a non-counted phase too")
    a = ap.parse_args()
    rows = load(a.phase)
    rng = np.random.default_rng(SEED)
    agents = sorted({r["model"] for r in rows}, key=lambda m: list(SHORT).index(m) if m in SHORT else 99)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["hiding"], r["arm"])].append(r)

    cells: dict = {}
    draws: dict = {}
    for c in CELLS:
        cells[c] = {"agents": {}, "pooled": {}}
        for m in agents:
            d = {}
            for arm in ARMS:
                runs = groups.get((m, c, arm), [])
                if not runs:
                    continue
                k, n = sum(r["fit"] for r in runs), len(runs)
                bd = boot_rate(runs, rng)
                draws[(m, c, arm)] = bd
                d[arm] = {"k": k, "n": n, "rate": round(k / n, 4), "ci": ci(bd),
                          "cue": sum(r["cue"] for r in runs), "cue_css": sum(r["cue_css"] for r in runs),
                          "css_fetched": sum(r["css_fetched"] for r in runs),
                          "notice_hidden": sum(r["notice_hidden"] for r in runs), "place_word": sum(r["place_word"] for r in runs),
                          "place_word_nofollow": sum(r["place_word"] and not r["fit"] for r in runs), "noticed_parent": sum(r["noticed"] for r in runs),
                          "honeypot_named": sum(r["honeypot_named"] for r in runs), "off": sum(r["off"] for r in runs),
                          "correct": sum(r["correct"] for r in runs)}
            if all(x in d for x in ARMS):
                cut = draws[(m, c, "neutral")] - draws[(m, c, "stacked")]
                d["cut"] = round(d["neutral"]["rate"] - d["stacked"]["rate"], 4)
                d["cut_ci"] = ci(cut)
                d["shuffle_p"] = round(shuffle_p(groups[(m, c, "neutral")], groups[(m, c, "stacked")], rng), 4)
            cells[c]["agents"][m] = d
        have = [m for m in agents if "cut" in cells[c]["agents"].get(m, {})]
        if have and len(have) == len(agents):
            for arm in ARMS:
                pd = np.mean([draws[(m, c, arm)] for m in agents], axis=0)
                cells[c]["pooled"][arm] = {"rate": round(float(np.mean([cells[c]["agents"][m][arm]["rate"] for m in agents])), 4),
                                           "ci": ci(pd), "k": sum(cells[c]["agents"][m][arm]["k"] for m in agents),
                                           "n": sum(cells[c]["agents"][m][arm]["n"] for m in agents)}
            pc = np.mean([draws[(m, c, "neutral")] - draws[(m, c, "stacked")] for m in agents], axis=0)
            cells[c]["pooled"]["cut"] = round(cells[c]["pooled"]["neutral"]["rate"] - cells[c]["pooled"]["stacked"]["rate"], 4)
            cells[c]["pooled"]["cut_ci"] = ci(pc)
            cells[c]["cue_score"] = round(np.mean([r["cue"] for r in rows if r["hiding"] == c]), 4)

    complete = all(cells[c]["pooled"] for c in CELLS)
    out: dict = {"phase": a.phase, "agents": agents, "n_runs": len(rows), "bootstrap": B, "seed": SEED, "cells": cells}
    if complete:
        # PREREG s5 status
        gate_agents = [m for m in agents if cells["A"]["agents"][m]["cut"] >= 0.25]
        gate = bool(gate_agents)
        yes = all(cells[c]["pooled"]["stacked"]["ci"][0] >= 0.70 and
                  all(cells[c]["agents"][m]["cut_ci"][1] < 0.30 for m in agents) for c in FREE_CELLS)
        no_reasons = []
        for c in FREE_CELLS:
            if cells[c]["pooled"]["stacked"]["ci"][1] < 0.70:
                no_reasons.append(f"{c}: pooled stacked CI-high {cells[c]['pooled']['stacked']['ci'][1]} < 0.70")
            for m in agents:
                ag = cells[c]["agents"][m]
                if ag["cut"] >= 0.30 and ag["cut_ci"][0] > 0.15:
                    no_reasons.append(f"{c}: {m} cut {ag['cut']} >= 0.30 and CI-low {ag['cut_ci'][0]} > 0.15")
        if not gate:
            status = "NOT SETTLED"
        elif yes:
            status = "YES"
        elif no_reasons:
            status = "NO"
        else:
            status = "NOT SETTLED"
        # The law (reported): per agent that passes the A gate, and pooled; >= 5 of 6 cells match.
        def law(cuts: dict) -> dict:
            match = {c: (cuts[c] >= 0.25) if c in CUE_CELLS else (cuts[c] < 0.15) for c in CELLS}
            return {"cuts": cuts, "match": match, "n_match": sum(match.values()), "holds": sum(match.values()) >= 5}
        law_agents = {m: law({c: cells[c]["agents"][m]["cut"] for c in CELLS}) for m in gate_agents}
        law_pooled = law({c: cells[c]["pooled"]["cut"] for c in CELLS})
        # Prereg legs (point estimates as worded)
        leg_i = cells["A"]["pooled"]["cut"] >= 0.25
        surv = [c for c in FREE_CELLS if cells[c]["pooled"]["stacked"]["rate"] >= 0.70 and
                all(cells[c]["agents"][m]["cut"] < 0.30 for m in agents)]
        leg_ii = bool(surv)
        leg_iii = law_pooled["holds"]
        vc_path = ROOT / "viewcheck" / "viewcheck.json"
        absent = None
        if vc_path.exists():
            absent = json.loads(vc_path.read_text())
        leg_iv_cells = []
        if absent is not None:
            leg_iv_cells = [c for c in surv if _absent_all(absent, c)]
        leg_iv = bool(leg_iv_cells) if absent is not None else None
        legs = {"i_A_pooled_cut_ge_25": leg_i, "ii_survivor": surv, "iii_law_pooled_5of6": leg_iii,
                "iv_survivor_absent_a11y_reader_print": leg_iv_cells if absent is not None else "viewcheck.json missing"}
        held = leg_i and leg_ii and leg_iii and bool(leg_iv)
        out |= {"status": status, "gate": {"passed": gate, "agents": gate_agents}, "yes": yes, "no_reasons": no_reasons,
                "law_agents": law_agents, "law_pooled": law_pooled, "prereg_legs": legs,
                "prereg_outcome": "held" if held else "failed"}
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "cells").mkdir(exist_ok=True)
    for c in CELLS:  # one artifact per proof-matrix cell (agent x cell, both arms + cut + shuffle null; pooled)
        for m, d in cells[c]["agents"].items():
            (res / "cells" / f"{a.phase}__{c}__{SHORT.get(m, m)}.json").write_text(json.dumps({"cell": c, "model": m, **d}, indent=1))
        if cells[c]["pooled"]:
            (res / "cells" / f"{a.phase}__{c}__pooled.json").write_text(json.dumps({"cell": c, "model": "pooled", **cells[c]["pooled"]}, indent=1))
    (res / f"analysis_{a.phase}.json").write_text(json.dumps(out, indent=1))
    with (res / f"cells_{a.phase}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "model", "cell", "arm", "k", "n", "rate", "lo", "hi", "cue_runs", "notice_hidden", "css_fetched"])
        for c in CELLS:
            for m, d in cells[c]["agents"].items():
                for arm in ARMS:
                    if arm in d:
                        x = d[arm]
                        w.writerow([a.phase, m, c, arm, x["k"], x["n"], x["rate"], x["ci"][0], x["ci"][1], x["cue"], x["notice_hidden"], x["css_fetched"]])
            for arm in ARMS:
                if arm in cells[c]["pooled"]:
                    x = cells[c]["pooled"][arm]
                    w.writerow([a.phase, "pooled", c, arm, x["k"], x["n"], x["rate"], x["ci"][0], x["ci"][1], "", "", ""])
    if a.phase == "counted" or a.figure:
        fig = {"_phase": a.phase, "_source": f"results/analysis_{a.phase}.json", "cells": {}}
        for c in CELLS:
            fig["cells"][c] = {"cue_score": cells[c].get("cue_score")}
            for arm in ARMS:
                fig["cells"][c][arm] = {SHORT.get(m, m): {k: cells[c]["agents"][m][arm][k] for k in ("k", "n")} |
                                        {"rate": cells[c]["agents"][m][arm]["rate"], "lo": cells[c]["agents"][m][arm]["ci"][0],
                                         "hi": cells[c]["agents"][m][arm]["ci"][1]}
                                        for m in agents if arm in cells[c]["agents"].get(m, {})}
                if arm in cells[c]["pooled"]:
                    p = cells[c]["pooled"][arm]
                    fig["cells"][c][arm]["pooled"] = {"k": p["k"], "n": p["n"], "rate": p["rate"], "lo": p["ci"][0], "hi": p["ci"][1]}
            if "cut" in cells[c]["pooled"]:
                fig["cells"][c]["pooled_cut"] = {"cut": cells[c]["pooled"]["cut"], "lo": cells[c]["pooled"]["cut_ci"][0],
                                                 "hi": cells[c]["pooled"]["cut_ci"][1]}
        (res / "figure_data.json").write_text(json.dumps(fig, indent=1))
    # Print
    for c in CELLS:
        line = [f"{c:3s} cue={cells[c].get('cue_score')}"]
        for m in agents:
            d = cells[c]["agents"].get(m, {})
            if "cut" in d:
                line.append(f"{SHORT.get(m, m)} {d['neutral']['k']}/{d['neutral']['n']}->{d['stacked']['k']}/{d['stacked']['n']} "
                            f"cut {d['cut']:+.3f} [{d['cut_ci'][0]:+.2f},{d['cut_ci'][1]:+.2f}] p={d['shuffle_p']} nh {d['stacked']['notice_hidden']}")
        if cells[c]["pooled"]:
            p = cells[c]["pooled"]
            line.append(f"POOLED stacked {p['stacked']['rate']:.3f} {p['stacked']['ci']} cut {p['cut']:+.3f} {p['cut_ci']}")
        print(" | ".join(line))
    if complete:
        print("STATUS", out["status"], "gate", out["gate"], "no_reasons", out["no_reasons"])
        print("LAW pooled", out["law_pooled"]["n_match"], "of 6; per agent", {SHORT.get(m, m): v["n_match"] for m, v in out["law_agents"].items()})
        print("PREREG legs", out["prereg_legs"], "->", out["prereg_outcome"])


def _absent_all(vc: list[dict], cell: str) -> bool:
    """Iris's viewcheck.json (one row per template x hiding, booleans): the cell's fitting link is absent from the
    Chromium a11y tree, print and Reader View in every template (PREREG s6 / prereg leg iv)."""
    rows = [r for r in vc if r["hiding"] == cell]
    if not rows:
        return False
    return all(not r["a11y"] and not r["print"] and not r["reader"] for r in rows)


if __name__ == "__main__":
    main()
