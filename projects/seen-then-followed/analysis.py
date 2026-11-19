"""PREREG s4/s5 estimator: status of the question and the scout's three clauses, from score.py's per-run rows.

    .venv/bin/python score.py --phase main && .venv/bin/python score.py --phase gate
    .venv/bin/python analysis.py            -> results/analysis.json + stdout

slots.json (written at the s1 switch decision) maps slot A / B to the chosen model ids.
Cells: FIT (raw,A) (raw,B) (mcpfetch,B) (lcpw,A); HELD OUT (mcpfetch,A) (lcpw,B).
Scored set (s2): the largest k-prefix complete in all 6 main cells (run i counts when i mod 30 < K), floor K >= 20.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HARN = ("raw", "mcpfetch", "lcpw")
CELL = {h: f"{h}:display_none:fit:footer:class" for h in HARN}
GATE_CELL = "raw:visible:fit:footer:class"
FIT = [("raw", "A"), ("raw", "B"), ("mcpfetch", "B"), ("lcpw", "A")]
HELD = [("mcpfetch", "A"), ("lcpw", "B")]
OTHER = {"A": "B", "B": "A"}
N_BOOT, SEED = 2000, 1118
T975_DF2 = 4.302652729911275  # t quantile 0.975, 2 degrees of freedom (3 blocks)


def load(phase: str) -> list[dict]:
    p = ROOT / "results" / f"runs_{phase}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def arrs(rows: list[dict]) -> dict[str, np.ndarray]:
    return {"reached": np.array([r["reached"] for r in rows], bool), "followed": np.array([r["followed"] for r in rows], bool)}


def fit_values(cells: dict, idx: dict | None = None) -> tuple[dict, dict, dict]:
    """reach of every FIT cell, run-weighted pooled follow per slot over its FIT cells, and the pooled follow over all
    4 FIT cells (model-blind null). idx = optional bootstrap index per cell."""
    def get(key):
        a = cells[key]
        return {k: v[idx[key]] for k, v in a.items()} if idx is not None else a
    reach, num, den = {}, {"A": 0, "B": 0}, {"A": 0, "B": 0}
    for h, s in FIT:
        a = get((h, s))
        reach[(h, s)] = a["reached"].mean()
        num[s] += int((a["followed"] & a["reached"]).sum())
        den[s] += int(a["reached"].sum())
    follow = {s: (num[s] / den[s] if den[s] else float("nan")) for s in "AB"}
    pooled = (num["A"] + num["B"]) / (den["A"] + den["B"]) if den["A"] + den["B"] else float("nan")
    return reach, follow, {"all": pooled}


def predict(h: str, s: str, reach: dict, follow: dict, pooled: dict | None = None) -> float:
    f = follow[s] if pooled is None else pooled["all"]
    return reach[(h, OTHER[s])] * f  # identity-safe: reach from the other slot's fit cell in the same harness


def main() -> None:
    slots = json.loads((ROOT / "slots.json").read_text())
    model_of = {"A": slots["A"], "B": slots["B"]}
    main_rows = load("main")
    by = {}
    for h in HARN:
        for s in "AB":
            by[(h, s)] = [r for r in main_rows if r["cell"] == CELL[h] and r["model"] == model_of[s]]
    # scored set: largest K with runs k, 30+k, 60+k present in all 6 cells for every k < K
    K = 0
    while K < 30 and all({K, 30 + K, 60 + K} <= {r["i"] for r in v} for v in by.values()):
        K += 1
    scored = {c: sorted([r for r in v if r["i"] % 30 < K], key=lambda r: r["i"]) for c, v in by.items()}
    cells = {c: arrs(v) for c, v in scored.items()}
    n = {c: len(v) for c, v in scored.items()}

    reach, follow, pooled = fit_values(cells)
    # the bootstrap: resample runs within every cell (all 6 are used by at least one prediction)
    rng = np.random.default_rng(SEED)
    boot = {c: [] for c in HELD}
    boot_null = {c: [] for c in HELD}
    boot_catch = {c: [] for c in scored}
    boot_follow = {"A": [], "B": []}
    for _ in range(N_BOOT):
        idx = {c: rng.integers(0, n[c], n[c]) for c in scored}
        r_b, f_b, p_b = fit_values(cells, idx)
        for s in "AB":
            boot_follow[s].append(f_b[s])
        for c in scored:
            boot_catch[c].append(cells[c]["followed"][idx[c]].mean())
        for h, s in HELD:
            obs = cells[(h, s)]["followed"][idx[(h, s)]].mean()
            boot[(h, s)].append(obs - predict(h, s, r_b, f_b))
            boot_null[(h, s)].append(obs - predict(h, s, r_b, f_b, p_b))

    def ci(x):
        x = np.array(x, float)
        x = x[~np.isnan(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]

    # seed level: block b uses only block-b runs of the held-out cell and of every fit cell
    seed = {}
    for h, s in HELD:
        errs, preds, catches, ns = [], [], [], []
        for b in range(3):
            cb = {c: arrs([r for r in v if r["i"] // 30 == b]) for c, v in scored.items()}
            rb, fb, _ = fit_values(cb)
            preds.append(float(predict(h, s, rb, fb)))
            catches.append(float(cb[(h, s)]["followed"].mean()))
            ns.append(int(len(cb[(h, s)]["followed"])))
            errs.append(catches[-1] - preds[-1])
        m, sd = float(np.mean(errs)), float(np.std(errs, ddof=1))
        half = T975_DF2 * sd / math.sqrt(3)
        seed[(h, s)] = {"err_by_block": errs, "pred_by_block": preds, "catch_by_block": catches, "n_by_block": ns, "mean": m, "ci": [m - half, m + half], "method": "t-interval over 3 blocks (df 2)"}

    # gates
    gate_rows = load("gate")
    gate = {}
    for s in "AB":
        g = [r for r in gate_rows if r["cell"] == GATE_CELL and r["model"] == model_of[s] and r["i"] < 60]
        ok = sum(r["followed"] and r["correct"] for r in g)
        gate[s] = {"model": model_of[s], "n": len(g), "success": ok, "pass": len(g) == 60 and ok >= 42,
                   "by_block": [sum(r["followed"] and r["correct"] for r in g if r["i"] // 30 == b) for b in (0, 1)]}
    herr = {c: sum(r["api_error"] for r in v) for c, v in scored.items()}
    gates = {"skill": all(gate[s]["pass"] for s in "AB"),
             "harness": all(herr[c] <= 0.02 * n[c] for c in scored),
             "completeness": K >= 20}

    held_out = {}
    for h, s in HELD:
        pred = predict(h, s, reach, follow)
        obs = float(cells[(h, s)]["followed"].mean())
        pnull = predict(h, s, reach, follow, pooled)
        crawl = json.loads((ROOT / "results" / "crawl.json").read_text())["crawl_reach_home"] \
            if (ROOT / "results" / "crawl.json").exists() else None
        pcrawl = crawl[h] * follow[s] if crawl else None  # reported row: crawl-guess reach (crawl.py) x follow_M
        held_out[(h, s)] = {"pred": pred, "obs": obs, "err": obs - pred, "err_ci": ci(boot[(h, s)]),
                            "pred_crawl": pcrawl, "err_crawl": (obs - pcrawl) if pcrawl is not None else None,
                            "pred_null": pnull, "err_null": obs - pnull, "err_null_ci": ci(boot_null[(h, s)])}
    within = all(abs(v["err"]) <= 0.10 for v in held_out.values())
    in_range = all(0.10 <= v["pred"] <= 0.90 for v in held_out.values())
    outside = any(v["err_ci"][0] > 0.10 or v["err_ci"][1] < -0.10 for v in held_out.values())
    if not all(gates.values()):
        status = "UNCONFIRMED"
    elif within and in_range:
        status = "CONFIRMED"
    elif outside:
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"

    # s5: the scout's three clauses on all 6 cells
    fgr = {c: (float((cells[c]["followed"] & cells[c]["reached"]).sum() / cells[c]["reached"].sum())
               if cells[c]["reached"].sum() else None) for c in scored}
    c1 = within
    c2_by = {h: (abs(fgr[(h, "A")] - fgr[(h, "B")]) if fgr[(h, "A")] is not None and fgr[(h, "B")] is not None else None)
             for h in HARN}
    c2 = None if any(v is None for v in c2_by.values()) else all(v < 0.15 for v in c2_by.values())
    hreach = {h: float((cells[(h, "A")]["reached"].sum() + cells[(h, "B")]["reached"].sum()) / (n[(h, "A")] + n[(h, "B")]))
              for h in HARN}
    c3 = max(hreach.values()) - min(hreach.values()) > 0.40
    pairs = {f"{a}-{b}": abs(hreach[a] - hreach[b]) for i, a in enumerate(HARN) for b in HARN[i + 1:]}
    if c1 and c2 and c3 and all(gates.values()):
        outcome = "held"
    elif (c1 is False) or (c2 is False) or (c3 is False):
        outcome = "failed"
    else:
        outcome = "unsettled"

    # fit-cell ticks for the figure: the two-factor model fitted on the 4 FIT cells
    reach_h = {"raw": float((cells[("raw", "A")]["reached"].sum() + cells[("raw", "B")]["reached"].sum()) / (n[("raw", "A")] + n[("raw", "B")])),
               "mcpfetch": float(reach[("mcpfetch", "B")]), "lcpw": float(reach[("lcpw", "A")])}
    out_cells = []
    for h in HARN:
        for s in "AB":
            c, v = (h, s), scored[(h, s)]
            role = "heldout" if c in HELD else "fit"
            pr = held_out[c]["pred"] if role == "heldout" else reach_h[h] * follow[s]
            row = {"harness": h, "slot": s, "model": model_of[s], "role": role, "n": n[c],
                   "reached": int(cells[c]["reached"].sum()), "followed": int(cells[c]["followed"].sum()),
                   "follow_given_reach": fgr[c],
                   "follow_without_reach": int((cells[c]["followed"] & ~cells[c]["reached"]).sum()),
                   "catch": float(cells[c]["followed"].mean()), "catch_ci": ci(boot_catch[c]),
                   "pred": float(pr), "harness_errors": herr[c],
                   "seed_catch": [float(np.mean([r["followed"] for r in v if r["i"] // 30 == b])) for b in range(3)],
                   "raw_mode_runs": sum(r["raw_mode"] for r in v),
                   "first_reach_tool": {t: sum(r["first_reach_tool"] == t for r in v) for t in sorted({str(r["first_reach_tool"]) for r in v})},
                   "correct": sum(r["correct"] for r in v)}
            if role == "heldout":
                row.update({"err": held_out[c]["err"], "err_ci": held_out[c]["err_ci"], "seed_level": seed[c],
                            "pred_null": held_out[c]["pred_null"], "err_null": held_out[c]["err_null"],
                            "err_null_ci": held_out[c]["err_null_ci"], "pred_crawl": held_out[c]["pred_crawl"],
                            "err_crawl": held_out[c]["err_crawl"]})
            out_cells.append(row)
    res = {
        "slots": model_of, "K": K, "status": status, "gates": gates, "gate": gate,
        "rule": {"both_within_10": within, "both_preds_in_10_90": in_range, "some_ci_outside_10": outside},
        "reach_by_fit_cell": {f"{h},{s}": float(v) for (h, s), v in reach.items()},
        "reach_by_harness_fit": reach_h, "reach_by_harness_all_cells": hreach,
        "follow_by_slot": {s: {"value": follow[s], "ci": ci(boot_follow[s])} for s in "AB"},
        "follow_pooled_all_fit": pooled["all"],
        "prereg_clauses": {"C1": c1, "C2": c2, "C2_gap_by_harness": c2_by, "C3": c3, "C3_pairs": pairs, "outcome": outcome},
        "cells": out_cells,
        "bootstrap": {"draws": N_BOOT, "seed": SEED, "kind": "percentile 95%, runs resampled within every cell"},
        "spend_usd_main": sum(r["cost"] for r in main_rows), "spend_usd_gate": sum(r["cost"] for r in gate_rows),
    }
    out = ROOT / "results" / "analysis.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"K={K} (n per cell {sorted(set(n.values()))})  status {status}  gates {gates}")
    for s in "AB":
        print(f"  gate {s} {model_of[s]}: {gate[s]['success']}/{gate[s]['n']} blocks {gate[s]['by_block']}")
    for row in out_cells:
        extra = (f" err {row['err']:+.3f} CI [{row['err_ci'][0]:+.3f},{row['err_ci'][1]:+.3f}] seed-level {row['seed_level']['mean']:+.3f} "
                 f"[{row['seed_level']['ci'][0]:+.3f},{row['seed_level']['ci'][1]:+.3f}] null err {row['err_null']:+.3f}"
                 if row["role"] == "heldout" else "")
        print(f"  {row['role']:7s} {row['harness']:8s} {row['slot']} n {row['n']:3d} reach {row['reached']:3d} "
              f"f|r {row['follow_given_reach'] if row['follow_given_reach'] is None else round(row['follow_given_reach'], 3)} "
              f"catch {row['catch']:.3f} pred {row['pred']:.3f} f!r {row['follow_without_reach']} raw_mode {row['raw_mode_runs']}{extra}")
    print(f"  follow A {follow['A']:.3f} B {follow['B']:.3f}; clauses C1 {c1} C2 {c2} {c2_by} C3 {c3} {hreach} -> {outcome}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
