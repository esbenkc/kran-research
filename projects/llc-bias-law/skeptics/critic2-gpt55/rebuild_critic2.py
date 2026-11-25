#!/usr/bin/env python3
"""critic2 cross-family rebuild for theory-1125-llc-bias-law.

Inputs: projects/llc-bias-law/predictions.json, data/cells_S/*.json,
        projects/llc-bias-law/results.json for stored-aggregate comparison only.
Outputs: skeptics/critic2-gpt55/rebuild_output.json.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "skeptics" / "critic2-gpt55" / "rebuild_output.json"


def skey(row: dict) -> str:
    return f"nb{row['nbeta']:g}_q{row['q']:g}_c{row['c']:g}"


def load_raw_cells() -> dict[str, list[dict]]:
    raw: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((ROOT / "data" / "cells_S").glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            raw[f"{row['case']}|{skey(row)}"].append(row)
    return raw


def measure(rows: list[dict]) -> dict:
    ok = all(row["alive"] and row["halves_ok"] and math.isfinite(row["e"]) for row in rows)
    e = sum(row["e"] for row in rows) / len(rows)
    return {
        "e_meas": e,
        "converged": ok,
        "n_seeds": len(rows),
        "seed_e": {str(row["seed"]): row["e"] for row in rows},
    }


def miss(cell: dict, law: str = "L", use_seed: str | None = None) -> float:
    if not cell["converged"]:
        return 100.0
    if use_seed is None:
        e = cell["e_meas"]
    else:
        e = cell["seed_e"].get(use_seed)
        if e is None:
            return 100.0
    if not math.isfinite(e):
        return 100.0
    return 100.0 * abs(cell["pred"][law] - e)


def score(cells: list[dict], law: str = "L") -> dict:
    misses = [miss(cell, law) for cell in cells]
    return {
        "n": len(cells),
        "within_5": sum(value <= 5 for value in misses),
        "share_within_5": sum(value <= 5 for value in misses) / len(misses) if misses else None,
        "median_miss": statistics.median(misses) if misses else None,
        "max_miss": max(misses) if misses else None,
        "unsettled": sum(not cell["converged"] for cell in cells),
    }


def main() -> None:
    pred_payload = json.loads((ROOT / "predictions.json").read_text())
    predictions = pred_payload["cells"]
    raw = load_raw_cells()
    cells: list[dict] = []
    for key, pred in predictions.items():
        if key not in raw:
            continue
        measured = measure(raw[key])
        cells.append({
            "key": key,
            "case": pred["case"],
            "group": pred["group"],
            "primary": pred["primary"],
            "nbeta": pred["nbeta"],
            "q": pred["q"],
            "c": pred["c"],
            **measured,
            "pred": {
                "L": pred["e_L"],
                "L_z": pred["e_Lz"],
                "N1": pred["e_N1"],
                "N2": pred["e_N2"],
                "N3": pred["e_N3"],
                "N4": pred["e_N4"],
            },
        })

    primary = [cell for cell in cells if cell["primary"]]
    primary_scores = {law: score(primary, law) for law in ("L", "L_z", "N1", "N2", "N3", "N4")}
    seed_shares = {}
    for seed in ("0", "1", "2"):
        values = [miss(cell, "L", seed) for cell in primary]
        seed_shares[seed] = {
            "within_5": sum(value <= 5 for value in values),
            "n": len(values),
            "share_within_5": sum(value <= 5 for value in values) / len(values),
        }

    # Shape-transfer numbers: all 180 primary cells, grouped by nbeta/q/c.
    by_setting: dict[tuple, list[float]] = defaultdict(list)
    for cell in primary:
        by_setting[(cell["nbeta"], cell["q"], cell["c"])].append(100.0 * cell["e_meas"])
    ranges = [max(values) - min(values) for values in by_setting.values() if len(values) == 5]
    ys = [100.0 * cell["e_meas"] for cell in primary]
    mean_y = sum(ys) / len(ys)
    total_ss = sum((y - mean_y) ** 2 for y in ys)
    within_ss = sum(
        sum((y - sum(values) / len(values)) ** 2 for y in values)
        for values in by_setting.values()
    )

    seen = [cell for cell in cells if cell["group"] == "seen" and cell["q"] >= 0.1]
    seen_by_shape = {}
    for case in sorted({cell["case"] for cell in seen}):
        shape_cells = [cell for cell in seen if cell["case"] == case]
        shape_score = score(shape_cells, "L")
        settled = [cell for cell in shape_cells if cell["converged"]]
        settled_hits = sum(miss(cell, "L") <= 5 for cell in settled)
        shape_score.update({
            "settled_n": len(settled),
            "settled_within_5": settled_hits,
            "settled_share_within_5": settled_hits / len(settled) if settled else None,
        })
        seen_by_shape[case] = shape_score

    control = [cell for cell in cells if cell["group"] == "control" and cell["q"] >= 0.1]
    control_score = score(control, "L")
    control_settled = [cell for cell in control if cell["converged"]]
    control_score.update({
        "settled_n": len(control_settled),
        "settled_within_5": sum(miss(cell, "L") <= 5 for cell in control_settled),
    })

    stored = json.loads((ROOT / "results.json").read_text())
    output = {
        "inputs": {
            "recomputed_from_raw": ["predictions.json", "data/cells_S/*.json"],
            "stored_comparison_only": "results.json",
            "prereg_commit": pred_payload.get("prereg_commit"),
        },
        "primary": {
            "n": len(primary),
            "scores": primary_scores,
            "seed_shares": seed_shares,
        },
        "shape_transfer_all_primary": {
            "n_settings": len(ranges),
            "median_fixed_setting_range_pts": statistics.median(ranges),
            "setting_only_r2": 1.0 - within_ss / total_ss,
        },
        "s6_seen_recomputed": {
            **score(seen, "L"),
            "by_shape": seen_by_shape,
        },
        "control_16x16x8r8_recomputed": control_score,
        "results_json_comparison": {
            "primary_L": stored["primary"]["L"],
            "S6_seen": stored["secondary"]["S6_seen"],
            "control_16x16x8r8": stored["secondary"]["control_16x16x8r8"],
        },
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
