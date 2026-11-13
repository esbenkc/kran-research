"""Score the counted runs against the frozen predictions (PREREG v2.2). No model is called.

    .venv/bin/python score.py [--phase counted] [--out results/score.json]

follow = the server access log has a GET of the run's hidden page /<token>/r/<hid>.html (the only follow source).
Scored n = the largest n that EVERY scored route combo reached (its first n finished runs by index; a harness fault
is replaced by the next index, DEVIATIONS D1); floor 30, else NOT SETTLED.
clear miss = Newcombe 95% interval of (measured - 1109 calibration rate, n=30) lies fully outside +-0.10.
YES = route MAE <= 0.10 AND <= 1 clear miss AND 0 follows in every reach-0 combo.
NO  = route MAE > 0.15 OR >= 2 clear misses OR any reach-0 follow. Else NOT SETTLED.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROUTE_ACTIONS = {"lcpw": {"extract_hyperlinks", "get_elements"}, "pwmcp": {"browser_evaluate"},
                 "bufull": {"evaluate", "find_elements", "extract"}, "mcpfetch": {"fetch:raw"}}
Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% interval of p1 - p2 (Newcombe 1998, method 10, Wilson score limits)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def route_called(tool: str, a: dict) -> bool:
    t = a.get("tool")
    if tool == "mcpfetch":
        return t == "fetch" and bool((a.get("mcp_args") or {}).get("raw"))
    return t in ROUTE_ACTIONS.get(tool, set())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--out", default="results/score.json")
    a = ap.parse_args()
    pred = json.loads((ROOT / "predictions.json").read_text())
    P = {(c["tool"], c["method"], c["agent"]): c for c in pred["cells"]}
    gets = set(re.findall(r'\tGET\t(/[0-9a-f]+/r/[0-9a-f]+\.html)\t', (ROOT / "logs" / "access.log").read_text()))
    man = [json.loads(l) for l in (ROOT / "raw" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    man = [m for m in man if m["phase"] == a.phase]
    runs = collections.defaultdict(list)
    for m in man:
        rec = json.loads((ROOT / m["raw_file"]).read_text())
        acts = rec["actions"]
        first_route = next((x["step"] for x in acts if route_called(m["tool"], x)), None)
        first_hid = next((x for x in acts if x.get("hid_in_obs")), None)
        runs[(m["tool"], m["method"], m["model"])].append({
            "i": m["i"], "followed": m["hidden_path"] in gets, "route_called": first_route is not None,
            "route_step": first_route, "hid_seen": first_hid is not None,
            "hid_first_action": (first_hid or {}).get("tool"), "answered": rec["answer"] is not None,
            "steps": rec["steps_used"], "parse_fail": rec["parse_fail"], "cost": rec["cost_usd_est"],
            "saw_route_page": any(str(x.get("url", "") or "").endswith(m["route_path"].split("/")[-1]) for x in acts),
            "raw_file": m["raw_file"]})
    route_keys = [k for k, c in P.items() if c["role"] == "scored_route"]
    n_scored = min((len({r["i"] for r in runs.get(k, [])}) for k in route_keys), default=0)
    # equal-n truncation: every route combo is scored on its first n_scored finished runs by index (D1)
    cells = []
    for k, c in sorted(P.items()):
        if c["role"] == "crawl_only" and k not in runs:
            continue
        rs = sorted(runs.get(k, []), key=lambda r: r["i"])
        if c["role"] == "scored_route":
            rs = rs[:n_scored]  # first n_scored runs by index; a D1 harness fault is replaced by the next index
        n = len(rs)
        f = sum(r["followed"] for r in rs)
        meas = f / n if n else None
        row = {"tool": k[0], "method": k[1], "agent": k[2], "role": c["role"], "class": c["class"], "n": n, "follows": f,
               "measured": meas, "ci": wilson(f, n) if n else None, "pred_two_factor": c["pred_two_factor"],
               "pred_one_factor": c["pred_one_factor"], "pred_first_view": c["pred_first_view"],
               "route_called": sum(r["route_called"] for r in rs), "hid_seen": sum(r["hid_seen"] for r in rs),
               "followed_given_hid": sum(r["followed"] for r in rs if r["hid_seen"]),
               "follow_without_hid_seen": sum(r["followed"] and not r["hid_seen"] for r in rs),
               "hid_first_action": dict(collections.Counter(r["hid_first_action"] for r in rs if r["hid_seen"])),
               "answered_before_cap": sum(r["answered"] for r in rs), "answered_step0": sum(r["answered"] and r["steps"] <= 1 for r in rs),
               "parse_fail_runs": sum(r["parse_fail"] > 0 for r in rs), "saw_route_page": sum(r["saw_route_page"] for r in rs),
               "cost": round(sum(r["cost"] for r in rs), 4)}
        if n and c["role"] in ("scored_route", "replication"):
            k_cal, n_cal = c["cal"]
            lo, hi = newcombe(f, n, k_cal, n_cal)
            row.update(gap=meas - c["pred_two_factor"], gap_vs_cal_ci=(lo, hi), clear_miss=(hi < -0.10 or lo > 0.10),
                       point_hit=abs(meas - c["pred_two_factor"]) <= 0.10)
        cells.append(row)
    sr = [r for r in cells if r["role"] == "scored_route"]
    r0 = [r for r in cells if r["role"] == "scored_reach0"]
    mae = sum(abs(r["measured"] - r["pred_two_factor"]) for r in sr) / len(sr) if sr and n_scored else None
    misses = sum(r["clear_miss"] for r in sr) if n_scored else None
    r0_follows = sum(r["follows"] for r in r0)
    if n_scored < 30 or mae is None:
        status = "NOT SETTLED"
    elif mae > 0.15 or misses >= 2 or r0_follows > 0:
        status = "NO"
    elif mae <= 0.10 and misses <= 1 and r0_follows == 0:
        status = "YES"
    else:
        status = "NOT SETTLED"
    rep = {
        "mae_two_factor": mae, "clear_misses": misses, "reach0_follows": r0_follows, "reach0_n": [r["n"] for r in r0],
        "n_scored_route": n_scored, "status": status,
        "reported_point_hit_share": (sum(r["point_hit"] for r in sr) / len(sr)) if sr and n_scored else None,
        "reported_mae_one_factor": (sum(abs(r["measured"] - r["pred_one_factor"]) for r in sr) / len(sr)) if sr and n_scored else None,
        "reported_mae_const99": (sum(abs(r["measured"] - 0.99) for r in sr) / len(sr)) if sr and n_scored else None,
        "reported_mae_first_view": (sum(abs(r["measured"] - r["pred_first_view"]) for r in sr) / len(sr)) if sr and n_scored else None,
    }
    out = {"phase": a.phase, "predictions_sha256": (ROOT / "predictions.json.sha256").read_text().strip(), **rep, "cells": cells}
    (ROOT / a.out).parent.mkdir(exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"STATUS {status}: route MAE {mae if mae is None else round(mae, 3)}, clear misses {misses}, reach-0 follows "
          f"{r0_follows} (n {rep['reach0_n']}), scored n per route combo {n_scored}")
    for k in ("reported_point_hit_share", "reported_mae_one_factor", "reported_mae_const99", "reported_mae_first_view"):
        print(f"  {k} {rep[k] if rep[k] is None else round(rep[k], 3)}")
    for r in cells:
        extra = ""
        if "gap" in r:
            extra = f" gap {r['gap']:+.2f} vsCal[{r['gap_vs_cal_ci'][0]:+.2f},{r['gap_vs_cal_ci'][1]:+.2f}]{' CLEAR-MISS' if r['clear_miss'] else ''}"
        m = "-" if r["measured"] is None else f"{r['measured']:.2f}"
        print(f"  {r['role']:13s} {r['tool']:10s} {r['method']:12s} {r['agent']:22s} {r['follows']:3d}/{r['n']:<3d} meas {m} "
              f"pred {r['pred_two_factor']:.2f}{extra} | route_called {r['route_called']} hid_seen {r['hid_seen']} "
              f"f|hid {r['followed_given_hid']} f_nohid {r['follow_without_hid_seen']} ans0 {r['answered_step0']} via {r['hid_first_action']}")


if __name__ == "__main__":
    main()
