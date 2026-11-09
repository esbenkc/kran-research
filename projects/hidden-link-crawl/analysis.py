"""Score the frozen law against the agent runs (PREREG s5 + s8.8). No model.

    .venv/bin/python analysis.py [--phase counted] [--pred predictions.json]

Inputs: results/scores_<phase>.jsonl (score.py: follow = GET of the run's own address in the access log) and
predictions.json (predict.py). Each run's cell (tool:method) maps to its context; runs of all cells in one context pool.
Per combo (context x agent x anchor): measured = followed/runs, pred = exposure x frozen rate, d = measured - pred,
95% Newcombe hybrid score interval of d (Wilson for the measured rate; Wilson for the frozen rate, scaled by exposure).
  HIT  |d| <= 0.10          MISS  interval fully outside [-0.10, +0.10]
  OFF  |d| > 0.10 and the interval excludes 0
Headline combos: off-task combos on exposed contexts, every combo on an interior context, fit combos on reachable
zeros. Sanity strip: fit combos at exposure 1 and other exposure-0 combos (count for MISS/OFF, not the HIT share).
HIT share (PREREG v1.1.2): the 14 non-free headline combos = 9 off-task family combos (mcpraw/sr_only, crawl4ai,
browseruse/sr_only x 3 agents), 3 mcpfetch interior fit combos, 2 interior mini off-task combos.
Status: Yes = <=1 MISS (exposed combos) AND <=1 OFF (headline + exposed fit) AND >=80% HIT-share HIT AND law MAE <
all1 and < all0 (ties = not lower); No = >=2 MISS or >=2 OFF; else Not settled.
Writes results/analysis.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BAND = 0.10


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def newcombe(k1: int, n1: int, k2: int, n2: int, scale: float) -> tuple[float, float, float]:
    """d = k1/n1 - scale*k2/n2 with Newcombe's hybrid score interval (method 10); the second proportion and its
    Wilson bounds are multiplied by `scale` (the crawl exposure)."""
    p1, p2 = k1 / n1, scale * k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = (scale * x for x in wilson(k2, n2))
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="counted")
    ap.add_argument("--pred", default="predictions.json")
    ap.add_argument("--out", default="results/analysis.json")
    a = ap.parse_args()
    P = json.loads((ROOT / a.pred).read_text())
    cell2ctx = {cell: c["id"] for c in P["contexts"] for cell in c["cells"]}
    ctx = {c["id"]: c for c in P["contexts"]}
    pred = {(p["context"], p["agent"], p["anchor"]): p for p in P["predictions"]}
    runs = collections.defaultdict(list)
    other = collections.defaultdict(list)
    for line in (ROOT / "results" / f"scores_{a.phase}.jsonl").read_text().splitlines():
        r = json.loads(line)
        cell = f"{r['tool']}:{r['method']}"
        if r["slot"] != "footer" or r["css"] != "inline" or cell not in cell2ctx:
            other[(r["cell"], r["model"])].append(r)  # replication row and any non-counted cell
            continue
        runs[(cell2ctx[cell], r["model"], r["anchor"])].append(r)

    combos = []
    for key, rs in sorted(runs.items()):
        c, ag, k = key
        p = pred[key]
        n, f = len(rs), sum(bool(r["followed"]) for r in rs)
        fk, fn = p["rate"]
        d, lo, hi = newcombe(f, n, fk, fn, p["exposure"])
        kind = ctx[c]["kind"]
        headline = (k == "nonfit" and kind == "exposed") or kind == "interior" or (k == "fit" and kind == "reachable_zero")
        hit = abs(d) <= BAND + 1e-12
        miss = lo > BAND or hi < -BAND
        off = abs(d) > BAND and (lo > 0 or hi < 0)
        combos.append({"context": c, "cells": ctx[c]["cells"], "agent": ag, "anchor": k, "kind": kind,
                       "format": ctx[c]["format"], "exposure": p["exposure"], "n": n, "followed": f,
                       "measured": round(f / n, 4), "pred": p["pred"], "d": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "hit": hit, "miss": miss, "off": off, "headline": headline,
                       "all1": p["all1"], "all0": p["all0"], "all0strict": p["all0strict"],
                       "paged_runs": sum(bool(r.get("paged")) for r in rs), "raw_switch_runs": sum(bool(r.get("raw_switch")) for r in rs),
                       "scrolled_runs": sum(bool(r.get("scrolled")) for r in rs),
                       "click_error_runs": sum(bool(r.get("click_errors")) for r in rs),
                       "cells_runs": dict(collections.Counter(f"{r['tool']}:{r['method']}" for r in rs))})
    cdir = ROOT / "results" / "cells"
    cdir.mkdir(parents=True, exist_ok=True)
    for x in combos:  # one artifact per combo: the runs it pools (token, cell, template, wording, followed)
        rs = runs[(x["context"], x["agent"], x["anchor"])]
        x["artifact"] = f"results/cells/{x['context'].replace('/', '_')}__{x['agent']}__{x['anchor']}.json"
        (ROOT / x["artifact"]).write_text(json.dumps({**{k: v for k, v in x.items() if k != "artifact"}, "runs": [
            {k: r[k] for k in ("token", "i", "tool", "method", "template", "paraphrase", "followed", "follow_fetch_index",
                               "exposed_hid", "exposed_anchor", "steps", "trace")} for r in rs]}, indent=1) + "\n")
    exposed = [x for x in combos if x["exposure"] > 0]
    head = [x for x in combos if x["headline"]]
    FAMILIES = {"mcpraw/sr_only", "browseruse/sr_only", next(c["id"] for c in P["contexts"] if "crawl4ai:sr_only" in c["cells"])}
    MCPFETCH_INT = next(c["id"] for c in P["contexts"] if "mcpfetch:sr_only" in c["cells"])
    share = [x for x in combos if (x["anchor"] == "nonfit" and x["context"] in FAMILIES)
             or (x["kind"] == "interior" and x["anchor"] == "nonfit" and x["agent"] == "gpt-5.4-mini")
             or (x["context"] == MCPFETCH_INT and x["anchor"] == "fit")]
    for x in combos:
        x["in_hit_share"] = x in share
    off_scope = [x for x in combos if x["headline"] or (x["anchor"] == "fit" and x["exposure"] == 1)]
    n_miss = sum(x["miss"] for x in exposed)
    n_off = sum(x["off"] for x in off_scope)
    hit_share = sum(x["hit"] for x in share) / len(share) if share else float("nan")

    def mae(key):
        return sum(abs(x["measured"] - x[key]) for x in combos) / len(combos)
    m_law, m_all1, m_all0, m_strict = mae("pred"), mae("all1"), mae("all0"), mae("all0strict")
    beats = m_law < m_all1 and m_law < m_all0
    if n_miss >= 2 or n_off >= 2:
        status = "REJECTED"
    elif n_miss <= 1 and n_off <= 1 and hit_share >= 0.8 and beats:
        status = "CONFIRMED"
    else:
        status = "UNCONFIRMED"

    # frozen goal prediction, scored as written (PREREG s0): clause 3 per agent x tool:method cell with fit runs
    # (every tool:method cell of a context takes the context's measured rate, so each cell counts once per agent)
    cells_fit = [abs(x["d"]) <= BAND for x in combos if x["anchor"] == "fit" for _ in x["cells"]]
    clause3 = sum(cells_fit) / len(cells_fit) if cells_fit else float("nan")
    zero_follow = [x for x in combos if x["exposure"] == 0]
    clause4 = all(x["measured"] <= 0.02 for x in zero_follow)
    rep = {k[0] + "|" + k[1]: {"n": len(v), "followed": sum(bool(r["followed"]) for r in v)} for k, v in other.items()}

    out = {"phase": a.phase, "status": status, "n_combos": len(combos), "n_headline": len(head),
           "n_exposed_combos": len(exposed), "miss": n_miss, "off": n_off, "headline_hit_share": round(hit_share, 4), "n_hit_share": len(share), "hit_share_hits": sum(x["hit"] for x in share),
           "mae": {"law": round(m_law, 4), "all1": round(m_all1, 4), "all0": round(m_all0, 4), "all0strict": round(m_strict, 4)},
           "law_beats_both": beats, "prereg_clause3_fit_cells_within_10": round(clause3, 4),
           "prereg_clause4_zero_cells_le_2pct": clause4, "other_rows": rep, "combos": combos}
    (ROOT / a.out).parent.mkdir(exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"STATUS {status}: MISS {n_miss} (of {len(exposed)} exposed), OFF {n_off} (of {len(off_scope)}), "
          f"HIT share {hit_share:.2f} ({sum(x['hit'] for x in share)}/{len(share)}), MAE law {m_law:.3f} all1 {m_all1:.3f} all0 {m_all0:.3f}")
    for x in combos:
        flag = "MISS" if x["miss"] else "OFF" if x["off"] else "HIT" if x["hit"] else "UNS"
        print(f"  {flag:4s} {'S' if x['in_hit_share'] else 'H' if x['headline'] else ' '} {x['context']:28s} {x['agent']:22s} {x['anchor']:6s} "
              f"exp {x['exposure']:.2f} {x['followed']:3d}/{x['n']:<3d} meas {x['measured']:.2f} pred {x['pred']:.2f} "
              f"d {x['d']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}]")
    for k, v in rep.items():
        print(f"  other {k}: {v['followed']}/{v['n']}")


if __name__ == "__main__":
    main()
