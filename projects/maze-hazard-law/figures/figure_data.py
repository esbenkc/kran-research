"""Write the blog's main-figure data (data.json) for the maze-hazard-law post.

Reads analysis.json (analysis.py output) and keeps only what the figure draws. Every number comes from analysis.json:
law.km (Kaplan-Meier), law.pred_curve (the 3-page forecast), law.G, law.km_median, law.pred_median, dead_end. This
script only selects and labels (no recomputation of the frozen rule). Status agents that are not in analysis.json yet
get an empty panel that names their laptop-queue job.

  python3 figures/figure_data.py [--analysis results/analysis.json] [--waiting gpt-oss-20b=JOB] [--out ...]
  python3 figures/figure_data.py --mock      # layout test with made-up numbers, marked NOT COUNTED
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BLOG = PROJ.parents[2] / "blog.kran.ai"

STATUS_ORDER = ["gpt-oss-20b", "Qwen3-8B"]  # analysis.py STATUS_ORDER; the question names them
CAP = 80  # model calls per run (PREREG s.2); the x axis runs to it
# law.verdict (analysis.py) in plain words for the panel head
VERDICT = {"HOLDS": "rule: same chance on every page holds", "FAILS": "rule: the chance changes with depth",
           "no give-up seen": "not tested: no give-ups", "too few give-ups": "not tested: too few give-ups",
           "never quits: set by the cap": "not tested: most runs hit the limit",
           "shape untestable": "not tested: all give-ups at one depth", "pilot too small": "not tested: pages 1 to 3 too few"}


def mock() -> dict:
    """Made-up analysis.json in analysis.py's key shape, for layout only (status agents absent = still queued)."""
    rng = random.Random(0)
    side = {}
    for name, h in [("Qwen3-1.7B", 0.03), ("Gemma-3-1B", 0.06)]:
        n = 40
        depths, gave = [], []
        for _ in range(n):
            d = 1
            while d < CAP and rng.random() > h * (1.6 if d > 30 else 1.0):
                d += 1
            depths.append(d)
            gave.append(d < CAP)
        km, s = [{"depth": 0, "surv": 1.0, "lo": 1.0, "hi": 1.0, "at_risk": n}], 1.0
        for d in range(1, CAP + 1):
            at = sum(1 for x in depths if x >= d)
            ev = sum(1 for x, g in zip(depths, gave) if x == d and g)
            s *= 1 - ev / at if at else 1
            se = s * math.sqrt(max(1e-9, ev / max(1, at * (at - ev or 1))))
            km.append({"depth": d, "surv": s, "lo": max(0.0, s - 1.96 * se), "hi": min(1.0, s + 1.96 * se), "at_risk": at})
        h3 = h * 0.8
        side[name] = {"label": name, "gated": True, "n_entered": n, "dead_end": {"n_entered": 20, "exit_page1": 1.0},
                      "law": {"G": sum(gave), "h3": h3, "km": km, "verdict": "HOLDS", "K": CAP - 1,
                              "km_median": next((p["depth"] for p in km if p["surv"] <= 0.5), None),
                              "pred_median": math.log(0.5) / math.log(1 - h3),
                              "pred_curve": [{"depth": d, "surv": (1 - h3) ** d} for d in range(CAP + 1)]}}
    return {"phase": "mock", "agents": {}, "side_row": side}


def _row(name: str, a: dict, side: bool) -> dict:
    """One figure row from analysis.json agents.<name> or side_row.<name> (analysis.py keys, no recomputation)."""
    label = (name if "(" in name else  # analysis.py SIDE labels may already say what the row is
             f"{name} (closed model, side row)" if side and name.lower().startswith("gpt-5") else
             f"{name} (side row)" if side else name)
    L = a.get("law") or {}
    km = L.get("km") or []
    if km:  # fail loudly if analysis.py's keys moved (the post must never draw a number analysis.json lacks)
        need = ["G", "km_median", "pred_median", "pred_curve", "verdict", "K"]
        miss = [k for k in need if k not in L] + [k for k in ("n_entered", "dead_end") if k not in a]
        assert not miss, f"{name}: analysis.json lacks {miss}"
    else:
        return {"id": name, "label": label, "side": side, "sub": "", "note": f"no maze data: {L.get('verdict') or 'not gated'}"}
    de = a.get("dead_end") or {}
    dn, d1 = de.get("n_entered"), de.get("exit_page1")  # exit_page1 = share of entered dead-end runs that left at page 1
    med, pmed = L.get("km_median"), L.get("pred_median")
    K = L.get("K") or CAP
    pm = None if pmed is None else (f"{pmed:.0f}" if pmed < K - 1e-9 else "past the limit")
    return {
        "id": name,
        "label": label,
        "side": side,
        "sub": f"{a.get('n_entered')} maze runs, {L.get('G')} gave up before the limit"
               + (f" · {VERDICT.get(L.get('verdict'), L.get('verdict'))}" if L.get("verdict") else ""),
        "n": a.get("n_entered"),
        "verdict": L.get("verdict"),
        "km": ([{"depth": 0, "surv": 1.0, "lo": 1.0, "hi": 1.0, "at_risk": a.get("n_entered")}] if km[0]["depth"] > 0 else [])
              + [{q: p[q] for q in ("depth", "surv", "lo", "hi", "at_risk")} for p in km],
        "fit": [{"depth": p["depth"], "surv": p["surv"]} for p in (L.get("pred_curve") or [])],
        "dead": {"n": dn, "left": round(d1 * dn), "surv": 1 - d1} if dn and d1 is not None else None,
        "end_label": (f"median {med:.0f} pages, forecast {pm}" if med is not None and pm else
                      (f"median not reached, forecast {pm}" if pm else "median not reached")),
    }


def build(an: dict, waiting: dict[str, str]) -> dict:
    rows = []
    for name in STATUS_ORDER:
        if name in (an.get("agents") or {}):
            rows.append(_row(name, an["agents"][name], False))
        else:
            job = waiting.get(name)
            rows.append({"id": name, "label": name, "sub": "named in our question",
                         "note": f"runs still in our laptop queue{f' (job {job})' if job else ''}"})
    for name, a in (an.get("side_row") or {}).items():
        if a.get("gated"):
            rows.append(_row(name, a, True))
    return {
        "phase": an.get("phase", "counted"),
        "cap": CAP,
        "title": "How long agents stay in a decoy maze",
        "subtitle": "Share of maze runs still in the maze after each page, with the forecast from the first 3 pages",
        "legend": {
            "km": "measured",
            "fit": "forecast from pages 1 to 3 (same chance to leave on every page)",
            "dead": "dead end (one page)",
            "pilot": "pages 1 to 3",
            "cap": f"{CAP}-call limit",
            "x": "Decoy pages read",
            "y": "Still in the maze",
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(PROJ / "results" / "analysis.json"))
    ap.add_argument("--out", default=str(BLOG / "assets" / "figures" / "maze-hazard-law" / "data.json"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--waiting", action="append", default=[], help="NAME=JOBID for a status agent still in labq")
    args = ap.parse_args()
    an = mock() if args.mock else json.loads(Path(args.analysis).read_text())
    waiting = dict(w.split("=", 1) for w in args.waiting)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(an, waiting), indent=1) + "\n")
    print(f"wrote {out} (phase {an.get('phase', 'counted')}, {len(an.get('agents') or {})} status + "
          f"{len(an.get('side_row') or {})} side agents)")


if __name__ == "__main__":
    main()
