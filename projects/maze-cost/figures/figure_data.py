"""Write the blog's main-figure data (data.json) for the maze-cost post.

Reads analysis.json (Bruno's analysis.py output) and keeps only what the figure draws. Every number in the figure
comes from analysis.json; this script only selects and labels (no recomputation of the status rule).

  uv run python figures/figure_data.py [--analysis out/analysis.json] [--out <blog>/assets/figures/maze-cost/data.json]
  uv run python figures/figure_data.py --mock      # layout test with made-up numbers, marked NOT COUNTED
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
CAP = 12  # maze cap K in pages after entry (PREREG section 1)

OUTCOME_TEXT = {
    "quits inside the pilot": "not tested, because half of its runs left within the first 3 pages",
    "never quits: spend set by the cap": "not tested, because more than half of its runs were still in the maze at the call limit",
}


def mock(never: bool = False) -> dict:
    """Made-up analysis.json in the agreed key shape, for layout only. never=True: agents almost never leave."""
    rng = random.Random(0)
    agents = {}
    hs = [0.0, 0.01] if never else [0.12, 0.2]
    for (aid, label), h in zip([("gemma-4-31b-it", "Gemma 4 31B"), ("gemma-4-26b-a4b-it", "Gemma 4 26B-A4B")], hs):
        n = 42
        depths, left = [], []
        for _ in range(n):
            d = 1
            while d < CAP and rng.random() > h:
                d += 1
            depths.append(d)
            left.append(d < CAP or rng.random() < h)
        km, s = [{"depth": 0, "surv": 1.0, "lo": 1.0, "hi": 1.0, "at_risk": n}], 1.0
        for d in range(1, CAP + 1):
            at = sum(1 for x in depths if x >= d)
            ev = sum(1 for x, l in zip(depths, left) if x == d and l)
            s *= 1 - ev / at if at else 1
            se = s * math.sqrt(max(1e-9, ev / max(1, at * (at - ev or 1))))
            km.append({"depth": d, "surv": s, "lo": max(0.0, s - 1.96 * se), "hi": min(1.0, s + 1.96 * se), "at_risk": at})
        hh = h * (0.9 + 0.2 * rng.random())
        d_star = math.ceil(math.log(0.5) / math.log(1 - hh)) if hh > 0 else None
        a, b = 1400.0, 900.0
        P = sum(a + b * j for j in range(1, d_star + 1)) if d_star else None
        R = 0.8 + 0.5 * rng.random()
        dec = sum(min(x, CAP - 1) for x in depths)
        lv = sum(left)
        agents[aid] = {
            "label": label, "scored": True, "gate": {"k": 9, "n": 10}, "n_runs": 44, "n_entered": n,
            "outcome": "never quits: spend set by the cap" if never else "TESTED", "h": hh, "d_star": d_star,
            "a": a, "b": b, "P": P, "O": P / R if P else None, "R": R,
            "R_ci": [R * 0.7, R * 1.4], "R_seed": [R * 0.5, R * 1.9], "R_blocks": [R * 0.9, R, R * 1.1], "km": km,
            "leaves_all": lv, "decisions_all": dec, "leave_rate_upper95": clopper_pearson(lv, dec, 0.1)[1],
            "median_depth_lower95": None, "runs_leaving": lv, "run_leave_upper95": clopper_pearson(lv, n, 0.1)[1],
            "K": 10, "constancy": {"L_A": min(lv, 3), "E_A": min(dec, 120), "L_B": max(0, lv - 3), "E_B": max(0, dec - 120)},
            "lr_p": 0.4, "share_depth4": 0.6, "ratio_usd_median": 900.0, "ratio_usd_band": [120.0, 9000.0],
            "dead_end": {"n_entered": 40, "median_spend": 2100, "ratio_usd_median": 350.0},
            "maze_over_dead_end_tokens": 9.5,
        }
    return {"phase": "mock", "status": "UNCONFIRMED", "prereg_outcome": "failed", "agents": agents}


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    term, total = (1 - p) ** n, 0.0
    for i in range(k + 1):
        total += term
        term *= (n - i) / (i + 1) * p / (1 - p)
    return min(1.0, total)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided 95% interval for a binomial rate (bisection on the binomial CDF)."""
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def build(an: dict) -> dict:
    rows = []
    status_agents = set(an.get("status_agents") or [])
    for aid, a in an["agents"].items():
        if not (a.get("scored") or aid in status_agents):
            continue
        tested = a.get("outcome") == "TESTED"
        h = a.get("h")
        # reported rows added after maze run 000 was seen (DEVIATIONS D5; analysis.py keys, not recomputed here):
        # leaves over leave-or-go-on decisions at every maze depth with a one-sided Clopper-Pearson 95% upper bound,
        # and the run-level row (share of entered maze runs that leave within their budget)
        k_all, n_all = a.get("leaves_all"), a.get("decisions_all")
        K = a.get("K")
        # dead-end arm: share of entered runs still behind the hidden link after reading its one page
        de = [r for r in (a.get("runs") or {}).get("deadend", []) if r.get("entered")]
        dead = ([{"depth": 0, "surv": 1.0}, {"depth": 1, "surv": sum(1 for r in de if not r["leave"]) / len(de)}]
                if de else [])
        rows.append({
            "id": aid,
            "label": a["label"],
            "n": a["n_entered"],
            "km": [{k: p[k] for k in ("depth", "surv", "lo", "hi", "at_risk")} for p in a["km"]],
            # the pilot fit drawn as a line: share still in the maze after d pages = (1 - h)^d
            "fit": [{"depth": d, "surv": (1 - h) ** d} for d in range(0, CAP + 1)] if h is not None else [],
            "dead": dead,
            # exploratory cell (DEVIATIONS D10, reported only): the same maze without the sentence that promises the
            # figures in a later part
            "neutral": ({"km": [{k: p[k] for k in ("depth", "surv", "lo", "hi", "at_risk")} for p in nz["km"]],
                         "n": nz.get("n_entered"), "runs_leaving": nz.get("runs_leaving"),
                         "rate": {"k": nz.get("leaves_all"), "n": nz.get("decisions_all"),
                                  "est": (nz["leaves_all"] / nz["decisions_all"]) if nz.get("decisions_all") else None,
                                  "hi": nz.get("leave_rate_upper95")}}
                        if (nz := a.get("maze_neutral")) and nz.get("km") else None),
            "dead_n": len(de),
            "dead_left": sum(1 for r in de if r["leave"]),
            "h": h,
            "d_star": a.get("d_star"),
            "rate": {"k": k_all, "n": n_all, "est": k_all / n_all if n_all else None,
                     "hi": a.get("leave_rate_upper95"), "depth_lb": a.get("median_depth_lower95")},
            "runs_leaving": {"k": a.get("runs_leaving"), "n": a["n_entered"], "hi": a.get("run_leave_upper95")},
            # the frozen TESTED window in give-up rate: d* > 3 and (1 - h)^K < 0.5
            "window": [1 - 0.5 ** (1 / K), 1 - 0.5 ** (1 / 3)] if K else None,
            "K": K,
            "tested": tested,
            "note": None if tested else OUTCOME_TEXT.get(a.get("outcome"), a.get("outcome")),
            "R": a.get("R") if tested else None,
            "R_ci": a.get("R_ci") if tested else None,
            "R_seed": a.get("R_seed") if tested else None,
            "P": a.get("P"),
            "O": a.get("O"),
        })
    return {
        "phase": an.get("phase", "counted"),
        "cap": CAP,
        "title": "Agents in a decoy maze, page by page",
        "subtitle": ("Share of runs still behind the hidden link, and how well a 3-page pilot predicted their spend"
                     if any(r["tested"] for r in rows) else
                     "Share of runs still behind the hidden link, and how often agents chose to leave"),
        "legend": {
            "km": "decoy maze",
            "dead": "dead end (one page)",
            "neutral": "decoy maze without the promise",
            "fit": "prediction from pages 1 to 3 (same rate on every page)",
            "pilot": "pilot: pages 1 to 3",
            "half": "half have left",
            "x": "Decoy pages read",
            "y": "Still behind the hidden link",
            "rate_head": "Give-up rate per page",
            "rate_x": "Share of leave-or-go-on choices that ended in a leave",
            "window": "range our test can decide",
            "r_head": "Predicted median spend ÷ measured median spend",
            "band": "within 2x: Yes zone",
            "run_ci": "95% interval over runs",
            "seed_ci": "seed-level interval (3 seed blocks)",
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(PROJ / "results" / "analysis.json"))
    ap.add_argument("--out", default=str(BLOG / "assets" / "figures" / "maze-cost" / "data.json"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-never", action="store_true")
    args = ap.parse_args()
    an = mock(args.mock_never) if (args.mock or args.mock_never) else json.loads(Path(args.analysis).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(an), indent=1) + "\n")
    print(f"wrote {out} (phase {an.get('phase')}, {len(an['agents'])} agents)")


if __name__ == "__main__":
    main()
