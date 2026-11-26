"""Build the post's main figure data (athanorscience.com assets/figures/second-tell-8b/data.json).

Reads ONLY the frozen analysis output (data/analysis.json, written by Bruno's analysis script) and the
parent's law. It recomputes nothing that enters the status: the dots are the analysis T values, the
"twice as fast" bar is 0.5 x the smallest reference geometric mean (the yes bar of the frozen rule:
MAIN below half of EVERY reference), and the law line is the parent's fitted law at cue B's size.

usage: python figures/build_main.py [--analysis data/analysis.json] [--kB 4] [--out <blog>/assets/figures/second-tell-8b/data.json] [--check]
"""
import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BLOG = PROJ.parents[2] / "athanorscience.com"

# Parent law (cue-size-law-8b-rep, seeds 5-7): T at share 0.342% (k4) = 296 / 280 / 304, slope -0.956 per ln share.
PARENT_K4 = [296, 280, 304]
PARENT_SLOPE = -0.956
S0 = 5.469  # share in % at k = 0

ARMS = [
    ("fresh", "Fresh tail (no stage 1)", "Fresh"),
    ("coin", "Coin flip, no cue (256 updates)", "Coin flip, no cue"),
    ("shuf", "Coin flip, cue A shown (256 updates)", "Coin flip, cue A"),
    ("main", "Learned cue A first (256 updates)", "Learned cue A"),
]


def geomean(xs):
    return math.exp(sum(math.log(max(x, 8)) for x in xs) / len(xs))


def law_T(kB):
    """Parent law: geometric mean of the parent's k4 seeds, moved along the slope to share S0/2^kB."""
    t4 = geomean(PARENT_K4)
    return t4 * (2.0 ** (kB - 4)) ** (-PARENT_SLOPE) if kB != 4 else t4


def load_runs(an, kB):
    """Normalise the analysis runs to {arm, seed, T, cens}. T is stage-2 updates to the cue-B switch."""
    out = []
    for r in an["runs"]:
        if int(r.get("kB", kB)) != kB:
            continue
        arm = r["arm"]
        out.append({"arm": arm, "seed": int(r["seed"]), "T": None if r.get("censored") else r["T"], "cens": bool(r.get("censored"))})
    return out


def build(an, kB):
    runs = load_runs(an, kB)
    cap = int(an.get("cap", 1536))
    keys = [a[0] for a in ARMS]
    for k in keys:
        assert any(r["arm"] == k for r in runs), f"no runs for arm {k} at kB={kB}"
    seen = {(r["arm"], r["seed"]) for r in runs}
    assert len(seen) == len(runs), "duplicate (arm, seed) cell"
    geo = {}
    for k in keys:
        ts = [cap if r["cens"] else r["T"] for r in runs if r["arm"] == k]
        geo[k] = round(geomean(ts), 1)
    ref_min = min(geo[k] for k in ("fresh", "coin", "shuf"))
    half = 0.5 * ref_min
    law = law_T(kB)
    seeds = sorted({r["seed"] for r in runs})
    vals = [cap if r["cens"] else max(r["T"], 8) for r in runs] + [law, half]
    zero = any((not r["cens"]) and r["T"] == 0 for r in runs)
    ylo = 10 ** math.floor(math.log10(min(vals) / 1.3))
    if zero:  # T = 0 (switched before any cue-B update) sits on its own row at the bottom, labelled 0
        ylo = min(ylo, 5)
    yhi = max(vals) * (2.2 if any(r["cens"] for r in runs) else 1.25)
    yticks = [t for t in [1, 2, 10, 20, 50, 100, 150, 200, 300, 400, 600, 1000, 2000] if ylo <= t <= yhi]
    share = S0 / 2 ** kB
    n = {k: sum(1 for r in runs if r["arm"] == k) for k in keys}
    return {
        "title": "PENDING",
        "subtitle": "PENDING",
        "kB": kB,
        "share": round(share, 3),
        "arms": [{"key": k, "label": l, "short": s, "n": n[k]} for k, l, s in ARMS],
        "runs": runs,
        "geo": geo,
        "cap": cap,
        "law": {"T": round(law, 1)},
        "half": {"T": round(half, 1), "ref": min(("fresh", "coin", "shuf"), key=lambda k: geo[k])},
        "ratio_fresh": an.get("k4", {}).get("ratios", {}).get("fresh", {}).get("R") if kB == 4 else None,
        "zero": zero,
        "seed_mid": (seeds[0] + seeds[-1]) / 2,
        "ylo": ylo,
        "yhi": round(yhi),
        "yticks": yticks,
        "y_label": "Updates until the tail learned the switch on cue B",
        "y_short": "Updates to learn cue B",
        "legend": {
            "dot": "one training seed",
            "geo": "group mean (geometric)",
            "law": "cue-size law for a fresh tail (last post)",
            "half": "half the updates of the fastest comparison group",
        },
        "notes": {
            "law": f"law: {round(law):,} updates",
            "half": f"half: {round(half):,} updates",
            "zero": "0: switched before any cue-B update",
            "main": "{r:.2f} × the fresh tails",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(PROJ / "data" / "analysis.json"))
    ap.add_argument("--kB", type=int, default=4)
    ap.add_argument("--out", default=str(BLOG / "assets" / "figures" / "second-tell-8b" / "data.json"))
    ap.add_argument("--check", action="store_true", help="print the numbers, write nothing")
    a = ap.parse_args()
    an = json.load(open(a.analysis))
    d = build(an, a.kB)
    old = {}
    if Path(a.out).exists():
        old = json.load(open(a.out))
    # Text owned by the post (title, subtitle) survives a rebuild unless it still says PENDING.
    for k in ("title", "subtitle"):
        if old.get(k) and not str(old[k]).startswith("PENDING"):
            d[k] = old[k]
    print(json.dumps({"geo": d["geo"], "law": d["law"], "half": d["half"], "n": {x["key"]: x["n"] for x in d["arms"]}}, indent=1))
    if not a.check:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(d, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
