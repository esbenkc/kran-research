"""Build the post's main figure data (athanorscience.com assets/figures/tell-chain-8b/data.json).

What the figure shows: for each tell in the chain (2nd = cue B, 3rd = cue C, 4th = cue D), the ratio of the
updates the chain tail needed to learn that cue to the updates of (a) the same-update coin-flip control and
(b) a fresh tail, one dot per seed, with the geometric mean and its seed-level 95% interval. Next to the dots:
the two models frozen in PREREG.md before the run, from the parent's single measured factor 0.74:
multiplying (0.74, 0.74^2 = 0.55, 0.74^3 = 0.41) and flat (0.74 at every tell), and the frozen bar 0.55 at the
4th tell.

Reads ONLY the frozen analysis outputs: data/analysis.json (stages C and D, Bruno's analyze.py) and
ref/parent_analysis.json (stage B, the parent run second-tell-8b, k4: 'coin' = the no-cue coin-flip control).
It recomputes the geometric mean and the t-interval from the per-seed switch times and asserts they match the
analysis numbers, so the figure cannot drift from the status.

usage: python figures/build_main.py [--analysis data/analysis.json] [--parent ref/parent_analysis.json]
                                     [--out <blog>/assets/figures/tell-chain-8b/data.json] [--synthetic flat|mult]
"""
import argparse
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
BLOG = PROJ.parents[2] / "athanorscience.com"
FIG = "tell-chain-8b"

R1 = 0.74  # the parent's single measured factor (PREREG: frozen input of both models)
MODELS = {"mult": [R1, R1**2, R1**3], "flat": [R1, R1, R1]}
BAR = 0.55  # frozen bar at the 4th tell
FLOOR = 8  # switch times are floored at 8 updates (parent rule)
T975 = {2: 12.7062047, 3: 4.3026527, 4: 3.1824463, 5: 2.7764451, 6: 2.5705818, 7: 2.4469119, 8: 2.3646243, 9: 2.3060041}  # t quantiles, df n-1

STAGES = [
    ("B", "2nd cue", "B"),
    ("C", "3rd cue", "C"),
    ("D", "4th cue", "D"),
]
REFS = [
    ("ctrl", "Left of each pair: vs coin-flip controls (same updates, no switch to learn)", "vs coin-flip controls"),
    ("fresh", "Right of each pair: vs fresh tails (no earlier training)", "vs fresh tails"),
]


def tint(lnr):
    """Geometric mean and t-interval over seeds of the paired log ratios (df n-1)."""
    n = len(lnr)
    m = sum(lnr) / n
    if n < 2:
        return math.exp(m), None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in lnr) / (n - 1))
    h = T975[n] * sd / math.sqrt(n)
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def close(a, b, tol=1e-6):
    return a is None and b is None or (a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b)))


def parent_stage_B(par):
    """Stage B from the parent: per-seed ratios of the cue-A tail to the no-cue coin flips ('coin') and fresh tails."""
    rat = par["k4"]["ratios"]
    out = {}
    for ref, key in (("ctrl", "coin"), ("fresh", "fresh")):
        ps = {int(s): v for s, v in rat[key]["per_seed"].items()}
        g, lo, hi = tint([math.log(v) for v in ps.values()])
        assert close(g, rat[key]["R"], 1e-9) and close(lo, rat[key]["R_lo"], 1e-9) and close(hi, rat[key]["R_hi"], 1e-9), (ref, g, rat[key])
        out[ref] = {"per_seed": ps, "cens": {s: None for s in ps}, "geo": g, "lo": lo, "hi": hi}
    return out


def stage_from_T(an, stage):
    """Ratios for stage C or D recomputed from per-seed switch times, checked against the analysis ratios."""
    T, cens = an["T"], an.get("censored", {})
    main = f"main_{stage}"
    out = {}
    for ref in ("ctrl", "fresh"):
        arm = f"{ref}_{stage}"
        seeds = sorted(set(int(s) for s in T.get(main, {})) & set(int(s) for s in T.get(arm, {})))
        ps, cf = {}, {}
        for s in seeds:
            tm, tr = T[main][str(s)], T[arm][str(s)]
            if tm is None or tr is None:
                continue
            ps[s] = max(tm, FLOOR) / max(tr, FLOOR)
            cm, cr = bool(cens.get(main, {}).get(str(s))), bool(cens.get(arm, {}).get(str(s)))
            cf[s] = "main" if cm and not cr else "ref" if cr and not cm else "both" if cm and cr else None
        if not ps:
            continue
        g, lo, hi = tint([math.log(v) for v in ps.values()])
        got = an.get("ratios", {}).get(stage, {}).get(ref)
        if got:
            assert close(g, got["geo"]) and close(lo, got["lo"]) and close(hi, got["hi"]), (stage, ref, g, lo, hi, got)
        out[ref] = {"per_seed": ps, "cens": cf, "geo": g, "lo": lo, "hi": hi}
    return out


def synthetic(model, seed=0):
    """A fake analysis.json in the schema asked of analyze.py, for testing the figure before data exist."""
    rng = random.Random(seed)
    seeds = [5, 6, 7, 8, 9]
    T, cens = {}, {}
    for i, st in enumerate(("C", "D")):
        target = MODELS[model][i + 1]
        for arm in ("ctrl", "fresh"):
            T[f"{arm}_{st}"] = {str(s): 8 * round(rng.gauss(290, 15) / 8) for s in seeds}
        T[f"main_{st}"] = {str(s): 8 * max(1, round(T[f"ctrl_{st}"][str(s)] * target * math.exp(rng.gauss(0, 0.08)) / 8)) for s in seeds}
    if model == "flat":  # exercise the censored path once
        cens["ctrl_D"] = {"9": True}
    return {"seeds": seeds, "T": T, "censored": cens, "ratios": {}, "synthetic": model}


def build(an, par):
    stages = {"B": parent_stage_B(par)}
    for st in ("C", "D"):
        s = stage_from_T(an, st)
        if s:
            stages[st] = s
    points, summary = [], []
    for i, (st, lab, cue) in enumerate(STAGES):
        for ref, _, _ in REFS:
            v = stages.get(st, {}).get(ref)
            if not v:
                continue
            for s, r in sorted(v["per_seed"].items()):
                points.append({"x": i, "stage": st, "ref": ref, "seed": s, "R": round(r, 4), "cens": v["cens"].get(s)})
            summary.append({"x": i, "stage": st, "ref": ref, "geo": round(v["geo"], 4),
                            "lo": None if v["lo"] is None else round(v["lo"], 4),
                            "hi": None if v["hi"] is None else round(v["hi"], 4), "n": len(v["per_seed"])})
    allR = [p["R"] for p in points] + [x for s in summary for x in (s["lo"], s["hi"]) if x] + MODELS["mult"]
    ylo, yhi = min(0.3, min(allR) * 0.92), max(1.15, max(allR) * 1.08)
    d = {
        "title": "The 3rd and 4th cues cost only 0.2 to 0.3 of the comparison tails' updates, below both pre-run predictions",
        "subtitle": ("A tail is Qwen3-8B's last two layers, the only part we train. Each dot is one seed: the RL updates the "
                     "main tail needed to learn the switch on a new cue, divided by a comparison tail's updates (1 = no saving). "
                     "Lines: the two outcomes we wrote down before the run, from the one factor (0.74) our last run measured."),
        "y_label": "Share of comparison tail's updates (log, lower = cheaper)",
        "y_short": "Share of comparison updates (log)",
        "stages": [{"x": i, "key": st, "label": f"{lab} ({cue})", "short": lab, "note": "last run" if st == "B" else ""}
                   for i, (st, lab, cue) in enumerate(STAGES)],
        "refs": [{"key": k, "label": lab, "short": sh} for k, lab, sh in REFS],
        "models": {"mult": [round(x, 4) for x in MODELS["mult"]], "flat": [round(x, 4) for x in MODELS["flat"]]},
        "bar": {"x": 2, "R": BAR},
        "legend": {
            "mult": "Multiplying saving: each cue cuts the next by 0.74 (0.74, 0.55, 0.41)",
            "flat": "Flat, one-off saving: 0.74 at every cue",
            "bar": "Pre-set test: the answer is yes if the 4th cue falls below 0.55 against both comparison tails",
            "geo": "Geometric mean, seed-level 95% interval",
            "last": "Hollow: 2nd cue, from our last run",
        },
        "points": points,
        **({"legend_cens": "Cross: a run hit the update cap"} if any(p["cens"] for p in points) else {}),
        "summary": summary,
        "ylo": round(ylo, 3),
        "yhi": round(yhi, 3),
        "yticks": [t for t in (0.15, 0.2, 0.3, 0.4, 0.55, 0.74, 1.0) if ylo <= t <= yhi],
    }
    if an.get("synthetic"):
        d["title"] = f"SYNTHETIC ({an['synthetic']}): " + d["title"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=str(PROJ / "data" / "analysis.json"))
    ap.add_argument("--parent", default=str(PROJ / "ref" / "parent_analysis.json"))
    ap.add_argument("--out", default=str(BLOG / "assets" / "figures" / FIG / "data.json"))
    ap.add_argument("--synthetic", choices=["flat", "mult"])
    a = ap.parse_args()
    par = json.loads(Path(a.parent).read_text())
    an = synthetic(a.synthetic) if a.synthetic else json.loads(Path(a.analysis).read_text())
    d = build(an, par)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=1) + "\n")
    for s in d["summary"]:
        print(s)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
