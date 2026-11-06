"""Build the main figure's data.json (blog: assets/figures/shaping-watched/data.json) from the run files.

Reads ONLY (file names from the build owner, room #15367):
  data/runs/<arm>_<trainer>_s<seed>/curve.jsonl  per checkpoint, 64 dev prompts, sampler graph:
                                                 step, cheat_noprobe, cheat_leaky, n_dev
  data/status/<run>.json                         end read, real compiled graph, held-out prompts:
                                                 cheat_noprobe, cheat_leaky, cheat_bitexact, gap, ci_lo, ci_hi, gap_bitexact, n
  data/check/<run>.json                          check probe: auroc, bal_acc, auroc_orig, auroc_null
Every number in data.json is copied from those files; check() re-reads them and compares.

Usage: uv run python figures/figure_data.py --answer "No" --title "..." [--out <blog>/assets/figures/shaping-watched/data.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # projects/shaping-watched
DATA = HERE / "data"
DEFAULT_OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "shaping-watched" / "data.json"

# plain names used in the post (Selma's nouns)
VARIANT = {"A": "Reward-plus-gradient shaping", "B": "Reward-only shaping", "O": "Output penalty"}
SHORT = {"A": "Reward + gradient", "B": "Reward-only", "O": "Output penalty"}
TRAINER = {
    "standard": "Standard trainer: it recomputes the model, so it never sees the probe's footprint",
    "fed": "Trainer fed the sampler's numbers (the footprint reaches the update)",
}
TRAINER_SHORT = {"standard": "Standard trainer (recomputes the model)", "fed": "Trainer fed the sampler's numbers"}
ROW_ORDER = [("standard", "A"), ("standard", "B"), ("fed", "B"), ("fed", "A"), ("fed", "O")]
PANELS = [("standard", "B"), ("fed", "B")]  # top: the same shaping version, the trainer is the one change
assert len({a for _, a in PANELS}) == 1, "top_title names one shaping version"
NO_PENALTY_RUN = "pilot_fed_s0"  # outcome reward only


def parse(run: str) -> tuple[str, str, int]:
    """'B_standard_s1' -> ('B', 'standard', 2): arm letter, trainer, 1-based run number shown to readers."""
    arm, trainer, seed = run.split("_")
    return arm, trainer, int(seed[1:]) + 1


def read_curve(run: str) -> list[dict]:
    rows = [json.loads(x) for x in (DATA / "runs" / run / "curve.jsonl").read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["step"])


def read_json(kind: str, run: str) -> dict | None:
    f = DATA / kind / f"{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def finished_runs() -> list[str]:
    runs = [f.stem for f in (DATA / "status").glob("*.json") if not f.stem.startswith("pilot")]
    key = {k: i for i, k in enumerate(ROW_ORDER)}
    return sorted(runs, key=lambda r: (key[(parse(r)[1], parse(r)[0])], parse(r)[2]))


def build(runs: list[str], answer: str, title: str) -> dict:
    curves = []
    for run in sorted((DATA / "runs").iterdir()):
        if not (run / "curve.jsonl").exists() or run.name.startswith("pilot"):
            continue
        arm, trainer, n = parse(run.name)
        if (trainer, arm) not in PANELS:
            continue
        for r in read_curve(run.name):
            curves.append({"panel": TRAINER_SHORT[trainer], "run": n, "update": r["step"],
                           "with": round(r["cheat_leaky"], 4), "without": round(r["cheat_noprobe"], 4)})
    no_penalty = [{"update": r["step"], "cheating": round(r["cheat_noprobe"], 4)} for r in read_curve(NO_PENALTY_RUN)]
    rows, ns = [], set()
    for run in runs:
        arm, trainer, n = parse(run)
        s, c = read_json("status", run), read_json("check", run)
        ns.add(s["n"])
        rows.append({"group": TRAINER[trainer], "variant": VARIANT[arm], "short": f"{SHORT[arm]} · {n}", "run": n,
                     "gap": round(s["gap"], 4), "low": round(s["ci_lo"], 4), "high": round(s["ci_hi"], 4),
                     "bitexact_gap": round(s["gap_bitexact"], 4),
                     "with": round(s["cheat_leaky"], 4), "without": round(s["cheat_noprobe"], 4),
                     "check_auroc": None if c is None else round(c["auroc"], 2)})
    assert len(ns) == 1, f"end reads use different prompt counts: {ns}"
    return {
        "title": title,
        "subtitle": "Qwen3-0.6B trained by reward on a task it can cheat on. A cheating probe lowers the reward on half of the episodes.",
        "answer": answer,
        "legend": {"with": "with the leaky probe", "without": "without a probe", "none": "no penalty at all",
                   "interval": "gap, 95% interval", "diamond": "same run, bit-exact probe",
                   "bar": "yes bar: 15 points", "band": "no gap: under 5 points", "auroc": "check probe AUROC"},
        "top_title": f"{VARIANT[PANELS[0][1]]}: cheating during training (64 practice prompts)",
        "x_top": "training updates",
        "y_top": "cheating (%)",
        "bottom_title": f"Every run at the end: {ns.pop()} held-out prompts",
        "bottom_note": "Gap = cheating without a probe minus cheating with the leaky probe.",
        "x_bottom": "gap (points)",
        "yes_bar": 0.15,
        "no_band": 0.05,
        "panels": [TRAINER_SHORT[t] for t, a in PANELS],
        "rows": rows,
        "no_penalty": no_penalty,
        "curves": curves,
    }


def check(data: dict, runs: list[str]) -> None:
    """Re-read the sources and compare every drawn number."""
    assert len(data["rows"]) == len(runs)
    for row, run in zip(data["rows"], runs):
        s = read_json("status", run)
        assert abs(row["gap"] - s["gap"]) < 1e-3, run
        assert abs(s["gap"] - (s["cheat_noprobe"] - s["cheat_leaky"])) < 2e-3, f"{run}: gap != without - with"
        assert row["low"] - 1e-4 <= row["gap"] <= row["high"] + 1e-4, run
        assert abs(row["bitexact_gap"] - s["gap_bitexact"]) < 1e-3, run
    for c in data["curves"]:
        assert 0 <= c["with"] <= 1 and 0 <= c["without"] <= 1
    assert data["title"] and data["answer"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--answer", required=True, help="plain answer to the title: Yes / No / Not settled")
    ap.add_argument("--title", required=True, help="figure title that states the result in numbers")
    a = ap.parse_args()
    runs = finished_runs()
    data = build(runs, a.answer, a.title)
    check(data, runs)
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(a.out, len(data["rows"]), "rows", len(data["curves"]), "curve points")
    for r in data["rows"]:
        print(f"  {r['group'][:16]:16s} {r['variant']:28s} run {r['run']}: gap {r['gap']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]"
              f" bit-exact {r['bitexact_gap']:+.3f} with {r['with']:.3f} without {r['without']:.3f} AUROC {r['check_auroc']}")


if __name__ == "__main__":
    main()
