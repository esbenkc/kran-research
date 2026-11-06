"""Build the main figure's data.json (blog: assets/figures/shaping-watched/data.json) from the run files.

Reads ONLY (file names from the build owner, room #15367):
  data/runs/<arm>_<trainer>_s<seed>/curve.jsonl  per checkpoint, 64 dev prompts, sampler graph:
                                                 step, cheat_noprobe, cheat_leaky, n_dev
  data/status/<run>.json                         end read, real compiled graph, held-out prompts:
                                                 cheat_noprobe, cheat_leaky, cheat_bitexact, gap, ci_lo, ci_hi, gap_bitexact, n
  data/check/<run>.json                          check probe: auroc_refit (the retrained check probe), auroc_orig, ...
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

# Plain names used in the post (Selma's nouns). One entry per (arm, update) pair that can appear in data/status/.
# arm: A = reward-plus-gradient shaping, B = reward-only shaping, O = output penalty, P0 = output penalty with the
# earlier post's reward. update: standard = the usual update (recomputes the model), fed = the update is fed the
# served numbers; *pilot = the run started from the pilot model that already cheats (reading-only cells, D4/D5).
G_STD = "Usual update: it recomputes the model, so it never sees the served footprint"
G_FED = "Update fed the served numbers (the footprint reaches the update)"
G_PILOT = "Reading only: started from a model that already cheats 39% of the time"
ROWS = {  # (arm, update) -> (group, row label, short row label for phones); dict order = row order
    ("B", "standard"): (G_STD, "Reward-only shaping", "Reward-only"),
    ("A", "standard"): (G_STD, "Reward-plus-gradient shaping", "Reward + gradient"),
    ("B", "fed"): (G_FED, "Reward-only shaping", "Reward-only"),
    ("O", "fed"): (G_FED, "Output penalty", "Output penalty"),
    ("P0", "fed"): (G_FED, "Output penalty, earlier post's reward", "Output pen., earlier"),
    ("B", "stdpilot"): (G_PILOT, "Reward-only shaping, usual update", "Reward-only, usual"),
    ("B", "fedpilot"): (G_PILOT, "Reward-only shaping, update fed", "Reward-only, fed"),
    ("P0", "fedpilot"): (G_PILOT, "Output penalty (earlier), update fed", "Output pen., fed"),
}
# Top panels: the one-variable pair (same shaping version, only the update changes).
PANELS = [(("B", "standard"), "Usual update (recomputes the model)"), (("B", "fed"), "Update fed the served numbers")]
TOP_TITLE = "Reward-only shaping: cheating during training (64 practice prompts)"


def no_penalty_run() -> str:
    """The outcome-reward-only pilot (no probe penalty); its no-probe read is the dashed line."""
    pilots = sorted(d.name for d in (DATA / "runs").iterdir() if d.name.startswith("pilot_") and (d / "curve.jsonl").exists())
    assert len(pilots) == 1, f"expected one pilot run, found {pilots}"
    return pilots[0]


def parse(run: str) -> tuple[str, str, int]:
    """'B_standard_s1' -> ('B', 'standard', 2): arm letter, trainer, 1-based run number shown to readers."""
    arm, trainer, seed = run.removesuffix(CONT).split("_")
    assert (arm, trainer) in ROWS, f"no plain name for run {run}"
    return arm, trainer, int(seed[1:]) + 1


CONT = "_10k"  # a run continued to 10,000 updates (DEVIATIONS D3) writes <run>_10k next to its 5,000-update parent


def chosen(run: str) -> str:
    """The run whose end read is the status: the continuation once its own end read exists, else the parent.
    Curve and end read always come from the same choice, so the figure never mixes budgets."""
    return run + CONT if (DATA / "status" / f"{run}{CONT}.json").exists() else run


def read_curve(run: str) -> list[dict]:
    rows = [json.loads(x) for x in (DATA / "runs" / run / "curve.jsonl").read_text().splitlines() if x.strip()]
    return sorted(rows, key=lambda r: r["step"])


def read_json(kind: str, run: str) -> dict | None:
    f = DATA / kind / f"{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def finished_runs() -> list[str]:
    runs = sorted({f.stem.removesuffix(CONT) for f in (DATA / "status").glob("*.json") if not f.stem.startswith("pilot")})
    key = {k: i for i, k in enumerate(ROWS)}
    return sorted(runs, key=lambda r: (key[parse(r)[:2]], parse(r)[2]))


def build(runs: list[str], answer: str, title: str) -> dict:
    curves = []
    for run in sorted(d.name for d in (DATA / "runs").iterdir() if (d / "curve.jsonl").exists() and not d.name.endswith(CONT)):
        if run.startswith("pilot"):
            continue
        arm, trainer, n = parse(run)
        panel = dict(PANELS).get((arm, trainer))
        if panel is None:
            continue
        for r in read_curve(chosen(run)):
            curves.append({"panel": panel, "run": n, "update": r["step"],
                           "with": round(r["cheat_leaky"], 4), "without": round(r["cheat_noprobe"], 4)})
    no_penalty = [{"update": r["step"], "cheating": round(r["cheat_noprobe"], 4)} for r in read_curve(no_penalty_run())]
    top = max((c["update"] for c in curves), default=0)
    if no_penalty and no_penalty[-1]["update"] < top:  # the pilot ran fewer updates than the arms: say so in its legend
        none_label = f"no penalty at all (ran {no_penalty[-1]['update']:,} updates)"
    else:
        none_label = "no penalty at all"
    rows, ns = [], set()
    for run in runs:
        arm, trainer, n = parse(run)
        s, c = read_json("status", chosen(run)), read_json("check", chosen(run))
        updates = read_curve(chosen(run))[-1]["step"]
        ns.add(s["n"])
        group, label, short = ROWS[(arm, trainer)]
        rows.append({"group": group, "variant": label, "short": f"{short} · {n}", "run": n,
                     "gap": round(s["gap"], 4), "low": round(s["ci_lo"], 4), "high": round(s["ci_hi"], 4),
                     "bitexact_gap": round(s["gap_bitexact"], 4),
                     "with": round(s["cheat_leaky"], 4), "without": round(s["cheat_noprobe"], 4),
                     "check_auroc": None if c is None else round(c["auroc_refit"], 2), "updates": updates})
    assert len(ns) == 1, f"end reads use different prompt counts: {ns}"
    budgets = sorted({r["updates"] for r in rows})
    long_rows = [r for r in rows if r["updates"] != budgets[0]]
    budget_note = "" if not long_rows else " " + "; ".join(
        f"{r['variant']} ran {r['updates']:,} updates" for r in long_rows) + f"; every other run ran {budgets[0]:,}."
    return {
        "title": title,
        "subtitle": "Qwen3-0.6B is trained by reward. Its first word is \u201cread\u201d (peek at the answer file: cheating) or \u201csolve\u201d. A cheating probe lowers the reward on half of the episodes.",
        "answer": answer,
        "legend": {"with": "leaky probe attached", "without": "no probe", "none": none_label,
                   "interval": "gap, 95% interval", "levels": "leaky / none", "diamond": "same run, bit-exact probe",
                   "bar": "yes bar: 15 points", "band": "no gap: under 5 points", "auroc": "check probe AUROC"},
        "top_title": TOP_TITLE,
        "x_top": "training updates",
        "y_top": "cheating chance (%)",
        "bottom_title": f"Every run at the end: {ns.pop()} held-out prompts",
        "bottom_note": "Gap = cheating chance with no probe minus with the leaky probe. Cheating chance = share of \u201cread\u201d among \u201cread\u201d or \u201csolve\u201d. Right column: the two chances, leaky probe / no probe." + budget_note,
        "x_bottom": "gap (points)",
        "yes_bar": 0.15,
        "no_band": 0.05,
        "panels": [name for _, name in PANELS],
        "rows": rows,
        "no_penalty": no_penalty,
        "curves": curves,
    }


def check(data: dict, runs: list[str]) -> None:
    """Re-read the sources and compare every drawn number."""
    assert len(data["rows"]) == len(runs)
    for row, run in zip(data["rows"], runs):
        s = read_json("status", chosen(run))
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
    ap.add_argument("--final", action="store_true", help="refuse unfinished continuations (a _10k run with no end read)")
    a = ap.parse_args()
    if a.final:
        open_ = [d.name for d in (DATA / "runs").glob(f"*{CONT}") if not (DATA / "status" / f"{d.name}.json").exists()]
        assert not open_, f"continued runs without an end read: {open_}"
    runs = finished_runs()
    data = build(runs, a.answer, a.title)
    check(data, runs)
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(a.out, len(data["rows"]), "rows", len(data["curves"]), "curve points")
    for r in data["rows"]:
        print(f"  {r['group'][:16]:16s} {r['variant']:28s} run {r['run']} ({r['updates']:,} updates): gap {r['gap']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]"
              f" bit-exact {r['bitexact_gap']:+.3f} with {r['with']:.3f} without {r['without']:.3f} AUROC {r['check_auroc']}")


if __name__ == "__main__":
    main()
