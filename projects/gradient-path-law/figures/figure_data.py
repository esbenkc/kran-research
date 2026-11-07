"""Build the main figure's data.json (blog: assets/figures/gradient-path-law/data.json) from the run files.

Reads only Bruno's run outputs (contract #15680):
  data/status/<run>.json   end read on the served build, 200 held-out puzzles: cheat_noprobe, cheat_leaky (probe
                           attached), gap (= noprobe - leaky, a share), ci_lo, ci_hi; carry.py adds trainer_gap,
                           trainer_gap_logit, served_gap_logit, carry, carry_ci [lo, hi] (Bruno #15764)
  data/runs/<run>/curve.jsonl  practice puzzles every 250 updates: step, gap (served cue), gap_trainer (the
                           update's own no-probe vs probed input), gap_trainer_logit
A run continued past 5,000 updates writes <run>_10k next to its parent (as in the parent project); the
continuation is used once its own status exists, and curve and end read always come from the same choice.
Draws only runs whose status file exists. check() re-reads every source and compares every drawn number.

Usage: python figures/figure_data.py [--out PATH] [--answer TEXT]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # projects/gradient-path-law
DATA = HERE / "data"
DEFAULT_OUT = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures" / "gradient-path-law" / "data.json"
CONT = "_10k"

# Colours (plot.js): end to end pink, importance weights orange (a hue a reader tells apart from pink on a phone,
# Kasper #15846), probed rerun muted (Kasper #15808), reused-numbers teal,
# clean rerun muted; the group headers also name every row (Kasper #15719).
# Plain names from the post (Selma #15658). Group order = drawing order. Each group lists the run-name prefixes
# that belong to it (run = <prefix>_s<seed>). Only tonight's runs are drawn; the earlier night's pair is in the text.
# The build-mismatch control (Ta) explains the collapse, not the title: text only (Tove #15879).
GROUPS = [  # D4 (Felix #15771, Tove #15773): the end-to-end training build carries the title; names from Selma #15776
    {"id": "endtoend", "name": "Training build end to end",
     "note": ("The training build writes the answers and computes the update. The served build is used only for "
              "the test."),
     "prefixes": ["Tc"]},
    {"id": "tis", "name": "Probed rerun with importance weights",
     "note": ("The served build writes the answers. The update reruns them with the probe attached, on a training "
              "build set up to train from the probe's layer on. Each answer is reweighted by how much the two builds "
              "disagree on it (capped at 2)."),
     "prefixes": ["Tb"]},
    {"id": "probed", "name": "Probed rerun, served answers",
     "note": "The same rerun without the weights.",
     "prefixes": ["T"]},
    {"id": "reused", "name": "Reused-numbers update",
     "note": ("The update reuses the served numbers. A rerun that trains only layers after the probe gets these "
              "same numbers, bit for bit."),
     "prefixes": ["fed"]},
    {"id": "clean", "name": "Clean rerun",
     "note": "The update reruns the model with no probe, so it cannot learn a switch.",
     "prefixes": ["rec"]},
]
# Three carry predictions for the probed rerun, frozen in PREREG.md (v1 59c47d4; P1 in carry units from v1.1 fcf3a4c) before any
# probed-rerun update; check() re-reads them from PREREG.md. Plain names named by what each assumes; the post names no single law (Tove #15760, Kasper #15752).
PREDICTIONS = [
    {"id": "P1", "name": "Overlap prediction: how much the training-build change overlaps the served one", "short": "overlap rule", "lo": 0.036, "hi": 0.053,
     "band": [0.02, 0.10], "src": r"P1 in carry units = 0\.045-0\.059 x 0\.80-0\.89 = 0\.036-0\.053; P1's band is 0\.02-0\.10\."},
    {"id": "P2", "name": "Chat-start push prediction: how the update's push on the first chat words overlaps", "short": "chat-start push", "lo": 0.43, "hi": 0.43,
     "src": r"positions 1-7: 0\.43\."},
    {"id": "P3", "name": "All-words push prediction: how the update's push on every word overlaps", "short": "all-words push", "lo": -0.22, "hi": -0.22,
     "src": r"on all positions: -0\.22 "},
]
TITLE = "Does a probe-off switch reach the served model only when the update sees the served numbers?"
SUBTITLE = ("Qwen3-0.6B on Countdown puzzles. The first word is \u201cread\u201d (cheat) or \u201csolve\u201d. "
            "A probe on half the training episodes lowers the reward for cheating.")
LEGEND = {
    "trainer": "Switch on the training build: the numbers the update trained on",
    "served": "Switch on the served model: 200 new puzzles, 95% interval",
    "bars": "Dashed line: the 15-point bar a switch had to reach on the training build",
    "carry_title": "How much of the switch reached the served model",
    "carry_note": ("Runs with the probe in the update, against three predictions written down before training (hollow diamonds), and the "
                   "reused-numbers runs for comparison. Measured on the model's read-vs-solve preference, 95% "
                   "interval. Shaded band: within 15% either way, the switch did not reach the served model."),
    "carry_axis": "share of the training-build switch seen on the served model",
    "carry_axis_short": "share seen on the served model",
    "carry_band": "did not\nreach",
    "carry_ref": "Reused numbers, every run",
    "carry_none": "no switch learned, so nothing to carry",
    "carry_nudge": ("small hollow dots: what a random nudge of the same size on the chat-start words would carry "
                    "(measured on the last two layers alone)"),
    "short": {"endtoend": "End to end", "probed": "Served answers", "tis": "Importance weights"},
    "short_narrow": {"endtoend": "End to end", "probed": "No weights", "tis": "With weights"},
    "carry_ref_short": "Reused, every run",
    "carry": ("Right column: the share of the training-build switch that reached the served model, measured on "
              "the model's read-vs-solve preference, with its 95% interval. A whole interval under 15% means the "
              "switch did not reach it."),
}


def parse(run: str) -> tuple[str, int]:
    """'T_s1' -> ('T', 2); 'B_fed14pilot_s0' -> ('B_fed14pilot', 1): prefix and 1-based run number for readers."""
    prefix, seed = run.rsplit("_s", 1)
    return prefix, int(seed) + 1


def chosen(run: str) -> str:
    return run + CONT if (DATA / "status" / f"{run}{CONT}.json").exists() else run


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def read_curve(run: str) -> list[dict]:
    f = DATA / "runs" / run / "curve.jsonl"
    if not f.exists():
        return []
    return sorted((json.loads(x) for x in f.read_text().splitlines() if x.strip()), key=lambda r: r["step"])


def finished_runs() -> list[str]:
    stems = {f.stem.removesuffix(CONT) for f in (DATA / "status").glob("*.json")}
    return sorted(s for s in stems if "_s" in s and any(parse(s)[0] in g["prefixes"] for g in GROUPS))


def row(run: str) -> dict:
    use = chosen(run)
    st = read_json(DATA / "status" / f"{use}.json")
    curve = read_curve(use)
    last = next((c for c in reversed(curve) if c.get("gap_trainer") is not None), None)
    prefix, _ = parse(run)
    return {
        "run": use, "label": None, "updates": curve[-1]["step"] if curve else None,
        "earlier_night": prefix.startswith("B_"),
        "served": {"gap": st["gap"], "lo": st["ci_lo"], "hi": st["ci_hi"],
                   "probe_on": st["cheat_leaky"], "no_probe": st["cheat_noprobe"],
                   "gap_logit": st.get("gap_logit"), "ci_logit": st.get("ci_logit")},
        # the update's own switch: the end read on the 200 status puzzles when status.py writes it, else the
        # last practice-puzzle point of the curve
        "trainer": ({"gap": st["trainer_gap"], "gap_logit": st.get("trainer_gap_logit"), "step": None, "source": "status"}
                    if st.get("trainer_gap") is not None else
                    None if last is None else {"gap": last["gap_trainer"], "gap_logit": last.get("gap_trainer_logit"),
                                               "step": last["step"], "source": "curve"}),
        "carry": None if st.get("carry") is None else {"value": st["carry"], "lo": st["carry_ci"][0], "hi": st["carry_ci"][1]},
        "curve": [{"step": c["step"], "served": c.get("gap"), "trainer": c.get("gap_trainer")} for c in curve],
    }


TAKEOFF = 0.15  # PREREG status rule: a run took off when its switch on the training build is >= 15 points
DEAD = 0.01     # cheating under 1% with and without the probe: the run stopped cheating altogether


def annotate(group_id: str, r: dict) -> None:
    """Plain notes the figure prints next to a row, and whether its carry may be drawn. The carry is a ratio of
    two switches; with no switch on the training build (no takeoff) the ratio means nothing and is not drawn."""
    s, t = r["served"], r["trainer"]
    dead = s["no_probe"] < DEAD and s["probe_on"] < DEAD
    took_off = t is not None and abs(t["gap"]) >= TAKEOFF
    r["took_off"] = took_off
    r["note"] = ("stopped cheating: 0% with and without the probe" if dead else
                 "no switch on the training build" if group_id in ("endtoend", "probed", "tis", "reused") and not took_off
                 else None)
    if r["carry"] is not None and not took_off:
        r["carry_withheld"] = r.pop("carry")  # kept for the record, never drawn
        r["carry"] = None


def nudges(run: str) -> list[float] | None:
    """Random-nudge null (Vera #15784, Bruno #15797): the carry that each random constant offset on the chat-start
    words, at the served cue's size, would give on this run's adapter. Only the end-to-end rows use it."""
    f = DATA / "randprefix.json"
    if not f.exists():
        return None
    rp = json.loads(f.read_text())
    runs = rp.get("runs", rp)
    rec = runs.get(run) if isinstance(runs, dict) else next((x for x in runs if x.get("run") == run), None)
    if not rec:
        return None
    ce = rec["carry_eq"]
    vals = [v for k, v in sorted(ce.items()) if k.startswith("rand_normS")]
    return vals or None


def build(runs: list[str], answer: str) -> dict:
    groups = []
    for g in GROUPS:
        rows = [row(r) for r in runs if parse(r)[0] in g["prefixes"]]
        for i, r in enumerate(rows):  # readers count runs inside each group (Selma #15694); the seed stays in "run"
            r["label"] = f"run {i + 1}"
            annotate(g["id"], r)
            if g["id"] == "endtoend" and r["carry"]:
                r["nudge"] = nudges(r["run"])
        if rows:
            groups.append({"id": g["id"], "name": g["name"], "note": g["note"], "rows": rows})
    # legend and plain words BEFORE the cells (the figure_alone judge reads the head of data.json)
    preds = [{k: p[k] for k in ("id", "name", "short", "lo", "hi", "band") if k in p} for p in PREDICTIONS]
    return {"title": TITLE, "subtitle": SUBTITLE, "answer": answer, "legend": LEGEND, "predictions": preds,
            "bars": {"switch_at": 0.15, "carry_silent_below": 0.15}, "groups": groups}


def check(data: dict) -> None:
    n = 0
    for g in data["groups"]:
        for r in g["rows"]:
            st = read_json(DATA / "status" / f"{r['run']}.json")
            s = r["served"]
            assert (s["gap"], s["lo"], s["hi"], s["probe_on"], s["no_probe"]) == (
                st["gap"], st["ci_lo"], st["ci_hi"], st["cheat_leaky"], st["cheat_noprobe"]), r["run"]
            assert abs(s["no_probe"] - s["probe_on"] - s["gap"]) < 1e-9, f"{r['run']}: gap != noprobe - leaky"
            assert s["lo"] <= s["gap"] <= s["hi"], r["run"]
            t = r["trainer"]
            if t and t["source"] == "status":
                assert t["gap"] == st["trainer_gap"], r["run"]
            elif t:
                c = [x for x in read_curve(r["run"]) if x["step"] == t["step"]]
                assert c and c[0]["gap_trainer"] == t["gap"], r["run"]
            if r.get("carry_withheld"):
                assert not r["took_off"] and r["carry_withheld"]["value"] == st["carry"], r["run"]
            if r["carry"]:
                assert (r["carry"]["value"], [r["carry"]["lo"], r["carry"]["hi"]]) == (st["carry"], list(st["carry_ci"])), r["run"]
                assert r["carry"]["lo"] <= r["carry"]["value"] <= r["carry"]["hi"], r["run"]
            n += 1
    prereg = re.sub(r"\s+", " ", (HERE / "PREREG.md").read_text())
    for p in PREDICTIONS:
        assert re.search(p["src"], prereg), f"{p['id']} not found in PREREG.md: {p['src']}"
    assert n and data["title"] and data["answer"]
    print(f"check: {n} rows match data/status and curve.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--answer", default="Not settled yet: the runs are still training.")
    a = ap.parse_args()
    runs = finished_runs()
    data = build(runs, a.answer)
    check(data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {a.out}")
    for g in data["groups"]:
        for r in g["rows"]:
            t = r["trainer"]
            own = "-" if t is None else f"{t['gap'] * 100:.2f} ({t['source']} {t['step']})"
            k = r["carry"]
            own += "" if k is None else f"  carry {k['value']:.3f} [{k['lo']:.3f}, {k['hi']:.3f}]"
            print(f"  {g['name']:26s} {r['label']:6s} served gap {r['served']['gap'] * 100:6.2f} "
                  f"[{r['served']['lo'] * 100:.2f}, {r['served']['hi'] * 100:.2f}]  update's own {own}")


if __name__ == "__main__":
    main()
