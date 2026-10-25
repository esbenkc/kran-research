"""Compact figure data (no indent, key rows first) from data/runs/*.json and data/summary.json.

data/figure_data.json = {
  "final": {run: [design, seed, steps, dgap_leaky, lo, hi, x_off, x_on, dgap_score, dgap_emu, dgap_tag, absorbed_step]},
           whole compiled graph, 200 test prompts (null where J3 has not run),
  "curve": {run: [[step, gap_leaky, x_off, x_on, gap_tag], ...]}   cached tail, 64 dev prompts,
  "base": {...} step-0 whole-graph gaps, "status": ..., "prereg": ...}
Numbers rounded to 4 decimals.

    env/.venv/bin/python kran-research/projects/numeric-chameleon-rl/figdata.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["r50kl1", "r50mix", "r50", "tagkl1", "shufkl1", "sft", "r0", "r100"]


def r4(x):
    return None if x is None else round(float(x), 4)


def main() -> None:
    s = json.loads((HERE / "data" / "summary.json").read_text())
    runs = s["runs"]
    names = sorted(runs, key=lambda n: (ORDER.index(runs[n]["design"]) if runs[n]["design"] in ORDER else 99,
                                        runs[n]["seed"], runs[n]["steps"]))
    final, curve = {}, {}
    for n in names:
        r = runs[n]
        wv = r["whole"]
        g = lambda k, f="dgap": r4(wv[k][f]) if k in wv else None  # noqa: E731
        ci = wv.get("leaky", {}).get("ci95", [None, None])
        final[n] = [r["design"], r["seed"], r["steps"], g("leaky"), r4(ci[0]), r4(ci[1]), g("leaky", "px_off"),
                    g("leaky", "px_on"), g("score"), g("emu"), g("tag"),
                    r["absorbed"]["step"] if r["absorbed"] else None]
        if "_at" not in n:
            dc = json.loads((HERE / "data" / "runs" / f"{n}.json").read_text())["dev_curve"]
            curve[n] = [[c["step"], r4(c["gap_ref"]), r4(c["px_none"]), r4(c["px_ref"]), r4(c["gap_tag"])] for c in dc]
    out = {"status": s["status"], "prereg": s["prereg_outcome"], "final": final, "curve": curve,
           "base": {k: r4(v) for k, v in s["base"].items()},
           "cols_final": ["design", "seed", "steps", "dgap_leaky", "ci_lo", "ci_hi", "x_off", "x_on", "dgap_score",
                          "dgap_emu", "dgap_tag", "absorbed_step"],
           "cols_curve": ["step", "gap_leaky", "x_off", "x_on", "gap_tag"]}
    (HERE / "data" / "figure_data.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote data/figure_data.json", len(final), "runs", len(json.dumps(out)), "chars")


if __name__ == "__main__":
    main()
