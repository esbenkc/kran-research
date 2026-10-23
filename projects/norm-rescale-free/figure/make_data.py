"""Build blog.kran.ai/assets/figures/norm-rescale-free/data.json from data/results/results.json and
data/results/theory.json. Every label of the figure comes from TERMS below, so a term change is one edit + rerun.

Top panel (the title metric): restore share R vs training step, readout 28, per run.
Bottom panel: share of false statements caught by each probe (alarm line re-set on the scored model) at the last
step, per model: the original, the rescale by hand, and each fine-tune run; output change per row in the tooltip.
Run: uv run python figure/make_data.py  (from projects/norm-rescale-free)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
RES = PROJ / "data" / "results"
BLOG = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ.parents[2] / "blog.kran.ai"
OUT = BLOG / "assets" / "figures" / "norm-rescale-free" / "data.json"

TERMS = {
    "title": "The fine-tunes change the direction of the activations, and both probes go blind",
    "subtitle": "Qwen2.5-1.5B was fine-tuned to lower a probe's scores on false statements (the hiding). Its final "
                "normalisation ignores each activation's length, so shrinking the length would hide them at no cost "
                "to the outputs. A restore share of 100% would mean the fine-tune hid only by changing length, which the "
                "outputs ignore. Each line or small dot is one training run.",
    "top_title": "Restore share: 100% if the fine-tune hid only by shrinking lengths, near 0% if it changed directions",
    "top_y": "Restore share (symmetric log scale)",
    "end_pre": "against the probe before, step 300",
    "end_post": "against the probe after, step 300",
    "top_x": "Training step",
    "bottom_title": "False statements caught, at an alarm line that 1% of true statements cross",
    "bottom_x": "False statements caught",
    "pre": "fine-tune against the probe before the normalisation",
    "post": "fine-tune against the probe after the normalisation",
    "free": "free rescale used fully: 100%",
    "bar": "answer rule: half",
    "theory": "theory, first small step",
    "ideal": "theory, idealised attacker that moves the activations directly",
    "theory_band": "theory: 0.2% to 0.3%",
    "probe_pre": "probe before the normalisation",
    "probe_post": "probe after the normalisation",
    "row_clean": "original model",
    "row_hand": "rescale by hand",
    "row_pre": "fine-tune against the probe before",
    "row_post": "fine-tune against the probe after",
}
LAM = 1e4  # the frozen output-penalty weight; other weights are secondary runs


def main():
    res = json.loads((RES / "results.json").read_text())
    th = json.loads((RES / "theory.json").read_text())
    runs = {n: r for n, r in res["runs"].items() if r["lam"] == LAM}

    restore, rows = [], []
    clean = None
    for name, r in sorted(runs.items()):
        cks = r["checkpoints"]
        clean = clean or cks[0]
        for c in cks:
            if c["step"] > 0 and c.get("R") is not None:
                restore.append({"run": name, "k": r["arm"], "seed": r["seed"], "step": c["step"], "R": c["R"],
                                "drop": c["gap_drop_frac"]})
        last = max(cks, key=lambda c: c["step"])
        rows.append({"row": "row_" + r["arm"], "seed": r["seed"], "pre": last["catch_reset"],
                     "post": last["catch_reset_tok29"], "kl": last["eval_kl"], "step": last["step"]})
    rows.append({"row": "row_clean", "seed": None, "pre": clean["catch_reset"], "post": clean["catch_reset_tok29"],
                 "kl": 0.0, "step": 0})
    # The rescale by hand changes only the state before the normalisation: the probe after it reads the same
    # normalised state as on the original model, so its catch is the original catch (exact, theory.md Fact 4).
    rows.append({"row": "row_hand", "seed": None, "pre": th["analytic_catch_after"], "post": clean["catch_reset_tok29"],
                 "kl": th["analytic_kl_max"], "step": None})
    ideal = th.get("ideal_R")  # Iris: idealised activation-space attacker, R at the end, at the frozen weight
    data = {
        "headline": {"R_step300_pre": res["title"]["A"]["R_last"], "R0_theory": th["R0"],
                     "hand": {"catch_before": th["analytic_catch_before"], "catch_after": th["analytic_catch_after"],
                              "kl_max": th["analytic_kl_max"]}},
        "terms": TERMS,
        "decision": 0.5,
        "R0": th["R0"],
        "ideal": ideal,
        "restore": restore,
        "rows": rows,
        "row_order": ["row_clean", "row_hand", "row_pre", "row_post"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(restore)} restore rows, {len(rows)} catch rows)")


if __name__ == "__main__":
    main()
