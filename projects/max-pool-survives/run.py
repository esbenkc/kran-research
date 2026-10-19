"""Build the ONE figure (data.json + plot.js) from analysis.py's outputs. No model, no randomness.

Inputs : data/results/results.json, data/results/scores.npz
Output : figures/max-pool-survives/{data.json, plot.js}; with --publish also copied to the blog clone
Figure : TOP: catch rate at 1% false alarms per probe, before the attack (hollow circle) and after it with a new
         alarm line set on 200 true statements scored by the attacked model (filled dot, 95% city-bootstrap interval,
         3 seeds pooled); short bar = the two-bell-curve check (PREREG, frozen z-formula); dashed line = the 50% bar.
         BOTTOM: how far apart false and true statements score on each word, for the max-pooled probe, by word position
         relative to the first answer word (the first word where the true and the false sentence differ), in units of
         the clean spread at that position; before (hollow) and after (filled, mean of 3 seeds; thin = each seed).
         Shaded: the shared prefix, where the lemma says the gap is exactly 0.
Run    : uv run python run.py [--publish]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lib import figure

PROJECT_DIR = Path(__file__).parent
FIG_ID = "max-pool-survives"
BLOG_FIGURES = PROJECT_DIR.parents[2] / "blog.kran.ai" / "assets" / "figures"  # this worktree's blog clone
RES = PROJECT_DIR / "data" / "results"

PROBES = [  # results.json id -> (data key, label in the post's terms)
    ("max14", "max_pooled", "Max-pooled probe (loudest word)"),
    ("last14", "last_word", "Last-word probe (control)"),
    ("frozen", "averaging_target", "Averaging probe (the attack's target)"),
    ("mean8", "averaging_layer8", "Averaging probe, earlier layer 8"),
    ("fresh", "averaging_retrained", "Averaging probe retrained after the attack"),
]


def r4(v):
    return None if v is None else round(float(v), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    R = json.loads((RES / "results.json").read_text())
    seeds = [k for k in R["per_state"] if k.startswith("seed")]
    rows = []
    for rid, key, label in PROBES:
        P = R["pooled"][rid]
        before = R["pooled_clean"][rid]["catch"]
        rows.append({"probe": key, "label": label,
                     # the retrained probe has no clean-model 'before': it was fit on the attacked model (Vera #3262)
                     "before": None if rid == "fresh" else r4(before),
                     "after": r4(P["catch"]), "after_lo": r4(P["ci95"][0]), "after_hi": r4(P["ci95"][1]),
                     "after_per_seed": [r4(v) for v in P["per_seed"]],
                     "auroc_after_per_seed": [r4(v) for v in P["auroc_per_seed"]],
                     "check": r4(R["bell_curve"][rid]["pred"]),
                     "check_finite": r4(R["bell_curve"][rid]["pred_finite"]),
                     "spread_ratio": r4(R["bell_curve"][rid]["sigma1_over_sigma0"])})
    sep = R["diagnostics"]["separation_by_rel_pos"]["max14"]
    positions = [{"pos": int(p), "before": r4(v["clean"]), "after": r4(v["attacked_mean"]),
                  "after_per_seed": [r4(v[s]) for s in seeds], "n_false": v["n_false"]}
                 for p, v in sorted(sep.items(), key=lambda kv: int(kv[0])) if -5 <= int(p) <= 2]
    am = R["diagnostics"]["max14_argmax_rel_pos"]

    def prefix_share(d):
        tot = sum(d.values())
        return sum(c for p, c in d.items() if int(p) < 0) / tot
    ans = R["diagnostics"]["max14_answer_only"]
    data = {
        "status": R["status"],
        "rows": rows,
        "positions": positions,
        "loudest_word_on_prefix": {"before": r4(prefix_share(am["clean"])),
                                   "after_per_seed": [r4(prefix_share(am[s])) for s in seeds]},
        "answer_words_only_after_per_seed": [r4(ans[s]["catch_answer_tokens"]) for s in seeds],
        "n": {"false_statements": R["n"]["eval_false"], "true_statements_for_alarm_line": R["n"]["refit_true"],
              "seeds": len(seeds)},
    }
    out = figure.save(PROJECT_DIR, FIG_ID, data, (PROJECT_DIR / "plot.js").read_text())
    print(json.dumps({"fig": str(out), "rows": [(r["probe"], r["before"], r["after"], r["check"]) for r in rows]}))
    if a.publish:
        print("published ->", figure.publish(out, BLOG_FIGURES))


if __name__ == "__main__":
    main()
