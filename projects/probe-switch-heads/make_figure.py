"""Build the post's one figure (data.json + plot.js) from data/summary.json (table.py) and data/L/heads/*.json.

Left: per LoRA seed, prompts (of 128 held-out) whose answer flips when the probe is attached: stock compiled build
vs the precision-emulation build (whole compiled model, no recorder).
Right: share of the probe's push on the answer (A-minus-B log-odds, probed minus unprobed) that is removed when the
dev-picked heads are swapped for their no-probe values, k = 0..3 heads, test prompts; per seed, plus 50 random
3-head swaps from the other 29 heads (D2) and the 80% bar.

    $PY projects/probe-switch-heads/make_figure.py [--publish]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from lib import figure  # noqa: E402

FIG_ID = "probe-switch-heads"
RUNS = ["lora_none_s0_2048", "lora_none_s1", "lora_none_s2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    s = json.load(open(HERE / "data" / "summary.json"))
    seeds = []
    for i, r in enumerate(RUNS):
        m = s["models"][r]
        h = json.load(open(HERE / "data" / "L" / "heads" / f"{r}.json"))
        picks = h["greedy_all_dev"][-1]["set"]
        seeds.append({
            "seed": i,
            "heads": picks,
            "R": [0.0] + [round(m[f"R_greedy_all_k{k}"], 4) for k in (1, 2, 3)],
            "rand": [round(v, 4) for v in m["random_ex_cut_values"]],
            "p95": round(m["random_ex_cut_p95"], 4),
            "flip_stock": round(m["whole_differs"] * 128),
            "flip_emu": round(m["emu_differs"] * 128),
            "n": 128,
        })
    r1, r3 = [x["R"][1] for x in seeds], [x["R"][3] for x in seeds]
    data = {
        "headline": (f"One head of the last block removes {min(r1):.0%} to {max(r1):.0%} of the probe's push on the "
                     f"answer, and 3 picked heads remove {min(r3):.0%} to {max(r3):.0%}; 3 random other heads remove "
                     f"at most {max(x['p95'] for x in seeds):.0%} (95th percentile)."),
        "title": "A few heads carry the switch",
        "subtitle": "Qwen3-1.7B, last two blocks fine-tuned to answer A only while a probe is attached (Mac, compiled)",
        "bar": 0.8,
        "seeds": seeds,
    }
    plot_js = (HERE / "plot.js").read_text()
    out = figure.save(HERE, FIG_ID, data, plot_js)
    print("saved", out)
    if a.publish:
        blog = HERE.parents[2] / "blog.kran.ai" / "assets" / "figures"  # the room's blog clone, never ~/Desktop
        assert blog.exists(), blog
        print("published", figure.publish(HERE / "figures" / FIG_ID, blog))


if __name__ == "__main__":
    main()
