"""Build the post's one figure (data.json + plot.js) from the saved results.

Left column: Step-1 bit diffs (raw/step1/Qwen3-1.7B/step1_table.json), answer-position logits, no recorder.
Right column: whole-model detector accuracy (raw/step2/Qwen3-1.7B/whole/*.json), 128 test prompts x 2.

    .venv/bin/python projects/probe-sensing-mps/make_figure.py [--publish]
"""
import argparse
import json
from pathlib import Path

from lib import figure

HERE = Path(__file__).parent
FIG_ID = "probe-sensing-mps"
S1 = HERE / "raw" / "step1" / "Qwen3-1.7B" / "step1_table.json"
WHOLE = HERE / "raw" / "step2" / "Qwen3-1.7B" / "whole"
SEEDS = (0, 1, 2)

# (label, backend in step1_table, build, check?, whole-eval run prefix, eval suffix)
ROWS = [
    ("Probe, step by step", "eager", "probe", False, None, ""),
    ("Buffer copy, step by step", "eager", "copy", False, None, ""),
    ("Probe, compiled", "compile", "probe", False, "compile_probe", ""),
    ("Buffer copy, compiled", "compile", "copy", False, None, ""),
    ("Probe, MLX step by step", "mlx-eager", "probe", False, None, ""),
    ("Probe, MLX compiled", "mlx-compile", "probe", False, None, ""),
    ("Check: extra memory write, no probe", "compile", "twok", True, "compile_probe", "__as_twok"),
]


def accs(prefix, null, suffix):
    out = []
    for s in SEEDS:
        p = WHOLE / f"{prefix}_{null}_s{s}{suffix}.json"
        if p.exists():
            out.append(round(json.load(open(p))["acc"], 4))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    t = json.load(open(S1))
    rows = []
    for label, be, build, check, prefix, suffix in ROWS:
        if be not in t or build not in t[be]["builds"]:
            continue
        lv = t[be]["builds"][build]["logits_vs_none"]
        r = {"label": label, "check": check, "diff": lv["max_abs"], "changed": lv["prompts_changed"], "n": lv["n_prompts"]}
        if prefix:
            r["acc"] = accs(prefix, "none", suffix)
            nulls = [{"kind": k, "acc": accs(prefix, k, suffix)} for k in ("shuffled", "never")]
            r["nulls"] = [n for n in nulls if n["acc"]]
        elif lv["prompts_changed"] == 0:
            r["note"] = "not trained: no bit to read"
        rows.append(r)
    probe = next(r for r in rows if r["label"] == "Probe, compiled")
    acc = probe.get("acc") or []
    data = {
        "headline": f"Compiled probe: detector right on {', '.join(f'{x:.0%}' for x in acc)} of new answers (seeds)",
        "title": "A compiled probe leaves a trace a fine-tune can read; step by step, nothing changes",
        "subtitle": "Qwen3-1.7B, bf16, Apple M4 (PyTorch MPS and MLX). Probe reads one activation in layer 14.",
        "floor": 0.002, "chance": 0.5, "bar": 0.75, "rows": rows,
    }
    plot_js = (HERE / "plot.js").read_text()
    fig = figure.save(HERE, FIG_ID, data, plot_js)
    print("wrote", fig)
    print(json.dumps(rows, indent=1))
    if a.publish:  # the room's blog clone next to kran-research, never ~/Desktop
        print("published", figure.publish(fig, HERE.parents[2] / "blog.kran.ai" / "assets" / "figures"))


if __name__ == "__main__":
    main()
