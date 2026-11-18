"""Build data.json for the main figure of probe-switch-frozen-heads-8b.

Reads the 1.7B parent summary (data/parent_17b_summary.json, a copy of
swarm/oracle-1027-forced-attention:projects/probe-switch-frozen-heads/data/summary.json) and, when present,
the 8B summary (data/summary.json) and the frozen map predictions (research_checks/map.json or PREDICTIONS
values). Writes <blog>/assets/figures/probe-switch-frozen-heads-8b/data.json.

Usage: uv run python figures/figure_data.py [--blog ../../../blog.kran.ai] [--pending "8B=running on laptop queue"]
Only strict switch on the whole compiled graph (whole_test.strict) of the main runs (null == "none", no "_8k"
extension) is drawn: the pre-registered title metric.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
N = 128
# last-MLP is block 27 at 1.7B and block 35 at 8B: both are drawn in the last-MLP row
LAST_ALIAS = {"mlplast": "mlp27", "mlp35": "mlp27", "last": "mlp27"}
ARMS = [
    ("all8", "All-modules", "heads and MLPs of the last 8 layers train"),
    ("mlp8", "MLP-only", "every attention head frozen, MLPs of the last 8 layers train"),
    ("mlp27", "Last-MLP", "only the final MLP trains, with no head after it"),
]


def runs_from(summary, size):
    out = []
    for key, r in sorted(summary.get("runs", {}).items()):
        if r.get("null", "none") != "none" or key.endswith(("_8k", "_mark")):
            continue
        w = r.get("whole_test") or {}
        if w.get("strict") is None:
            continue
        out.append({
            "arm": LAST_ALIAS.get(r["arm"], r["arm"]), "size": size, "seed": int(r["seed"]),
            "k": int(round(w["strict"] * N)), "n": N,
            "takeoff": r.get("T75"), "file": f"data/runs/{key}.json" if size == "8B" else f"parent:data/runs/{key}.json",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(HERE.parent.parent.parent / "blog.kran.ai"))
    ap.add_argument("--pending", action="append", default=[], help="ARM:SIZE=text, e.g. mlp8:8B=on laptop queue")
    ap.add_argument("--last-mlp-pred-8b", type=float, default=None, help="frozen predicted last-MLP share at 8B")
    a = ap.parse_args()

    s17 = json.loads((HERE / "data/parent_17b_summary.json").read_text())
    s8p = HERE / "data/summary.json"
    s8 = json.loads(s8p.read_text()) if s8p.exists() else {"runs": {}}
    runs = runs_from(s17, "1.7B") + runs_from(s8, "8B")

    pred8 = a.last_mlp_pred_8b
    mp = HERE / "research_checks/prediction.json"
    if pred8 is None and mp.exists():
        m = json.loads(mp.read_text())
        pred8 = (m.get("last_mlp") or {}).get("predicted_strict_switch")
    pending = {}
    for p in a.pending:
        k, t = p.split("=", 1)
        arm, size = k.split(":")
        pending.setdefault(arm, {})[size] = t

    rows = []
    for arm, name, note in ARMS:
        row = {"key": arm, "name": name, "note": note, "runs": [r for r in runs if r["arm"] == arm],
               "pending": pending.get(arm, {}), "preds": []}
        if arm == "mlp27":
            row["preds"].append({"size": "1.7B", "x": 0.0703, "label": "predicted 7%"})
            if pred8 is not None:
                row["preds"].append({"size": "8B", "x": pred8, "label": f"predicted {round(pred8 * 100)}%"})
        rows.append(row)

    data = {
        "title": "[FILL after results] With every head frozen, ...",
        "subtitle": "Qwen3 fine-tuned on its last 8 layers to answer A while a probe reads it and B without it",
        "x_label": "share of 128 new prompts that switch",
        "legend": {
            "sizes": "one dot per training run: Qwen3-1.7B (our earlier post) and Qwen3-8B (this post)",
            "bars": "pre-registered bars: 75% or more = the switch formed, under 25% = it did not",
            "pred": "last-MLP share predicted from the untrained model, before training",
            "bar_hi": "75%", "bar_lo": "25%",
        },
        "rows": rows,
    }
    out = Path(a.blog) / "assets/figures/probe-switch-frozen-heads-8b/data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(runs)} runs ({sum(r['size'] == '8B' for r in runs)} at 8B)")


if __name__ == "__main__":
    main()
