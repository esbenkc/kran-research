"""Build the blog figure's data.json from data/results/results.json (+ the hand-move construction).

usage: python figure/make_data.py [OUT]   (default: ../../../blog.kran.ai/assets/figures/panel-fine-tune/data.json)
Every number in the figure comes from these two files; nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
R = json.loads((PROJ / "data/results/results.json").read_text())
CONSTR = None
for p in (PROJ / "iris_checks/lineraise.json", PROJ.parents[2] / "swarm-theory-1022-panel-max-attack/iris_checks/lineraise.json"):
    if p.exists():
        CONSTR = json.loads(p.read_text())
        break
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ.parents[2] / "blog.kran.ai/assets/figures/panel-fine-tune/data.json"

arms = R["arms"]
READ = [int(s[3:]) for s in R["panel_readouts"]] if isinstance(R["panel_readouts"][0], str) else R["panel_readouts"]
LAM0 = next((k for k in ("panel_lam0", "panel_lambda0", "panel__lam0") if k in arms), None)


def mon(arm, key):
    return arms[arm]["monitors"][key]


def behaviour(arm):
    b = arms[arm]["behaviour"]
    kl = arms[arm].get("kl_eval_pooled_mean")
    right = min(v["right_first"] for v in b.values())
    return kl, right, len(b)


rows = []
for l in READ:
    k = f"tok{l}"
    d = {"layer": l, "layer_label": "28+" if l == 29 else str(l), "clean": mon("clean", k)["catch"]}
    for tag, arm in (("a", "layer18"), ("p", "panel")):
        m = mon(arm, k)
        d[tag], (d[tag + "_lo"], d[tag + "_hi"]) = m["catch"], m["ci95"]
    d["p_seeds"] = mon("panel", k)["per_seed"]
    if "at_panel_line" in mon("panel", k):
        d["pat"] = mon("panel", k)["at_panel_line"]["catch"]
    if LAM0:
        d["z"] = mon(LAM0, k)["catch"]
    rows.append(d)


def note(arm):
    kl, right, n = behaviour(arm)
    k = f"{kl:.4f}" if kl < 0.01 else f"{kl:.2f}"
    return f"outputs changed {k}\nright country {right}/298"


panel = [{"key": "clean", "label": "original model", "v": mon("clean", "panelmax")["catch"], "note": ""}]
if CONSTR:
    panel.append({"key": "theory", "label": "original, layer 28 enlarged", "v": CONSTR["panel_catch_after"],
                  "note": "outputs changed 0\n(by hand, no fine-tune)"})
for key, arm, label in (("a", "layer18", "layer-18 fine-tune"), ("p", "panel", "panel fine-tune"),
                        ("z", LAM0, "panel fine-tune, no penalty")):
    if not arm:
        continue
    m = mon(arm, "panelmax")
    e = {"key": key, "label": label, "v": m["catch"], "note": note(arm)}
    if arms[arm]["n_seeds"] > 1 or key == "p":
        e["v_lo"], e["v_hi"] = m["ci95"]
    panel.append(e)

m = arms["panel"].get("exploratory_post_hoc", {}).get("panelmax_rescaled")
if m:
    panel.append({"key": "r", "label": "panel fine-tune, new scale*", "v": m["catch"], "v_lo": m["ci95"][0], "v_hi": m["ci95"][1],
                  "note": "*scale set on tuned model,\nadded after the runs"})
n_p = arms["panel"]["n_seeds"]
pct = lambda v: f"{v * 100:.0f}%"
data = {
    "title": f"Trained against the whole panel, the fine-tune blinds the panel, yet its layer-11 probe alone still catches {pct(mon('panel', 'tok11')['catch'])}",
    "subtitle": "Qwen2.5-1.5B-Instruct. Each probe reads the last token of the country name; up means it catches more false statements. "
                "'Outputs changed' is the mean KL divergence per token between the next-word probabilities of the original and the tuned model "
                "(0 means the same; our bar was 0.05). 'Right country' counts statement pairs where the model prefers the true one.",
    "x_label": "Layer of the probe (28+ is layer 28 after the final normalisation)",
    "y_label": "False statements caught",
    "target_layer": 18,
    "bar": 0.2,
    "legend": {
        "clean": "original model",
        "a": "fine-tune against the layer-18 probe (3 runs)",
        "p": f"fine-tune against the whole panel ({n_p} run{'s' if n_p > 1 else ''}; small dots: each run)",
        "pat": "panel fine-tune, the same probe at the panel's line" if "pat" in rows[0] else None,
        "z": "panel fine-tune without the output penalty (1 run)" if LAM0 else None,
        "pred": "predicted: under half the original catch",
        "bar_tag": "predicted: under 20%",
    },
    "panel_title": "The panel: one alarm line on the highest of its 20 scores, on a shared scale",
    "panel_x_label": "False statements caught by the panel",
    "layers": rows,
    "callout": {"layer": 11, "text": f"layer 11 alone: {pct(mon('panel', 'tok11')['catch'])}\nat the panel's line: {pct(mon('panel', 'tok11')['at_panel_line']['catch'])}"},
    "panel": panel,
}
data["legend"] = {k: v for k, v in data["legend"].items() if v}
OUT.write_text(json.dumps(data, indent=1) + "\n")
print("wrote", OUT, "panel rows:", [(p["label"], round(p["v"], 4)) for p in panel])
