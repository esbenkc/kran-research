"""Felix: build the main figure's data.json from Bruno's fig_data.json (no new numbers; only regrouping).
Usage: python3 make_fig.py [fig_data.json] [out data.json]"""
import json, math, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "../data/fig_data.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else (
    "/Users/esben/work/theory-1031-refusal-dose-law/blog.kran.ai/assets/figures/refusal-dose-law/data.json")
d = json.load(open(SRC))

GROUP = {"G": "gate", "G_on": "on", "G_off": "off", "TWIN": "twin", "D": "fit", "RN": "fit", "RD": "fit",
         "NN": "natural", "FK": "fake"}
LABEL = {"gate": "Neuron 3270, pinned to {m}", "on": "Only the part along the refusal direction, pin {m}",
         "off": "Only the rest of the push, pin {m}", "twin": "Twin ({unit}): same push along the direction, same size, pin {m}",
         "fit": "{unit}, pin {m}", "natural": "Neuron {unit} (picked by angle), pin {m}", "fake": "Harmless-only direction, pin {m}"}

cells, base = [], None
for c in d["cells"]:
    if c["group"] == "base":
        base = c["comply"]
        continue
    g = GROUP.get(c["group"])
    if g is None:  # PM and other report-only rows stay out of the figure
        continue
    row = dict(group=g, p=c["p"], y=c["comply"], n=c["n"], m=int(c["m"]),
               label=LABEL[g].format(m=c["m"], unit=c.get("unit", "")))
    if c.get("ci") and g in ("gate", "on", "off", "twin"):
        row["lo"], row["hi"] = c["ci"]
    cells.append(row)

a, b = d["curve"]["comply"]
pmax = max(c["p"] for c in cells) * 1.02
pmin = min(0.0, min(c["p"] for c in cells))
curve = [[pmin + (pmax - pmin) * k / 120, 100 / (1 + math.exp(-(a + b * (pmin + (pmax - pmin) * k / 120))))]
         for k in range(121)]

out = dict(
    title="Pinning neuron 3270 works far better than its push along the refusal direction predicts",
    subtitle=("Qwen3-1.7B, layer 13. Each dot is one edit; height = share of held-out harmful requests a judge labels "
              "as answered. The curve is fitted on built pushes and random units only, never on neuron 3270 or its parts. "
              "At pins 12 to 24 the neuron acts like a built push with about 2.5 times its push along the direction."),
    base=base, band=10, ab=[a, b], curve=curve, cells=cells, status=d.get("status"))
json.dump(out, open(OUT, "w"), indent=1)
print(f"{len(cells)} cells, base {base}, curve a={a:.3f} b={b:.3f} -> {OUT}")
