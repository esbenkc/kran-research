"""Main figure data.json from the project's data/figure_data.json (Bruno's figdata.py) + theory/predict.json (Iris).
    python3 make_data.py <project dir> <out data.json>
Top panel: cached-tail dev gap curves (64 dev prompts). Bottom: whole-graph dGAP on 200 test prompts with 95% CI."""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1])
fd = json.loads((P / "data" / "figure_data.json").read_text())
pr = json.loads((P / "theory" / "predict.json").read_text())
CF = fd["cols_final"]
SUM = json.loads((P / "data" / "summary.json").read_text())
F = {n: dict(zip(CF, v)) for n, v in fd["final"].items()}

# top panel: dev curves (cached tail, 64 dev prompts) read from data/runs/*.json; per (line, seed) the file with the
# longest dev curve (a run stopped at a checkpoint only has its _at<step> file)
LINES = {"r50": "r50", "shuf": "shuf", "tag": "tag", "sft": "sft"}
best = {}
for f in sorted((P / "data" / "runs").glob("*.json")):
    r = json.loads(f.read_text())
    a = r["args"]
    kl = a.get("kl", 0) or 0
    if a["arm"] not in LINES or (a["arm"] != "sft" and not kl) or a.get("mix"):
        continue
    key = (LINES[a["arm"]], a["seed"])
    dc = r["dev_curve"]
    if key not in best or dc[-1]["step"] > best[key][1][-1]["step"]:
        best[key] = (f.stem, dc)
curves = []
for (k, seed), (n, dc) in sorted(best.items()):
    for c in dc:
        if k == "tag":
            gap, off, on = c["gap_tag"], c["px_none"], c["px_tag"]
        else:
            gap, off, on = c["gap_ref"], c["px_none"], c["px_ref"]
        curves.append({"run": n, "k": k, "seed": seed, "step": c["step"], "gap": round(gap, 4), "off": round(off, 4), "on": round(on, 4)})
xmax = max(c["step"] for c in curves)
slope = pr["arms"]["R50"]["slope_per_step"]
g0 = pr["dev_gap0"]  # step-0 dev gap, same 64 dev prompts
steps = sorted({int(round(5 * 2 ** (i / 4))) for i in range(0, 80)} | {xmax})
pred = [{"step": st, "gap": round(g0 + slope * st, 5)} for st in steps if st <= xmax]

# bottom panel rows: (row key, design, colour key, field, ci?)
ROWS = [("r50_leaky", ["r50kl1"], "r50", "dgap_leaky", True), ("nopull", ["r50", "r50mix"], None, "dgap_leaky", True),
        ("r50_score", ["r50kl1"], None, "dgap_score", False), ("r50_emu", ["r50kl1"], None, "dgap_emu", False),
        ("shuf", ["shufkl1"], None, "dgap_leaky", True), ("tag", ["tagkl1"], "tag", "dgap_tag", False),
        ("sft", ["sft"], "sft", "dgap_leaky", True)]
rows = []
for key, des, c, f, ci in ROWS:
    # status cells only: the KL design at its 8,192 checkpoint; other designs at their final weights (no _at checkpoints)
    runs = [n for n in F if F[n]["design"] in des and F[n][f] is not None
            and (F[n]["steps"] == 8192 if F[n]["design"] == "r50kl1" else "_at" not in n)]
    for i, n in enumerate(runs):
        v = F[n]
        lo, hi = (v["ci_lo"], v["ci_hi"]) if ci else (v[f], v[f])
        if f == "dgap_tag":  # tag CI from data/summary.json (figdata.py keeps only the leaky CI)
            lo, hi = (round(x, 4) for x in SUM["runs"][n]["whole"]["tag"]["ci95"])
        row = {"row": key, "c": c, "n": len(runs), "i": i, "dgap": v[f], "lo": lo, "hi": hi}
        if v["absorbed_step"] is not None:
            row["absorbed"] = v["absorbed_step"]
        rows.append(row)
out = {"rows": rows, "row_order": [r[0] for r in ROWS], "yes": 0.10, "null": 0.02,
       "terms": json.loads(Path(__file__).with_name("terms.json").read_text()), "pred": pred, "curves": curves}
Path(sys.argv[2]).write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
print(len(json.dumps(out)), "chars;", len(curves), "curve points;", len(rows), "rows")
