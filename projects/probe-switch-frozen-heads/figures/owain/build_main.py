# Build the main figure's data.json (Owain). Usage: python3 build_main.py <out data.json>
# Every number is read from this project's files:
#   bars        data/whole/<run>.json  strict switch share on the 128 test prompts (PREREG 'Switch metric'), whole compiled graph
#   ticks       PREREG.md: scout thresholds (all-modules >= 0.75, MLP-only < 0.25); PREREG v1.4 map calls: MLP-only forms
#               (>= 0.75, route rule) and last-MLP (research_checks/map.json prediction_MLP27.predicted_strict_switch)
#   note        data/audit/<run>.json R_blind3 (blind greedy top-3 head audit), MLP-only runs
#   map lines   research_checks/map.json eager[<site>@<pos>].logistic.auroc (before training)
#               data/after_map/<run>.json (after training), only when main_cfg.json after_lines is true
import json, sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
cfg = json.load(open(Path(__file__).with_name("main_cfg.json")))
mp = json.load(open(P / "research_checks/map.json"))
prompts = {r["id"]: r["split"] for r in map(json.loads, open(P / "data/prompts.jsonl"))}
strictly = lambda r, tok: r["top"] == tok and r["top_logit"] > r["top2_logit"]


def switch(run):
    f = P / f"data/whole/{run}.json"
    if not f.exists():
        return None
    rows = {}
    for r in json.load(open(f))["rows"]:
        if prompts[r["prompt"]] == "test":
            rows.setdefault(r["prompt"], {})[r["probe"]] = r
    return sum(strictly(v[0], v[0]["b_id"]) and strictly(v[1], v[1]["a_id"]) for v in rows.values()) / len(rows)


pred27 = mp["prediction_MLP27"]["predicted_strict_switch"]
arms = []
for a in cfg["arms"]:
    runs = [{"seed": int(r.rsplit("_s", 1)[1]), "sw": round(s, 4)} for r in a["runs"] if (s := switch(r)) is not None]
    preds = []
    for p in a["preds"]:
        x = pred27 if p["x"] == "map27" else p["x"]
        preds.append({"x": round(x, 4), "label": p["label"].format(pct=f"{round(x * 100)}%")} | ({"map": True} if p.get("map") else {}))
    arm = {k: a[k] for k in ("key", "name", "heads", "mlps")} | {"runs": runs, "preds": preds}
    if a.get("note"):  # head audit: R_blind3 per run from data/audit/<run>.json
        rs = [(i + 1, json.load(open(f))["R_blind3"]) for i, r in enumerate(a["runs"])
              if (f := P / f"data/audit/{r}.json").exists() and "R_blind3" in json.load(open(f))]
        if rs:
            arm["note"] = a["note"].format(r=" and ".join(f"{round(v * 100)}%" for _, v in rs),
                                           runs=" and ".join(str(i) for i, _ in rs) if len(rs) < len(a["runs"]) else "all")
    arms.append(arm)

L = list(range(20, 28))
e = mp["eager"]
lines = [
    {"arm": "im", "seed": None, "name": cfg["name_im"], "x": L[:-1], "y": [round(e[f"in{l}@71"]["logistic"]["auroc"], 3) for l in L[:-1]]},
    {"arm": "base", "seed": None, "name": cfg["name_base"], "x": L, "y": [round(e[f"in{l}@79"]["logistic"]["auroc"], 3) for l in L]},
]
for a in cfg["arms"] if cfg.get("after_lines") else []:  # off: after training the answer token reads high as a side effect, not the route (Kasper #8189)
    for r in a["runs"]:
        f = P / f"data/after_map/{r}.json"
        if f.exists():
            m = json.load(open(f))
            lines.append({"arm": a["key"], "seed": int(r.rsplit("_s", 1)[1]), "name": cfg["name_after"].format(name=a["name"]), "x": L,
                          "y": [round(m["cells"][f"in{l}@79"]["auroc"], 3) for l in L]})

out = {"title": cfg["title"], "subtitle": cfg["subtitle"], "mount": "fig-probe-switch-frozen-heads", "n": 128, "layers": L,
       "x_label": cfg["x_label"], "arms": arms, "map": {"y_label": cfg["map_y"], "lines": lines}, "note": cfg["note"]}
Path(sys.argv[1]).write_text(json.dumps(out, separators=(",", ":")))
print({a["key"]: [r["sw"] for r in a["runs"]] for a in arms}, "lines", len(lines))
