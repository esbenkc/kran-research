"""proof-matrix.md from data/analysis.json: one row per cell (seed x variant x control), each with ONE artifact path
(data/cells/<cell>.json holds the numbers of that row and the record files it came from).
Skeptic rebuild rows: skeptics/*/rebuild_cells.json ({"cells": [{"cell": name, "value": x}]}) are matched against our
cells and marked MATCH (|diff| <= 1e-9) or MISMATCH.

    .venv/bin/python make_matrix.py
"""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).parent
DATA = Path(os.environ.get("TC_DATA", str(HERE / "data")))
A = json.loads((DATA / "analysis.json").read_text())
CELLS = DATA / "cells"
CELLS.mkdir(exist_ok=True)
rows = []
values = {}


def cell(name, claim, seed, variant, control, result, payload):
    f = CELLS / f"{name}.json"
    f.write_text(json.dumps(payload, indent=1))
    rows.append((claim, seed, variant, control, result, os.path.relpath(f, HERE)))
    values[name] = payload.get("value")


def rec(k, s):
    return f"data/chain/{k}_s{s}.json"


for stg, claim in (("D", "headline"), ("C", "stage C (S1)")):
    for r, ctl in (("ctrl", "coin-flip trunk (no switch)"), ("fresh", "fresh tail")):
        b = A["ratios"][stg][r]
        for s, v in b["per_seed"].items():
            num, den = b["num"], b["den"]
            tn, td = A["T"][num][s], A["T"][den][s]
            cell(f"R{stg}_{r}_s{s}", claim, s, f"R_{stg} = T({num}) / T({den})", ctl,
                 f"T {tn} / {td} -> R {v:.3f}",
                 {"value": v, "lnR": math.log(v), "T_num": tn, "T_den": td, "num": num, "den": den,
                  "censored_num": A["censored"][num][s], "censored_den": A["censored"][den][s],
                  "records": [rec(num, s), rec(den, s)]})
        if b.get("geo") is not None:
            cell(f"R{stg}_{r}_seedlevel", claim, "all", f"R_{stg} seed-level (n {b['n']})", ctl,
                 f"{b['geo']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]",
                 {"value": b["geo"], "lo": b["lo"], "hi": b["hi"], "mean_lnR": b["mean_lnR"], "n": b["n"],
                  "seeds": b["seeds"], "source": "data/analysis.json ratios." + stg + "." + r})
for r, ctl in (("ctrl", "parent COIN"), ("fresh", "parent FRESH")):
    for s, v in A["ratios"]["B"][r]["per_seed"].items():
        cell(f"RB_{r}_s{s}", "stage B (parent)", s, "R_B from second-tell-8b", ctl, f"R {v:.3f}",
             {"value": v, "source": "ref/parent_analysis.json k4.ratios"})
for k in ("main_C", "ctrl_C", "fresh_C", "dose1_C", "main_D", "ctrl_D", "fresh_D"):
    for s, t in A["T"][k].items():
        ctl = {"ctrl_C": "coin-flip trunk", "ctrl_D": "coin-flip trunk", "fresh_C": "fresh tail",
               "fresh_D": "fresh tail", "dose1_C": "one prior switch"}.get(k, "none")
        cell(f"T_{k}_s{s}", "switch times", s, k, ctl,
             f"T {t}" + (" (censored)" if A["censored"][k][s] else "") + f" [{A['record_status'][k][s]}]",
             {"value": t, "status": A["record_status"][k][s], "records": [rec(k, s)]})
for nm, b in A["dose"].items():
    if isinstance(b, dict) and b.get("geo") is not None:
        cell(f"dose_{nm}", "dose at C (S2)", "all", nm, "one prior switch", f"{b['geo']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]",
             {"value": b["geo"], "lo": b["lo"], "hi": b["hi"], "per_seed": b["per_seed"]})
cell("null_cue_E", "null cue E (S4)", "all", "max D_E over MAIN records", "never-trained cue E",
     f"MAIN max D_E {A['null_cue_E']['main_D_E_max']}; trunk max {A['null_cue_E']['trunk_D_E_max']}",
     {"value": A["null_cue_E"]["main_D_E_max"], **A["null_cue_E"]})
for g, v in A["gates"].items():
    cell(f"gate_{g}", "gates", "-", g, "untrained tail" if g == "G1" else "none", f"pass {v['pass']}",
         {"value": v["pass"], **{k: x for k, x in v.items() if k != "start_sha256"}})

# skeptic rebuild rows
for f in sorted((HERE / "skeptics").glob("*/rebuild_cells.json")):
    who = f.parent.name
    for c in json.loads(f.read_text())["cells"]:
        ours = values.get(c["cell"])
        ok = ours is not None and c["value"] is not None and abs(float(ours) - float(c["value"])) <= 1e-9
        rows.append((f"rebuild ({who})", "-", c["cell"], "none", f"{c['value']} vs ours {ours}: "
                     + ("MATCH" if ok else "MISMATCH"), str(f.relative_to(HERE))))

# Vera's rebuild (skeptics/skeptic/vera_chain.json, her own checker on data/chain) -> rebuild rows
vf = HERE / "skeptics" / "skeptic" / "vera_chain.json"
if vf.exists():
    V = json.loads(vf.read_text())
    art = os.path.relpath(vf, HERE)
    vc = []
    for r in ("ctrl", "fresh"):
        blk = V["ratios"][f"{r}_D"]
        vc += [(f"RD_{r}_s{s}", v, 1e-9) for s, v in blk["per_seed"].items()]
        vc.append((f"RD_{r}_seedlevel", blk["geo"], 1e-9))
        sec = V["secondaries"].get(f"R_C_vs_{r}")
        if sec:
            vc += [(f"RC_{r}_s{s}", v, 5e-5) for s, v in sec["per_seed"].items()]
            vc.append((f"RC_{r}_seedlevel", sec["geo"], 1e-9))
    vc += [(f"T_{k}", v["T"], 0) for k, v in sorted(V["T"].items())]
    for name, v, tol in vc:
        ours = values.get(name)
        ok = ours is not None and v is not None and abs(float(ours) - float(v)) <= tol
        rows.append(("rebuild (Vera, skeptic)", "-", name, "none",
                     f"{v:.6g} vs ours {ours if ours is None else format(ours, '.6g')}: " + ("MATCH" if ok else "MISMATCH"),
                     art))
    rows.append(("rebuild (Vera, skeptic)", "-", "status + prediction", "none",
                 f"{V['status']} / {V['prediction']} vs ours {A['status']} / {A['prediction']['outcome']}: "
                 + ("MATCH" if (V["status"], V["prediction"]) == (A["status"], A["prediction"]["outcome"]) else "MISMATCH"),
                 art))

hd = A["ratios"]["D"]
lines = ["# Proof matrix — tell-chain-8b", "",
         f"Status (frozen rule, PREREG.md section 3): **{A['status']}** (rule: {A['status_rule']}, gates ok: {A['gates_ok']}). "
         f"Prediction outcome: {A['prediction']['outcome']}. Built by make_matrix.py from data/analysis.json.", ""]
for r in ("ctrl", "fresh"):
    b = hd[r]
    if b.get("geo") is not None:
        lines.append(f"seed-level: {b['geo']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] (R_D vs {r}, geometric mean, "
                     f"t-interval on ln R over n {b['n']} paired seeds, df n-1)")
lines += ["", "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
lines += [f"| {' | '.join(str(x) for x in r)} |" for r in rows]
(HERE / "proof-matrix.md").write_text("\n".join(lines) + "\n")
print(f"proof-matrix.md: {len(rows)} rows, {len(list(CELLS.glob('*.json')))} cell files")
