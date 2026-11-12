"""Iris, 2026-11-12 (exploratory, after the first headline cells landed; answers Tove #20355).
Split each run's switch time into (how far LoRA B must grow) / (how fast it grows):
  B_need  = normB(T) - normB(16)       (normB at the switch minus after the alpha-free first 16 updates)
  B_speed = B_need / (T - 16)          (mean growth per update between update 16 and the switch)
T is T_dense (rl.py, union of dev and dense evals); normB is logged in dev_curve (rl.py:355).
Then log-log slopes on the cue size x, per optimiser. If T - 16 = B_need / B_speed, the T slope is the B_need slope minus
the B_speed slope. A step that is linear in the raw gradient (SGD) should give a B_speed slope near 1.
Rebuild: /usr/bin/python3 decompose.py [runs_dir]  -> decompose.json next to this file."""
import glob, json, os, sys
import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/work/theory-1112-optimizer-sets/kran-research/projects/optimizer-cue-size/data/runs")
rows = []
for f in sorted(glob.glob(f"{RUNS}/*.json")):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f)); a = d["args"]
    if a.get("arm") != "cue" or name.startswith(("smoke", "p0cast")) or a.get("freeze_a") or a.get("group", 8) != 8:
        continue
    if name.startswith("sgdpilot") and a.get("lr") != 0.01:  # lr 0.01 pilot = the SGD x_top seed-0 headline cell
        continue
    if a.get("opt") == "sgd" and a.get("lr") != 0.01:
        continue
    T = d.get("T_dense"); c = [r for r in d["dev_curve"] if "normB" in r]
    if not T or T <= 16 or not c:
        continue
    t = np.array([r["step"] for r in c], float); B = np.array([r["normB"] for r in c])
    b16, bT = float(np.interp(16, t, B)), float(np.interp(T, t, B))
    rows.append(dict(run=name, opt=a["opt"], seed=a["seed"], x=d["cue_size"]["x"], T=T, B16=b16, BT=bT,
                     B_need=bT - b16, B_speed=(bT - b16) / (T - 16)))
out = {"runs": rows, "fits": {}}
for opt in sorted({r["opt"] for r in rows}):
    rr = [r for r in rows if r["opt"] == opt]
    if len({round(r["x"], 2) for r in rr}) < 2:
        continue
    lx = np.log([r["x"] for r in rr])
    f = lambda k: float(np.polyfit(lx, np.log([r[k] for r in rr]), 1)[0])
    out["fits"][opt] = {"n": len(rr), "slope_T": f("T"),
                        "slope_T_minus_16": float(np.polyfit(lx, np.log([r["T"] - 16 for r in rr]), 1)[0]),
                        "slope_B_need": f("B_need"), "slope_B_speed": f("B_speed")}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "decompose.json"), "w"), indent=1)
for r in rows:
    print(f"{r['opt']:5s} x {r['x']:5.2f} s{r['seed']} T {r['T']:5d}  B16 {r['B16']:.3f} B_T {r['BT']:.3f} "
          f"need {r['B_need']:.3f} speed {r['B_speed']:.5f}")
for k, v in out["fits"].items():
    print(k, {kk: round(vv, 3) for kk, vv in v.items()})
