"""M3 (mini, 2026-11-22) checks: C2 seed 6 max |gap| over all evals; S9 cell m8u1127_k5_s7 args; its evals vs the
partial laptop log data/laptop/R3/l8u1127_k5_s7.log (in git before M3). Usage: python vera_m3.py <project dir>"""
import json, re, ast, sys
P = sys.argv[1]
d = json.load(open(P + "/data/runs/m8_c2_s6.json")); pts = d["dev_curve"] + d["dense_curve"]
print("C2 s6: cue_alpha", d["args"].get("cue_alpha"), "seed", d["args"]["seed"], "machine", d.get("machine"), "steps_run", d.get("steps_run"),
      "n evals", len(pts), "max |px_off-px_on|", max(abs(e["px_off"] - e["px_on"]) for e in pts), "last", max(e["step"] for e in pts))
m = json.load(open(P + "/data/runs_explore/m8u1127_k5_s7.json"))
print("S9 m8u1127_k5_s7: machine", m.get("machine"), "share", m["args"]["share"], "alpha", m["args"]["cue_alpha"], "u", m["args"]["u_file"],
      "x_offset", m["args"]["x_offset"], "T_dense", m.get("T_dense"), "T(64-grid)", m.get("T"), "steps_run", m.get("steps_run"))
lap = {}
for line in open(P + "/data/laptop/R3/l8u1127_k5_s7.log"):
    mm = re.match(r".*?step (\d+) (\{[^{}]*\})", line.strip())
    if mm:
        e = ast.literal_eval(mm.group(2)); lap[e["step"]] = e
mini = {e["step"]: e for e in m["dev_curve"] + m["dense_curve"]}
sh = sorted(set(lap) & set(mini)); eq = sum(all(lap[s][k] == mini[s][k] for k in ("px_off", "px_on")) for s in sh)
print(f"laptop partial log vs mini JSON: laptop last {max(lap)}, shared evals {len(sh)}, bit-equal {eq}")
