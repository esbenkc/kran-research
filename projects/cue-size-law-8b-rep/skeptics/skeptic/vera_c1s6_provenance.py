"""C1 seed 6 provenance (Vera, theory-1122). (1) test gap of the laptop file vs the log line written before
labq 5f3rwdo6um failed ('10:05:25 [B] OK l8_c1_s6 DONE T None test_gap 0.014066543337612529').
(2) bit-equality of the killed mini rerun M2 (log only, stopped at step 2624) vs the laptop file."""
import json, re, ast, sys, statistics as st
run = sys.argv[1]; m2log = sys.argv[2]
d = json.load(open(run)); t = d["test"]
print("stored test gap", repr(t["gap"]), "recomputed", repr(st.fmean(t["px_off"]) - st.fmean(t["px_on"])),
      "log line 0.014066543337612529 equal:", repr(t["gap"]) == "0.014066543337612529")
lap = {e["step"]: e for e in d["dev_curve"] + d["dense_curve"]}
n = eq = 0; last = None
for line in open(m2log):
    m = re.match(r"step (\d+) (\{[^{}]*\})", line.strip())
    if not m: continue
    e = ast.literal_eval(m.group(2)); last = e["step"]
    if last in lap:
        n += 1; eq += all(e[k] == lap[last][k] for k in ("px_off", "px_on"))
print(f"mini M2 last step {last}; shared evals {n}; bit-equal px_off and px_on {eq}/{n}")
