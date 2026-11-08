"""Vera: assembly-layer row from my own rebuilds. Assembly layer per model = first h_L with readout R^2 >= 0.9 x max
(readout by layer from build's ridge file results/readout_layers.json; the rule is Vera #17320). Recovery at that layer
from my rebuild of the saved npz; x = first-token mass of block L from A0_all_blocks."""
import json, os
import numpy as np
from scipy import stats
from vera_status import law_status
RL = json.load(open("../../kran-research/projects/gated-sinks/results/readout_layers.json"))
rl = RL.get("models", RL) if isinstance(RL, dict) else RL
rows = []
for run in ["softmax_s0","softmax_s1","softmax_s2","softmax_s3","softmax_s4","softmax1_s0","sinklogit_s0","kbias_s0","kbias_s1","gated_s0","gated_s1","gated_s2"]:
    by = RL[f"{run}:12000"]["readout_r2_by_layer"]
    by = np.array(by[:5] if len(by) >= 5 else by)
    L = int(np.argmax(by >= 0.9 * by.max()))
    f = {2: f"vr_{run}.json", 3: f"vr3_{run}.json", 4: f"vr4_{run}.json"}.get(L)
    if f is None or not os.path.exists(f):
        rows.append(dict(run=run, L=L, missing=f)); continue
    r = json.load(open(f))[0]
    x = r["D_sink_blocks"][L - 1] if "D_sink_blocks" in r else r["D_sink"]
    rows.append(dict(run=run, arm=r["arm"], gated=r["arm"] == "gated", L=L, readout=[round(v, 3) for v in by], sink=x,
                     recovery=r["A_build_sae_subbdec"]["recovery"], rec_own=r["B_own_sae"]["recovery"]))
for x in rows: print(x)
ok = [x for x in rows if "recovery" in x]
st = law_status(ok); st_own = law_status([dict(x, recovery=x["rec_own"]) for x in ok])
print("assembly row (build SAEs):", st["status"], "b", round(st["b"], 4), "CI", [round(v, 4) for v in st["b_ci"]], "swing_hi", round(st["swing_hi"], 4))
print("assembly row (own SAEs):", st_own["status"], "b", round(st_own["b"], 4), "CI", [round(v, 4) for v in st_own["b_ci"]], "swing_hi", round(st_own["swing_hi"], 4))
json.dump(dict(rows=rows, status_build=st, status_own=st_own), open("vera_assembly.json", "w"), indent=1, default=float)
