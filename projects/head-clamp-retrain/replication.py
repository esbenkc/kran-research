"""ENGINE REPLICATION (PREREG v2 + D3): our clamped s0 ALL-32 SWAP dev curve vs the parent's head-free cell
(probe-switch-audit-penalty data/runs/hf_s0_2048.json, curve[].mon.switch_all32cut = the same model scored WITH all
32 heads swapped; copied to data/parent/hf_s0/). Pass = both stay <= 0.11 at every check to 2,048 updates and both
train CE stay near ln 2 (0.693).

    $PY replication.py  -> data/replication_hf_s0.json
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
ours = json.load(open(HERE / "data" / "runs" / "all32_swap_s0.json"))
hf = json.load(open(HERE / "data" / "parent" / "hf_s0" / "hf_s0_2048.json"))
o = [(c["step"], c["switch_correct"], c["shift_mean"]) for c in ours["dev_curve"]]
h = [(c["step"], c["mon"]["switch_all32cut"]) for c in hf["curve"]]
o2048 = [x for x in o if x[0] <= 2048]
loss = ours["loss"]
out = {"ours_dev_all32": o, "hf_s0_dev_all32cut": h,
       "ours_max_to_2048": max(x[1] for x in o2048), "ours_max_to_4096": max(x[1] for x in o),
       "hf_max_to_2048": max(x[1] for x in h),
       "ours_train_ce_mean_after_256": sum(loss[256:]) / len(loss[256:]), "ln2": math.log(2),
       "ours_max_abs_push": max(abs(x[2]) for x in o)}
out["pass"] = out["ours_max_to_2048"] <= 0.11 and out["hf_max_to_2048"] <= 0.11 and \
    abs(out["ours_train_ce_mean_after_256"] - math.log(2)) < 0.01
(HERE / "data" / "replication_hf_s0.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if not isinstance(v, list)})
