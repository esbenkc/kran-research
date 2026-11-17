"""data/sens_17b.json: 1.7B T_f on the 64 grid (sens_17b.sh) and the band ratios with it (SENSITIVITY, not the frozen
read), plus the bit-identity check of the rerun vs the parent's switch runs at the parent's 128-grid evals."""
import json
from pathlib import Path

HERE = Path(__file__).parent
PAR = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}
out = {"seeds": {}}
band = json.load(open(HERE / "data" / "band_17b.json"))
for s in (0, 1, 2):
    r = json.load(open(HERE / "data" / "sens17b" / "runs" / f"switch_s{s}.json"))
    p = json.load(open(HERE / "inputs" / "parent17" / f"{PAR[s]}.json"))
    mine = {e["step"]: e for e in r["dev_curve"]}
    same, n = 0, 0
    for e in p["dev_curve"]:
        if e["step"] in mine and e["step"] > 0:
            n += 1
            m = mine[e["step"]]
            same += (m["switch_correct"] == e["switch_correct"] and m["shift_mean"] == e["shift_mean"]
                     and abs(r["loss"][e["step"] - 1] - round(e["train_loss"], 6)) < 1e-6)
    ret = band["set"][str(s)]["return_step"]
    out["seeds"][s] = {"T_f_64": r["t_f"], "T_f_parent_128": band["parent_T_f"][str(s)], "set_return_64": ret,
                       "ratio_sens": None if (ret is None or not r["t_f"]) else ret / r["t_f"],
                       "ratio_frozen": band["set"][str(s)]["ratio"], "evals_equal_parent": f"{same}/{n}"}
rs = [v["ratio_sens"] for v in out["seeds"].values()]
if all(x is not None for x in rs):
    w = 64 / min(v["T_f_64"] for v in out["seeds"].values())
    out["band_sens"] = [min(rs) - w, max(rs) + w]
(HERE / "data" / "sens_17b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
