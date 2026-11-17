"""data/band_17b.json: the frozen 1.7B band. r_s = SET return update (dev grid 64, this code) / the parent's T_f
(1,280 / 1,408 / 1,280 on the parent's 128 grid, inputs/parent17/status_17b.json 'formation'); band = [min r - 64/T_f_min,
max r + 64/T_f_min]."""
import json
from pathlib import Path

HERE = Path(__file__).parent
st = json.load(open(HERE / "inputs" / "parent17" / "status_17b.json"))["runs"]
tf = {s: st[f"set_swap_s{s}"]["formation"] for s in (0, 1, 2)}
out = {"parent_T_f": tf, "set": {}, "top1": {}}
for arm in ("set", "top1"):
    for s in (0, 1, 2):
        r = json.load(open(HERE / "data" / "band17b" / "runs" / f"{arm}_s{s}.json"))
        out[arm][s] = {"return_step": r["return_step"], "stop_reason": r["stop_reason"],
                       "final_test_strict": r["final_test_strict"],
                       "ratio": None if r["return_step"] is None else r["return_step"] / tf[s]}
rs = [out["set"][s]["ratio"] for s in (0, 1, 2)]
assert all(x is not None for x in rs), rs
w = 64 / min(tf.values())
out["band"] = [min(rs) - w, max(rs) + w]
out["rule"] = __doc__.strip()
(HERE / "data" / "band_17b.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
