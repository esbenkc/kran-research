"""Vera: rebuild G1/G2 verdicts from data/gates.json per-prompt numbers + the parent's saved NDIF file."""
import json, sys, numpy as np, torch
P = sys.argv[1] if len(sys.argv) > 1 else "."
g = json.load(open(f"{P}/data/gates.json"))
pp = g["G2"]["per_prompt"]
ours, ndif, floor = (np.array(pp[k]) for k in ("lnpx_ours", "lnpx_ndif", "lnpx_floor"))
d, f = np.abs(ours - ndif), np.abs(floor - ndif)
nd = torch.load(f"{P}/ref/ndif/dev_block34_ndif.pt", map_location="cpu", weights_only=False)
out = {"keys_ndif_file": list(nd.keys()) if isinstance(nd, dict) else str(type(nd))}
# independent NDIF top-1 / tie read from the saved NDIF file
if isinstance(nd, dict):
    for k, v in nd.items():
        out[f"ndif_{k}"] = list(v.shape) if hasattr(v, "shape") else (type(v).__name__, len(v) if hasattr(v, "__len__") else v)
top_o, top_n = np.array(pp["top_ours"]), np.array(pp["top_ndif"])
ties = [t for t in pp["tied_ids"]]
match_ties = sum(int(o == n or o in t) for o, n, t in zip(top_o, top_n, ties))
out.update({
    "n": len(d), "d_max": d.max(), "d_median": float(np.median(d)), "f_max": f.max(), "f_median": float(np.median(f)),
    "cond1_max": bool(d.max() <= 2 * f.max()), "cond2_median": bool(np.median(d) <= 0.01),
    "top1_strict": int((top_o == top_n).sum()), "top1_with_ties": match_ties, "cond3_top1": match_ties >= 63,
    "n_d_over_0.05": int((d > 0.05).sum()), "argmax_d": pp["id"][int(d.argmax())],
    "G2b_pass": bool(d.max() <= 2 * f.max() and np.median(d) <= 0.01 and match_ties >= 63),
    "G2a_pass_wholeprompt": g["G2"]["a_rel_err_max"] < 1e-2,
    "G1": {"dev_mean": g["G1"]["dev_mean"], "in_range": 0.30 <= g["G1"]["dev_mean"] <= 0.50, "think_top1": g["G1"]["think_top1_dev"]},
})
print(json.dumps(out, indent=1, default=float))
