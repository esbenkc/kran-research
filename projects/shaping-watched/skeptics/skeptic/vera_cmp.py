# Vera: per-prompt diff of my independent read (reads/<run>.npz) vs Bruno's raw status logits.
import sys, json, os, numpy as np
SUF = os.environ.get("SUF", "")
D = "kran-research/projects/shaping-watched/data/status/"
out = {}
for run in sys.argv[1:]:
    v = np.load(f"swarm-oracle-1106-shaping-watched/skeptic/reads/{run}{SUF}.npz")
    o = {}
    for b in ("none", "ref", "score"):
        z = np.load(f"{D}{run}_raw/{b}_plain.npz")
        assert (z["prompt_ids"] == v["prompt_ids"]).all()
        dv = v[f"{b}_l_read"] - v[f"{b}_l_solve"]; dz = z["l_read"] - z["l_solve"]
        o[b] = {"bit_equal_l_read": int((v[f"{b}_l_read"] == z["l_read"]).sum()),
                "bit_equal_l_solve": int((v[f"{b}_l_solve"] == z["l_solve"]).sum()),
                "max_abs_dD": float(np.abs(dv - dz).max()), "corr_D": float(np.corrcoef(dv, dz)[0, 1])}
    # footprint agreement: (ref - none) in my read vs in Bruno's
    fv = (v["ref_l_read"] - v["ref_l_solve"]) - (v["none_l_read"] - v["none_l_solve"])
    zr = np.load(f"{D}{run}_raw/ref_plain.npz"); zn = np.load(f"{D}{run}_raw/none_plain.npz")
    fz = (zr["l_read"] - zr["l_solve"]) - (zn["l_read"] - zn["l_solve"])
    o["footprint_corr_mine_vs_build"] = float(np.corrcoef(fv, fz)[0, 1]) if fv.std() > 0 and fz.std() > 0 else None
    o["footprint_bit_equal_prompts"] = int((fv == fz).sum())
    out[run] = o
print(json.dumps(out, indent=1))
json.dump(out, open(f"swarm-oracle-1106-shaping-watched/skeptic/vera_cmp{SUF}.json", "w"), indent=1)
