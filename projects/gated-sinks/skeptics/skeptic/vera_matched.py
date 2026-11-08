"""Vera: registered-protocol row (matched-loss picks) + floors, rebuilt from raw npz with my own code."""
import json, os
import numpy as np
from vera_rebuild import rebuild_one

ROOT = "../../kran-research/projects/gated-sinks/runs"
L_STAR = json.load(open("../../kran-research/projects/gated-sinks/results/lstar.json"))
cells = {
    "softmax_s0": "softmax_s0/acts_L2_m256_k8_c4200.npz",
    "softmax_s1": "softmax_s1_fine/acts_L2_m256_k8_c510.npz",
    "softmax_s2": "softmax_s2/acts_L2_m256_k8_c4300.npz",
    "gated_s0": "gated_s0/acts_L2_m256_k8_c2600.npz",
    "gated_s1": "gated_s1/acts_L2_m256_k8_c2400.npz",
    "gated_s2": "gated_s2/acts_L2_m256_k8_c2500.npz",
}
floors = {f"untrained_{r}": f"{r}/acts_untrained_L2_m256_k8.npz" for r in ("softmax_s0", "gated_s0", "kbias_s0", "sinklogit_s0")}
floors["embed_softmax_s0"] = "softmax_s0/acts_embed_L0.npz"
out = {"L_star_file": L_STAR, "matched": {}, "floors": {}}
for group, d in (("matched", cells), ("floors", floors)):
    for name, rel in d.items():
        npz = os.path.join(ROOT, rel)
        if not os.path.exists(npz):
            out[group][name] = "missing " + rel
            continue
        r = rebuild_one(npz, npz.replace("acts_", "measure_").replace(".npz", ".json"), own_steps=3000)
        m = json.load(open(npz.replace("acts_", "measure_").replace(".npz", ".json")))
        out[group][name] = dict(file=rel, ckpt=m.get("ckpt"), eval_loss=m.get("eval_loss"), sink=r.get("D_sink"),
                                rec_build=r["A_build_sae_subbdec"]["recovery"], json_rec=m["recovery"],
                                rec_own=r["B_own_sae"]["recovery"], floor=r["C_random_dict_floor"])
        print(group, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[group][name].items()}, flush=True)
mm = out["matched"]
pairs = []
for s in (0, 1, 2):
    a, b = mm.get(f"softmax_s{s}"), mm.get(f"gated_s{s}")
    if isinstance(a, dict) and isinstance(b, dict):
        pairs.append(dict(seed=s, d_rec=b["rec_build"] - a["rec_build"], d_rec_own=b["rec_own"] - a["rec_own"],
                          sink_ratio=b["sink"] / a["sink"] if a["sink"] else None))
out["registered_pairs_matched"] = pairs
print("registered (matched-loss picks):", pairs)
json.dump(out, open("vera_matched.json", "w"), indent=1, default=float)
