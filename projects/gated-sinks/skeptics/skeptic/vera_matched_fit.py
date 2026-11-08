"""Vera: law fit on the registered matched-loss picks (all 12), own code from raw npz."""
import json, os
from vera_rebuild import rebuild_one
from vera_status import law_status
R = "../../kran-research/projects/gated-sinks/runs"
picks = {"softmax_s0": "softmax_s0/acts_L2_m256_k8_c4200.npz", "softmax_s1": "softmax_s1_fine/acts_L2_m256_k8_c510.npz",
         "softmax_s2": "softmax_s2/acts_L2_m256_k8_c4300.npz", "softmax_s3": "softmax_s3/acts_L2_m256_k8_c600.npz",
         "softmax_s4": "softmax_s4_fine/acts_L2_m256_k8_c510.npz", "softmax1_s0": "softmax1_s0/acts_L2_m256_k8_c4600.npz",
         "sinklogit_s0": "sinklogit_s0/acts_L2_m256_k8_c4200.npz", "kbias_s0": "kbias_s0/acts_L2_m256_k8_c3600.npz",
         "kbias_s1": "kbias_s1_fine/acts_L2_m256_k8_c610.npz", "gated_s0": "gated_s0/acts_L2_m256_k8_c2600.npz",
         "gated_s1": "gated_s1/acts_L2_m256_k8_c2400.npz", "gated_s2": "gated_s2/acts_L2_m256_k8_c2500.npz"}
ms = []
for run, rel in picks.items():
    p = os.path.join(R, rel)
    m = json.load(open(p.replace("acts_", "measure_").replace(".npz", ".json")))
    r = rebuild_one(p, p.replace("acts_", "measure_").replace(".npz", ".json"), own_steps=1)
    ms.append(dict(run=run, arm=run.split("_")[0], gated=run.startswith("gated"), sink=r["D_sink"],
                   recovery=r["A_build_sae_subbdec"]["recovery"], json_rec=m["recovery"], loss=m["eval_loss"]))
    print(run, rel.split("/")[-1], round(r["D_sink"], 4), round(ms[-1]["recovery"], 4), "json", round(m["recovery"], 4), "loss", round(m["eval_loss"], 4))
st = law_status(ms)
print(st["status"], "b", round(st["b"], 4), "CI", [round(v, 4) for v in st["b_ci"]], "s_tot", round(st["s_tot"], 4), "R", round(st["R"], 4), "swing_hi", round(st["swing_hi"], 4), "n_in", st["n_in"])
json.dump(dict(models=ms, status=st), open("vera_matched_fit.json", "w"), indent=1, default=float)
