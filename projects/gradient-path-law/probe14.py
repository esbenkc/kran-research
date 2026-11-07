"""D6 (reading only): shaping probe at layers[14].self_attn.q_norm. Fits v14 on the BASE model's q_norm(14) output at
the answer token (read rows = cheat, solve rows = honest; train prompts), re-encoded in the real compiled graph with
the leaky hook at 14 (cache.py --L 14 --passes enc --builds ref, all prompts). Writes data/probe_v14.pt and the score
table data/probe_M14.npy [N, 2] (layers <= 14 are frozen in every site-14 run, so the table is the probe's score).
    python probe14.py
"""
import json
import numpy as np
import torch
from probe import auroc, fit_logistic
from tail import HERE

E = np.load(HERE / "raw/cache/enc14/ref_enc.npy").astype(np.float64)      # [646, 2, 2048] in prompts.jsonl order
split = np.array([json.loads(l)["split"] for l in open(HERE / "raw/prompts.jsonl")])
tr = np.where(split == "train")[0]
X = np.r_[E[tr, 0], E[tr, 1]]; y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
v, b = fit_logistic(X, y)
M = E @ v.numpy()
res = {s: auroc(np.r_[M[split == s, 0], M[split == s, 1]], np.r_[np.ones((split == s).sum()), np.zeros((split == s).sum())])
       for s in ("train", "dev", "test")}
res.update(sd_M0=float(np.r_[M[tr, 0], M[tr, 1]].std()), margin_train=float((M[tr, 0] - M[tr, 1]).mean()))
torch.save({"v": v.float(), "b": b, "sd_M0": res["sd_M0"], "layer": 14, "site": "self_attn.q_norm", "pos": "answer"},
           HERE / "data/probe_v14.pt")
np.save(HERE / "data/probe_M14.npy", M)
json.dump(res, open(HERE / "data/probe14_fit.json", "w"), indent=1)
print(res)
