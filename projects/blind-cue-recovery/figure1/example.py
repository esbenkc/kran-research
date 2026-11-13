"""Figure 1 example numbers (weights only, CPU, small): for one cue model and read t, the top singular value and top input
direction of the stacked block-26 q,k,v LoRA delta of (a) the cue model, (b) the mean of its 4 auditor copies (SM twins),
(c) the remainder after subtraction. Same delta convention as blind.py:67 (scale = lora_alpha / r = 2).
Run: env/.venv/bin/python example.py m0 16 > example_m0_t16.json"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

P = Path(__file__).resolve().parents[2] / "kran-research/projects/blind-cue-recovery"
MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
RUN = {f"m{k}": f"dir_u111{k}_a0.03125_s{k}" for k in range(6)}


def deltas(path):
    ad = torch.load(path, map_location="cpu", weights_only=True)
    out = []
    for m in MODS:
        B = ad[f"base_model.model.layers.0.{m}.lora_B.default.weight"].double().numpy()
        A = ad[f"base_model.model.layers.0.{m}.lora_A.default.weight"].double().numpy()
        out.append(2.0 * B @ A)
    return out


def top(Ws):
    G = sum(W.T @ W for W in Ws)
    ev, V = np.linalg.eigh(G)
    return V[:, -1], float(np.sqrt(max(ev[-1], 0.0))), float(np.sqrt(sum((W ** 2).sum() for W in Ws)))


m, t = sys.argv[1], int(sys.argv[2])
tgt = deltas(P / f"raw/runs/{RUN[m]}/step{t}.pt")
tw = [deltas(P / f"raw/runs/sm_{m}_t{t}_d{s}/step{t}.pt") for s in range(int(m[1:]) + 100, int(m[1:]) + 104)]
mean = [sum(x[i] for x in tw) / len(tw) for i in range(3)]
res = [tgt[i] - mean[i] for i in range(3)]
vt, st, nt = top(tgt)
vm, sm, nm = top(mean)
vr, sr, nr = top(res)
print(json.dumps({"model": m, "run": RUN[m], "t": t, "s1_target": st, "s1_twinmean": sm, "s1_resid": sr,
                  "fro_target": nt, "fro_twinmean": nm, "fro_resid": nr,
                  "cos_target_twinmean": abs(float(vt @ vm)), "cos_resid_target": abs(float(vr @ vt)),
                  "proj_twin_on_target_v1": float(np.sqrt(sum(((W @ vt) ** 2).sum() for W in mean))),
                  "proj_resid_on_target_v1": float(np.sqrt(sum(((W @ vt) ** 2).sum() for W in res)))}, indent=1))
