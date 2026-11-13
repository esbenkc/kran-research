"""Cue-free diagnostics (PREREG v2, run before any true-cue read). No cue vector is loaded.

1. Rebuild check: nc_s0 / nc_s1 (this room, --no-eval) vs c2_a0_s0 / c2_a0_s1 (lora-switch-audit) at every shared snapshot.
2. Iris #21730: |cos(v1 of held-out no-cue target h6 at t = 16, v1 of the mean of its 4 SM twins)|; < 0.9 means the twin
   does not reproduce the generic direction. Also for h7-h9 and every read that has 4 twins (reported, no bar).
    python diag.py   -> data/diag.json
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
RUNS = os.environ.get("RUNS_DIR", "raw/runs")  # data/adapters_qkv26 = committed copies (pack.py)
MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]


def load(p):
    return torch.load(HERE / p, map_location="cpu", weights_only=True)


def dW(ad):
    k = "base_model.model.layers.0.{}.lora_{}.default.weight"
    return [2.0 * ad[k.format(m, "B")].double().numpy() @ ad[k.format(m, "A")].double().numpy() for m in MODS]


def v1(Ws):
    return np.linalg.eigh(sum(W.T @ W for W in Ws))[1][:, -1]


out = {"rebuild": {}, "twin_v1": {}}
for new, old in [("nc_s0", "c2_a0_s0"), ("nc_s1", "c2_a0_s1")]:
    for f in sorted((HERE / RUNS / old).glob("step*.pt")):
        g = HERE / RUNS / new / f.name
        if g.exists():
            a, b = load(f.relative_to(HERE)), load(g.relative_to(HERE))
            out["rebuild"][f"{new}/{f.name} vs {old}/{f.name}"] = max(float((a[k] - b[k]).abs().max()) for k in a)
for j in (6, 7, 8, 9):
    for t in (8, 16, 24, 32):
        tw = [HERE / RUNS / f"sm_h{j}_t{t}_d{j + 100 + d}/step{t}.pt" for d in range(4)]
        if not all(p.exists() for p in tw):
            continue
        Wt = dW(load(f"{RUNS}/nc_s{j}/step{t}.pt"))
        Wr = [dW(load(p.relative_to(HERE))) for p in tw]
        mean = [np.mean([w[i] for w in Wr], axis=0) for i in range(3)]
        res = [Wt[i] - mean[i] for i in range(3)]
        out["twin_v1"][f"h{j}@{t}"] = {
            "cos_v1_target_v1_twinmean": float(abs(v1(Wt) @ v1(mean))),
            "cos_v1_target_each_twin": [float(abs(v1(Wt) @ v1(w))) for w in Wr],
            "res_norm_over_target": float(np.sqrt(sum((r ** 2).sum() for r in res) / sum((w ** 2).sum() for w in Wt)))}
(HERE / "data" / ("diag.json" if RUNS == "raw/runs" else "diag_packed.json")).write_text(json.dumps(out, indent=1))
print("rebuild max diff:", max(out["rebuild"].values()) if out["rebuild"] else None, "over", len(out["rebuild"]), "snapshots")
for k, v in out["twin_v1"].items():
    print(k, "cos(v1 target, v1 twin mean) =", round(v["cos_v1_target_v1_twinmean"], 3),
          "| residual norm / target norm =", round(v["res_norm_over_target"], 3))
