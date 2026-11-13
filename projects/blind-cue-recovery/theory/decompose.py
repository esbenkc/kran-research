"""Cue-free decomposition (Iris): how much of an adapter's early delta can a slice-matched twin reproduce?

For a target delta W (stacked block-26 q,k,v, dW = 2 B A) at update t and its K = 4 SM twins T_1..T_4 (A := target's
A_t, B = 0, other data seeds, t updates; Bruno's raw/runs/sm_<tgt>_t<t>_d<seed>/step<t>.pt), model
    W = mu + n,   T_k = mu + n_k,   n, n_k independent with the same expected energy sigma^2,
where mu is the part fixed by the slice and the reward (what any run from this A learns) and n is the part fixed by
the run's own prompt order and samples. Estimators:
    sigma^2  = (1/(K-1)) sum_k ||T_k - Tbar||^2                      (spread among the twins)
    data-order share of the target = sigma^2 / ||W||^2                (1 - this = the share a twin can reproduce)
    predicted residual ratio (no cue) = sqrt(sigma^2 (1 + 1/K) / ||W||^2)
    measured residual ratio            = ||W - Tbar|| / ||W||
For a no-cue target (h6..h9) the two residual ratios should agree; for a cue target (m0..m5) an excess over the
prediction is energy the twins do not explain (cue-related or a larger generic term, Vera #21763).
Also |cos(v1(W), v1(Tbar))|. No cue vector is read.
Run: ../../../env/.venv/bin/python theory/decompose.py -> theory/decompose.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "raw/runs"
PRE = "base_model.model.layers.0.self_attn."
MODS = ["q_proj", "k_proj", "v_proj"]
TGT = {f"m{k}": f"dir_u{1110 + k}_a0.03125_s{k}" for k in range(6)}
TGT.update({f"h{k}": f"nc_s{k}" for k in range(6, 10)})


def dW(f):
    sd = torch.load(f, map_location="cpu", weights_only=False)
    return np.concatenate([2.0 * (sd[PRE + m + ".lora_B.default.weight"].double()
                                  @ sd[PRE + m + ".lora_A.default.weight"].double()).numpy() for m in MODS], 0)


def v1(M):
    return np.linalg.svd(M, full_matrices=False)[2][0]


groups = defaultdict(list)
for p in RUNS.iterdir():
    m = re.fullmatch(r"sm_(m\d|h\d)_t(\d+)_d(\d+)", p.name)
    if m and p.is_dir() and (p / f"step{m.group(2)}.pt").exists():
        groups[(m.group(1), int(m.group(2)))].append(p / f"step{m.group(2)}.pt")

rows = []
for (tg, t), files in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    if len(files) < 4:
        continue
    tf = RUNS / TGT[tg] / f"step{t}.pt"
    if not tf.exists():
        continue
    W = dW(tf)
    Ts = np.stack([dW(f) for f in sorted(files)])
    K = len(Ts)
    Tbar = Ts.mean(0)
    sig2 = float(np.sum((Ts - Tbar) ** 2) / (K - 1))
    w2 = float(np.sum(W ** 2))
    rows.append({"target": tg, "run": TGT[tg], "t": t, "K": K,
                 "data_order_share": sig2 / w2,
                 "resid_ratio_pred_nocue": float(np.sqrt(sig2 * (1 + 1 / K) / w2)),
                 "resid_ratio_measured": float(np.linalg.norm(W - Tbar) / np.sqrt(w2)),
                 "twin_norm_over_target": float(np.mean([np.linalg.norm(x) for x in Ts]) / np.sqrt(w2)),
                 "cos_v1_target_twinmean": abs(float(v1(W) @ v1(Tbar)))})
json.dump(rows, open(HERE / "decompose.json", "w"), indent=1)
print(f"{'tgt':4} {'t':>4} {'order share':>11} {'resid pred':>10} {'resid meas':>10} {'twin/tgt':>8} {'cos v1':>7}")
for r in rows:
    print(f"{r['target']:4} {r['t']:>4} {r['data_order_share']:11.3f} {r['resid_ratio_pred_nocue']:10.3f} "
          f"{r['resid_ratio_measured']:10.3f} {r['twin_norm_over_target']:8.3f} {r['cos_v1_target_twinmean']:7.3f}")
