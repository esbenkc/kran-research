"""Vera: onset rebuild from the saved dev curves (PREREG s4). T_obs = first dev check where gap(t) - gap(0) >= 0.10
and the NEXT check is also >= 0.10; none by 8,192 = censored. Scored: T_obs / T_naive in [0.5, 2] in >= 2 of 3 main
seeds (censored = miss). Second: same with T_x33. T_naive recomputed from s with the frozen formula.
usage: python3 vera_onset.py <runs_dir> <t_pred.json> [rows02.npz]
rows02.npz (optional): eager [3, D], compiled [3, D] block-34 inputs at rows 0-2 and w = block-34 input_layernorm
weight; then s is recomputed here instead of read."""
import json, os, sys, math
import numpy as np

runs_dir, tp_path = sys.argv[1], sys.argv[2]
tp = json.load(open(tp_path))
s = None
if len(sys.argv) > 3:
    z = np.load(sys.argv[3])
    bf = lambda u: (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)  # bf16 bits -> value
    he, hc, w, eps = bf(z["eager_bf16_bits"]), bf(z["compiled_bf16_bits"]), z["w_fp32"].astype(np.float64), float(z["eps"])
    same_e = bool((z["eager_bf16_bits"] == z["eager_bf16_bits"][0:1]).all()); same_c = bool((z["compiled_bf16_bits"] == z["compiled_bf16_bits"][0:1]).all())
    print(f"rows 0-2 identical in all {he.shape[0]} dev prompts: eager {same_e}, compiled {same_c}")
    def N(h):
        return h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps) * w
    ng, ns = N(he), N(hc)  # [P, 3, D]
    sp = np.sqrt(((ns - ng) ** 2).sum((1, 2))) / np.sqrt((ng ** 2).sum((1, 2)))  # Frobenius over rows 0-2, per prompt
    s = float(sp.mean())
    sq = ((ns - ng) ** 2).sum(-1).mean(0)
    per_row = (sq / sq.sum()).tolist()
    raw_row = [float(np.abs(hc[:, i] - he[:, i]).max()) for i in range(3)]
    print(f"s recomputed {s:.6g} (job: {tp.get('s')}); squared-offset share rows 0/1/2 {[round(x, 4) for x in per_row]}; "
          f"max |raw diff| rows 0/1/2 {raw_row}; post-norm share per row {[float(np.linalg.norm(ns[:, i] - ng[:, i]) / np.linalg.norm(ng[:, i])) for i in range(3)]}")
s = s if s is not None else float(tp["s"])
t_naive = 24 * (s / 0.05469) ** -0.956
t_x33 = 33.2 * t_naive
print(f"T_naive {t_naive:.1f} (job {tp.get('T_naive')}), T_x33 {t_x33:.1f} (job {tp.get('T_x33')}), extrapolated {not (0.0017 <= s <= 0.0547)}")
hits_n = hits_x = 0
for f in sorted(os.listdir(runs_dir)):
    if not f.endswith(".json") or "_at" in f:
        continue
    d = json.load(open(os.path.join(runs_dir, f)))
    c = sorted(d["dev_curve"], key=lambda r: r["step"])
    g0 = c[0]["gap"] if c[0]["step"] == 0 else None
    T = None
    for a, b in zip(c, c[1:]):
        if a["step"] > 0 and a["gap"] - g0 >= 0.10 and b["gap"] - g0 >= 0.10:
            T = a["step"]; break
    mx = max(r["gap"] - g0 for r in c)
    last = c[-1]["step"]
    line = f"{f:40s} checks {len(c)} last {last} gap0 {g0:+.4f} max dgap {mx:+.4f} T_obs {T}"
    if f.startswith("stack"):
        rn = T / t_naive if T else None; rx = T / t_x33 if T else None
        hn = rn is not None and 0.5 <= rn <= 2; hx = rx is not None and 0.5 <= rx <= 2
        hits_n += hn; hits_x += hx
        line += f"  T/T_naive {rn and round(rn, 3)} hit {hn}  T/T_x33 {rx and round(rx, 3)} hit {hx}"
    print(line)
print(f"scored clause (plain law) hits {hits_n}/3 -> {'holds' if hits_n >= 2 else 'fails'}; x33 hits {hits_x}/3 -> {'hit' if hits_x >= 2 else 'miss'}")
