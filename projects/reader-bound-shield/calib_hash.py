"""Hash (scattered) cue calibration on the START policy (PREREG v1 'Arms'): sigma so that its exact-MC D = D_base.

One process: sample k-token thoughts on the NONE build (per-row Gumbel noise, as read.py), then teacher-force the same
thoughts on the none build + the hash cue at each sigma in a grid. Per row L = sum_t log P_cue(x_t) - log P_off(x_t);
D(sigma) = mean (1 - exp(L))_+ (read.py:11 estimator). sigma* = linear interpolation of D(sigma) at D_target in log-log.
Rows: TRAIN prompts (the test prompts stay for the status reads). Also logs, per sigma, the per-token log-ratio sd and
KL both ways (Monte-Carlo over the off draws, importance-weighted for KL(on||off)) to compare with the real cue.

    env/.venv/bin/python calib_hash.py --D 0.104 --n 64 --R 2 --out data/gate/calib_hash.json
"""
import argparse
import json
import time

import numpy as np
import torch

import mt
from read import gens

HERE = mt.HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="data/runs/pilot/adapter.pt")
    ap.add_argument("--D", type=float, required=True, help="D_target = D_base of the real start read")
    ap.add_argument("--sigmas", default="0.01,0.02,0.03,0.045,0.065")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sig = [float(x) for x in a.sigmas.split(",")]
    pm, inner = mt.load_policy(HERE / a.policy)
    rows = [r for r in mt.prompts() if r["split"] == "train"][: a.n]
    pidx = np.array([int(r["id"][2:]) for r in rows])
    rpi, rr = np.repeat(pidx, a.R), np.tile(np.arange(a.R), len(rows))
    ids_all = torch.tensor([r["ids"] for r in rows], device=mt.DEV).repeat_interleave(a.R, 0)
    N, B = len(rpi), a.B
    S = mt.Sampler(inner, B, a.k)
    one = torch.ones(B, dtype=torch.bool, device=mt.DEV)
    L = {s: [] for s in sig}
    for s0 in range(0, N, B):
        j = list(range(s0, min(s0 + B, N)))
        nreal = len(j)
        j = j + [j[-1]] * (B - nreal)
        o1 = S.rollout(ids_all[j], "none", gen=gens(rpi[j], rr[j]))
        for s in sig:
            o2 = S.rollout(ids_all[j], "none", forced=o1["toks"], hcue=(s, one))
            L[s].append((o2["lp_tok"] - o1["lp_tok"])[:nreal].cpu().double().numpy())
        print(f"batch {s0 // B + 1} {time.time() - t0:.0f}s", flush=True)
    out = {"policy": a.policy, "k": a.k, "n_rows": int(N), "D_target": a.D, "grid": []}
    for s in sig:
        lt = np.concatenate(L[s])  # [rows, k] per-token log-ratio log P_cue - log P_off on off draws
        Ls = lt.sum(1)
        w = np.exp(Ls)
        out["grid"].append({"sigma": s, "D": float(np.maximum(0, 1 - w).mean()), "mean_w": float(w.mean()),
                            "tok_logratio_sd": float(lt.std()), "kl_off_on": float((-Ls).mean()),
                            "kl_on_off": float((w * Ls).mean() / w.mean())})
    g = out["grid"]
    xs, ys = np.log([r["sigma"] for r in g]), np.log([max(r["D"], 1e-9) for r in g])
    out["sigma_star"] = float(np.exp(np.interp(np.log(a.D), ys, xs)))
    out["calib_s"] = round(time.time() - t0, 1)
    (HERE / a.out).parent.mkdir(parents=True, exist_ok=True)
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
