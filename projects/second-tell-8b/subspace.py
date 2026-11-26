"""PREREG section 3, S4: principal angles between LoRA output subspaces (secondary read, never the status).

For each stage-2 tail (arm, seed) at size kB:
  dW2 = s * (B A)_stage-2-end - s * (B A)_stage-2-start   (end = the adapter at its parent early-stop point,
        raw/adapters/<arm>_s<seed>_k<kB>_stop.pt; a tail without a stop snapshot uses _k<kB>_end.pt; start = its
        stage-1 adapter <arm>_s<seed>_s1.pt, or the seed's LoRA init (B = 0, so s * B A = 0) for FRESH)
  dW1 = s * (B A) of MAIN's stage-1 adapter for the same seed (stage 1 starts at B = 0)
  on o_proj and down_proj of blocks 34 and 35 (tail layers 0, 1); s = lora_alpha / r = 2.
Output subspace of a delta = its top-16 left singular vectors (dW2 can have rank up to 32). Principal angles between
the two 16-dim subspaces: arccos of the singular values of U1^T U2 (degrees). Per tail and module: the smallest angle
and the mean of the 16 angles; per tail: the mean over the 4 modules of the mean angle.
The goal's clause holds if MAIN's per-tail mean angle is below that of SHUF-A and of COIN for at least 3 of seeds 5-7.
Null = FRESH's dW2 vs MAIN's dW1 (same seed). Random reference: two random 16-dim subspaces of R^4096 (numpy seed 0).

    .venv/bin/python subspace.py [--kB 4]   # -> data/subspace_k4.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path(__import__("os").environ.get("TT_ROOT", str(HERE)))
AD = ROOT / "raw" / "adapters"
MODS = [(0, "self_attn.o_proj"), (0, "mlp.down_proj"), (1, "self_attn.o_proj"), (1, "mlp.down_proj")]
SCALE, K = 2.0, 16


def factors(ad, li, mod):
    """(B [out, r], A [r, in]) float64 for tail layer li and module path mod."""
    pre = f"layers.{li}.{mod}."
    a = [v for k, v in ad.items() if pre in k and ".lora_A." in k]
    b = [v for k, v in ad.items() if pre in k and ".lora_B." in k]
    assert len(a) == 1 and len(b) == 1, (li, mod, len(a), len(b))
    return b[0].double().numpy(), a[0].double().numpy()


def top_left(Bs, As, k=K):
    """Top-k left singular vectors of sum_i Bs[i] @ As[i] without forming the out x in matrix."""
    Bc, Ac = np.concatenate(Bs, 1), np.concatenate(As, 0)  # [out, m], [m, in]
    Q, Rb = np.linalg.qr(Bc)
    U, S, _ = np.linalg.svd(Rb @ Ac, full_matrices=False)
    return Q @ U[:, :k], S[:k]


def angles(U1, U2):
    c = np.clip(np.linalg.svd(U1.T @ U2, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(c))  # ascending angles (descending cosines)


def load(name):
    f = AD / name
    return torch.load(f, map_location="cpu"), hashlib.sha256(f.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kB", type=int, default=4)
    a = ap.parse_args()
    shas, out = {}, {"kB": a.kB, "modules": [f"block{34 + li}.{m}" for li, m in MODS], "tails": {}}
    rng = np.random.default_rng(0)
    rnd = [angles(np.linalg.qr(rng.standard_normal((4096, K)))[0], np.linalg.qr(rng.standard_normal((4096, K)))[0])
           for _ in range(20)]
    out["random_reference"] = {"mean_angle": float(np.mean([r.mean() for r in rnd])),
                               "smallest_angle": float(np.mean([r.min() for r in rnd]))}
    for seed in range(5, 10):
        f_main1 = f"main_s{seed}_s1.pt"
        if not (AD / f_main1).exists():
            continue
        m1, shas[f_main1] = load(f_main1)
        U1 = {mm: top_left([SCALE * factors(m1, *mm)[0]], [factors(m1, *mm)[1]])[0] for mm in MODS}
        for arm in ("main", "shuf", "coin", "fresh"):
            end = f"{arm}_s{seed}_k{a.kB}_stop.pt"
            if not (AD / end).exists():
                end = f"{arm}_s{seed}_k{a.kB}_end.pt"
            if not (AD / end).exists():
                continue
            e, shas[end] = load(end)
            st = None
            if arm != "fresh":
                st, shas[f"{arm}_s{seed}_s1.pt"] = load(f"{arm}_s{seed}_s1.pt")
            per = {}
            for mm in MODS:
                Be, Ae = factors(e, *mm)
                Bs, As = [SCALE * Be], [Ae]
                if st is not None:
                    Bo, Ao = factors(st, *mm)
                    Bs.append(-SCALE * Bo)
                    As.append(Ao)
                U2, S2 = top_left(Bs, As)
                ang = angles(U1[mm], U2)
                per[f"block{34 + mm[0]}.{mm[1]}"] = {"smallest": float(ang.min()), "mean": float(ang.mean()),
                                                     "angles": [round(float(x), 4) for x in ang],
                                                     "dW2_top_sv": [float(x) for x in S2[:4]]}
            out["tails"][f"{arm}_s{seed}"] = {"arm": arm, "seed": seed, "end_file": end, "modules": per,
                                              "mean_angle": float(np.mean([v["mean"] for v in per.values()])),
                                              "smallest_angle": float(np.mean([v["smallest"] for v in per.values()]))}
    T = out["tails"]
    wins = {}
    for s in (5, 6, 7):
        need = [f"{x}_s{s}" for x in ("main", "shuf", "coin")]
        if all(n in T for n in need):
            wins[str(s)] = T[need[0]]["mean_angle"] < min(T[need[1]]["mean_angle"], T[need[2]]["mean_angle"])
    out["clause"] = {"seeds_5_7_main_below_both": wins, "holds": sum(wins.values()) >= 3}
    out["arm_mean_angle"] = {arm: float(np.mean([v["mean_angle"] for v in T.values() if v["arm"] == arm]))
                             for arm in ("main", "shuf", "coin", "fresh") if any(v["arm"] == arm for v in T.values())}
    out["inputs_sha256"] = shas
    (ROOT / "data" / f"subspace_k{a.kB}.json").write_text(json.dumps(out, indent=1))
    print("arm mean angle (deg)", {k: round(v, 2) for k, v in out["arm_mean_angle"].items()}, "clause", out["clause"],
          "random", out["random_reference"])


if __name__ == "__main__":
    main()
