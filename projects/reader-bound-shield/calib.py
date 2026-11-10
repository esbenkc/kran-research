"""Planted cue calibration on the START policy (PREREG v1 rev2): token X and bias eps.

X = the token whose mean probability at prefix position 1 (no-probe build, train prompts) is closest to 0.03 among
tokens with mean probability in [0.01, 0.05]. eps solves mean_prompts TV_1(eps) = D_plant, where TV_1 = TV between the
no-probe position-1 distribution and the planted sampler's (leaky build + eps on X); D_plant = D_base clipped to
[0.05, 0.5]. The planted cue lives at position 1 only, so the prefix TV of the planted arm ~ TV_1 + the real cue.

    env/.venv/bin/python calib.py --policy data/runs/pilot/adapter.pt --D 0.08 --out data/gate/calib.json
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer

import mt

HERE = mt.HERE


@torch.no_grad()
def pos1_probs(S, ids, build, B):
    out = []
    for s0 in range(0, len(ids), B):
        j = list(range(s0, min(s0 + B, len(ids))))
        n = len(j)
        j = j + [j[-1]] * (B - n)
        mt.FLAG["leaky"] = build == "leaky"
        S.cache.reset()
        mt.STORE.clear(); mt.REC.clear()
        lg = S.fn(ids[j], torch.arange(mt.T, device=mt.DEV)[None]).float()
        mt.STORE.clear(); mt.REC.clear()
        lg[:, mt.FS:] = -float("inf")
        out.append(torch.softmax(lg, -1)[:n].cpu().double())
    mt.FLAG["leaky"] = False
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--D", type=float, required=True, help="D_base (clipped to [0.05, 0.5] here)")
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noclip", action="store_true", help="use --D as D_plant (Iris's C-reader text at TV 0.03)")
    ap.add_argument("--plant-build", default="leaky", choices=["leaky", "none"],
                    help="build of the probed sampler (none = the cue alone; long-rollout-shield)")
    a = ap.parse_args()
    pm, inner = mt.load_policy(None if a.policy == "base" else HERE / a.policy)
    tok = AutoTokenizer.from_pretrained(mt.MODEL)
    S = mt.Sampler(inner, a.B, 16)
    tr = [r for r in mt.prompts() if r["split"] == "train"]
    ids = torch.tensor([r["ids"] for r in tr], device=mt.DEV)
    pn = pos1_probs(S, ids, "none", a.B)
    pl = pos1_probs(S, ids, a.plant_build, a.B) if a.plant_build == "leaky" else pn.clone()
    mean = pn.mean(0)
    cand = torch.where((mean >= 0.01) & (mean <= 0.05))[0]
    assert len(cand), "no token with mean p in [0.01, 0.05] at position 1"
    X = int(cand[(mean[cand] - 0.03).abs().argmin()])
    D_plant = float(a.D) if a.noclip else float(min(max(a.D, 0.05), 0.5))

    def tv1(eps):
        q = pl.clone()
        q[:, X] = q[:, X] * np.exp(eps)
        q = q / q.sum(1, keepdim=True)
        return float((0.5 * (q - pn).abs().sum(1)).mean())

    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tv1(mid) < D_plant else (lo, mid)
    eps = (lo + hi) / 2
    pX = pn[:, X].numpy()
    top = torch.topk(mean, 10)
    out = {"policy": a.policy, "X": X, "X_text": tok.decode([X]), "mean_pX": float(mean[X]),
           "pX_quartiles": [float(np.percentile(pX, q)) for q in (25, 50, 75)],
           "D_base_in": a.D, "plant_build": a.plant_build, "D_plant": D_plant, "eps": eps, "tv1_at_eps": tv1(eps), "tv1_real_cue": tv1(0.0),
           "candidates": [[int(c), tok.decode([int(c)]), float(mean[c])] for c in cand],
           "top10_pos1": [[int(i), tok.decode([int(i)]), float(p)] for p, i in zip(top.values, top.indices)]}
    (HERE / a.out).parent.mkdir(parents=True, exist_ok=True)
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
