"""Vera: rebuild the registered band from build's raw calibration arrays (own code), check the split matches my
own data rebuild, and run band-selection nulls.
Usage: python vera_band.py
"""
import json, math, sys
import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))  # swarm dir: ../data ; kran-research skeptics/skeptic: ../../data
D = (HERE + "/../data/" if os.path.isdir(HERE + "/../data") else HERE + "/../../data/") + "Qwen3-1.7B/"
band = json.load(open(D + "band.json"))


def rule(G):
    o = np.argsort(-G, kind="stable")
    cs = np.cumsum(G[o])
    k = int(np.searchsorted(cs, 0.5 * G.sum()) + 1)
    return sorted(o[:k].tolist()), k


out = {}
for nm in ["cal", "cal2"]:
    z = np.load(D + f"calib_{nm}.npz")
    s = z["sink"].mean(0)  # [28,16] mean over prompts (each prompt already mean over t>=16)
    cbar = z["gap"].mean(0)  # [28,16,64]
    G = (s[..., None] * cbar).sum((0, 1))
    B, k = rule(G)
    tot = G.sum()
    o = np.argsort(-G)
    out[nm] = dict(B=B, k=k, Gsum=float(tot), share_B=float(G[B].sum() / tot), S=float(s.mean()),
                   n_sinkheads=int((s >= 0.5).sum()), top6=[(int(j), round(float(G[j] / tot), 3)) for j in o[:6]],
                   n_neg=int((G < 0).sum()), docs=z["docs"][:4].tolist())
    if nm == "cal":
        Gcal = G
        print("max |G_mine - G_build| =", float(np.abs(G - np.array(band["G"])).max()), "max |G| =", float(np.abs(G).max()))
        # per-prompt bootstrap of the band (resample calibration prompts): how stable is B?
        rng = np.random.default_rng(0)
        cnt = {}
        n = z["sink"].shape[0]
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            Gb = (z["sink"][idx].mean(0)[..., None] * z["gap"][idx].mean(0)).sum((0, 1))
            cnt[tuple(rule(Gb)[0])] = cnt.get(tuple(rule(Gb)[0]), 0) + 1
        out["bootstrap_band"] = sorted(((v, list(k_)) for k_, v in cnt.items()), reverse=True)[:5]
        # alternative weightings
        sh = cbar / np.where(np.abs(cbar.sum(-1, keepdims=True)) > 1e-9, cbar.sum(-1, keepdims=True), np.nan)
        out["alt_headnorm"] = rule(np.nansum(s[..., None] * sh, (0, 1)))
        out["alt_unweighted"] = rule(cbar.sum((0, 1)))
        out["alt_sinkheads_only"] = rule(cbar[s >= 0.5].sum(0))
        # per-layer bands
        out["per_layer"] = {L: (rule((s[L][:, None] * cbar[L]).sum(0)) if (s[L][:, None] * cbar[L]).sum() > 0 else "no band")
                            for L in range(28)}
        # energy: e_j per pair, rank of band pairs
        e = z["energy"].mean(0).mean((0, 1))
        out["energy_rank_of_B"] = {j: int((e > e[j]).sum()) + 1 for j in B}
        out["energy_e_B_vs_build"] = (float(e[B].sum()), "build e_B 32.8")
for k_, v in out.items():
    print(k_, v)
json.dump(out, open("vera_band.json", "w"), indent=1, default=str)
