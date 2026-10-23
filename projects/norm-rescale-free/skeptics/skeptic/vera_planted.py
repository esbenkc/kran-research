"""Vera (1023): planted-signal calibration of the norm-restore share R (does R see a rescale when one is there?).
Start from A s0's clean states Hc and step-300 attacked states Ha. Build synthetic attacked states with a KNOWN mix:
  turn part: Ht = Hc + f * (Ha - Hc)          (a fraction f of the real fine-tune's edit)
  rescale part: false statements multiplied by k_i > 0 chosen so the final gap closes to the real attack's g_att.
Known share of the gap drop carried by the rescale (forward order): (g(Ht) - g_target) / (g_clean - g_target).
Report R (frozen, end-state restore), R_forward and R_shapley next to the planted share.
"""
import json

import numpy as np

from vera_core import EVF, HERE, PAIRS, PROJ, gap, probe, restore, score

D = PROJ / "data/runs/r28__lam10000__seed0"


def h28(step):
    z = np.load(D / f"ckpt_{step}.npz"); a = np.load(D / f"acts_{step}.npy", mmap_mode="r")
    return np.asarray(a[:, list(z["acts_readouts"]).index(28)], np.float64)


Hc, Ha = h28(0), h28(300)
mu, sd, w, b = probe(28); v = w / sd; b0 = b - v @ mu
gc, gt = gap(score(Hc, 28)), gap(score(Ha, 28))
out = []
for f in (0.0, 0.25, 0.5, 0.75, 0.9):
    Ht = Hc + f * (Ha - Hc)
    st = score(Ht, 28)
    gturn = gap(st)
    # one common k for all false statements so that the gap reaches the real g_att: k * mean(a_f) - mean(a_t) ... solve exactly
    af = Ht[PAIRS[:, 0]] @ v; at = Ht[PAIRS[:, 1]] @ v
    k = (gt + at.mean()) / af.mean()  # mean((k af + b0) - (at + b0)) = gt
    Hp = Ht.copy(); Hp[PAIRS[:, 0]] *= k
    gp = gap(score(Hp, 28))
    planted = (gturn - gp) / (gc - gp)
    gr = gap(score(restore(Hp, Hc), 28))
    R = (gr - gp) / (gc - gp)
    nc, na = np.linalg.norm(Hc, axis=1), np.linalg.norm(Hp, axis=1)
    gfwd = gap(score(Hc * (na / nc)[:, None], 28))
    Rf = (gc - gfwd) / (gc - gp)
    uc, ua = (Hc @ v) / nc, (Hp @ v) / na
    npart = (na - nc) * (uc + ua) / 2; ds = Hp @ v - Hc @ v
    Rs = (npart[PAIRS[:, 0]] - npart[PAIRS[:, 1]]).sum() / (ds[PAIRS[:, 0]] - ds[PAIRS[:, 1]]).sum()
    row = {"turn_fraction_f": f, "k_false": float(k), "planted_share": float(planted), "R": float(R), "R_forward": float(Rf),
           "R_shapley": float(Rs), "g_check": [float(gp), float(gt)]}
    out.append(row); print(row)
json.dump(out, open(HERE / "vera_planted.json", "w"), indent=1)
