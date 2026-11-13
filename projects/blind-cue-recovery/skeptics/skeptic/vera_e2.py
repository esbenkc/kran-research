"""Vera: E2 (PREREG.md E2, frozen 06:32 PST) rebuilt with own code. Parent sweep runs sw_*_s0 (cue u1110, rows 0-2,
bf16 cast), reads T/4, T/2, T. RAW top-1 and E1-RAW (project off own top input direction). Beats = (a) max of 1,000
random cues at the run's alpha (seed 20261112) AND (b) the true cue on c2_a0_s0 / c2_a0_s1 at the snapshot nearest in
log updates, same treatment AND (c) slice-only p95 (200 Z, seed 20261113) from the run's A_t [projected for E1-RAW].
"""
import glob
import json
import os

import numpy as np
import torch

from vera_check import slice_null
from vera_lib import T8, cue_diff, e1, load, remainder, v1

P = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit/raw/runs")
OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = [("sw_a0.00390625_s0", 192), ("sw_a0.00195312_s0", 320), ("sw_a0.000976562_s0", 704),
        ("sw_a0.000488281_s0", 1728), ("sw_a0.000244141_s0", 4352)]
U = np.load(f"{T8}/data/cue_u.npy").astype(np.float64)
R = np.random.default_rng(20261112).standard_normal((1000, 3, 1024))
R /= np.linalg.norm(R, axis=-1, keepdims=True)
DEV = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit/data/runs/")


def snaps(run):
    return sorted(int(os.path.basename(p)[4:-3]) for p in glob.glob(f"{P}/{run}/step*.pt"))


def nearest(run, t):
    s = [x for x in snaps(run) if x > 0]
    return min(s, key=lambda x: abs(np.log(x) - np.log(t)))


def read(M, P_=None):
    return v1(M @ P_ if P_ is not None else M)[0]


def proj_off(w):
    return torch.eye(1024, dtype=torch.float64) - torch.outer(w, w)


res = {}
for run, T in RUNS:
    a = json.load(open(f"{P}/{run}/args.json"))
    alpha = a["cue_size"]["alpha_eff"]
    cue = cue_diff(U, alpha)
    nulls = cue_diff(R, alpha)
    dev = {e["step"]: e["D"] for e in json.load(open(DEV + run + ".json"))["dev_curve"]}
    for lab, t in (("T/4", T // 4), ("T/2", T // 2), ("T", T)):
        if t not in snaps(run):
            print(f"{run} {lab} t={t}: no snapshot")
            continue
        sd = load(f"{P}/{run}/step{t}.pt")
        M = remainder(sd, [])
        w = v1(M)[0]
        Pw = proj_off(w)
        clean = [(c, nearest(c, t)) for c in ("c2_a0_s0", "c2_a0_s1")]
        row = {"t": t, "alpha": alpha, "D": dev.get(t), "clean_snaps": clean}
        for name, v, Pn in (("RAW", w, None), ("E1-RAW", read(M, Pw), Pw)):
            et = float(e1(v, cue)[0])
            ea = e1(v, nulls)
            hb = []
            for c, tc in clean:
                Mc = remainder(load(f"{P}/{c}/step{tc}.pt"), [])
                vc = v1(Mc)[0] if name == "RAW" else read(Mc, proj_off(v1(Mc)[0]))
                hb.append(float(e1(vc, cue)[0]))
            p95 = float(np.quantile(slice_null(sd, cue, P=Pn), 0.95))
            rank = int(1 + (ea >= et).sum())
            beats = bool(rank == 1 and et > max(hb) and et > p95)
            row[name] = {"e1": et, "rank_a": rank, "max_a": float(ea.max()), "max_b": max(hb), "p95_c": p95,
                         "beats": beats}
        res[f"{run}|{lab}"] = row
        print(f"{run} x={a['cue_size']['x']:.3f} {lab:3s} t={t:>5} D={row['D'] if row['D'] is None else round(100 * row['D'], 1)} "
              f"RAW rank {row['RAW']['rank_a']:>4} beats {row['RAW']['beats']} | E1-RAW rank {row['E1-RAW']['rank_a']:>4} "
              f"beats {row['E1-RAW']['beats']} (b {row['E1-RAW']['max_b']:.4f} c {row['E1-RAW']['p95_c']:.4f} "
              f"e1 {row['E1-RAW']['e1']:.4f}) clean {clean}", flush=True)
json.dump(res, open(f"{OUT}/vera_e2.json", "w"), indent=1)
