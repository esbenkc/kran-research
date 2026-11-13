"""Numeric check of the post's displayed equation, written apart from blind.py's code path:

  R_m = dW_m^target - (1/K) sum_k dW_m^(k),   e1(u) = sum_p (v1 . c_p)^2 / sum_p |c_p|^2,
  v1 = top RIGHT singular vector of [R_q; R_k; R_v] (here by SVD; blind.py uses eigh of sum_m R_m^T R_m).

Cells: SM at the status read for every model (K = 4 twins), plus RAW at 256 (K = 0, the phase-0 cell). The cue rows c_p come
from the base blocks (audit.Spaces.planted + embed, the parent's cue map); the 1,000 null cues are numpy seed 20261112.
Compares e1 and the rank with data/summary.json.
    NC_DEV=cpu python post/eq_check.py -> post/eq_check.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from audit import Spaces  # noqa: E402

TH = {0: 16, 1: 16, 2: 16, 3: 16, 4: 8, 5: 16}
KEY = "base_model.model.layers.0.self_attn.{}_proj.lora_{}.default.weight"
summ = json.load(open(HERE / "data" / "summary.json"))["cells"]
sp = Spaces()
h0 = sp.h0
g = np.random.default_rng(20261112).standard_normal((1000, 3, 1024))
U0 = (g / np.linalg.norm(g, axis=-1, keepdims=True)).astype(np.float32)


def dW(path):
    sd = torch.load(HERE / path, map_location="cpu", weights_only=True)
    return [2.0 * sd[KEY.format(m, "B")].double().numpy() @ sd[KEY.format(m, "A")].double().numpy() for m in "qkv"]


def cue_rows(u, alpha):
    """c_p for cue directions u [N, 3, 1024]: normed block-26 input with the cue planted minus without (bf16 cast as fed)."""
    on = sp.planted(u, alpha, (0, 1, 2), True)
    with torch.no_grad():
        e_on = sp.embed(on)["in26"].double().numpy()
        e_off = sp.embed(h0[None])["in26"].double().numpy()
    return e_on - e_off


out = []
for k in range(6):
    run = f"dir_u{1110 + k}_a0.03125_s{k}"
    u = np.load(HERE / ("data/cue_u.npy" if k == 0 else f"data/cue_u_{1110 + k}.npy")).astype(np.float32)
    C = cue_rows(np.concatenate([u[None], U0]), 0.03125)  # [1001, 3, 1024]
    for arm, t, refs in [("SM", TH[k], [f"raw/runs/sm_m{k}_t{TH[k]}_d{k + 100 + d}/step{TH[k]}.pt" for d in range(4)]),
                         ("RAW", 256, [])]:
        Wt = dW(f"raw/runs/{run}/step{t}.pt")
        Wr = [dW(r) for r in refs]
        R = [Wt[i] - (sum(w[i] for w in Wr) / len(Wr) if Wr else 0.0) for i in range(3)]
        v1 = np.linalg.svd(np.vstack(R), full_matrices=False)[2][0]
        e1 = (np.einsum("npd,d->np", C, v1) ** 2).sum(1) / (C ** 2).sum(axis=(1, 2))
        rank = int(1 + (e1[1:] >= e1[0]).sum())
        ref = summ[arm][f"m{k}"][str(t)]
        out.append({"model": f"m{k}", "arm": arm, "t": t, "e1_here": float(e1[0]), "e1_summary": ref["e1_true"],
                    "rel_diff": float(abs(e1[0] - ref["e1_true"]) / ref["e1_true"]), "rank_here": rank, "rank_summary": ref["rank"]})
(HERE / "post" / "eq_check.json").write_text(json.dumps(out, indent=1))
print("max relative e1 difference:", max(r["rel_diff"] for r in out), "| ranks equal in",
      sum(r["rank_here"] == r["rank_summary"] for r in out), "of", len(out), "cells")
