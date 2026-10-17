"""T17 (Felix): fixed-point quantization-aware finetune of a pruned net (from Bruno's prune_ft.py).

Same setup as Bruno's prune_qat.py (sparsity mask fixed, training in the rms-1 gauge, KL to the model's
logits at temperature T, fresh N(0,1) every batch, eval on dev seed 61), but the grid is fixed-point
in the rms-1 gauge (every neuron has rms activation 1, so entries of a row are comparable; in the +-1
gauge column scales differ by orders of magnitude and a per-row grid zeroes most entries):

  W_hi:        16 literals kept at full precision (counted as 32 bits each)

  W_hh row i:  w = q_hh[i] * n,  n integer, |n| <= 2^(b-1) - 1   (one step per row, 16 steps)
  W_oh:        w = q_oh[j] * n   (one step per output row j; row 0 is 0 after the row shift)

Steps are set from the start weights (max |row| / (2^(b-1)-1)) and kept fixed; the STE rounds to the grid.
Description bits := b * (#nonzero integers) + 32 * (#steps + #W_hi) + sparsity mask (log2 C(256, nnz_hh) +
log2 C(150, nnz_oh)), the mask counted as an enumerative code.

Run: .venv/bin/python prune_qatF.py prune_ft_T5_200_140.npz --bits 6 --steps 6000
Output: <stem>_fp<b>.npz (+ .json); npz holds the dequantized rms-gauge weights and the integer grids.
"""
import argparse
import json
from math import lgamma, log
from pathlib import Path

import numpy as np
import torch

from prune_ft import HERE, agree, batch, kl_loss, literals, rms_gauge

torch.set_num_threads(4)


def log2_binom(n, k):
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--bits", type=int, default=6)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    d = np.load(HERE / a.npz)
    Wi_pm, Wh_pm, Wo_pm = d["W_hi"], d["W_hh"], d["W_oh"]
    alive = Wi_pm != 0
    Wi_r, Wh_r, Wo_r = rms_gauge(np.where(alive, Wi_pm, 0.0), Wh_pm, Wo_pm)
    Wi_r, Wh_r, Wo_r = (np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in (Wi_r, Wh_r, Wo_r))
    Wi = torch.tensor(Wi_r)
    Wh = torch.nn.Parameter(torch.tensor(Wh_r))
    Wo = torch.nn.Parameter(torch.tensor(Wo_r))
    mh = torch.tensor((Wh_pm != 0).astype(float))
    mo = torch.tensor((Wo_pm != 0).astype(float))
    nmax = 2 ** (a.bits - 1) - 1
    q_hh = torch.tensor(np.maximum(np.abs(Wh_r * (Wh_pm != 0)).max(1), 1e-12) / nmax)[:, None]
    q_oh = torch.tensor(np.maximum(np.abs(Wo_r * (Wo_pm != 0)).max(1), 1e-12) / nmax)[:, None]

    def quant(Wh_, Wo_):
        hp, op = Wh_ * mh, Wo_ * mo
        hq = torch.clamp(torch.round(hp.detach() / q_hh), -nmax, nmax) * q_hh
        oq = torch.clamp(torch.round(op.detach() / q_oh), -nmax, nmax) * q_oh
        return Wi, hp + (hq - hp).detach(), op + (oq - op).detach()

    def fwd(x, W):
        Wi_, Wh_, Wo_ = W
        h = torch.zeros(x.shape[0], Wh_.shape[0], dtype=x.dtype)
        for t in range(x.shape[1]):
            h = torch.relu(h @ Wh_.T + x[:, t : t + 1] * Wi_[None, :])
        return h @ Wo_.T

    def export():
        Wi_, hq, oq = quant(Wh.detach(), Wo.detach())
        nh = torch.round(hq / q_hh).numpy().astype(int)
        no = torch.round(oq / q_oh).numpy().astype(int)
        return (Wi_.numpy().copy(), nh * q_hh.numpy(), no * q_oh.numpy()), nh, no

    def bits(nh, no):
        nnz_h, nnz_o = int((nh != 0).sum()), int((no != 0).sum())
        n_steps = int((np.abs(nh).max(1) > 0).sum() + (np.abs(no).max(1) > 0).sum())
        mask = log2_binom(256, nnz_h) + log2_binom(150, nnz_o)
        n_steps += int((Wi_r != 0).sum())
        return a.bits * (nnz_h + nnz_o) + 32 * n_steps + mask, nnz_h + nnz_o, n_steps

    W0, nh0, no0 = export()
    row = {"src": a.npz, "bits_per": a.bits, "agree_grid_before": agree(W0)}
    gen = torch.Generator().manual_seed(a.seed)
    opt = torch.optim.Adam([Wh, Wo], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    best = (row["agree_grid_before"], Wh.detach().clone(), Wo.detach().clone())
    for i in range(a.steps):
        x, t = batch(gen, 4096)
        loss = kl_loss(fwd(x, quant(Wh, Wo)), t, a.T)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([Wh, Wo], 1.0)
        opt.step()
        sched.step()
        if (i + 1) % 500 == 0:  # keep the best grid point (selected on dev seed 61; holdout untouched)
            ag = agree(export()[0])
            if ag > best[0]:
                best = (ag, Wh.detach().clone(), Wo.detach().clone())
    with torch.no_grad():
        Wh.copy_(best[1]); Wo.copy_(best[2])
    W, nh, no = export()
    tb, nnz, n_steps = bits(nh, no)
    row.update(agree_grid_after=agree(W), nonzero_ints=nnz, scale_literals=n_steps,
               literals_equiv=nnz + n_steps, description_bits=round(tb, 1), loss=float(loss.detach()))
    out = str(Path(a.npz).with_suffix("")) + f"_fp{a.bits}"
    np.savez(HERE / f"{out}.npz", W_hi=W[0], W_hh=W[1], W_oh=W[2], n_hh=nh, n_oh=no,
             q_hh=q_hh.numpy()[:, 0], q_oh=q_oh.numpy()[:, 0])
    json.dump({"args": vars(a), "row": row}, open(HERE / f"{out}.json", "w"), indent=1)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
