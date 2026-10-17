"""N1b (Bruno): rounding-aware finetune of a pruned net from prune_ft.py.

Takes a saved npz (W_hi = +-1 gauge, as prune_ft.py writes it), keeps its sparsity mask fixed, and
finetunes with a straight-through estimator so that the weights stay good AFTER rounding every literal
to k significant digits in the +-1 gauge (the gauge in which literals are counted and printed).
Training happens in the rms-1 gauge (same as prune_ft.py); the STE rounds in the +-1 gauge.
Loss: KL to the model's logits at temperature T, fresh N(0,1) every batch. Eval: dev seed 61 (prune_ft).

Modes: ste = straight-through rounding (made the 340-literal net worse: 0.982 -> 0.972-0.980);
       freeze = progressive freezing: round + freeze the largest entries first, finetune the rest, repeat.
Run: .venv/bin/python prune_qat.py prune_ft_T5_200_140.npz --k 3 --steps 8000 --mode freeze
Output: <npz stem>_q<k>.npz (exactly rounded weights) and a json row on stdout + <stem>_q<k>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prune_ft import HERE, TEACHER, W0_RMS, agree, batch, gauge, kl_loss, literals, rms_gauge, round_sig

torch.set_num_threads(4)


def round_sig_t(w, k):
    nz = (w != 0) & (w.abs() != 1.0)
    a = torch.where(nz, w.abs(), torch.ones_like(w))
    e = torch.floor(torch.log10(a))
    q = torch.round(w / 10 ** (e - k + 1)) * 10 ** (e - k + 1)
    return torch.where(nz, q, w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", choices=["ste", "freeze"], default="freeze")
    p.add_argument("--fracs", default="0.5,0.75,0.9,0.97,1.0", help="freeze schedule (freeze mode)")
    a = p.parse_args()
    d = np.load(HERE / a.npz)
    Wi_pm, Wh_pm, Wo_pm = d["W_hi"], d["W_hh"], d["W_oh"]
    alive = Wi_pm != 0
    # rms gauge of the pruned net (exact rescale); dead neurons keep scale 1
    Wi_r, Wh_r, Wo_r = rms_gauge(np.where(alive, Wi_pm, 0.0), Wh_pm, Wo_pm)
    Wi_r, Wh_r, Wo_r = (np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in (Wi_r, Wh_r, Wo_r))
    Wi = torch.tensor(Wi_r)
    Wh = torch.nn.Parameter(torch.tensor(Wh_r))
    Wo = torch.nn.Parameter(torch.tensor(Wo_r))
    mh = torch.tensor((Wh_pm != 0).astype(float))
    mo = torch.tensor((Wo_pm != 0).astype(float))
    s = torch.tensor(np.where(alive, 1.0 / np.where(alive, np.abs(Wi_r), 1.0), 1.0))  # rms -> +-1 gauge

    def to_pm(Wh_, Wo_):
        return (s[:, None] * Wh_) / s[None, :], Wo_ / s[None, :]

    fh = torch.zeros_like(mh)  # frozen (already rounded) entries, freeze mode
    fo = torch.zeros_like(mo)
    vh = torch.zeros_like(mh)  # frozen values in the +-1 gauge
    vo = torch.zeros_like(mo)

    def quant(Wh_, Wo_):
        hp, op = to_pm(Wh_ * mh, Wo_ * mo)
        if a.mode == "ste":
            hq, oq = round_sig_t(hp.detach(), a.k), round_sig_t(op.detach(), a.k)
            hp, op = hp + (hq - hp).detach(), op + (oq - op).detach()  # STE
        else:
            hp, op = torch.where(fh > 0, vh, hp), torch.where(fo > 0, vo, op)
        return torch.sign(Wi), hp, op

    def fwd(x, W):
        Wi_, Wh_, Wo_ = W
        h = torch.zeros(x.shape[0], Wh_.shape[0], dtype=x.dtype)
        for t in range(x.shape[1]):
            h = torch.relu(h @ Wh_.T + x[:, t : t + 1] * Wi_[None, :])
        return h @ Wo_.T

    def export():
        _, hq, oq = quant(Wh.detach(), Wo.detach())
        W = (torch.sign(Wi).numpy(), hq.numpy().copy(), oq.numpy().copy())
        W = (W[0], round_sig(W[1], a.k), round_sig(W[2], a.k))  # exact numpy rounding (idempotent)
        return W

    W_start = export()
    row = {"src": a.npz, "k": a.k, "mode": a.mode, "literals": literals(*W_start),
           "agree_round_before": agree(W_start)}
    gen = torch.Generator().manual_seed(a.seed)

    def train(steps):
        opt = torch.optim.Adam([Wh, Wo], lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=a.lr * 0.05)
        loss = torch.tensor(float("nan"))
        for i in range(steps):
            x, t = batch(gen, 4096)
            loss = kl_loss(fwd(x, quant(Wh, Wo)), t, a.T)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Wh, Wo], 1.0)
            opt.step()
            sched.step()
        return loss

    if a.mode == "ste":
        loss = train(a.steps)
    else:
        # progressive freezing: round + freeze the largest |w| (+-1 gauge) first, finetune the rest
        fracs = [float(f) for f in a.fracs.split(",")]
        loss = torch.tensor(float("nan"))
        for f in fracs:
            with torch.no_grad():
                hp, op = to_pm(Wh * mh, Wo * mo)
                allv = torch.cat([hp.flatten(), op.flatten()])
                live = torch.cat([(mh * (1 - fh)).flatten(), (mo * (1 - fo)).flatten()]) > 0
                n_total = int(torch.cat([mh.flatten(), mo.flatten()]).sum())
                n_new = int(round(f * n_total)) - int(fh.sum() + fo.sum())
                if n_new > 0:
                    score = torch.where(live, allv.abs(), torch.full_like(allv, -1.0))
                    idx = torch.topk(score, n_new).indices
                    sel = torch.zeros_like(allv)
                    sel[idx] = 1.0
                    sh, so = sel[: hp.numel()].view_as(hp), sel[hp.numel():].view_as(op)
                    vh.copy_(torch.where(sh > 0, round_sig_t(hp, a.k), vh))
                    vo.copy_(torch.where(so > 0, round_sig_t(op, a.k), vo))
                    fh.copy_(torch.maximum(fh, sh))
                    fo.copy_(torch.maximum(fo, so))
            if f < 1.0:
                loss = train(a.steps // max(1, len(fracs) - 1))
            row[f"agree_frozen_{f}"] = agree(export())
    W = export()
    row.update(agree_round_after=agree(W), literals_after=literals(*W), loss=float(loss.detach()),
               W_hh_nz=int((W[1] != 0).sum()), W_oh_nz=int((W[2] != 0).sum()))
    out = str(Path(a.npz).with_suffix("")) + f"_q{a.k}" + ("" if a.mode == "ste" else "f")
    np.savez(HERE / f"{out}.npz", W_hi=W[0], W_hh=W[1], W_oh=W[2])
    json.dump({"args": vars(a), "row": row}, open(HERE / f"{out}.json", "w"), indent=1)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
