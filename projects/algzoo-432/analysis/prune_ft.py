"""N1 (Bruno): prune + finetune M_{16,10} from its own weights.

Question: how many free numbers does a network-shaped copy need to agree with the model on >= 0.98671
(the frozen noise floor) when the kept weights may be RE-TRAINED (KL to the model's logits), not only
zeroed and rounded as in ladder.py? This is a description-length curve. Finetuned weights are fitted to
the model's outputs (PREREG item 3(ii)), so no point on it is an "explanation"; it bounds how much detail
an explanation would need.

Setup
  start: exact gauge of m16_10 (W_hi = +-1 for all 16 neurons, W_oh <- W_oh - W_oh[0]; 400 free numbers).
  W_hi stays fixed at +-1 (gauge). W_oh row 0 stays exactly 0 (gauge). Trained: kept W_hh, W_oh rows 1-9.
  loss: KL(softmax(model logits / T) || softmax(student logits / T)), fresh x ~ N(0,1)^10 every batch.
  training gauge: every neuron rescaled to rms 1 (exact); weights are mapped back to the +-1 gauge for
  eval and saving. pruning: gradual; saliency = |W_ij| * rms(h_j) (contribution size to unit i / logit k).
  literal = nonzero entry that is not exactly +-1 (ladder.py convention).
  eval: float64 numpy forward (core.run), dev seed 61, 2e5 sequences; also after rounding to 3 / 4
  significant digits. Holdout 432432 is never used here.

Run:
  .venv/bin/python prune_ft.py prune  --out prune_ft_joint   # stages listed in STAGES
  .venv/bin/python prune_ft.py neurons --out prune_ft_neurons  # warm-started neuron removal d=12/10/8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core import load_raw, run

HERE = Path(__file__).resolve().parent
BAR = 0.98671
torch.set_num_threads(4)


def gauge(Wi, Wh, Wo):
    s = np.where(Wi != 0, 1.0 / np.where(Wi != 0, np.abs(Wi), 1.0), 1.0)
    Wi, Wh, Wo = np.sign(Wi), (s[:, None] * Wh) / s[None, :], Wo / s[None, :]
    return Wi, Wh, Wo - Wo[0:1]


def literals(Wi, Wh, Wo):
    allw = np.concatenate([Wi, Wh.ravel(), Wo.ravel()])
    return int(((allw != 0) & (np.abs(allw) != 1.0)).sum())


def round_sig(W, k):
    out = W.copy()
    nz = (out != 0) & (np.abs(out) != 1.0)
    e = np.floor(np.log10(np.abs(out[nz])))
    out[nz] = np.round(out[nz] / 10 ** (e - k + 1)) * 10 ** (e - k + 1)
    return out


W0 = gauge(*load_raw())


def rms_gauge(Wi, Wh, Wo, n=20_000):
    """Exact rescale so every neuron has rms 1 over t=1..10 (training gauge; Adam steps are then comparable)."""
    _, hs = run(np.random.default_rng(60).standard_normal((n, 10)), Wi, Wh, Wo, return_states=True)
    r = np.sqrt((hs[:, 1:] ** 2).mean((0, 1)))
    return Wi / r, Wh * r[None, :] / r[:, None], Wo * r[None, :]


W0_RMS = rms_gauge(*W0)
X_DEV = np.random.default_rng(61).standard_normal((200_000, 10))
REF_DEV = run(X_DEV, *W0).argmax(1)


def agree(W):
    return float((run(X_DEV, *W).argmax(1) == REF_DEV).mean())


class Student(torch.nn.Module):
    def __init__(self, Wi, Wh, Wo):
        super().__init__()
        self.Wi = torch.tensor(Wi)  # fixed
        self.Wh = torch.nn.Parameter(torch.tensor(Wh))
        self.Wo = torch.nn.Parameter(torch.tensor(Wo))
        self.mh = torch.ones_like(self.Wh)
        self.mo = torch.ones_like(self.Wo)
        self.mo[0] = 0.0
        self.alive = torch.ones(Wh.shape[0], dtype=torch.float64)

    def weights(self):
        a = self.alive
        Wh = self.Wh * self.mh * a[:, None] * a[None, :]
        Wo = self.Wo * self.mo * a[None, :]
        return self.Wi * a, Wh, Wo

    def forward(self, x, return_states=False):
        Wi, Wh, Wo = self.weights()
        h = torch.zeros(x.shape[0], Wh.shape[0], dtype=x.dtype)
        hs = []
        for t in range(x.shape[1]):
            h = torch.relu(h @ Wh.T + x[:, t : t + 1] * Wi[None, :])
            hs.append(h)
        out = h @ Wo.T
        return (out, torch.stack(hs, 1)) if return_states else out

    def numpy(self):
        """Weights converted back to the +-1 gauge (exact), for eval, literal count and saving."""
        return gauge(*(w.detach().numpy().copy() for w in self.weights()))


TEACHER = Student(*W0_RMS)


def batch(gen, B):
    x = torch.randn(B, 10, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        t = TEACHER(x)
    return x, t


def kl_loss(s, t, T):
    return torch.nn.functional.kl_div(
        torch.log_softmax(s / T, 1), torch.log_softmax(t / T, 1), log_target=True, reduction="batchmean"
    )


def train(st, steps, lr, gen, B=4096, T=1.0, prune_fn=None, prune_every=0):
    opt = torch.optim.Adam([st.Wh, st.Wo], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for i in range(steps):
        if prune_fn is not None and prune_every and i % prune_every == 0:
            prune_fn(i)
        x, t = batch(gen, B)
        loss = kl_loss(st(x), t, T)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([st.Wh, st.Wo], 1.0)
        opt.step()
        sched.step()
    return float(loss.detach())


def saliency(st, gen):
    with torch.no_grad():
        x, _ = batch(gen, 8192)
        _, hs = st(x, return_states=True)
        rms = hs.pow(2).mean((0, 1)).sqrt()  # (16,)
        _, Wh, Wo = st.weights()
        return (Wh.abs() * rms[None, :]), (Wo.abs() * rms[None, :])


def set_mask(st, gen, hh_n, oh_n):
    """Keep the hh_n / oh_n most salient currently-kept entries of W_hh / W_oh (rows 1-9)."""
    sh, so = saliency(st, gen)
    for sal, m, n in ((sh, st.mh, hh_n), (so, st.mo, oh_n)):
        sal = sal * m
        flat = sal.flatten()
        keep = torch.zeros_like(flat)
        if n > 0:
            keep[torch.topk(flat, min(n, int((flat > 0).sum()))).indices] = 1.0
        m.copy_(keep.view_as(m) * m)


def evaluate(st):
    W = st.numpy()
    row = {"literals": literals(*W), "W_hh_nz": int((W[1] != 0).sum()), "W_oh_nz": int((W[2] != 0).sum()),
           "alive": int(st.alive.sum()), "agree": agree(W)}
    for k in (3, 4):
        Wr = (W[0], round_sig(W[1], k), round_sig(W[2], k))
        row[f"agree_{k}d"] = agree(Wr)
    return row, W


def ramp(st, gen, hh0, oh0, hh1, oh1, steps, lr, n_prunes=40, T=1.0):
    every = max(1, steps // n_prunes)

    def fn(i):
        f = min(1.0, (i // every + 1) / n_prunes)
        f = 1 - (1 - f) ** 3  # cubic schedule
        set_mask(st, gen, int(round(hh0 + (hh1 - hh0) * f)), int(round(oh0 + (oh1 - oh0) * f)))

    train(st, steps, lr, gen, T=T, prune_fn=fn, prune_every=every)
    set_mask(st, gen, hh1, oh1)


STAGES = [  # (W_hh kept, W_oh kept); W_oh has 144 free (rows 1-9)
    (200, 140), (160, 130), (130, 120), (110, 110), (95, 105), (80, 100), (80, 90), (70, 80), (60, 70),
    (50, 60), (40, 50),
]


def run_prune(args):
    gen = torch.Generator().manual_seed(args.seed)
    st = Student(*W0_RMS)
    rows = []
    hh, oh = 256, 144
    stages = [tuple(map(int, s.split(":"))) for s in args.stages.split(",")] if args.stages else STAGES
    for hh1, oh1 in stages:
        t0 = time.time()
        ramp(st, gen, hh, oh, hh1, oh1, args.ramp, args.lr, T=args.T)
        loss = train(st, args.ft, args.lr, gen, T=args.T)
        hh, oh = hh1, oh1
        row, W = evaluate(st)
        row.update(target=[hh1, oh1], loss=loss, sec=round(time.time() - t0, 1))
        rows.append(row)
        np.savez(HERE / f"{args.out}_{hh1}_{oh1}.npz", W_hi=W[0], W_hh=W[1], W_oh=W[2])
        print(json.dumps(row), flush=True)
        json.dump({"args": vars(args), "rows": rows}, open(HERE / f"{args.out}.json", "w"), indent=1)


def run_neurons(args):
    """Warm-started neuron removal: drop the least-needed neuron (largest agreement after zeroing it)."""
    gen = torch.Generator().manual_seed(args.seed)
    st = Student(*W0_RMS)
    rows = []
    while int(st.alive.sum()) > args.min_d:
        best = None
        for j in np.nonzero(st.alive.numpy())[0]:
            st.alive[j] = 0.0
            a = agree(st.numpy())
            st.alive[j] = 1.0
            if best is None or a > best[0]:
                best = (a, int(j))
        st.alive[best[1]] = 0.0
        t0 = time.time()
        loss = train(st, args.ft, args.lr, gen, T=args.T)
        row, W = evaluate(st)
        row.update(dropped=best[1], agree_before_ft=best[0], loss=loss, sec=round(time.time() - t0, 1))
        rows.append(row)
        np.savez(HERE / f"{args.out}_d{row['alive']}.npz", W_hi=W[0], W_hh=W[1], W_oh=W[2])
        print(json.dumps(row), flush=True)
        json.dump({"args": vars(args), "rows": rows}, open(HERE / f"{args.out}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prune", "neurons"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ramp", type=int, default=1000)
    p.add_argument("--ft", type=int, default=3000)
    p.add_argument("--min_d", type=int, default=8)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--stages", default="", help="override STAGES, e.g. 200:140,160:130")
    a = p.parse_args()
    print(json.dumps({"start": evaluate(Student(*W0_RMS))[0]}), flush=True)
    (run_prune if a.mode == "prune" else run_neurons)(a)
