"""Discovery (dev seeds 31/32/33 only): is the model ~ exact 2nd-argmax of y = G x (+ phantom zero)?
G is 10x10 (100 fitted constants). Trained with a smooth 2nd-argmax surrogate:
  P(j is 2nd) = sum_{k != j} softmax(b y)_k * softmax_{i != k}(b y)_j
Phantom zero: append y_{-1} = 0 at position "-1" which decodes to output 0 (Felix's C1 rule).
Run: PYTHONPATH=. .venv/bin/python scratch/distorted_2nd_argmax.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # find core.py

import numpy as np
import torch

from core import agreement, load_raw, run, sample

torch.manual_seed(0)
W = load_raw()


def data(n, seed):
    x = sample(n, seed)
    return torch.tensor(x), torch.tensor(run(x, *W).argmax(1))


def scores(G, x, beta, phantom):
    y = x @ G.T
    if phantom:
        y = torch.cat([y, torch.zeros(len(y), 1, dtype=y.dtype)], 1)  # index 10 = phantom
    return beta * y


def logp2(s):
    # log P(j is 2nd) = logsumexp_k [ log softmax(s)_k + log softmax_{-k}(s)_j ], k != j
    n = s.shape[1]
    lsm = torch.log_softmax(s, 1)  # (B,n)
    big = s[:, None, :].expand(-1, n, -1).clone()  # (B,k,j)
    eye = torch.eye(n, dtype=torch.bool)
    big[:, eye] = -1e9
    inner = torch.log_softmax(big, 2)  # softmax over j excluding k
    tot = lsm[:, :, None] + inner
    tot[:, eye] = -1e9
    return torch.logsumexp(tot, 1)  # (B,j)


def predict(G, x, phantom):
    y = x @ G.T
    if phantom:
        y = torch.cat([y, torch.zeros(len(y), 1, dtype=y.dtype)], 1)
    o = torch.argsort(-y, 1)
    p = o[:, 1]
    if phantom:
        p = torch.where(p == 10, torch.zeros_like(p), p)
    return p


def fit(phantom, diag_only=False, steps=1500):
    xtr, ptr = data(400_000, 31)
    xte, pte = data(200_000, 32)
    G = torch.eye(10, dtype=torch.float64, requires_grad=True)
    lb = torch.tensor(np.log(30.0), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([G, lb], lr=3e-3)
    tgt = ptr.clone()
    for it in range(steps):
        idx = torch.randint(0, len(xtr), (8192,))
        Gm = torch.diag(torch.diagonal(G)) if diag_only else G
        lp = logp2(scores(Gm, xtr[idx], lb.exp(), phantom))
        if phantom:  # model output 0 may come from real position 0 or the phantom
            lp0 = torch.logsumexp(torch.stack([lp[:, 0], lp[:, 10]], 1), 1)
            lp = torch.cat([lp0[:, None], lp[:, 1:10]], 1)
        loss = -lp.gather(1, tgt[idx][:, None]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            with torch.no_grad():
                Gm = torch.diag(torch.diagonal(G)) if diag_only else G
                a = agreement(predict(Gm, xte, phantom).numpy(), pte.numpy())
            print(f"  it {it} loss {loss.item():.4f} beta {lb.exp().item():.1f} test agree {a:.4f}")
    return (torch.diag(torch.diagonal(G)) if diag_only else G).detach().numpy()


if __name__ == "__main__":
    for phantom, diag in [(True, True), (True, False)]:
        print(f"phantom={phantom} diag_only={diag}")
        G = fit(phantom, diag)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print(G)
        np.save(f"scratch/G_phantom{int(phantom)}_diag{int(diag)}.npy", G)
