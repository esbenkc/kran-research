"""T14: description-length curve. How many parameters does ANY network need to agree with M_{16,10}
at the noise-floor bar 0.98671? Distill the model (cross-entropy to its argmax) into:
  (a) 1-hidden-layer ReLU MLPs on raw x (10 -> w -> 10, with biases): params = 21w + 10
  (b) ReLU RNNs of the same architecture as the model (no biases): params = d + d^2 + 10d
      d = 16 is a calibration point: it shows how close this training recipe gets with the model's own size.
Data: fresh N(0,1) batches from a torch generator (seed 50 + config index); test = core.sample(200_000, 52).
Dev seeds only, never 432432. CPU float32 training, float64 teacher.

Run: .venv/bin/python distill_curve.py --steps 15000 --target kl  -> distill_curve_kl.json
"""
import argparse
import json
import time

import numpy as np
import torch

from core import agreement, load_raw, run, sample

torch.set_num_threads(4)
W_hi, W_hh, W_oh = (torch.tensor(w) for w in load_raw())


def teacher(x64):
    """Model logits (float64)."""
    h = torch.zeros(x64.shape[0], 16, dtype=torch.float64)
    for t in range(10):
        h = torch.relu(h @ W_hh.T + x64[:, t : t + 1] * W_hi)
    return h @ W_oh.T


class MLP(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, w), torch.nn.ReLU(), torch.nn.Linear(w, 10))

    def forward(self, x):
        return self.net(x)


class RNN(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.rnn = torch.nn.RNN(1, d, nonlinearity="relu", bias=False, batch_first=True)
        self.out = torch.nn.Linear(d, 10, bias=False)

    def forward(self, x):
        _, h = self.rnn(x[..., None])
        return self.out(h[0])


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def train(model, steps, seed, target, bs=4096, lr=3e-3, T=1.0):
    """target='argmax': cross-entropy to the model's argmax; 'kl': KL to softmax(model logits / T)
    (keeps the near-tie margins, Kasper #1598)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        x = torch.randn(bs, 10, generator=g, dtype=torch.float64)
        lt = teacher(x)
        out = model(x.float())
        if target == "argmax":
            loss = torch.nn.functional.cross_entropy(out, lt.argmax(1))
        else:
            q = torch.softmax(lt / T, 1).float()
            loss = torch.nn.functional.kl_div(torch.log_softmax(out / T, 1), q, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--target", choices=["argmax", "kl"], default="kl")
    args = ap.parse_args()
    Wn = [w.numpy() for w in (W_hi, W_hh, W_oh)]
    xt = sample(200_000, 52)
    pt = run(xt, *Wn).argmax(1)
    xt_t = torch.tensor(xt, dtype=torch.float32)
    xs = sample(200_000, 51)  # second dev set: fresh-data training cannot overfit, so 51 ~ 52 is expected
    ps = run(xs, *Wn).argmax(1)
    xs_t = torch.tensor(xs, dtype=torch.float32)
    configs = [("mlp", w) for w in (8, 16, 32, 64, 128, 256)] + [("rnn", d) for d in (6, 8, 10, 12, 16)]
    rows = []
    for i, (kind, size) in enumerate(configs):
        torch.manual_seed(100 + i)
        m = MLP(size) if kind == "mlp" else RNN(size)
        t0 = time.time()
        loss = train(m, args.steps, 50 + i, args.target)
        with torch.no_grad():
            a = agreement(m(xt_t).argmax(1).numpy(), pt)
            a51 = agreement(m(xs_t).argmax(1).numpy(), ps)
        row = {"kind": kind, "size": size, "params": nparams(m), "target": args.target,
               "agree_dev52": a, "agree_dev51": a51, "final_train_loss": loss,
               "steps": args.steps, "sec": round(time.time() - t0, 1)}
        rows.append(row)
        print(row, flush=True)
        json.dump({"bar": 0.98671, "rows": rows}, open(f"distill_curve_{args.target}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
