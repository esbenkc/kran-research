"""Train the quadratic net on modular addition until it groks. Save a memorising and a grokked checkpoint.

Full-batch AdamW, centred squared loss + decoupled weight decay.
Memorising checkpoint: first step where train acc = 1 and test acc < 0.5, kept 'mem_hold' steps later
(the run is still memorising there). Grokked checkpoint: test acc >= 0.99 held for 'grok_hold' steps,
then 'grok_extra' more steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import qnet

HERE = Path(__file__).parent


def adamw_train(p, K, frac, seed, lr, wd, init_scale, max_steps, log_every=100, out_dir=None,
                mem_hold=200, grok_hold=500, grok_extra=2000):
    Xtr, Ytr, Xte, Yte, tr_idx, te_idx = qnet.modadd_data(p, frac, seed, return_idx=True)
    d = 2 * p
    rng = np.random.default_rng(1000 + seed)
    W = init_scale * rng.standard_normal((d, K)) / np.sqrt(d)
    V = init_scale * rng.standard_normal((p, K)) / np.sqrt(K)
    params = [W, V]
    m = [np.zeros_like(x) for x in params]
    v = [np.zeros_like(x) for x in params]
    b1, b2, eps = 0.9, 0.98, 1e-8
    hist = []
    mem_at = None
    mem_ckpt = None
    grok_first = None
    grok_ckpt = None
    for t in range(1, max_steps + 1):
        L, gW, gV = qnet.loss_grad(params[0], params[1], Xtr, Ytr)
        for i, g in enumerate((gW, gV)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            params[i] = params[i] - lr * (mh / (np.sqrt(vh) + eps) + wd * params[i])
        if t % log_every == 0 or t == 1:
            W, V = params
            tra = qnet.accuracy(W, V, Xtr, Ytr)
            tea = qnet.accuracy(W, V, Xte, Yte)
            Lte = qnet.loss(W, V, Xte, Yte)
            hist.append(dict(step=t, train_loss=L, test_loss=Lte, train_acc=tra, test_acc=tea))
            if mem_at is None and tra == 1.0 and tea < 0.5:
                mem_at = t
            if mem_at is not None and mem_ckpt is None and t >= mem_at + mem_hold:
                if tea < 0.5 and tra == 1.0:
                    mem_ckpt = dict(step=t, W=W.copy(), V=V.copy(), train_acc=tra, test_acc=tea,
                                    train_loss=L, test_loss=Lte)
                else:
                    mem_at = None
            if tea >= 0.99 and tra == 1.0:
                if grok_first is None:
                    grok_first = t
                if t >= grok_first + grok_hold + grok_extra:
                    grok_ckpt = dict(step=t, W=W.copy(), V=V.copy(), train_acc=tra, test_acc=tea,
                                     train_loss=L, test_loss=Lte)
                    break
            else:
                grok_first = None
    return dict(p=p, K=K, frac=frac, seed=seed, lr=lr, wd=wd, init_scale=init_scale, hist=hist,
                mem=mem_ckpt, grok=grok_ckpt, grok_first=grok_first, n_train=len(Xtr),
                train_idx=tr_idx, test_idx=te_idx)


def save_run(res, out_dir: Path):
    """Checkpoints: out_dir/p{p}_K{K}_s{seed}_{mem|grok}.npz (W, V, train_idx, test_idx, step, accs, losses,
    lr, wd, frac). Training curve: out_dir/p{p}_K{K}_s{seed}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"p{res['p']}_K{res['K']}_s{res['seed']}"
    meta = {k: res[k] for k in ("p", "K", "frac", "seed", "lr", "wd", "init_scale", "n_train", "grok_first")}
    meta["hist"] = res["hist"]
    for phase in ("mem", "grok"):
        ck = res[phase]
        if ck is not None:
            np.savez(out_dir / f"{tag}_{phase}.npz", W=ck["W"], V=ck["V"], train_idx=res["train_idx"],
                     test_idx=res["test_idx"], step=ck["step"], train_acc=ck["train_acc"],
                     test_acc=ck["test_acc"], train_loss=ck["train_loss"], test_loss=ck["test_loss"],
                     lr=res["lr"], wd=res["wd"], frac=res["frac"], p=res["p"], K=res["K"], seed=res["seed"])
            meta[phase] = {k: ck[k] for k in ("step", "train_acc", "test_acc", "train_loss", "test_loss")}
        else:
            meta[phase] = None
    (out_dir / f"{tag}.json").write_text(json.dumps(meta, indent=1, default=float))
    return tag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--init", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--out", type=str, default=str(HERE / "data" / "checkpoints"))
    a = ap.parse_args()
    r = adamw_train(a.p, a.K, a.frac, a.seed, a.lr, a.wd, a.init, a.max_steps)
    tag = save_run(r, Path(a.out))
    print(tag, "mem", r["mem"] and r["mem"]["step"], "grok_first", r["grok_first"],
          "grok", r["grok"] and r["grok"]["step"], "last", r["hist"][-1])
