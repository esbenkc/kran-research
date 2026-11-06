"""Shaping-probe direction v: logistic probe on the BASE model's q_norm(24) output at the answer token,
'read' rows (peek = cheat, label 1) vs 'solve' rows (honest, label 0), train prompts (balanced by construction:
every prompt gives one row of each). Reported: train/dev/test AUROC; sd_M0 = std of v.q over the train rows (the
attached term's scale). Also the same probe on the layer-0 token embedding (token-identity null, Iris #15354).

    env/.venv/bin/python probe.py   -> data/probe_v.pt, data/probe_fit.json
"""
import json

import numpy as np
import torch

from tail import DEV, HERE, Rows, load_tail, splits


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X, y, l2=1e-2, iters=200):
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    v = (w / sd).detach()                                  # direction in raw activation units
    return v, float(b.detach() - (mu / sd * w.detach()).sum())


@torch.no_grad()
def readings(tail, w2, Z, idx, bs=32):
    out = []
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        _, qn = tail(Z.get(j, 0), w2, None, Z.get(j, 1))
        out.append(torch.stack([qn[: len(j)], qn[len(j):]], 1).float().cpu().double())
    return torch.cat(out)                                  # [n, 2, 2048]


def main():
    idx = splits()
    Z = Rows(HERE / "raw/cache/base/none_rec_tail.npy")
    cfg, tail, w2 = load_tail()
    tail.eval()
    R = {s: readings(tail, w2, Z, idx[s]) for s in ("train", "dev", "test")}

    def xy(s):
        r = R[s]
        return torch.cat([r[:, 0], r[:, 1]]).numpy(), np.r_[np.ones(len(r)), np.zeros(len(r))]
    Xtr, ytr = xy("train")
    v, b = fit_logistic(Xtr, ytr)
    res = {}
    for s in ("train", "dev", "test"):
        X, y = xy(s)
        res[f"auroc_{s}"] = auroc(X @ v.numpy(), y)
    Mtr = Xtr @ v.numpy()
    sd_M0 = float(Mtr.std())
    res.update(sd_M0=sd_M0, margin_train=float(Mtr[ytr == 1].mean() - Mtr[ytr == 0].mean()),
               n_train=len(idx["train"]), dims=int(Xtr.shape[1]))
    # token-identity null: the answer token's embedding row alone (layer 0), same probe recipe
    emb = Z.a[idx["test"], :, -1].astype(np.uint16)
    E = torch.from_numpy(emb.view(np.int16)).view(torch.bfloat16).double().numpy()
    Etr = Z.a[idx["train"], :, -1].astype(np.uint16)
    Etr = torch.from_numpy(Etr.view(np.int16)).view(torch.bfloat16).double().numpy()
    ve, _ = fit_logistic(np.r_[Etr[:, 0], Etr[:, 1]], ytr)
    res["auroc_test_block22_input_null"] = auroc(np.r_[E[:, 0], E[:, 1]] @ ve.numpy(),
                                                 np.r_[np.ones(len(E)), np.zeros(len(E))])
    torch.save({"v": v.float(), "b": b, "sd_M0": sd_M0, "layer": 24, "site": "self_attn.q_norm", "pos": "answer"},
               HERE / "data" / "probe_v.pt")
    json.dump(res, open(HERE / "data" / "probe_fit.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
