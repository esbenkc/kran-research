"""EXPLORATORY (PREREG 'Exploratory'): the direction gradient descent itself pushes layer 3 at the clean weights.

At the clean weights the retain-KL gradient is zero, so the first update follows the probe loss alone.
g = gradient of the attacker's probe loss (config attack.loss = softplus_pos: mean softplus(s_w + b_w) over 64
target-positive TRAIN texts) w.r.t. blocks 0-2.
Two parameter steps:
  'sgd'  : dtheta = -g / |g|           -> layer-3 edit J dtheta = -(J J' w-term): the tangent-kernel direction K w
  'adam' : dtheta = -sign(g)            (Adam's first step, m/sqrt(v) = sign(g)); eps 1e-5 = a tenth of the lr
For each, the class-mean edit on the target's TEST texts, measured by a finite difference
[h(theta + eps dtheta) - h(theta)] / eps (eps chosen so the edit is small), pos-mean minus neg-mean.
Saves data/runs/Kw__<stem>.npy (sgd) and data/runs/Kadam__<stem>.npy.
Run: uv run python projects/fisher-pullback-transfer/kernel_dir.py
"""
import json

import numpy as np
import torch

from attack import h3, pad, probe_table, source_data
from common import DATA, device, load_model, pooled_acts

N_HALF = 64


def main():
    dev = device()
    sel = json.loads((DATA / "selection.json").read_text())
    meta, W = probe_table()
    out = {}
    for ti, tname in enumerate(sel["targets"]):
        tok, model = load_model(dev)
        params = [p for n, p in model.named_parameters() if any(n.startswith(f"gpt_neox.layers.{i}.") for i in (0, 1, 2))]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        t = next(p for p in meta if p["name"] == tname)
        w = torch.tensor(W[t["idx"]], dtype=torch.float32, device=dev)
        sd = source_data(tok, t["source"])
        y = (sd["label"] == t["label"]).astype(int)
        rng = np.random.default_rng(ti)
        tr = sd["split"] == 0
        pos = rng.choice(np.where(tr & (y == 1))[0], N_HALF, replace=False)
        eos = tok.eos_token_id
        sp = torch.cat([h3(model, *pad([sd["ids"][i] for i in pos[k:k + 16]], eos, dev)) @ w for k in range(0, N_HALF, 16)])
        b_w = float(np.load(DATA / "probes" / "probe_weights.npz")["b"][t["idx"]])
        loss = torch.nn.functional.softplus(sp + b_w).mean()
        grads = torch.autograd.grad(loss, params)
        te = np.where(sd["split"] == 1)[0]
        te_ids = [sd["ids"][i] for i in te]
        yt = y[te] == 1
        X0 = pooled_acts(model, te_ids, eos, dev).astype(np.float64)
        base = [p.detach().clone() for p in params]
        gnorm = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
        stem = tname.replace(":", "__").replace(" ", "_").replace("&", "and")
        res = {}
        for kind in ("sgd", "adam"):
            step = [-(g / gnorm) if kind == "sgd" else -torch.sign(g) for g in grads]
            eps = 1e-2 if kind == "sgd" else 1e-5
            with torch.no_grad():
                for p, b, s in zip(params, base, step):
                    p.copy_(b + eps * s)
            X1 = pooled_acts(model, te_ids, eos, dev).astype(np.float64)
            with torch.no_grad():
                for p, b in zip(params, base):
                    p.copy_(b)
            dh = (X1 - X0) / eps
            d = dh[yt].mean(0) - dh[~yt].mean(0)
            np.save(DATA / "runs" / f"{'Kw' if kind == 'sgd' else 'Kadam'}__{stem}.npy", d)
            new = X1 @ W[t["idx"]]
            res[kind] = {"eps": eps, "edit_norm": float(np.linalg.norm(d)),
                         "target_gap_change_per_eps": float(((new[yt].mean() - new[~yt].mean())
                                                             - ((X0 @ W[t["idx"]])[yt].mean() - (X0 @ W[t["idx"]])[~yt].mean())) / eps)}
        out[tname] = res
        print(tname, json.dumps(res), flush=True)
    (DATA / "runs" / "kernel_dir.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
