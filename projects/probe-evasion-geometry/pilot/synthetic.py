"""Synthetic test of the whitened-cosine lemma.

Concept model: latent z | y ~ N(y * m, I_32); the probed representation is x = A z with a fixed,
badly conditioned A, so x has anisotropic within-class covariance Sigma = A A'.
Monitor w = difference-of-means probe on x. 50 other probes u = LDA probes on random subspaces.
An evader (residual MLP) is trained to make w call class-1 inputs "class 0", with a penalty on
how far it moves the representation. Two evaders:
  upstream : edits z (before the fixed map A), penalty |z' - z|^2  (= Sigma^-1 norm in x)
  in-place : edits x directly,               penalty |x' - x|^2  (Euclidean norm in x)
The lemma predicts the upstream evader moves the class-1 mean along Sigma w.

  uv run --with torch python projects/probe-evasion-geometry/synthetic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (auroc, auroc_from_d, class_stats, cos_raw, cos_white, d_from_auroc, lda,
                    probe_family, r2_corr, r2_identity, sep)

HERE = Path(__file__).parent
OUT = HERE / "data"
D, N, N_PROBES, MEAN_GAP = 32, 8000, 50, 3.0
import os
LAMBDA, STEPS, LR = float(os.environ.get("LAMBDA", 0.05)), 1500, 3e-3


def make_world(seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(D)
    m *= MEAN_GAP / np.linalg.norm(m)
    U, _ = np.linalg.qr(rng.standard_normal((D, D)))
    V, _ = np.linalg.qr(rng.standard_normal((D, D)))
    s = np.logspace(0, -1.5, D)
    A = U @ np.diag(s) @ V.T

    def sample(n):
        y = np.repeat([0, 1], n // 2)
        z = rng.standard_normal((n, D)) + y[:, None] * m
        return z, y

    return rng, m, A, sample


class Evader(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(D, 128), torch.nn.GELU(), torch.nn.Linear(128, D))
        torch.nn.init.zeros_(self.net[2].weight)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, h):
        return h + self.net(h)


def train_evader(arm, z, y, A, w, c, s_w, seed):
    """Return a function mapping latent z -> evaded representation x'."""
    At = torch.tensor(A.T, dtype=torch.float32)
    wt = torch.tensor(w, dtype=torch.float32)
    zt = torch.tensor(z, dtype=torch.float32)
    y1 = torch.tensor(y == 1)
    h = zt if arm == "upstream" else zt @ At
    ev = Evader(seed)
    opt = torch.optim.Adam(ev.parameters(), lr=LR)
    for _ in range(STEPS):
        h2 = ev(h)
        x2 = h2 @ At if arm == "upstream" else h2
        score = (x2 @ wt - c) / s_w
        evade = torch.nn.functional.softplus(score[y1]).mean()   # "call class 1 benign"
        keep = ((h2 - h) ** 2).sum(1).mean()
        loss = evade + LAMBDA * keep
        opt.zero_grad()
        loss.backward()
        opt.step()

    def encode(znew):
        with torch.no_grad():
            zz = torch.tensor(znew, dtype=torch.float32)
            hh = zz if arm == "upstream" else zz @ At
            out = ev(hh)
            return (out @ At if arm == "upstream" else out).numpy().astype(float)

    return encode


def run(seed, arm):
    rng, m, A, sample = make_world(seed)
    z_tr, y_tr = sample(N)
    z_te, y_te = sample(N)
    X_tr, X_te = z_tr @ A.T, z_te @ A.T
    mu0, mu1, Sigma = class_stats(X_tr, y_tr)
    delta = mu1 - mu0
    w = delta / np.linalg.norm(delta)  # difference-of-means monitor
    c, s_w = float(w @ (mu0 + mu1) / 2), float(np.sqrt(w @ Sigma @ w))
    probes = probe_family(delta, Sigma, w, N_PROBES, rng)

    enc = train_evader(arm, z_tr, y_tr, A, w, c, s_w, seed)
    Xe_tr, Xe_te = enc(z_tr), enc(z_te)

    a_w0, a_w1 = auroc(X_te @ w, y_te), auroc(Xe_te @ w, y_te)
    kappa = sep(w, delta, Sigma) - d_from_auroc(a_w1)  # separation the monitor lost (noise units)

    rows = []
    for i, u in enumerate(probes):
        du = sep(u, delta, Sigma)
        cw, cr = cos_white(u, w, Sigma), cos_raw(u, w)
        rows.append(dict(
            probe=i, d_before=du, cos_white=cw, cos_raw=cr,
            auroc_before=auroc(X_te @ u, y_te), auroc_after=auroc(Xe_te @ u, y_te),
            pred_after_white=float(auroc_from_d(du - kappa * cw)),
            pred_after_raw=float(auroc_from_d(du - kappa * cr)),
        ))
    for r in rows:
        r["drop"] = r["auroc_before"] - r["auroc_after"]
        r["pred_drop_white"] = float(auroc_from_d(r["d_before"])) - r["pred_after_white"]
        r["pred_drop_raw"] = float(auroc_from_d(r["d_before"])) - r["pred_after_raw"]

    # refit: fresh LDA probe on the evaded representation
    e0, e1, Se = class_stats(Xe_tr, y_tr)
    refit = lda(e1 - e0, Se)
    a_refit = auroc(Xe_te @ refit, y_te)

    # where did the class-1 mean actually move?
    shift = (e1 - mu1) - (e0 - mu0)
    Sw = Sigma @ w
    meas = np.array([r["drop"] for r in rows])
    summary = dict(
        seed=seed, arm=arm, auroc_w_before=a_w0, auroc_w_after=a_w1, kappa=kappa,
        auroc_refit=a_refit, auroc_lda_before=auroc(X_te @ lda(delta, Sigma), y_te),
        shift_cos_Sigma_w=cos_raw(shift, Sw), shift_cos_w=cos_raw(shift, w),
        shift_norm=float(np.linalg.norm(shift)),
        cov_change_rel=float(np.linalg.norm(Se - Sigma) / np.linalg.norm(Sigma)),
        r2_white=r2_identity(meas, [r["pred_drop_white"] for r in rows]),
        r2_raw=r2_identity(meas, [r["pred_drop_raw"] for r in rows]),
        corr2_white=r2_corr(meas, [r["pred_drop_white"] for r in rows]),
        corr2_raw=r2_corr(meas, [r["pred_drop_raw"] for r in rows]),
        corr2_cos_white_cos_raw=r2_corr([r["cos_white"] for r in rows], [r["cos_raw"] for r in rows]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_seed{seed}"
    np.savez(OUT / f"{tag}.npz", Sigma=Sigma, delta=delta, w=w, probes=probes, refit=refit,
             X_te=X_te, Xe_te=Xe_te, y_te=y_te, mu0=mu0, mu1=mu1, e0=e0, e1=e1)
    (OUT / f"{tag}_probes.json").write_text(json.dumps(rows, indent=1))
    return summary


if __name__ == "__main__":
    arms = sys.argv[1:] or ["upstream", "in-place"]
    out = []
    for arm in arms:
        for seed in (0, 1, 2):
            s = run(seed, arm)
            out.append(s)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
