"""Evasion-transfer test, exactly as frozen in PREREG.md (analysis plan, 2026-10-16).

Setting 1 (synthetic): d=64, Sigma = R diag(1/i) R', Delta random with optimal d' = 4,
  n = 20,000 train / 20,000 test per seed, balanced.
Setting 2 (lm): pythia-70m mean-pooled residual activations from data/lm_acts_L<l>.npz
  (built by lm_extract.py; layer picked by data/lm_selection.json).
Monitor w = mass-mean probe. 50 probes = 25 logistic + 25 mass-mean, each fit in a random
subspace of dimension log-uniform in [2, d]. Evader f(x) = x + MLP(x) (2 hidden layers of 256,
GELU, Adam), loss = lam * (batch d'_w)^2 + retain, retain = ||Sigma_hat^-1/2 (f(x)-x)||^2
("mahalanobis", the decision arm) or ||f(x)-x||^2 ("euclidean", diagnostic arm).
Training stops when the train AUROC of w <= 0.6, or at the step cap.

  uv run --with torch python projects/probe-evasion-geometry/experiment.py synthetic
  uv run --with torch python projects/probe-evasion-geometry/experiment.py lm
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
DATA = HERE / "data"
SEEDS = (0, 1, 2)
N_PROBES = 50
import os
# D2 (DEVIATIONS.md): decision runs use EARLY_STOP=0 and LAM from data/lam_selection.json;
# EARLY_STOP=1 LAM=10 reproduces the literal frozen stop rule (first train AUROC_w <= 0.6).
EARLY_STOP = os.environ.get("EARLY_STOP", "0") == "1"
_sel = HERE / "data" / "lam_selection.json"
LAM = float(os.environ.get("LAM", 10.0 if EARLY_STOP or not _sel.exists() else json.loads(_sel.read_text())["choice"]))
STEP_CAP = int(os.environ.get("STEP_CAP", 16000))
MODE_SUFFIX = "_earlystop" if EARLY_STOP else ""
LR, BATCH, CHECK_EVERY, STOP_AUROC = 1e-3, 2048, 25, 0.6
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def synthetic_data(seed, d=64, n=20_000, d_opt=4.0):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = 1.0 / np.arange(1, d + 1)
    Sigma = R @ np.diag(lam) @ R.T
    L = R @ np.diag(np.sqrt(lam)) @ R.T
    Delta = rng.standard_normal(d)
    Delta *= d_opt / np.sqrt(Delta @ np.linalg.solve(Sigma, Delta))

    def draw(n):
        y = np.repeat([0, 1], n // 2)
        X = rng.standard_normal((n, d)) @ L + y[:, None] * Delta
        return X, y

    X_tr, y_tr = draw(n)
    X_te, y_te = draw(n)
    return X_tr, y_tr, X_te, y_te, dict(Sigma_true=Sigma, Delta_true=Delta)


def lm_data():
    sel = json.loads((DATA / "lm_selection.json").read_text())
    z = np.load(DATA / f"lm_acts_L{sel['chosen_layer']}.npz")
    return (z["X_train"].astype(np.float64), z["y_train"].astype(int),
            z["X_test"].astype(np.float64), z["y_test"].astype(int), dict(layer=sel["chosen_layer"]))


# ---------------------------------------------------------------- stats
def auc(s, y):
    return float(roc_auc_score(y, s))


def dprime(a, n):
    """sqrt2 * Phi^-1(AUROC), AUROC clipped to [1/(2n), 1-1/(2n)] (DEVIATIONS.md D1)."""
    a = np.clip(a, 1 / (2 * n), 1 - 1 / (2 * n))
    return float(np.sqrt(2) * norm.ppf(a))


def pooled_cov(X, y):
    Xc = np.concatenate([X[y == 0] - X[y == 0].mean(0), X[y == 1] - X[y == 1].mean(0)])
    return LedoitWolf(assume_centered=True).fit(Xc).covariance_


def cos_S(u, w, S):
    return float(u @ S @ w / np.sqrt((u @ S @ u) * (w @ S @ w)))


def cos_raw(u, w):
    return float(u @ w / (np.linalg.norm(u) * np.linalg.norm(w)))


def probe_family(X, y, rng):
    """25 logistic + 25 mass-mean probes, each inside a random subspace of log-uniform dim in [2, d]."""
    d = X.shape[1]
    mu = X.mean(0)
    Xs = (X - mu)
    probes, kinds, dims = [], [], []
    for i in range(N_PROBES):
        k = int(np.round(np.exp(rng.uniform(np.log(2), np.log(d)))))
        Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
        Z = Xs @ Q
        if i % 2 == 0:
            sd = Z.std(0)
            clf = LogisticRegression(max_iter=2000).fit(Z / sd, y)
            u = Q @ (clf.coef_[0] / sd)
            kinds.append("logistic")
        else:
            u = Q @ (Z[y == 1].mean(0) - Z[y == 0].mean(0))
            kinds.append("mass-mean")
        probes.append(u / np.linalg.norm(u))
        dims.append(k)
    return np.array(probes), kinds, dims


# ---------------------------------------------------------------- evader
class Evader(torch.nn.Module):
    def __init__(self, d, m, s):
        super().__init__()
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 256), torch.nn.GELU(),
            torch.nn.Linear(256, d))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # inputs standardised per dimension, edit expressed in per-dimension std units
        return x + self.s * self.net((x - self.m) / self.s)


def train_evader(X, y, w, S, retain, seed):
    torch.manual_seed(seed)
    d = X.shape[1]
    ev = Evader(d, X.mean(0), X.std(0) + 1e-8)
    opt = torch.optim.Adam(ev.parameters(), lr=LR)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y)
    wt = torch.tensor(w, dtype=torch.float32)
    evals, evecs = np.linalg.eigh(S)
    S_mhalf = torch.tensor(evecs @ np.diag(evals ** -0.5) @ evecs.T, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    log, step, a_w = [], 0, auc(X @ w, y)
    for step in range(1, STEP_CAP + 1):
        idx = torch.randint(0, len(Xt), (BATCH,), generator=g)
        xb, yb = Xt[idx], yt[idx]
        x2 = ev(xb)
        s = x2 @ wt
        s1, s0 = s[yb == 1], s[yb == 0]
        pooled = torch.sqrt(((s1 - s1.mean()) ** 2).sum() + ((s0 - s0.mean()) ** 2).sum()) / np.sqrt(BATCH - 2)
        dw = (s1.mean() - s0.mean()) / pooled
        edit = x2 - xb
        keep = ((edit @ S_mhalf) ** 2).sum(1).mean() if retain == "mahalanobis" else (edit ** 2).sum(1).mean()
        loss = LAM * dw ** 2 + keep
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                a_w = auc((ev(Xt) @ wt).numpy(), y)
            log.append(dict(step=step, train_auroc_w=a_w, loss=float(loss), keep=float(keep)))
            if EARLY_STOP and a_w <= STOP_AUROC:
                break

    def encode(Xn):
        with torch.no_grad():
            return ev(torch.tensor(Xn, dtype=torch.float32)).numpy().astype(np.float64)

    return encode, dict(steps=step, stopped=a_w <= STOP_AUROC, final_train_auroc_w=a_w, log=log)


# ---------------------------------------------------------------- one run
def run(setting, seed, retain):
    t0 = time.time()
    if setting == "synthetic":
        X_tr, y_tr, X_te, y_te, extra = synthetic_data(seed)
    else:
        X_tr, y_tr, X_te, y_te, extra = lm_data()
    rng = np.random.default_rng(1000 + seed)
    n_te = len(y_te)
    Delta = X_tr[y_tr == 1].mean(0) - X_tr[y_tr == 0].mean(0)
    w = Delta / np.linalg.norm(Delta)                       # mass-mean monitor
    S = pooled_cov(X_tr, y_tr)                              # Ledoit-Wolf pooled within-class
    probes, kinds, dims = probe_family(X_tr, y_tr, rng)

    enc, trainlog = train_evader(X_tr, y_tr, w, S, retain, seed)
    E_tr, E_te = enc(X_tr), enc(X_te)

    a_w0, a_w1 = auc(X_te @ w, y_te), auc(E_te @ w, y_te)
    dw0, dw1 = dprime(a_w0, n_te), dprime(a_w1, n_te)
    k = 1 - dw1 / dw0
    dw_mom = float(w @ Delta / np.sqrt(w @ S @ w))  # secondary: d' from train means + Sigma_hat
    rows = []
    for i, u in enumerate(probes):
        a0, a1 = auc(X_te @ u, y_te), auc(E_te @ u, y_te)
        du = dprime(a0, n_te)
        cw, cr = cos_S(u, w, S), cos_raw(u, w)
        pw = float(norm.cdf((du - k * dw0 * cw) / np.sqrt(2)))
        pr = float(norm.cdf((du - k * dw0 * cr) / np.sqrt(2)))
        rows.append(dict(probe=i, kind=kinds[i], subspace_dim=dims[i], auroc_before=a0, auroc_after=a1,
                         d_before=du, cos_white=cw, cos_raw=cr, drop=a0 - a1,
                         pred_drop_white=a0 - pw, pred_drop_raw=a0 - pr,
                         d_before_moments=float(u @ Delta / np.sqrt(u @ S @ u))))
    meas = np.array([r["drop"] for r in rows])
    pw_ = np.array([r["pred_drop_white"] for r in rows])
    pr_ = np.array([r["pred_drop_raw"] for r in rows])

    def r2c(a, b):
        return float(np.corrcoef(a, b)[0, 1] ** 2)

    def r2i(a, b):
        return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())

    # refit: fresh logistic probe on encoded train (standardised), AUROC on encoded test
    mu, sd = E_tr.mean(0), E_tr.std(0) + 1e-8
    refit = LogisticRegression(max_iter=3000).fit((E_tr - mu) / sd, y_tr)
    a_refit = auc(refit.decision_function((E_te - mu) / sd), y_te)
    mu0, sd0 = X_tr.mean(0), X_tr.std(0) + 1e-8
    full0 = LogisticRegression(max_iter=3000).fit((X_tr - mu0) / sd0, y_tr)
    a_full0 = auc(full0.decision_function((X_te - mu0) / sd0), y_te)

    # diagnostics
    S_after = pooled_cov(E_tr, y_tr)
    shift = (E_tr[y_tr == 1].mean(0) - E_tr[y_tr == 0].mean(0)) - Delta
    sw = lambda Z: np.concatenate([Z[y_tr == c] @ w - (Z[y_tr == c] @ w).mean() for c in (0, 1)]).std()
    summary = dict(
        setting=setting, seed=seed, retain=retain, **{k_: v for k_, v in extra.items() if k_ == "layer"},
        lam=LAM, early_stop=EARLY_STOP,
        n_train=len(y_tr), n_test=n_te, dim=X_tr.shape[1],
        auroc_w_before=a_w0, auroc_w_after=a_w1, k=k, dprime_w_before=dw0,
        r2_white=r2c(meas, pw_), r2_raw=r2c(meas, pr_),
        r2_white_nofit=r2i(meas, pw_), r2_raw_nofit=r2i(meas, pr_),
        corr_raw_white_cos=float(np.corrcoef([r["cos_white"] for r in rows], [r["cos_raw"] for r in rows])[0, 1]),
        r2_white_moments=r2c(meas, [r["auroc_before"] - norm.cdf((r["d_before_moments"] - k * dw_mom * r["cos_white"]) / np.sqrt(2)) for r in rows]),
        auroc_refit=a_refit, auroc_full_logistic_before=a_full0,
        n_probes_auroc_gt_0995=int(sum(r["auroc_before"] > 0.995 for r in rows)),
        cov_change_rel=float(np.linalg.norm(S_after - S) / np.linalg.norm(S)),
        w_score_sd_ratio=float(sw(E_tr) / sw(X_tr)),
        shift_cos_Sigma_w=cos_raw(shift, S @ w), shift_cos_w=cos_raw(shift, w),
        train=dict(steps=trainlog["steps"], stopped=trainlog["stopped"],
                   final_train_auroc_w=trainlog["final_train_auroc_w"]),
        seconds=round(time.time() - t0, 1),
    )
    out = DATA / "runs"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{setting}_{retain}_seed{seed}" + MODE_SUFFIX
    (out / f"{tag}_probes.json").write_text(json.dumps(rows, indent=1))
    (out / f"{tag}_trainlog.json").write_text(json.dumps(trainlog["log"]))
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez_compressed(out / f"{tag}.npz", w=w, probes=probes, probe_kind=np.array(kinds),
                        probe_subspace_dim=np.array(dims), Sigma_hat=S, Delta_hat=Delta,
                        X_train=f32(X_tr), y_train=y_tr, E_train=f32(E_tr),
                        X_test=f32(X_te), y_test=y_te, E_test=f32(E_te),
                        refit_scores=refit.decision_function((E_te - mu) / sd))
    return summary


if __name__ == "__main__":
    setting = sys.argv[1]
    retains = sys.argv[2:] or ["mahalanobis", "euclidean"]
    path = DATA / "runs" / f"summary_{setting}{MODE_SUFFIX}.json"
    allsum = json.loads(path.read_text()) if path.exists() else []
    for retain in retains:
        for seed in SEEDS:
            s = run(setting, seed, retain)
            allsum = [a for a in allsum if not (a["seed"] == seed and a["retain"] == retain)] + [s]
            path.write_text(json.dumps(allsum, indent=1))
            print(json.dumps({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in s.items()}), flush=True)
