"""Convergence check (Esben, 2026-10-20 11:16: "are you sure it did not just fail to converge?").

Refits the megaprobe (arm c) and the published axial probe (arm c2) on ONE trained model, outer folds 0-4, draw 0,
for much longer than the main run, and logs the curves. Same folds, draws, PCA, seeds and init as the main run.
At every checkpoint it records: training loss (BCE without the L2 term), training AUROC, inner-validation AUROC, and
the test-fold scores. Pooling the test scores over the 5 folds gives an out-of-fold (OOF) AUROC curve per epoch:
the quantity the post reports, for every length of training.

    uv run python projects/megaprobe-spin-lies/convergence.py c  <policy> <lr> <epochs> <tag> [l2]
    uv run python projects/megaprobe-spin-lies/convergence.py c2 <policy> <lr> <epochs> <tag>
-> data/convergence/<arm>_<policy>_<tag>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes as P  # noqa: E402
from score import SUBSETS  # noqa: E402

OUTD = P.DATA / "convergence"


def subsets(rows):
    out = {}
    for name, (fp, fn) in SUBSETS.items():
        pos = np.array([fp(r) for r in rows]); neg = np.array([fn(r) for r in rows])
        out[name] = (pos | neg, pos.astype(int))
    return out


def run_c(Z, offs, y, tr, te, seed, hp, log_every=10, mb=400):
    """probes.fit_c with logging; identical split, init and update rule."""
    rng = np.random.default_rng(seed)
    tr = np.array(tr); perm = rng.permutation(len(tr))
    nv = int(len(tr) * 0.2)
    va_rows, fit_rows = tr[perm[:nv]], tr[perm[nv:]]
    Xf, Mf = P.padded(Z, offs, fit_rows); Xv, Mv = P.padded(Z, offs, va_rows)
    Xt, Mt = P.padded(Z, offs, te)
    yf = torch.tensor(y[fit_rows], dtype=torch.float32, device=P.DEV)
    L, k = Z.shape[1], Z.shape[2]
    g = torch.Generator().manual_seed(seed)
    Q = {"q": (torch.randn(L, k, generator=g) * 0.01).to(P.DEV).requires_grad_(),
         "c": torch.zeros(L, device=P.DEV, requires_grad=True),
         "w": (torch.randn(L, k, generator=g) * 0.01).to(P.DEV).requires_grad_(),
         "b": torch.zeros((), device=P.DEV, requires_grad=True)}
    opt = torch.optim.Adam(Q.values(), lr=hp["lr"])
    pw = (yf == 0).sum() / (yf == 1).sum().clamp_min(1)
    nf = len(fit_rows)
    log = []
    for ep in range(hp["epochs"]):
        order = torch.from_numpy(rng.permutation(nf)) if nf > mb else torch.arange(nf)
        for s in range(0, nf, mb):
            b = order[s:s + mb].to(P.DEV)
            z = P.mega_forward(Q, Xf[b].float(), Mf[b])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yf[b], pos_weight=pw) \
                + hp["l2"] * (Q["q"].pow(2).sum() + Q["w"].pow(2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % log_every == log_every - 1:
            with torch.no_grad():
                zf = P.mega_forward(Q, Xf.float(), Mf)
                bce = float(torch.nn.functional.binary_cross_entropy_with_logits(zf, yf, pos_weight=pw))
                sv = P.mega_forward(Q, Xv.float(), Mv).cpu().numpy()
                st = np.concatenate([P.mega_forward(Q, Xt[s:s + 200].float(), Mt[s:s + 200]).cpu().numpy()
                                     for s in range(0, len(te), 200)])
            log.append({"epoch": ep, "train_bce": bce, "train_auroc": float(roc_auc_score(y[fit_rows], zf.cpu().numpy())),
                        "val_auroc": float(roc_auc_score(y[va_rows], sv)), "test": st.tolist()})
    del Xf, Xv, Xt; torch.mps.empty_cache()
    return log


def run_c2(X, offs, y, tr, te, seed, hp, mb=32):
    """fit_c2.fit_c2 with per-epoch logging; identical split, init, scaling and update rule."""
    import fit_c2 as C2
    rng = np.random.default_rng(seed)
    tr = np.array(tr); perm = rng.permutation(len(tr))
    nv = int(len(tr) * 0.2)
    va, fit = tr[perm[:nv]], tr[perm[nv:]]
    torch.manual_seed(seed)
    m = C2.AxialProbe(X.shape[2], len(C2.LAYERS), normalize_input="unit_norm", use_checkpoint=False).to(C2.DEV)
    sq = torch.zeros(len(C2.LAYERS)); cnt = 0
    for i in fit:
        x = torch.from_numpy(np.asarray(X[offs[i]:offs[i + 1]], np.float32)); sq += x.pow(2).sum(-1).sum(0); cnt += len(x)
    m.input_scale.copy_((sq / cnt).sqrt().view(1, -1, 1, 1).to(C2.DEV))
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    yt = torch.tensor(y, dtype=torch.float32, device=C2.DEV)
    pw = (yt[fit] == 0).sum() / (yt[fit] == 1).sum().clamp_min(1)
    lossf = lambda z, t: torch.nn.functional.binary_cross_entropy_with_logits(z, t, pos_weight=pw)
    lens = np.diff(offs)[fit]
    log = []
    for ep in range(hp["epochs"]):
        m.train()
        srt = fit[np.argsort(lens + rng.random(len(fit)), kind="stable")]
        batches = [srt[s:s + mb] for s in range(0, len(srt), mb)]
        for bi in rng.permutation(len(batches)):
            rows = batches[bi]
            B, M, last = C2.batch(X, offs, rows)
            z = m(B, M)[..., 0][torch.arange(len(rows), device=C2.DEV), last]
            loss = lossf(z, yt[rows])
            opt.zero_grad(); loss.backward(); opt.step()
        sf = C2.score(m, X, offs, fit); sv = C2.score(m, X, offs, va); st = C2.score(m, X, offs, np.asarray(te))
        log.append({"epoch": ep, "train_bce": float(lossf(torch.tensor(sf, device=C2.DEV), yt[fit])),
                    "train_auroc": float(roc_auc_score(y[fit], sf)),
                    "val_bce": float(lossf(torch.tensor(sv, device=C2.DEV), yt[va])),
                    "val_auroc": float(roc_auc_score(y[va], sv)), "test": st.tolist()})
    return log


def main():
    arm, p, lr, epochs, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    OUTD.mkdir(parents=True, exist_ok=True)
    fm = P.folds(); rows = P.load_rows(p); D = P.draws_for(p, rows, fm); y = P.yvec(rows)
    fold = np.array([fm[r["prompt_id"]] for r in rows])
    cache = P.Cache(p, "own")
    if arm == "c2":
        import fit_c2 as C2
        X, offs = C2.build(cache, p)
        hp = dict(C2.HP, lr=lr, epochs=epochs)
    else:
        hp = dict(P.HP, lr=lr, epochs=epochs, **({"l2": float(sys.argv[6])} if len(sys.argv) > 6 else {}))
    t0 = time.time(); per_fold = {}
    for k in range(P.N_FOLDS):
        te = np.where(fold == k)[0]; tr = np.array(D[k][0]); seed = P.SEED + 100 * k + 0
        if arm == "c":
            pca = P.pca_fit([(cache, [i for i in range(P.N_ROWS) if fold[i] != k])], P.HP["k"])
            Z, o = P.pca_project(cache, pca)
            per_fold[k] = {"te": te.tolist(), "log": run_c(Z, o, y, tr.tolist(), te, seed, hp)}
            del Z; torch.mps.empty_cache()
        else:
            per_fold[k] = {"te": te.tolist(), "log": run_c2(X, offs, y, tr, te, seed, hp)}
        print(f"{arm} {p} {tag} fold {k} done {time.time() - t0:.0f}s", flush=True)
    # pooled OOF AUROC per checkpoint
    S = subsets(rows)
    n_ck = len(per_fold[0]["log"])
    curve = []
    for j in range(n_ck):
        s = np.full(P.N_ROWS, np.nan)
        for k, f in per_fold.items():
            s[f["te"]] = f["log"][j]["test"]
        row = {"epoch": per_fold[0]["log"][j]["epoch"]}
        for name, (use, yy) in S.items():
            ok = use & np.isfinite(s)
            row[name] = float(roc_auc_score(yy[ok], s[ok]))
        for key in ("train_bce", "train_auroc", "val_auroc") + (("val_bce",) if arm == "c2" else ()):
            row[key] = float(np.mean([f["log"][j][key] for f in per_fold.values()]))
        curve.append(row)
    # early stop replayed on the long log: per fold, the checkpoint with the best inner-validation AUROC (c: the
    # main run's rule) or the lowest validation loss (c2: the main run's rule), over all epochs
    sel = {}
    for horizon in sorted({hp["epochs"], 300 if arm == "c" else 10}):
        s = np.full(P.N_ROWS, np.nan); eps = []
        for f in per_fold.values():
            lg = [e for e in f["log"] if e["epoch"] < horizon]
            b = max(lg, key=lambda e: e["val_auroc"]) if arm == "c" else min(lg, key=lambda e: e["val_bce"])
            s[f["te"]] = b["test"]; eps.append(b["epoch"])
        sel[f"early_stop_within_{horizon}"] = {"epochs": eps, **{n: float(roc_auc_score(yy[use], s[use]))
                                                                  for n, (use, yy) in S.items()}}
    for f in per_fold.values():  # raw test scores kept (4 decimals) for reanalysis
        for e in f["log"]:
            e["test"] = [round(v, 4) for v in e["test"]]
    out = OUTD / f"{arm}_{p}_{tag}.json"
    out.write_text(json.dumps({"arm": arm, "policy": p, "hp": hp, "draw": 0, "curve": curve, "early_stop": sel,
                               "per_fold": per_fold, "seconds": round(time.time() - t0)}))
    print(json.dumps(sel), flush=True)
    for r in curve:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    main()
