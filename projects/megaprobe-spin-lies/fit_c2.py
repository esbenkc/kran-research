"""Arm c2 (DEVIATIONS D1): the FIBS AxialProbe (fibs_axial.py, vendored from AlignmentResearch/fibs-probes @6cae61d),
code defaults: d_proj 128, 4 attention heads, 1 axial block, d_ff 1024, CLS pooling over layers, output capped
10*tanh(x/10). Layers: block outputs 7, 10, 13, 16, 19, 22 (the FIBS choice for a 24-block Qwen).

Read window, exactly: the answer tokens (as in arm c) followed by ONE more position, the final token of the follow-up
turn ("...Did you respond truthfully?<|im_end|>\\n<|im_start|>assistant\\nYes", the last cached follow.npy token).
The other 16 follow-up tokens are not fed. Token attention is causal; the score is the logit at that last position.
Input scaling: per-layer RMS scale on the training draw (unit_norm, FIBS App. A.1).
Training: AdamW, batch 32, class-balanced BCE, early stop on the loss of an inner 20% split (FIBS keeps the lowest-
validation-loss checkpoint), at most E epochs. lr / weight decay / E from inner CV on the base model, outer fold 0
training rows only (pilot), frozen before any trained-model fit. Same folds, draws and label sets (all, fact) as arm c.

    uv run python projects/megaprobe-spin-lies/fit_c2.py pilot
    uv run python projects/megaprobe-spin-lies/fit_c2.py fit [policy ...]
-> data/hp_c2.json, data/fits/<policy>_own_c2.json (scores c2_all, c2_fact)
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
from fibs_axial import AxialProbe  # noqa: E402
from probes import (CACHE, DATA, N_DRAWS, N_FIT, N_FOLDS, N_ROWS, OUT, POLICIES, SEED, Cache, draws_for, fact_train,  # noqa: E402
                    folds, load_rows, save_scores, yvec)

LAYERS = [7, 10, 13, 16, 19, 22]
DEV = __import__("os").environ.get("C2_DEV", "cpu")  # CPU: on MPS this job's footprint reached 3.2-3.4 GB
HP_FILE = DATA / "hp_c2.json"
HP = json.loads(HP_FILE.read_text())["chosen"] if HP_FILE.exists() else {"lr": 3e-4, "wd": 0.01, "epochs": 10}
N_DRAWS_C2 = json.loads(HP_FILE.read_text()).get("n_draws_c2", N_DRAWS) if HP_FILE.exists() else N_DRAWS


def build(cache, p):
    """-> X (ntok + n, 6, 896) fp16 memmap on disk (row i = its answer tokens, then its final follow-up token), offsets.
    Kept on disk (clean file pages are not in the job's footprint); each minibatch is read from it."""
    lens = np.diff(cache.offsets) + 1
    offs = np.concatenate([[0], np.cumsum(lens)])
    f = CACHE / f"{p}_own" / "c2_x.npy"
    if not f.exists():
        Xn = np.lib.format.open_memmap(f.with_suffix(".tmp.npy"), mode="w+", dtype=np.float16,
                                       shape=(int(offs[-1]), len(LAYERS), cache.tok.shape[2]))
        li = [l - 1 for l in LAYERS]
        for s in range(0, N_ROWS, 100):
            e = min(N_ROWS, s + 100)
            T = np.asarray(cache.tok[cache.offsets[s]:cache.offsets[e]][:, li])
            Fz = np.asarray(cache.follow[s:e, -1][:, li])
            for i in range(s, e):
                a, b = cache.offsets[i] - cache.offsets[s], cache.offsets[i + 1] - cache.offsets[s]
                Xn[offs[i]:offs[i + 1] - 1] = T[a:b]
                Xn[offs[i + 1] - 1] = Fz[i - s]
        Xn.flush(); del Xn
        f.with_suffix(".tmp.npy").rename(f)
    return np.load(f, mmap_mode="r"), offs


def batch(X, offs, rows):
    T = -(-max(int(offs[i + 1] - offs[i]) for i in rows) // 32) * 32  # few distinct shapes
    B = np.zeros((len(rows), len(LAYERS), T, X.shape[2]), np.float32)
    M = np.zeros((len(rows), T), bool)
    last = np.zeros(len(rows), np.int64)
    for j, i in enumerate(rows):
        n = int(offs[i + 1] - offs[i])
        B[j, :, :n] = np.asarray(X[offs[i]:offs[i + 1]]).transpose(1, 0, 2)
        M[j, :n] = True; last[j] = n - 1
    return torch.from_numpy(B).to(DEV), torch.from_numpy(M).to(DEV), torch.from_numpy(last).to(DEV)


def score(m, X, offs, rows, mb=32):
    rows = np.asarray(rows)
    order = np.argsort(np.diff(offs)[rows], kind="stable")  # length-sorted batches, results put back in row order
    out = np.zeros(len(rows), np.float32)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(rows), mb):
            idx = order[s:s + mb]
            B, M, last = batch(X, offs, rows[idx])
            z = m(B, M)[..., 0]
            out[idx] = z[torch.arange(len(last), device=DEV), last].cpu().numpy()
    return out


def fit_c2(X, offs, y, tr, te, seed, hp=None, val_frac=0.2, mb=32):
    hp = hp or HP
    rng = np.random.default_rng(seed)
    tr = np.array(tr); perm = rng.permutation(len(tr))
    nv = int(len(tr) * val_frac)
    va, fit = tr[perm[:nv]], tr[perm[nv:]]
    torch.manual_seed(seed)
    m = AxialProbe(X.shape[2], len(LAYERS), normalize_input="unit_norm", use_checkpoint=False).to(DEV)
    sq = torch.zeros(len(LAYERS)); cnt = 0  # per-layer RMS scale over the fit rows' tokens
    for i in fit:
        x = torch.from_numpy(np.asarray(X[offs[i]:offs[i + 1]], np.float32)); sq += x.pow(2).sum(-1).sum(0); cnt += len(x)
    m.input_scale.copy_((sq / cnt).sqrt().view(1, -1, 1, 1).to(DEV))
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    pw = (yt[fit] == 0).sum() / (yt[fit] == 1).sum().clamp_min(1)
    lossf = lambda z, t: torch.nn.functional.binary_cross_entropy_with_logits(z, t, pos_weight=pw)
    best = (np.inf, None, -1, 0.5)
    lens = np.diff(offs)[fit]
    for ep in range(hp["epochs"]):
        m.train()
        # length-bucketed minibatches (rows of similar length together, less padding); batch order shuffled each epoch
        srt = fit[np.argsort(lens + rng.random(len(fit)), kind="stable")]
        batches = [srt[s:s + mb] for s in range(0, len(srt), mb)]
        for bi in rng.permutation(len(batches)):
            rows = batches[bi]
            B, M, last = batch(X, offs, rows)
            z = m(B, M)[..., 0][torch.arange(len(rows), device=DEV), last]
            loss = lossf(z, yt[rows])
            opt.zero_grad(); loss.backward(); opt.step()
        sv = score(m, X, offs, va)
        vl = float(lossf(torch.tensor(sv, device=DEV), yt[va]))
        if vl < best[0]:
            vau = roc_auc_score(y[va], sv) if len(set(y[va])) == 2 else 0.5
            best = (vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep, vau)
    m.load_state_dict(best[1])
    s = score(m, X, offs, np.asarray(te))
    del m, opt
    if DEV == "mps":
        torch.mps.empty_cache()
    return s, float(best[3]), int(best[2])


def pilot():
    rows, fm = load_rows("base"), folds()
    y = yvec(rows)
    tr0 = np.array([i for i, r in enumerate(rows) if fm[r["prompt_id"]] != 0 and r["y_judge"] is not None])
    X, offs = build(Cache("base", "own"), "base")
    rng = np.random.default_rng(SEED)
    inner = rng.permutation(len(tr0)) % 4
    res = []
    for lr in (1e-4, 3e-4, 1e-3):  # 3 configs only: one c2 fit costs ~20 s on MPS and D1 drops c2 after 04:00
        for wd in (0.01,):
            hp = {"lr": lr, "wd": wd, "epochs": 10}
            t1 = time.time(); au = []
            for j in range(4):
                itr, ite = tr0[inner != j], tr0[inner == j]
                itr = np.sort(rng.choice(itr, min(N_FIT, len(itr)), replace=False))
                s, _, _ = fit_c2(X, offs, y, itr, ite, seed=SEED + j, hp=hp)
                au.append(float(roc_auc_score(y[ite], s)))
            res.append({**hp, "inner_auroc": float(np.mean(au)), "per_fold": au,
                        "sec_per_fit": round((time.time() - t1) / 4, 2)})
            print(res[-1], flush=True)
    ch = max(res, key=lambda r: r["inner_auroc"])
    n_c2 = N_DRAWS if ch["sec_per_fit"] <= 10 else 3
    HP_FILE.write_text(json.dumps({"grid": res, "chosen": {k: ch[k] for k in ("lr", "wd", "epochs")}, "n_draws_c2": n_c2,
                                   "rule": "max mean inner AUROC (all lies vs truthful); base; outer fold 0 training "
                                           "rows; inner fits on 400-row subsamples"}, indent=1))
    print("chosen", ch, "n_draws_c2", n_c2)


def fit_policy(p):
    f_out = OUT / f"{p}_own_c2.json"
    if f_out.exists():
        print("skip", f_out.name); return
    fm, rows = folds(), load_rows(p)
    D = draws_for(p, rows, fm)
    y = yvec(rows)
    fold = np.array([fm[r["prompt_id"]] for r in rows])
    cache = Cache(p, "own")
    X, offs = build(cache, p)
    sc = {n: np.full((N_ROWS, N_DRAWS), np.nan) for n in ("c2_all", "c2_fact")}
    info = {}
    t0 = time.time()
    for k in range(N_FOLDS):
        te = np.where(fold == k)[0]
        for d in range(N_DRAWS_C2):
            s, vau, ep = fit_c2(X, offs, y, D[k][d], te, SEED + 100 * k + d)
            sc["c2_all"][te, d] = s; info[f"{k}_all_{d}"] = {"val_auroc": vau, "epoch": ep}
        s, vau, ep = fit_c2(X, offs, y, fact_train(rows, fm, k), te, SEED + 100 * k + 50)
        sc["c2_fact"][te, 0] = s; info[f"{k}_fact_0"] = {"val_auroc": vau, "epoch": ep}
        print(f"{p} c2 fold {k} done {time.time() - t0:.0f}s", flush=True)
    save_scores(f_out, {"policy": p, "read": "own", "arm": "c2", "hp": HP, "n_draws_c2": N_DRAWS_C2, "layers": LAYERS,
                        "info": info, "prompt_ids": cache.prompt_ids, "seconds": round(time.time() - t0)}, sc)


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot()
    else:
        for p in sys.argv[2:] or POLICIES:
            fit_policy(p)
