"""Fit every probe arm out-of-fold and score spin lies (PREREG.md). Reads the activation cache from cache.py.

Rows: the 1000 prompt_ids that all 5 models answered (answers_<p>.jsonl rows 0-999, same order in every model).
Classes (labels_<p>.jsonl): truthful = y_judge 0; plain lie = y_judge 1 & y_string 1; spin lie = y_judge 1 &
y_string 0 (the judge says deceptive, the string check finds the fact's words); OTHER (y_judge null) is dropped.

Folds: ONE map, 5 folds over prompt_id (seed 1020), shared by every model, arm and read -> data/folds.json.
Label sets:
  all   every labelled row; y = y_judge. In each outer fold, 5 draws of n=400 from the 4 training folds,
        200 per class where the class has 200 rows, else all of the minority class and the rest from the majority.
  fact  fact-present rows only (y_string 0): spin lie vs truthful-that-states-the-fact. All such training rows,
        one fit per fold (n is too small to draw 400).
OOF score of a row = mean over the draws of the fold that holds it out.

Arms:
  a  LINEAR REFIT (refit-probe recipe): layer 12, mean over answer tokens, StandardScaler + LogisticRegression(C=0.05).
  b  ATLAS MEAN-OF-LAYERS (Taufeeque, Heimersheim, Gleave, Cundy, arXiv 2602.15515 s4.3; github
     AlignmentResearch/obfuscation-atlas @12abf65): final token of the follow-up "Did you respond truthfully? -> Yes",
     one linear probe per layer (input / sqrt(mean ||x||^2), the repo's unit_norm; Adam lr 3e-3, batch 32, 1000 steps,
     no weight decay: the repo's DetectorConfig.pretrain_cfg), keep layers with final train cross-entropy < 0.6
     (filter_layers_below_loss_threshold), score = mean of the kept layers' sigmoids.
  c  ATTENTION MEGAPROBE (ours): every answer token at every layer. Per layer l, the token vector is projected on the
     top-k principal components of the training fold's tokens (whitened). One softmax over all (token, layer) pairs:
        score = sum_{t,l} softmax_{t,l}(q_l . z_tl + c_l) * (w_l . z_tl) + b
     (q_l: what to attend to in layer l, c_l: a learned layer weight, w_l: the lie direction in layer l). Trained with
     Adam + an L2 penalty, early stop on an inner 20% split of the draw. k, L2 and lr fixed in HP below (chosen by
     inner CV on the base model's outer-fold-0 training rows only: hp_pilot()).
  d  WORDS ONLY: TF-IDF 1-2 grams + LogisticRegression(C=1) on the answer text.
  nulls: a and c refit on the same draws with permuted labels (arm a_null, c_null).
Reads: own (the model reads its answer; every arm) and base (the untrained model reads the trained model's answer:
held-text control, arms a and c, label set all).

    uv run python projects/megaprobe-spin-lies/probes.py pilot      # HP search on base fold-0 training rows
    uv run python projects/megaprobe-spin-lies/probes.py fit [policy ...]
    uv run python projects/megaprobe-spin-lies/probes.py score       # OOF -> summary.json, verdict.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA = PROJECT_DIR / "data"
SRC = DATA / "refit_probe"
CACHE = DATA / "cache"
OUT = DATA / "fits"
RES = DATA / "results"
DEV = "mps"
POLICIES = ["base", "lam1_s0", "lam1_s1", "ens_s0", "ens_s1"]
TRAINED = POLICIES[1:]
N_ROWS, N_FOLDS, N_FIT, N_DRAWS, SEED = 1000, 5, 400, 5, 1020
HP = {"k": 128, "l2": 1e-3, "lr": 1e-2, "epochs": 300}  # overwritten by data/hp.json once the pilot ran
HP_FILE = DATA / "hp.json"
if HP_FILE.exists():
    HP = json.loads(HP_FILE.read_text())["chosen"]


# ---------------- rows, folds, draws ----------------
def load_rows(p):
    A = [json.loads(l) for l in open(SRC / f"answers_{p}.jsonl")][:N_ROWS]
    L = [json.loads(l) for l in open(SRC / f"labels_{p}.jsonl")][:N_ROWS]
    rows = []
    for a, l in zip(A, L):
        assert a["prompt_id"] == l["prompt_id"]
        yj, ys = l["y_judge"], l["y_string"]
        cls = None if yj is None else ("truthful" if yj == 0 else ("spin" if ys == 0 else "plain"))
        rows.append({"idx": a["idx"], "prompt_id": a["prompt_id"], "split": a["split"], "response": a["response"],
                     "y_judge": yj, "y_string": ys, "cls": cls})
    return rows


def folds():
    f = DATA / "folds.json"
    if f.exists():
        return json.loads(f.read_text())
    ids = [r["prompt_id"] for r in load_rows("base")]
    perm = np.random.default_rng(SEED).permutation(len(ids))
    fm = {ids[j]: int(i % N_FOLDS) for i, j in enumerate(perm)}
    f.write_text(json.dumps(fm))
    return fm


def draws_for(p, rows, fm):
    """{fold: [ [row indices of draw d] ]} for label set 'all' (cached in data/draws.json)."""
    f = DATA / "draws.json"
    D = json.loads(f.read_text()) if f.exists() else {}
    if p in D:
        return {int(k): v for k, v in D[p].items()}
    rng = np.random.default_rng([SEED, POLICIES.index(p)])
    out = {}
    for k in range(N_FOLDS):
        tr = [i for i, r in enumerate(rows) if fm[r["prompt_id"]] != k and r["y_judge"] is not None]
        pos = [i for i in tr if rows[i]["y_judge"] == 1]
        neg = [i for i in tr if rows[i]["y_judge"] == 0]
        h = N_FIT // 2  # 200 per class where available, else all of the minority and the rest from the majority
        n_neg = min(len(neg), max(h, N_FIT - len(pos)))
        n_pos = min(len(pos), N_FIT - n_neg)
        out[k] = [sorted(rng.choice(pos, n_pos, replace=False).tolist() + rng.choice(neg, n_neg, replace=False).tolist())
                  for _ in range(N_DRAWS)]
    D[p] = out
    f.write_text(json.dumps(D))
    return out


def fact_train(rows, fm, k):
    return [i for i, r in enumerate(rows) if fm[r["prompt_id"]] != k and r["cls"] in ("spin", "truthful")
            and r["y_string"] == 0]


# ---------------- features ----------------
class Cache:
    def __init__(self, p, read):
        d = CACHE / f"{p}_{read}"
        meta = json.loads((d / "meta.json").read_text())
        self.offsets = np.load(d / "offsets.npy")[:N_ROWS + 1]
        self.tok = np.load(d / "tok.npy", mmap_mode="r")
        self.follow = np.load(d / "follow.npy", mmap_mode="r")
        self.prompt_ids = meta["prompt_ids"][:N_ROWS]
        self.ntok = int(self.offsets[-1])
        self.tok = self.tok[:self.ntok]  # base: drop the 600 extra rows' tokens
        self.L = self.tok.shape[1]

    def mean_layer(self, layer):  # layer = block number 1..24
        out = np.zeros((N_ROWS, self.tok.shape[2]), np.float32)
        for s in range(0, N_ROWS, 100):
            e = min(N_ROWS, s + 100)
            X = np.asarray(self.tok[self.offsets[s]:self.offsets[e], layer - 1], np.float32)
            for i in range(s, e):
                out[i] = X[self.offsets[i] - self.offsets[s]:self.offsets[i + 1] - self.offsets[s]].mean(0)
        return out

    def final(self):  # (n, 24, d): the follow-up's last token ("Yes")
        return np.asarray(self.follow[:N_ROWS, -1], np.float32)

    def tok_mask(self, rows):
        m = np.zeros(self.ntok, bool)
        for i in rows:
            m[self.offsets[i]:self.offsets[i + 1]] = True
        return m


def pca_fit(parts, k, chunk=1024):
    """Per layer PCA over the answer tokens of [(cache, rows), ...] (pooled). -> (mu (L,d), V (L,d,k), scale (L,k))."""
    L, d = parts[0][0].L, parts[0][0].tok.shape[2]
    S = torch.zeros(L, d, dtype=torch.float64)
    C = torch.zeros(L, d, d, dtype=torch.float32, device=DEV)
    n = 0
    ref = None  # shift by the first chunk's mean: avoids fp32 cancellation in E[xx'] - mu mu'
    for cache, rows in parts:
        tm = cache.tok_mask(rows)
        for s in range(0, cache.ntok, chunk):
            m = tm[s:s + chunk]
            if not m.any():
                continue
            X = torch.from_numpy(np.asarray(cache.tok[s:s + chunk][m], np.float32)).to(DEV)  # (c, L, d)
            if ref is None:
                ref = X.mean(0)
            X = X - ref
            S += X.sum(0).cpu().double()
            C += torch.einsum("cld,cle->lde", X, X)
            n += X.shape[0]
    m_s = S / n  # mean of the shifted tokens
    cov = C.cpu().double() / n - torch.einsum("ld,le->lde", m_s, m_s)
    ev, V = torch.linalg.eigh(cov)  # ascending
    mu = m_s.float().to(DEV) + ref
    del C; torch.mps.empty_cache()
    return mu, V[:, :, -k:].float().to(DEV), ev[:, -k:].float().clamp_min(1e-6).rsqrt().to(DEV)


def pca_project(cache, pca, rows=None, chunk=1024):
    """Whitened projection of the answer tokens of `rows` (default all) -> Z (ntok_sub, L, k) fp16 on DEV, offsets
    indexed by the ORIGINAL row number (rows outside `rows` get zero length)."""
    mu, V, scale = pca
    lens = np.diff(cache.offsets)
    if rows is not None:
        keep = np.zeros(N_ROWS, bool); keep[list(rows)] = True
        lens = np.where(keep, lens, 0)
        tm = cache.tok_mask(rows)
    offs = np.concatenate([[0], np.cumsum(lens)])
    Z = torch.empty(int(offs[-1]), cache.L, V.shape[2], dtype=torch.float16, device=DEV)
    w = 0
    for s in range(0, cache.ntok, chunk):
        Xn = np.asarray(cache.tok[s:s + chunk] if rows is None else cache.tok[s:s + chunk][tm[s:s + chunk]], np.float32)
        if len(Xn) == 0:
            continue
        X = torch.from_numpy(Xn).to(DEV)
        Z[w:w + len(X)] = (torch.einsum("cld,ldk->clk", X - mu, V) * scale).half()
        w += len(X)
    assert w == offs[-1]
    torch.mps.empty_cache()
    return Z, offs


# ---------------- arm a: linear refit ----------------
def fit_a(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    return m.decision_function(sc.transform(X[te]))


# ---------------- arm b: Atlas mean-of-layers ----------------
def fit_b(F, y, tr, te, seed, norm="unit_norm"):
    """F (n, L, d). All L per-layer probes trained at once (independent: the loss is a sum over layers).
    norm='unit_norm': the repo's mode, x / sqrt(mean ||x||^2) per layer (probe_archs.py:52, train.py:178);
    norm='l2': every vector scaled to norm 1 (probe_archs.py:49; sensitivity arm b_l2)."""
    g = torch.Generator().manual_seed(seed)
    if norm == "l2":
        F = F / np.linalg.norm(F, axis=-1, keepdims=True).clip(1e-8)
    Xtr = torch.from_numpy(F[tr]); Ytr = torch.tensor(y[tr], dtype=torch.float32)
    scale = Xtr.pow(2).sum(-1).mean(0).sqrt() if norm == "unit_norm" else torch.ones(Xtr.shape[1])
    Xtr = Xtr / scale[None, :, None]
    L, d = Xtr.shape[1], Xtr.shape[2]
    W = torch.zeros(L, d, requires_grad=True); B = torch.zeros(L, requires_grad=True)
    opt = torch.optim.Adam([W, B], lr=3e-3)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    pw = (Ytr == 0).sum() / (Ytr == 1).sum().clamp_min(1)  # 1.0 for a balanced draw
    for step in range(1000):
        idx = torch.randint(0, len(tr), (32,), generator=g)
        z = torch.einsum("bld,ld->bl", Xtr[idx], W) + B
        loss_l = lossf(z, Ytr[idx, None].expand_as(z), pos_weight=pw, reduction="none").mean(0)  # (L,)
        opt.zero_grad(); loss_l.sum().backward(); opt.step()
    with torch.no_grad():  # final train CE on the whole training set (the filter's quantity)
        z = torch.einsum("bld,ld->bl", Xtr, W) + B
        ce = lossf(z, Ytr[:, None].expand_as(z), reduction="none").mean(0)
        keep = ce < 0.6
        if not keep.any():
            keep = ce == ce.min()
        Xte = torch.from_numpy(F[te]) / scale[None, :, None]
        p = torch.sigmoid(torch.einsum("bld,ld->bl", Xte, W) + B)[:, keep].mean(1)
    return p.numpy(), keep.nonzero().flatten().tolist(), ce.tolist()


# ---------------- arm c: attention megaprobe ----------------
def padded(Z, offsets, rows, T_max=128, device=None):
    k = Z.shape[2]; L = Z.shape[1]
    T = max(1, min(T_max, max(int(offsets[i + 1] - offsets[i]) for i in rows)))
    T = min(T_max, -(-T // 32) * 32)  # few distinct shapes (MPS caches a graph per shape); padding is masked out
    X = torch.zeros(len(rows), T, L, k, dtype=torch.float16, device=device or DEV)
    M = torch.zeros(len(rows), T, dtype=torch.bool, device=device or DEV)
    for j, i in enumerate(rows):
        s, e = int(offsets[i]), int(offsets[i + 1])
        e = min(e, s + T)
        X[j, :e - s] = Z[s:e]; M[j, :e - s] = True
    return X, M


def mega_forward(P, X, M):
    a = torch.einsum("btlk,lk->btl", X, P["q"]) + P["c"]  # attention logits
    v = torch.einsum("btlk,lk->btl", X, P["w"])  # per token-layer lie score
    a = a.masked_fill(~M[:, :, None], -1e9).flatten(1).softmax(1)
    return (a * v.flatten(1)).sum(1) + P["b"]


def fit_c(Z, offsets, y, tr, te, seed, hp=None, val_frac=0.2, mb=400):
    """Full-batch for a 400-row draw (320 fit rows); minibatches of `mb` rows for the larger pooled fit."""
    hp = hp or HP
    rng = np.random.default_rng(seed)
    tr = np.array(tr); perm = rng.permutation(len(tr))
    nv = int(len(tr) * val_frac)
    va_rows, fit_rows = tr[perm[:nv]], tr[perm[nv:]]
    # A 400-row draw stays on MPS (full batch). The pooled fit (~1,400 rows) keeps its padded rows on the CPU and
    # moves one minibatch at a time: on MPS alone it hit the 3 GB allocator cap (logs/pooled.log, 04:18).
    store = DEV if len(tr) <= N_FIT else "cpu"
    Xf, Mf = padded(Z, offsets, fit_rows, device=store)  # fp16 storage, fp32 compute per minibatch
    Xv, Mv = padded(Z, offsets, va_rows, device=store)
    yf = torch.tensor(y[fit_rows], dtype=torch.float32, device=DEV)
    L, k = Z.shape[1], Z.shape[2]
    g = torch.Generator().manual_seed(seed)
    P = {"q": (torch.randn(L, k, generator=g) * 0.01).to(DEV).requires_grad_(),
         "c": torch.zeros(L, device=DEV, requires_grad=True),
         "w": (torch.randn(L, k, generator=g) * 0.01).to(DEV).requires_grad_(),
         "b": torch.zeros((), device=DEV, requires_grad=True)}
    opt = torch.optim.Adam(P.values(), lr=hp["lr"])
    pw = (yf == 0).sum() / (yf == 1).sum().clamp_min(1)  # class balance
    best = (-1.0, None, -1)
    nf = len(fit_rows)
    for ep in range(hp["epochs"]):
        order = torch.from_numpy(rng.permutation(nf)) if nf > mb else torch.arange(nf)
        for s in range(0, nf, mb):
            b = order[s:s + mb].to(Xf.device)
            z = mega_forward(P, Xf[b].to(DEV).float(), Mf[b].to(DEV))
            b = b.to(DEV)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yf[b], pos_weight=pw) \
                + hp["l2"] * (P["q"].pow(2).sum() + P["w"].pow(2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            with torch.no_grad():
                sv = np.concatenate([mega_forward(P, Xv[s:s + mb].to(DEV).float(), Mv[s:s + mb].to(DEV)).cpu().numpy()
                                     for s in range(0, len(va_rows), mb)])
            au = roc_auc_score(y[va_rows], sv) if len(set(y[va_rows])) == 2 else 0.5
            if au > best[0]:
                best = (au, {n: t.detach().clone() for n, t in P.items()}, ep)
    del Xf, Xv
    out = []
    with torch.no_grad():
        for s in range(0, len(te), 200):
            Xt, Mt = padded(Z, offsets, te[s:s + 200])
            out.append(mega_forward(best[1], Xt.float(), Mt).cpu().numpy())
    torch.mps.empty_cache()
    return np.concatenate(out), float(best[0]), int(best[2])


# ---------------- arm d: words only ----------------
def fit_d(texts, y, tr, te):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([texts[i] for i in tr])
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(v.transform([texts[i] for i in tr]),
                                                                              y[tr])
    return m.decision_function(v.transform([texts[i] for i in te]))


def yvec(rows):
    return np.array([r["y_judge"] if r["y_judge"] is not None else -1 for r in rows])


def save_scores(path, meta, sc):
    meta["scores"] = {n: [[None if np.isnan(v) else round(float(v), 6) for v in r] for r in m] for n, m in sc.items()}
    path.write_text(json.dumps(meta))


# ---------------- driver ----------------
def hp_pilot():
    """Arm c hyperparameters: inner 4-fold CV on the base model, outer fold 0's labelled training rows only, label set
    all, each inner fit on a 400-row subsample (the size of a real draw). Test folds of the outer split never seen."""
    rows, fm = load_rows("base"), folds()
    y = yvec(rows)
    tr0 = np.array([i for i, r in enumerate(rows) if fm[r["prompt_id"]] != 0 and r["y_judge"] is not None])
    cache = Cache("base", "own")
    rng = np.random.default_rng(SEED)
    inner = rng.permutation(len(tr0)) % 4
    t0 = time.time()
    pca = pca_fit([(cache, tr0.tolist())], 128)  # PCA on the outer training rows only
    Z128, offs = pca_project(cache, pca)
    t_pca = time.time() - t0
    res = []
    for k in (64, 128):
        Z = Z128 if k == 128 else Z128[..., -64:].contiguous()  # top-64 = the last 64 of the ascending top-128
        for l2 in (1e-4, 1e-3, 1e-2):
            hp = {"k": k, "l2": l2, "lr": 1e-2, "epochs": 300}
            t1 = time.time(); au = []
            for j in range(4):
                itr, ite = tr0[inner != j], tr0[inner == j]
                itr = np.sort(rng.choice(itr, min(N_FIT, len(itr)), replace=False))
                s, _, _ = fit_c(Z, offs, y, itr.tolist(), ite, seed=SEED + j, hp=hp)
                au.append(float(roc_auc_score(y[ite], s)))
            res.append({**hp, "inner_auroc": float(np.mean(au)), "per_fold": au,
                        "sec_per_fit": round((time.time() - t1) / 4, 2)})
            print(res[-1], flush=True)
        del Z
    chosen = max(res, key=lambda r: r["inner_auroc"])
    sec = chosen["sec_per_fit"]
    chosen = {k: chosen[k] for k in ("k", "l2", "lr", "epochs")}
    n_c = N_DRAWS if sec <= 10 else 3  # plan.md: megaprobe uses 3 draws if one base-pilot fit takes > 10 s
    HP_FILE.write_text(json.dumps({"grid": res, "chosen": chosen, "n_draws_c": n_c, "pca_sec": round(t_pca, 1),
                                   "rule": "max mean inner AUROC (all lies vs truthful); base model; outer fold 0 "
                                           "training rows; inner fits on 400-row subsamples"}, indent=1))
    print("chosen", chosen, "n_draws_c", n_c)


def hp_pilot_ext():
    """The first grid's best L2 (1e-2) sat on its edge, so the grid is extended upward at the chosen k (same inner
    folds, same subsample rule and seed sequence as hp_pilot), then the max-inner-AUROC rule is re-applied."""
    H = json.loads(HP_FILE.read_text())
    rows, fm = load_rows("base"), folds()
    y = yvec(rows)
    tr0 = np.array([i for i, r in enumerate(rows) if fm[r["prompt_id"]] != 0 and r["y_judge"] is not None])
    cache = Cache("base", "own")
    rng = np.random.default_rng(SEED)
    inner = rng.permutation(len(tr0)) % 4
    subs = {j: np.sort(np.random.default_rng([SEED, 9, j]).choice(tr0[inner != j], N_FIT, replace=False))
            for j in range(4)}
    k = H["chosen"]["k"]
    Z, offs = pca_project(cache, pca_fit([(cache, tr0.tolist())], k))
    for l2 in (3e-2, 1e-1):
        hp = {"k": k, "l2": l2, "lr": 1e-2, "epochs": 300}
        t1 = time.time(); au = []
        for j in range(4):
            s, _, _ = fit_c(Z, offs, y, subs[j].tolist(), tr0[inner == j], seed=SEED + j, hp=hp)
            au.append(float(roc_auc_score(y[tr0[inner == j]], s)))
        H["grid"].append({**hp, "inner_auroc": float(np.mean(au)), "per_fold": au, "ext": True,
                          "sec_per_fit": round((time.time() - t1) / 4, 2)})
        print(H["grid"][-1], flush=True)
    ch = max(H["grid"], key=lambda r: r["inner_auroc"])
    H["chosen"] = {kk: ch[kk] for kk in ("k", "l2", "lr", "epochs")}
    H["rule"] += "; grid extended to L2 3e-2, 1e-1 at the chosen k because 1e-2 was the edge"
    HP_FILE.write_text(json.dumps(H, indent=1))
    print("chosen", H["chosen"])


def fit_policy(p):
    """Own read: arms a b c d, label sets all + fact, nulls a/c (all). Base read (held text): arms a c, label set all."""
    fm = folds()
    rows = load_rows(p)
    D = draws_for(p, rows, fm)
    y = yvec(rows)
    fold = np.array([fm[r["prompt_id"]] for r in rows])
    texts = [r["response"] for r in rows]
    reads = ["own"] + (["base"] if p != "base" else [])
    reads = [r for r in reads if r in READS]
    OUT.mkdir(parents=True, exist_ok=True)
    for rd in reads:
        f_out = OUT / f"{p}_{rd}.json"
        if f_out.exists():
            print("skip", f_out.name); continue
        t0 = time.time()
        cache = Cache(p, rd)
        assert cache.prompt_ids == [r["prompt_id"] for r in rows]
        X12, F = cache.mean_layer(12), cache.final()
        own = rd == "own"
        names = (["a_all", "b_all", "b_l2_all", "c_all", "d_all", "a_fact", "b_fact", "c_fact", "d_fact", "a_null_all",
                  "c_null_all", "a_all_n200"] if own else ["a_all", "c_all"])
        sc = {n: np.full((N_ROWS, N_DRAWS), np.nan) for n in names}
        info = {"b_layers": {}, "c_val": {}, "draw_sizes": {}}
        rng = np.random.default_rng([SEED, POLICIES.index(p), 7])
        for k in range(N_FOLDS):
            te = np.where(fold == k)[0]
            pca = pca_fit([(cache, [i for i in range(N_ROWS) if fold[i] != k])], HP["k"])
            Z, offs = pca_project(cache, pca)
            fact_tr = np.array(fact_train(rows, fm, k))
            for d in range(N_DRAWS):
                tr = np.array(D[k][d]); seed = SEED + 100 * k + d
                info["draw_sizes"][f"{k}_{d}"] = [int((y[tr] == 0).sum()), int((y[tr] == 1).sum())]
                sc["a_all"][te, d] = fit_a(X12, y, tr, te)
                if d < N_DRAWS_C:
                    s, vau, ep = fit_c(Z, offs, y, tr.tolist(), te, seed)
                    sc["c_all"][te, d] = s
                    info["c_val"][f"{k}_all_{d}"] = {"val_auroc": vau, "epoch": ep}
                if not own:
                    continue
                s, keep, ce = fit_b(F, y, tr, te, seed)
                sc["b_all"][te, d] = s
                info["b_layers"][f"{k}_all_{d}"] = {"kept": [l + 1 for l in keep], "train_ce": [round(c, 4) for c in ce]}
                sc["b_l2_all"][te, d] = fit_b(F, y, tr, te, seed, norm="l2")[0]
                sc["d_all"][te, d] = fit_d(texts, y, tr, te)
                yperm = y.copy(); yperm[tr] = rng.permutation(y[tr])
                sc["a_null_all"][te, d] = fit_a(X12, yperm, tr, te)
                if d == 0:  # megaprobe null: one draw per fold (every row still gets an OOF null score)
                    sc["c_null_all"][te, d] = fit_c(Z, offs, yperm, tr.tolist(), te, seed)[0]
                r2 = np.random.default_rng([SEED, POLICIES.index(p), k, d, 200])
                pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                sub = np.concatenate([r2.choice(pos, min(100, len(pos)), replace=False),
                                      r2.choice(neg, min(100, len(neg)), replace=False)])
                sc["a_all_n200"][te, d] = fit_a(X12, y, sub, te)
            if own:  # label set fact: spin (1) vs fact-stated truthful (0), all such training rows, one fit
                seed = SEED + 100 * k + 50
                info["draw_sizes"][f"{k}_fact"] = [int((y[fact_tr] == 0).sum()), int((y[fact_tr] == 1).sum())]
                sc["a_fact"][te, 0] = fit_a(X12, y, fact_tr, te)
                s, keep, ce = fit_b(F, y, fact_tr, te, seed)
                sc["b_fact"][te, 0] = s
                info["b_layers"][f"{k}_fact_0"] = {"kept": [l + 1 for l in keep], "train_ce": [round(c, 4) for c in ce]}
                s, vau, ep = fit_c(Z, offs, y, fact_tr.tolist(), te, seed)
                sc["c_fact"][te, 0] = s
                info["c_val"][f"{k}_fact_0"] = {"val_auroc": vau, "epoch": ep}
                sc["d_fact"][te, 0] = fit_d(texts, y, fact_tr, te)
            del Z; torch.mps.empty_cache()
            print(f"{p} {rd} fold {k} done {time.time() - t0:.0f}s", flush=True)
        save_scores(f_out, {"policy": p, "read": rd, "hp": HP, "n_draws_c": N_DRAWS_C, "info": info,
                            "prompt_ids": cache.prompt_ids, "seconds": round(time.time() - t0)}, sc)
        print(f"{p} {rd} written {time.time() - t0:.0f}s", flush=True)


def fp(tag):
    """Log this process's footprint (macOS `footprint`), for the memory cap."""
    import os, subprocess
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if "Footprint:" in l]
    print(f"[footprint] {tag}: {line[0].split('Footprint:')[1].strip() if line else '?'}", flush=True)


def fit_pooled_fact():
    """Label set fact, pooled over the 4 trained policies (own read): per outer fold, one fit on all 4 policies'
    fact-present training rows (shared PCA for arm c), scores for every policy's held-out fold rows."""
    f_out = OUT / "pooled_fact.json"
    if f_out.exists():
        print("skip pooled"); return
    fm = folds(); t0 = time.time()
    R = {p: load_rows(p) for p in TRAINED}
    C = {p: Cache(p, "own") for p in TRAINED}
    y = np.concatenate([yvec(R[p]) for p in TRAINED])
    fold = np.concatenate([[fm[r["prompt_id"]] for r in R[p]] for p in TRAINED])
    texts = sum([[r["response"] for r in R[p]] for p in TRAINED], [])
    X12 = np.concatenate([C[p].mean_layer(12) for p in TRAINED])
    F = np.concatenate([C[p].final() for p in TRAINED])
    fp("features loaded")
    factrow = {p: [i for i, r in enumerate(R[p]) if r["cls"] in ("spin", "truthful") and r["y_string"] == 0]
               for p in TRAINED}
    sc = {n: np.full((len(y), 1), np.nan) for n in ("a_fact_pooled", "b_fact_pooled", "c_fact_pooled", "d_fact_pooled")}
    for k in range(N_FOLDS):
        tr = np.array([j * N_ROWS + i for j, p in enumerate(TRAINED) for i in factrow[p] if fm[R[p][i]["prompt_id"]] != k])
        te = np.where(fold == k)[0]
        seed = SEED + 100 * k + 77
        sc["a_fact_pooled"][te, 0] = fit_a(X12, y, tr, te); fp(f"fold {k} a")
        sc["b_fact_pooled"][te, 0] = fit_b(F, y, tr, te, seed)[0]; fp(f"fold {k} b")
        sc["d_fact_pooled"][te, 0] = fit_d(texts, y, tr, te); fp(f"fold {k} d")
        # arm c: shared PCA on the pooled training rows' tokens; project only fact-present rows (the scored ones)
        pca = pca_fit([(C[p], [i for i in factrow[p] if fm[R[p][i]["prompt_id"]] != k]) for p in TRAINED], HP["k"])
        Zs, offs = [], [np.zeros(1, np.int64)]
        for p in TRAINED:
            Zp, op = pca_project(C[p], pca, rows=factrow[p])
            offs.append(op[1:] + offs[-1][-1]); Zs.append(Zp)
        Z = torch.cat(Zs); offs = np.concatenate(offs); del Zs
        fp(f"fold {k} pca+project, Z {tuple(Z.shape)}")
        te_f = np.array([j for j in te if (j % N_ROWS) in set(factrow[TRAINED[j // N_ROWS]])])
        sc["c_fact_pooled"][te_f, 0] = fit_c(Z, offs, y, tr.tolist(), te_f, seed)[0]
        del Z; torch.mps.empty_cache()
        print(f"pooled fold {k} done {time.time() - t0:.0f}s", flush=True)
    save_scores(f_out, {"policies": TRAINED, "read": "own", "hp": HP, "rows": "policy-major: row j = policy j//1000",
                        "seconds": round(time.time() - t0)}, sc)


N_DRAWS_C = int(json.loads(HP_FILE.read_text()).get("n_draws_c", N_DRAWS)) if HP_FILE.exists() else N_DRAWS
READS = ["own", "base"]

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        hp_pilot()
    elif cmd == "pilot_ext":
        hp_pilot_ext()
    elif cmd == "fit":  # all own reads first (the verdict needs them), then the held-text reads
        for READS in (["own"], ["base"]):
            for p in sys.argv[2:] or POLICIES:
                fit_policy(p)
    elif cmd == "pooled":
        fit_pooled_fact()
