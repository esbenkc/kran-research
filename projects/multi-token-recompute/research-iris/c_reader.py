"""C = the learnable number (PREREG v1 'Reads'): a held-out classifier that predicts the build (none vs leaky)
from the prefix text alone. Spec (Iris #16532, frozen before data):
  - train on TRAIN prompts, test on TEST prompts (split by prompt; both CRN members of a pair in the same split)
  - features = (position, token) one-hots + bag of tokens (counts)
  - sklearn LogisticRegression (L2, liblinear); reg C_reg from {0.01, 0.1, 1, 10} by 5-fold GroupKFold over train prompts
  - C = 2 * balanced_accuracy - 1 on test; 95% CI by bootstrap over test prompts (2,000, seed 0)
Also reported (not frozen): the soft paired gap G = mean over test pairs of q(x_none) - q(x_leaky), where
q = predicted P(none | text): the probe-off-minus-probe-on switch (in points) that a policy acting on q would show.
And the pair-discordance share f on the same data.

Input: an .npz with toks_none [N, k], toks_leaky [N, k] (int, CRN pairs: same prompt, same seed), prompt [N]
(int prompt index or id hash), split [N] (str 'train'/'test') -- or pass --split-from prompts.jsonl.
"""
import argparse, json, os
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

REGS = [0.01, 0.1, 1.0, 10.0]


def featurize(toks, V):
    n, k = toks.shape
    rows = np.repeat(np.arange(n), k)
    pos_tok = (np.tile(np.arange(k), n) * V + toks.reshape(-1))
    a = sparse.csr_matrix((np.ones(n * k), (rows, pos_tok)), shape=(n, k * V))
    b = sparse.csr_matrix((np.ones(n * k), (rows, toks.reshape(-1))), shape=(n, V))
    return sparse.hstack([a, b]).tocsr()


def fit_c(toks_none, toks_leaky, prompt, is_train, seed=0, n_boot=2000, _return_pred=False):
    k = toks_none.shape[1]
    V = int(max(toks_none.max(), toks_leaky.max())) + 1
    X = featurize(np.concatenate([toks_none, toks_leaky]), V)
    y = np.r_[np.zeros(len(toks_none)), np.ones(len(toks_leaky))]          # 1 = leaky (probe on)
    g = np.r_[prompt, prompt]
    tr = np.r_[is_train, is_train]
    te = ~tr
    # regularisation by grouped CV on train prompts
    cv = GroupKFold(n_splits=5)
    scores = {}
    for r in REGS:
        acc = []
        for a, b in cv.split(X[tr], y[tr], g[tr]):
            m = LogisticRegression(C=r, solver="liblinear", max_iter=2000).fit(X[tr][a], y[tr][a])
            acc.append(balanced_accuracy_score(y[tr][b], m.predict(X[tr][b])))
        scores[r] = float(np.mean(acc))
    best = max(REGS, key=lambda r: (scores[r], -r))
    m = LogisticRegression(C=best, solver="liblinear", max_iter=2000).fit(X[tr], y[tr])
    pred = m.predict(X[te])
    q_none = 1 - m.predict_proba(X)[:, 1]                                   # P(none | text)
    n = len(toks_none)
    C = 2 * balanced_accuracy_score(y[te], pred) - 1
    # soft paired gap on test pairs: q(x_none) - q(x_leaky)
    tp = ~is_train
    G_pair = q_none[:n][tp] - q_none[n:][tp]
    # bootstrap over test prompts
    rng = np.random.default_rng(seed)
    pt = prompt[tp]
    up = np.unique(pt)
    idx_by = {p: np.where(pt == p)[0] for p in up}
    yt_none, yt_leak = pred[: tp.sum()], pred[tp.sum():]                    # te order = [none test..., leaky test...]
    Cb, Gb = [], []
    for _ in range(n_boot):
        ps = rng.choice(up, size=len(up), replace=True)
        ii = np.concatenate([idx_by[p] for p in ps])
        # balanced accuracy = mean of (TNR on none, TPR on leaky)
        Cb.append((np.mean(yt_none[ii] == 0) + np.mean(yt_leak[ii] == 1)) - 1)
        Gb.append(G_pair[ii].mean())
    differ = (toks_none != toks_leaky).any(1)
    out = dict(C=float(C), C_ci=[float(np.quantile(Cb, .025)), float(np.quantile(Cb, .975))],
                G_pair_points=float(100 * G_pair.mean()),
                G_pair_ci=[float(100 * np.quantile(Gb, .025)), float(100 * np.quantile(Gb, .975))],
                reg=best, cv_bal_acc=scores, n_train_pairs=int(is_train.sum()), n_test_pairs=int(tp.sum()),
                n_test_prompts=int(len(up)), f_all=float(differ.mean()), f_test=float(differ[tp].mean()), k=int(k))
    if _return_pred:
        out["_contrib"] = (pred[tp.sum():] == 1).astype(float) - (pred[: tp.sum()] == 1).astype(float)
    return out


def fit_c_crossfit(toks_none, toks_leaky, prompt, seed=0, n_boot=2000):
    """For reads that only have TEST prompts (final adapters, R=8): split the prompts in two halves at random
    (seed 0), fit on one half with the same grouped CV, predict the other, swap, and pool. C = mean over all
    pairs of (pred on leaky text) - (pred on none text) = 2 x balanced accuracy - 1 on balanced pairs."""
    up = np.unique(prompt)
    half = set(np.random.default_rng(seed).permutation(up)[: len(up) // 2])
    in_a = np.array([p in half for p in prompt])
    contrib = np.zeros(len(prompt))
    regs = []
    for train_mask in (in_a, ~in_a):
        r = fit_c(toks_none, toks_leaky, prompt, train_mask, seed=seed, n_boot=1, _return_pred=True)
        contrib[~train_mask] = r["_contrib"]
        regs.append(r["reg"])
    rng = np.random.default_rng(seed)
    idx_by = {p: np.where(prompt == p)[0] for p in up}
    Cb = [contrib[np.concatenate([idx_by[p] for p in rng.choice(up, len(up))])].mean() for _ in range(n_boot)]
    differ = (toks_none != toks_leaky).any(1)
    return dict(C=float(contrib.mean()), C_ci=[float(np.quantile(Cb, .025)), float(np.quantile(Cb, .975))],
                mode="crossfit2_by_prompt", regs=regs, n_pairs=int(len(prompt)), n_prompts=int(len(up)),
                n_nonzero=int((contrib != 0).sum()), f_test=float(differ.mean()), k=int(toks_none.shape[1]))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--split-from", default=None, help="prompts.jsonl with id/split; npz 'prompt' holds row indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crossfit", action="store_true", help="2-fold cross-fit by prompt on the TEST rows only")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    kn = "toks_none" if "toks_none" in z.files else "toks_n"            # Bruno read.py: toks_n / toks_l / prompt_idx
    kl = "toks_leaky" if "toks_leaky" in z.files else "toks_l"
    tn, tl = z[kn].astype(np.int64), z[kl].astype(np.int64)
    prompt = z["prompt"] if "prompt" in z.files else z["prompt_idx"]
    if "split" in z.files:
        is_train = np.array([str(s) == "train" for s in z["split"]])
    else:
        sp = [json.loads(l)["split"] for l in open(a.split_from)]
        is_train = np.array([sp[int(p)] == "train" for p in prompt])
    if prompt.dtype.kind not in "iu":
        _, prompt = np.unique(prompt, return_inverse=True)
    prompt = prompt.astype(np.int64)
    if a.crossfit:
        te = ~is_train
        res = fit_c_crossfit(tn[te], tl[te], prompt[te])
    else:
        res = fit_c(tn, tl, prompt, is_train)
    res["source"] = a.npz
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
