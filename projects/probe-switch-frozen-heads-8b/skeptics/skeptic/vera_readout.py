"""Vera's own readout of a probe trace from raw bf16 vectors (no torch; numpy + sklearn only).

Input: two uint16 arrays (bf16 bits), probe off / on, shape (N, d) or (N, T, d) with a --pos index,
plus the prompt-id list (meta json with 'prompt_ids', split = id prefix train/dev/test).
Output (stdout + json):
  - G0: share of prompts per split whose vector differs in any bit; mean changed dims;
  - held-out linear readout: L2 logistic regression, C picked on dev, fit on train rows (both conditions),
    accuracy and AUROC on test rows; 2-layer MLP probe (width 16) accuracy;
  - prediction max(0, 2*acc - 1) with acc = the larger of the two accuracies (frozen form);
  - null: the same logistic fit with condition labels shuffled WITHIN each train pair (20 draws).
Usage: python3 vera_readout.py OFF.npy ON.npy META.json [--pos P] [--out FILE]
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def main():
    off_p, on_p, meta_p = sys.argv[1:4]
    pos = int(sys.argv[sys.argv.index('--pos') + 1]) if '--pos' in sys.argv else None
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    off = np.load(off_p, mmap_mode='r')
    on = np.load(on_p, mmap_mode='r')
    if pos is not None:
        off, on = off[:, pos, :], on[:, pos, :]
    off, on = np.asarray(off), np.asarray(on)
    assert off.dtype == np.uint16 and off.shape == on.shape, (off.dtype, off.shape, on.shape)
    ids = json.load(open(meta_p))['prompt_ids']
    assert len(ids) == off.shape[0]
    split = np.array([i.rstrip('0123456789') for i in ids])
    res = {'shape': list(off.shape), 'pos': pos}
    diff = off != on
    for s in ('train', 'dev', 'test'):
        m = split == s
        res[f'g0_{s}'] = float(diff[m].any(axis=1).mean())
        res[f'changed_dims_{s}'] = float(diff[m].sum(axis=1).mean())
    X0, X1 = bf16_to_f32(off), bf16_to_f32(on)

    def rows(s):
        m = split == s
        X = np.concatenate([X0[m], X1[m]]).astype(np.float64)
        y = np.concatenate([np.zeros(m.sum()), np.ones(m.sum())])
        return X, y

    Xtr, ytr = rows('train')
    Xdv, ydv = rows('dev')
    Xte, yte = rows('test')
    sc = StandardScaler().fit(Xtr)
    Xtr, Xdv, Xte = sc.transform(Xtr), sc.transform(Xdv), sc.transform(Xte)
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        a = lr.score(Xdv, ydv)
        if best is None or a > best[0]:
            best = (a, C, lr)
    _, C, lr = best
    acc_lr = float(lr.score(Xte, yte))
    auc_lr = float(roc_auc_score(yte, lr.decision_function(Xte)))
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0, early_stopping=True).fit(Xtr, ytr)
    acc_mlp = float(mlp.score(Xte, yte))
    acc = max(acc_lr, acc_mlp)
    rng = np.random.default_rng(0)
    n = len(ytr) // 2
    null = []
    for _ in range(20):
        flip = rng.random(n) < 0.5
        yy = ytr.copy()
        yy[:n][flip], yy[n:][flip] = 1, 0
        m2 = LogisticRegression(C=C, max_iter=3000).fit(Xtr, yy)
        null.append(float(roc_auc_score(yte, m2.decision_function(Xte))))
    res.update(C=C, acc_logistic=acc_lr, auroc_logistic=auc_lr, acc_mlp16=acc_mlp, acc=acc,
               predicted_switch=max(0.0, 2 * acc - 1), null_auroc_mean=float(np.mean(null)),
               null_auroc_max=float(np.max(null)))
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
