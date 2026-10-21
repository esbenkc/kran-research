#!/usr/bin/env python3
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def auroc(pos, neg):
    vals = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    order = np.argsort(vals, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(vals)+1)
    # average ranks for ties
    sv = vals[order]
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_pos = ranks[labels == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((sum_pos - n1*(n1+1)/2) / (n1*n0))


def signfree(x):
    return max(x, 1-x)


def compute(model_dir: Path, seed=123, n_random=200, n_shuffle=200):
    z = np.load(model_dir/'acts.npz')
    # Base direction by prereg: mean BASE fit harmful - harmless. For stripped model readout, still use dirs.npy externally.
    base = np.load('kran-research/projects/refusal-addback/runs/base/acts.npz')
    dirs = base['fit_harmful'].astype(np.float32).mean(axis=0) - base['fit_harmless'].astype(np.float32).mean(axis=0)
    stored = np.load('kran-research/projects/refusal-addback/runs/dirs.npy').astype(np.float32)
    cos = []
    for L in range(dirs.shape[0]):
        a, b = dirs[L], stored[L]
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        cos.append(float(a.dot(b)/denom) if denom else None)
    th = z['test_harmful'].astype(np.float32)
    tn = z['test_harmless'].astype(np.float32)
    base_auc = []
    for L in range(dirs.shape[0]):
        d = dirs[L]
        base_auc.append(auroc(th[:,L,:] @ d, tn[:,L,:] @ d) if np.linalg.norm(d) else 0.5)

    rng = np.random.default_rng(seed)
    rand_p95 = []
    rand_max = []
    for L in range(dirs.shape[0]):
        vals = []
        for _ in range(n_random):
            d = rng.normal(size=dirs.shape[1]).astype(np.float32)
            d /= np.linalg.norm(d)
            vals.append(signfree(auroc(th[:,L,:] @ d, tn[:,L,:] @ d)))
        rand_p95.append(float(np.quantile(vals, 0.95)))
        rand_max.append(float(max(vals)))

    fit = np.concatenate([base['fit_harmful'].astype(np.float32), base['fit_harmless'].astype(np.float32)], axis=0)
    labels = np.array([1]*len(base['fit_harmful']) + [0]*len(base['fit_harmless']))
    shuf_p95 = []
    shuf_max = []
    for L in range(dirs.shape[0]):
        vals = []
        acts_L = fit[:,L,:]
        for _ in range(n_shuffle):
            y = rng.permutation(labels)
            d = acts_L[y==1].mean(axis=0) - acts_L[y==0].mean(axis=0)
            if np.linalg.norm(d) == 0:
                vals.append(0.5)
            else:
                vals.append(signfree(auroc(th[:,L,:] @ d, tn[:,L,:] @ d)))
        shuf_p95.append(float(np.quantile(vals, 0.95)))
        shuf_max.append(float(max(vals)))

    return {
        'model_dir': str(model_dir),
        'dir_cos_min_nonzero': min(c for c in cos if c is not None),
        'base_dir_auc': base_auc,
        'base_dir_peak': {'layer': int(np.argmax(base_auc)), 'auc': float(max(base_auc))},
        'random_signfree_p95': rand_p95,
        'random_signfree_max': rand_max,
        'shuffle_fit_signfree_p95': shuf_p95,
        'shuffle_fit_signfree_max': shuf_max,
    }

out = {
    'base': compute(Path('kran-research/projects/refusal-addback/runs/base')),
    'strip_s0': compute(Path('kran-research/projects/refusal-addback/runs/strip_s0')),
}
Path('swarm-oracle-1021-refusal/skeptic2/readout_nulls.json').write_text(json.dumps(out, indent=2))
for key, val in out.items():
    peak = val['base_dir_peak']
    L = peak['layer']
    print(key, 'peak', peak, 'random_p95_at_peak', val['random_signfree_p95'][L], 'shuffle_p95_at_peak', val['shuffle_fit_signfree_p95'][L], 'shuffle_max_at_peak', val['shuffle_fit_signfree_max'][L])
