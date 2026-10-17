"""Shared data for T16 residual search (Iris). Dev seeds only; never the holdout 432432.

Model forward is numpy float64 from weights_raw.npz (same weights as m16_10.pth).
Phantom-augmented order: a 0 at virtual position -1 sits before x_0 (C1, algorithm_compact.py).
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_w = np.load(HERE / "weights_raw.npz")
W_hi, W_hh, W_oh = _w["W_hi"], _w["W_hh"], _w["W_oh"]


def run(x, states=False):
    h = np.zeros((x.shape[0], 16))
    hs = [h]
    for t in range(10):
        h = np.maximum(h @ W_hh.T + x[:, t:t + 1] * W_hi[None, :], 0.0)
        hs.append(h)
    L = h @ W_oh.T
    return (L, np.stack(hs, 1)) if states else L


def data(seed, n):
    assert seed != 432432, "holdout seed is reserved for Rasmus"
    X = np.random.default_rng(seed).standard_normal((n, 10))
    L = run(X)
    return X, L, L.argmax(1)


def c1(X):
    Xp = np.concatenate([np.zeros((len(X), 1)), X], 1)
    o = np.argsort(-Xp, 1, kind="stable")
    return np.maximum(o[:, 1] - 1, 0)


def aug_order(X):
    """Positions (-1..9) sorted by value, phantom 0 at -1 included. Shape (n, 11)."""
    Xp = np.concatenate([np.zeros((len(X), 1)), X], 1)
    return np.argsort(-Xp, 1, kind="stable") - 1


def cand_features(X, K, drop=()):
    """Candidate-centric features for residual_mlp.py. Returns F (n,K,d) float32 and output positions (n,K).

    Values are divided by s = |M| + rms(x), so the scorer is scale-invariant like the model (no biases)."""
    n = len(X)
    o = aug_order(X)
    Xp = np.concatenate([np.zeros((n, 1)), X], 1)
    vals = np.take_along_axis(Xp, o + 1, 1)  # sorted values incl phantom
    s = np.abs(vals[:, :1]) + np.sqrt((X ** 2).mean(1, keepdims=True))
    v = vals / s
    oK = o[:, :K]
    feats = []
    for k in range(K):
        f = [np.eye(K)[np.full(n, k)]]
        if "pos" not in drop: f.append(np.eye(11)[oK[:, k] + 1])
        if "pmax" not in drop: f.append(np.eye(11)[o[:, 0] + 1])
        if "vals" not in drop:
            f.append(v[:, k:k + 1]); f.append(v[:, :5])
            f.append(np.diff(v[:, :5], axis=1))
        if "xs" not in drop: f.append(X / s)
        feats.append(np.concatenate(f, 1))
    return np.stack(feats, 1).astype(np.float32), np.maximum(oK, 0)
