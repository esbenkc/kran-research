"""Vera (skeptic) own helpers. Independent of core.py and teammates' scripts.
Forward: h_0 = 0, h_t = relu(W_hi x_t + W_hh h_{t-1}), logits = W_oh h_9. No biases.
Checked against torch.nn.RNN(relu, bias=False) in vera_checks.py.
Dev seeds 900-999 only. Never the holdout 432432.
"""
import numpy as np
import torch

HERE = __file__.rsplit("/", 2)[0]  # swarm-algzoo-432 (mirror: projects/algzoo-432/analysis)

def load():
    sd = torch.load(f"{HERE}/m16_10.pth", map_location="cpu")
    Wi = sd["rnn.weight_ih_l0"].double().numpy()[:, 0]
    Wh = sd["rnn.weight_hh_l0"].double().numpy()
    Wo = sd["linear.weight"].double().numpy()
    return Wi, Wh, Wo


def logits(X, Wi, Wh, Wo):
    h = np.zeros((X.shape[0], Wh.shape[0]))
    for t in range(X.shape[1]):
        h = np.maximum(0.0, np.outer(X[:, t], Wi) + h @ Wh.T)
    return h @ Wo.T


def pred(X, W):
    return logits(X, *W).argmax(1)


def sample(seed, n):
    assert seed != 432432, "holdout"
    return np.random.default_rng(seed).standard_normal((n, 10))


def aug_order(X):
    """Phantom-augmented ranking: column 0 = phantom 0 at position -1, columns 1..10 = x_0..x_9.
    Returns positions (phantom mapped to real position 0) sorted by value, descending."""
    A = np.concatenate([np.zeros((X.shape[0], 1)), X], 1)
    o = np.argsort(-A, 1, kind="stable")
    pos = o - 1
    phantom = pos == -1
    pos[phantom] = 0
    return pos, phantom
