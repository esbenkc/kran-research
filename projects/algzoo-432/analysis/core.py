"""Shared helpers for M_{16,10} (ARC AlgZoo example_2nd_argmax).

Weights: m16_10.pth, fetched from
https://storage.googleapis.com/arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth
(the same file alg_zoo.example_2nd_argmax() loads).

Model: h_0 = 0; h_{t+1} = ReLU(W_hh h_t + W_hi x_t); logits = W_oh h_10. No biases.
All analysis runs in numpy float64 on CPU.
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N, D = 10, 16


def load_raw():
    sd = torch.load(HERE / "m16_10.pth", weights_only=True, map_location="cpu")
    W_hi = sd["rnn.weight_ih_l0"].double().numpy()[:, 0]  # (16,)
    W_hh = sd["rnn.weight_hh_l0"].double().numpy()  # (16,16)
    W_oh = sd["linear.weight"].double().numpy()  # (10,16)
    assert W_hi.shape == (D,) and W_hh.shape == (D, D) and W_oh.shape == (N, D)
    assert set(sd) == {"rnn.weight_ih_l0", "rnn.weight_hh_l0", "linear.weight"}
    return W_hi, W_hh, W_oh


def standardize(W_hi, W_hh, W_oh):
    """Rescale neuron i by s_i = 1/|W_hi_i| (exact equivalence, ReLU is positively homogeneous).

    h' = S h  =>  W_hi' = S W_hi, W_hh' = S W_hh S^-1, W_oh' = W_oh S^-1.
    Neurons with W_hi == 0 are left unscaled.
    """
    s = np.where(np.abs(W_hi) > 1e-12, 1.0 / np.abs(W_hi), 1.0)
    return s * W_hi, (s[:, None] * W_hh) / s[None, :], W_oh / s[None, :], s


def load_std():
    return standardize(*load_raw())[:3]


def sample(n, seed):
    return np.random.default_rng(seed).standard_normal((n, N))


def run(x, W_hi, W_hh, W_oh, return_states=False):
    """x: (B,10). Returns logits (B,10) and optionally states (B,11,16) with states[:,t]=h_t."""
    B = x.shape[0]
    h = np.zeros((B, D))
    states = [h]
    for t in range(N):
        h = np.maximum(h @ W_hh.T + x[:, t : t + 1] * W_hi[None, :], 0.0)
        states.append(h)
    logits = h @ W_oh.T
    if return_states:
        return logits, np.stack(states, axis=1)
    return logits


def second_argmax(x):
    return np.argsort(-x, axis=1)[:, 1]


def agreement(pred, ref):
    return float(np.mean(pred == ref))
