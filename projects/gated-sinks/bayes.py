"""Exact Bayes posterior of the generator, the best any model can do.

For a sequence x, every one of the C(64,3) = 41,664 possible active sets S has likelihood
    log p(x_1..x_t | S) = sum_{u <= t, u not a copied token} log softmax(base + beta * sum_{i in S} U_i)[x_u]
with a uniform prior over S. The posterior marginal P(z_j = 1 | x_1..x_t) sums the posterior over sets
that contain j. The Bayes next-token prediction is p(x_{t+1} | x_1..x_t) = sum_S P(S | x_1..x_t) p_S(x_{t+1}).
Copied tokens (repeated spans, known from the generator's mask) are left out of the likelihood, because
they are not fresh draws; the Bayes loss is scored on non-copied positions only.
"""
from __future__ import annotations

from itertools import combinations

import mlx.core as mx
import numpy as np


class Bayes:
    def __init__(self, world):
        c = world.cfg
        S = np.array(list(combinations(range(c.n_lat), c.k_active)), np.int32)   # [nS, k]
        logits = world.base[None, :] + c.beta * world.U[S].sum(1)                 # [nS, V]
        logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) \
            - logits.max(1, keepdims=True)
        inc = np.zeros((len(S), c.n_lat), np.float32)
        for j in range(c.k_active):
            inc[np.arange(len(S)), S[:, j]] = 1.0
        self.logp = mx.array(logp.astype(np.float32))                             # [nS, V]
        self.p = mx.exp(self.logp)
        self.inc = mx.array(inc)
        self.vocab = c.vocab

    def run(self, toks: np.ndarray, rep: np.ndarray, bs: int = 4):
        """toks [n, T] with BOS at 0. Returns marginals [n, T, n_lat] (index t = after seeing x_1..x_t;
        t = 0 is the prior) and Bayes log-loss of x_t for t = 1..T-1 [n, T-1] (nan where x_t is copied)."""
        n, T = toks.shape
        margs, losses = [], []
        for i in range(0, n, bs):
            tk = mx.array(toks[i:i + bs, 1:])                       # [b, T-1]
            m = mx.array((~rep[i:i + bs, 1:]).astype(np.float32))   # fresh draws
            ll = self.logp[:, tk]                                    # [nS, b, T-1]
            ll = ll.transpose(1, 0, 2) * m[:, None, :]               # [b, nS, T-1]
            C = mx.cumsum(ll, axis=2)
            C = mx.concatenate([mx.zeros(C.shape[:2] + (1,)), C], axis=2)   # prior at t = 0
            post = mx.softmax(C, axis=1)                             # [b, nS, T]
            marg = post.transpose(0, 2, 1) @ self.inc                # [b, T, n_lat]
            pred = post[:, :, :-1].transpose(0, 2, 1) @ self.p       # [b, T-1, V] predicts x_1..x_{T-1}
            px = mx.take_along_axis(pred, tk[..., None], axis=2)[..., 0]
            mx.eval(marg, px)
            margs.append(np.array(marg))
            lo = -np.log(np.array(px))
            lo[np.array(m) == 0] = np.nan
            losses.append(lo)
        return np.concatenate(margs), np.concatenate(losses)
