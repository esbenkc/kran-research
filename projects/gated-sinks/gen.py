"""Synthetic sequences with planted sparse ground-truth latents.

Each sequence has K of N_LAT binary latents switched on (the ground truth). Every latent boosts a
small fixed set of tokens. Tokens are drawn i.i.d. from softmax(base + beta * sum of active boosts),
so a model that predicts the next token well must infer which latents are on from the context so far.
Position 0 is always BOS (a fixed first token where an attention sink can form).

To give some heads a reason to "do nothing" sometimes (the usual cause of attention sinks), a share
of sequences also contains repeated spans: a span of SPAN tokens is copied to a later place. A head
that copies from the earlier occurrence has work only inside repeats; elsewhere it must idle.

Everything is a pure function of (generator seed, batch seed), so a skeptic can rebuild any batch.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class GenCfg:
    vocab: int = 128          # content tokens 0..vocab-1; BOS = vocab
    n_lat: int = 64           # planted latents
    k_active: int = 3         # latents on per sequence
    boost_tokens: int = 6     # tokens each latent boosts
    beta: float = 2.0         # boost size (logit units)
    base_std: float = 0.5     # spread of the base token logits
    seq_len: int = 128        # including BOS
    p_repeat: float = 0.5     # share of sequences with repeated spans
    n_spans: int = 2          # repeats per such sequence
    span: int = 8             # tokens per repeated span
    world_seed: int = 1234    # fixes base logits and latent->token map

    @property
    def bos(self) -> int:
        return self.vocab

    @property
    def n_tokens(self) -> int:
        return self.vocab + 1

    def dict(self):
        return asdict(self)


class World:
    """The fixed generator: base logits and the latent -> boosted-token map."""

    def __init__(self, cfg: GenCfg):
        self.cfg = cfg
        r = np.random.default_rng(cfg.world_seed)
        self.base = r.normal(0.0, cfg.base_std, cfg.vocab)
        self.U = np.zeros((cfg.n_lat, cfg.vocab))
        for i in range(cfg.n_lat):
            self.U[i, r.choice(cfg.vocab, cfg.boost_tokens, replace=False)] = 1.0

    def sample(self, n: int, seed: int):
        """Return tokens [n, T] int32, latents z [n, n_lat] (0/1 float32), repeat mask [n, T] bool."""
        c = self.cfg
        r = np.random.default_rng(seed)
        z = np.zeros((n, c.n_lat), np.float32)
        for b in range(n):
            z[b, r.choice(c.n_lat, c.k_active, replace=False)] = 1.0
        logits = self.base[None, :] + c.beta * z @ self.U          # [n, V]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        cdf = np.cumsum(p, 1)
        u = r.random((n, c.seq_len - 1))
        toks = np.empty((n, c.seq_len), np.int32)
        toks[:, 0] = c.bos
        for b in range(n):
            toks[b, 1:] = np.minimum(np.searchsorted(cdf[b], u[b]), c.vocab - 1)
        rep = np.zeros((n, c.seq_len), bool)
        has = r.random(n) < c.p_repeat
        for b in np.nonzero(has)[0]:
            for _ in range(c.n_spans):
                src = r.integers(1, c.seq_len // 2 - c.span)
                dst = r.integers(src + c.span + 4, c.seq_len - c.span)
                toks[b, dst:dst + c.span] = toks[b, src:src + c.span]
                rep[b, dst + 1:dst + c.span] = True   # positions predictable by copying
        return toks, z, rep
