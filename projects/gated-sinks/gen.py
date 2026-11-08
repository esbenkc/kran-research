"""Synthetic sequences with planted sparse ground-truth latents.

Each sequence has K of N_LAT binary latents switched on (the ground truth). Every latent boosts a small
fixed set of CONTENT tokens. An evidence token is drawn from softmax(base + beta * sum of active boosts),
so a model that predicts evidence tokens well must infer which latents are on from earlier evidence.
Position 0 is always BOS (a fixed first token where an attention sink can form).

Organisms (GenCfg.organism):
  iid  every position is an evidence token; optional repeated spans (the first, void pilot: no sink).
  sc   "conditional average" (Ran-Milo 2603.11487; Iris #17197): a TRIGGER token appears at ~p_trig of
       positions and the `burst` tokens right after it are evidence tokens. All other positions are FILLER tokens
       from a latent-free bigram chain on a disjoint filler vocab. A head that pools the evidence is needed
       only at trigger queries and must idle elsewhere.
  sa   iid evidence stream plus associative recall on a disjoint KEY/VALUE vocab: n_pairs key-value pairs
       placed early, each key reappears once later and the next token is its value.

Token ids: content 0..vocab-1, then filler (sc), key and value (sa), trigger (sc), then BOS last
(iid keeps the first pilot's layout: BOS = vocab, markers after it).
`sample` returns tokens, latents z, `fresh` (True where the token is a fresh evidence draw given z; the
Bayes likelihood uses exactly these) and `score` (True at the scored query positions P, before the
>= 32 cut: sc = trigger positions, sa = evidence-stream positions that are not recall queries,
iid = all positions).
Everything is a pure function of (cfg, batch seed), so a skeptic can rebuild any batch.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class GenCfg:
    organism: str = "iid"
    vocab: int = 128          # content tokens
    n_lat: int = 64           # planted latents
    k_active: int = 3         # latents on per sequence
    boost_tokens: int = 6     # tokens each latent boosts
    beta: float = 2.0         # boost size (logit units)
    base_std: float = 0.5     # spread of the base token logits
    seq_len: int = 128        # including BOS
    world_seed: int = 1234    # fixes base logits, latent->token map, filler chain
    # iid organism: repeated spans
    p_repeat: float = 0.5
    n_spans: int = 2
    span: int = 8
    markers: bool = False
    # sc organism
    p_trig: float = 0.1
    burst: int = 1             # evidence tokens after each trigger
    n_filler: int = 64
    filler_alpha: float = 0.1  # Dirichlet concentration of each filler bigram row
    # sa organism
    n_keys: int = 16
    n_pairs: int = 6

    # --- token layout ---
    @property
    def filler0(self) -> int:
        return self.vocab

    @property
    def key0(self) -> int:
        return self.vocab

    @property
    def val0(self) -> int:
        return self.vocab + self.n_keys

    @property
    def trigger(self) -> int:
        return self.vocab + self.n_filler

    @property
    def bos(self) -> int:
        if self.organism == "sc":
            return self.vocab + self.n_filler + 1
        if self.organism == "sa":
            return self.vocab + 2 * self.n_keys
        return self.vocab   # iid: BOS = vocab, markers S = vocab+1, M = vocab+2

    @property
    def n_tokens(self) -> int:
        if self.organism == "iid":
            return self.vocab + (3 if self.markers else 1)
        return self.bos + 1

    def dict(self):
        return asdict(self)


class World:
    """The fixed generator: base logits, the latent -> boosted-token map, the filler bigram chain."""

    def __init__(self, cfg: GenCfg):
        self.cfg = cfg
        r = np.random.default_rng(cfg.world_seed)
        self.base = r.normal(0.0, cfg.base_std, cfg.vocab)
        self.U = np.zeros((cfg.n_lat, cfg.vocab))
        for i in range(cfg.n_lat):
            self.U[i, r.choice(cfg.vocab, cfg.boost_tokens, replace=False)] = 1.0
        rf = np.random.default_rng(cfg.world_seed + 1)
        self.filler_P = rf.dirichlet(np.full(cfg.n_filler, cfg.filler_alpha), cfg.n_filler)

    def _evidence(self, r, z, n_draw):
        c = self.cfg
        logits = self.base[None, :] + c.beta * z @ self.U
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        cdf = np.cumsum(p, 1)
        u = r.random((len(z), n_draw))
        return np.stack([np.minimum(np.searchsorted(cdf[b], u[b]), c.vocab - 1)
                         for b in range(len(z))]).astype(np.int32)

    def sample(self, n: int, seed: int):
        c = self.cfg
        r = np.random.default_rng(seed)
        T = c.seq_len
        z = np.zeros((n, c.n_lat), np.float32)
        for b in range(n):
            z[b, r.choice(c.n_lat, c.k_active, replace=False)] = 1.0
        toks = np.empty((n, T), np.int32)
        toks[:, 0] = c.bos
        fresh = np.zeros((n, T), bool)
        score = np.zeros((n, T), bool)
        ev = self._evidence(r, z, T - 1)
        if c.organism == "iid":
            toks[:, 1:] = ev
            fresh[:, 1:] = True
            score[:, 1:] = True
            has = r.random(n) < c.p_repeat
            for b in np.nonzero(has)[0]:
                for _ in range(1 if c.markers else c.n_spans):
                    src = r.integers(2, T // 2 - c.span)
                    dst = r.integers(src + c.span + 4, T - c.span)
                    toks[b, dst:dst + c.span] = toks[b, src:src + c.span]
                    fresh[b, dst:dst + c.span] = False
                    if c.markers:
                        toks[b, src - 1] = c.vocab + 1
                        toks[b, dst - 1] = c.vocab + 2
                        fresh[b, [src - 1, dst - 1]] = False
        elif c.organism == "sc":
            u = r.random((n, T))
            fu = r.random((n, T))
            cdf = np.cumsum(self.filler_P, 1)
            for b in range(n):
                f = r.integers(c.n_filler)
                t, e = 1, 0
                while t < T:
                    if t < T - c.burst and u[b, t] < c.p_trig:
                        toks[b, t] = c.trigger
                        for j in range(c.burst):          # the query before each evidence token
                            score[b, t + j] = True
                            toks[b, t + 1 + j] = ev[b, e]
                            fresh[b, t + 1 + j] = True
                            e += 1
                        t += 1 + c.burst
                        continue
                    f = min(int(np.searchsorted(cdf[f], fu[b, t])), c.n_filler - 1)
                    toks[b, t] = c.filler0 + f
                    t += 1
        elif c.organism == "sa":
            toks[:, 1:] = ev
            fresh[:, 1:] = True
            score[:, 1:] = True
            for b in range(n):
                keys = r.choice(c.n_keys, c.n_pairs, replace=False)
                vals = r.integers(0, c.n_keys, c.n_pairs)
                kpos = 1 + 2 * np.sort(r.choice(14, c.n_pairs, replace=False))   # pairs in 1..28
                qpos = np.sort(r.choice(np.arange(30, T - 1, 2), c.n_pairs, replace=False))
                qord = r.permutation(c.n_pairs)
                for i in range(c.n_pairs):
                    for pos, key, val in ((kpos[i], keys[i], vals[i]),
                                          (qpos[i], keys[qord[i]], vals[qord[i]])):
                        toks[b, pos] = c.key0 + key
                        toks[b, pos + 1] = c.val0 + val
                        fresh[b, [pos, pos + 1]] = False
                        score[b, [pos, pos + 1]] = False
        else:
            raise ValueError(c.organism)
        return toks, z, fresh, score
