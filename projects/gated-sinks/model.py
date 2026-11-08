"""Tiny pre-norm transformer in MLX with switchable attention variants.

attn variants (all else identical):
  softmax    standard causal softmax attention
  gated      softmax attention, then an elementwise sigmoid gate on each head's output computed
             from the block input: o = SDPA(q,k,v) * sigmoid(x W_g + b)  (Qiu et al. 2025, arXiv
             2505.06708, the "G1" position: after scaled dot-product attention, head-specific)
  gate_indep same gate position, but the gate is a learned per-dimension constant sigmoid(b),
             independent of the input (2505.06708 Table 4 "input-independent" row)
  softmax1   "off-by-one" softmax: an extra null slot with logit 0 and value 0, so a head can put
             weight nowhere (Miller 2023)
  sinklogit  softmax with one learned null-slot logit per head (value 0), as in gpt-oss
  kbias      softmax with one learned null-slot KEY per head, logit q.k*/sqrt(dh), value 0
             (Gu et al. 2024, arXiv 2410.10781, Table 4: learnable k bias, v bias fixed to 0)
Each null slot is virtual; it is not position 0. First-token sink mass is measured on position 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn

VARIANTS = ("softmax", "gated", "gate_indep", "softmax1", "sinklogit", "kbias")


@dataclass(frozen=True)
class ModelCfg:
    n_tokens: int = 129
    d: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 1024
    seq_len: int = 128
    attn: str = "softmax"
    gate_bias_init: float = 0.0   # gated only: initial bias of the gate logits

    def dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        assert c.attn in VARIANTS, c.attn
        self.c = c
        self.dh = c.d // c.n_heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.o = nn.Linear(c.d, c.d, bias=False)
        self.rope = nn.RoPE(self.dh)
        if c.attn == "gated":
            self.g = nn.Linear(c.d, c.d, bias=True)
            self.g.bias = mx.full((c.d,), c.gate_bias_init)
        if c.attn == "gate_indep":
            self.gb = mx.full((c.d,), c.gate_bias_init)
        if c.attn == "sinklogit":
            self.sink = mx.zeros((c.n_heads,))
        if c.attn == "kbias":
            self.ksink = mx.random.normal((c.n_heads, self.dh)) * 0.02

    def __call__(self, x, mask, want_attn=False):
        B, T, _ = x.shape
        H, dh = self.c.n_heads, self.dh
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = self.rope(q.reshape(B, T, H, dh).transpose(0, 2, 1, 3))
        k = self.rope(k.reshape(B, T, H, dh).transpose(0, 2, 1, 3))
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * (dh ** -0.5) + mask       # [B,H,T,T]
        null_mass = None
        if self.c.attn in ("softmax1", "sinklogit", "kbias"):
            if self.c.attn == "kbias":
                null = (q @ self.ksink.reshape(1, H, dh, 1)) * (dh ** -0.5)   # [B,H,T,1]
            else:
                null = mx.zeros((B, H, T, 1), s.dtype)
            if self.c.attn == "sinklogit":
                null = null + self.sink.reshape(1, H, 1, 1)
            full = mx.softmax(mx.concatenate([s, null], axis=-1), axis=-1)
            a, null_mass = full[..., :T], full[..., T]
        else:
            a = mx.softmax(s, axis=-1)
        y = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        gate = None
        if self.c.attn == "gated":
            gate = mx.sigmoid(self.g(x))
            y = y * gate
        if self.c.attn == "gate_indep":
            y = y * mx.sigmoid(self.gb)
        out = self.o(y)
        if want_attn:
            return out, {"attn": a, "null": null_mass, "gate": gate}
        return out, None


class Block(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.n1 = nn.RMSNorm(c.d)
        self.attn = Attention(c)
        self.n2 = nn.RMSNorm(c.d)
        self.fc1 = nn.Linear(c.d, c.d_mlp, bias=False)
        self.fc2 = nn.Linear(c.d_mlp, c.d, bias=False)

    def __call__(self, x, mask, want_attn=False):
        h, a = self.attn(self.n1(x), mask, want_attn)
        x = x + h
        x = x + self.fc2(nn.gelu(self.fc1(self.n2(x))))
        return x, a


class TinyLM(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.n_tokens, c.d)
        self.blocks = [Block(c) for _ in range(c.n_layers)]
        self.nf = nn.RMSNorm(c.d)
        self.head = nn.Linear(c.d, c.n_tokens, bias=False)

    def mask(self, T):
        return nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.float32)

    def __call__(self, toks):
        x = self.emb(toks)
        m = self.mask(toks.shape[1])
        for b in self.blocks:
            x, _ = b(x, m)
        return self.head(self.nf(x))

    def trace(self, toks):
        """Forward pass that returns logits, the residual stream after every block (index 0 = the
        embeddings) and every layer's attention pattern."""
        x = self.emb(toks)
        m = self.mask(toks.shape[1])
        resid, attn = [x], []
        for b in self.blocks:
            x, a = b(x, m, want_attn=True)
            resid.append(x)
            attn.append(a)
        return self.head(self.nf(x)), resid, attn

    def lens(self, h):
        """Logit lens: read a residual vector with the final norm and unembedding."""
        return self.head(self.nf(h))
