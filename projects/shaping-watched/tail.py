"""The trained part of Qwen3-0.6B: blocks 22-27 + final norm + tied head, eager, bf16, on cached block-22 inputs.

The probe site is layers[24].self_attn.q_norm (L = 24). Layer 24's attention runs in a fused-kernel emulation:
q_norm and k_norm in fp32, rotary in fp32, one rounding to bf16 after the rotary (what Inductor's fused kernel does
when nothing else reads the q_norm output). With the probe attached through the LEAKY path (a hook keeps the bf16
q_norm output), the q_norm output must exist in bf16, so the rotary reads a rounded copy. EMULATED FOOTPRINT:
probed rows round q_norm's fp32 output to bf16 before the rotary; unprobed rows do not. Nothing else differs, so
everything up to and including the q_norm output (the probe's reading) is bit-identical across builds.

forward(h, probed) on rows [B, T+1, H] (prompt + one answer token) returns
  d2  [B, 2] fp32: logits of 'read' and 'solve' at the decision position T-1 (the policy's 2-way choice)
  qn  [B, heads*head_dim] fp32: q_norm(24) output at the answer position T (the probe's reading)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

DEV = os.environ.get("SW_DEV", "mps")
HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"
FIRST, L = 22, 24


def bf16(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def rms32(x, norm):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.float()


class Tail(torch.nn.Module):
    def __init__(self, cfg, first=FIRST):
        super().__init__()
        self.cfg = cfg
        self.first = first
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(cfg)
        self.li = L - first  # < 0 when the tail starts after the probe site: no emulation, no probe reading

    def attn_emu(self, attn, x, cos32, sin32, probed):
        B, S, _ = x.shape
        hd = attn.head_dim
        q = attn.q_proj(x).view(B, S, -1, hd)
        k = attn.k_proj(x).view(B, S, -1, hd)
        v = attn.v_proj(x).view(B, S, -1, hd).transpose(1, 2)
        qn = rms32(q, attn.q_norm)                                    # [B, S, nh, hd] fp32 = the probe's reading
        kn = rms32(k, attn.k_norm)
        if probed is not None:
            qin = torch.where(probed[:, None, None, None], qn.to(torch.bfloat16).float(), qn)
        else:
            qin = qn
        qin, kn = qin.transpose(1, 2), kn.transpose(1, 2)
        c, s = cos32[:, None], sin32[:, None]
        qr = (qin * c + rot_half(qin) * s).to(torch.bfloat16)
        kr = (kn * c + rot_half(kn) * s).to(torch.bfloat16)
        rep = qr.shape[1] // kr.shape[1]
        kr, v = kr.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        o = torch.nn.functional.scaled_dot_product_attention(qr, kr, v, is_causal=True, scale=attn.scaling)
        return attn.o_proj(o.transpose(1, 2).reshape(B, S, -1)), qn

    def forward(self, h, w2, probed=None, solve_rows=None):
        """h [B, S, H] bf16. w2 [2, H] head rows of (read, solve). probed: bool [B] or None (no footprint).
        solve_rows: optional [B2, S, H]: rows that only need the probe reading (blocks 22-23 + q_norm 24)."""
        B, S, _ = h.shape
        pos = torch.arange(S, device=h.device)[None]
        cos, sin = self.rotary(h, pos)                                 # bf16 tables for the plain layers
        with torch.autocast("mps", enabled=False):
            inv = 1.0 / (self.cfg.rope_parameters["rope_theta"] ** (
                torch.arange(0, self.cfg.head_dim, 2, device=h.device, dtype=torch.float32) / self.cfg.head_dim))
            fr = pos[0].float()[:, None] * inv[None]
            emb = torch.cat((fr, fr), -1)[None]
            cos32, sin32 = emb.cos(), emb.sin()
        n2 = 0
        if solve_rows is not None:
            n2 = solve_rows.shape[0]
            h = torch.cat([h, solve_rows])
        qn_out = None
        for i, layer in enumerate(self.layers):
            if i == self.li:
                if n2:
                    hs = h[B:]
                    a = layer.self_attn
                    x = layer.input_layernorm(hs[:, -1:])
                    qs = rms32(a.q_proj(x).view(n2, 1, -1, a.head_dim), a.q_norm)[:, 0].flatten(-2)
                    h = h[:B]
                res = h
                x = layer.input_layernorm(h)
                ao, qn = self.attn_emu(layer.self_attn, x, cos32, sin32, probed)
                h = res + ao
                res = h
                h = res + layer.mlp(layer.post_attention_layernorm(h))
                qn_out = qn[:, -1].flatten(-2)
                if n2:
                    qn_out = torch.cat([qn_out, qs])
            else:
                h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        hd = self.norm(h[:, -2])                                        # decision position T-1
        return (hd @ w2.T).float(), qn_out

    def probe_only(self, h):
        """The probe's reading only: blocks 22-23, then layer 24's input norm, q_proj and q_norm at the last position
        (the answer token). [B, S, H] -> [B, heads*head_dim] fp32. Same ops as forward()'s qn; the last bits may differ
        because q_proj runs on one position here (another matmul shape). Used for the reward score when no gradient
        flows through the probe (arms pilot/O/B); arm A reads qn from forward()."""
        S = h.shape[1]
        pos = torch.arange(S, device=h.device)[None]
        cos, sin = self.rotary(h, pos)
        for layer in self.layers[: self.li]:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        layer = self.layers[self.li]
        a = layer.self_attn
        x = layer.input_layernorm(h[:, -1:])
        return rms32(a.q_proj(x).view(h.shape[0], 1, -1, a.head_dim), a.q_norm)[:, 0].flatten(-2)


def load_tail(first=FIRST):
    path = Path(snapshot_download(MODEL))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    tail = Tail(cfg, first)
    want, w_out = {}, None
    out_key = "lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == out_key:
                    w_out = sf.get_tensor(k)
    missing, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    toks = json.load(open(HERE / "data" / "tokens.json"))
    w2 = w_out[[toks["read"], toks["solve"]]].to(torch.bfloat16).to(DEV)
    return cfg, tail.to(torch.bfloat16).to(DEV), w2


class Rows:
    """Cached block-22 inputs [N, 2, T+1, H] (row 0 = prompt + 'read', row 1 = prompt + 'solve'), disk memmap."""

    def __init__(self, path):
        self.a = np.load(path, mmap_mode="r")

    def get(self, idx, which):
        return bf16(self.a[list(idx), which]).to(DEV)


def splits():
    rows = [json.loads(l) for l in open(HERE / "raw" / "prompts.jsonl")]
    return {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ("train", "dev", "test")}
