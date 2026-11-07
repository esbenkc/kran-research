"""Vera: my own Qwen3-0.6B tail (blocks 26-27 + final norm + read/solve rows of the tied head), fp32 CPU.
Written from the Qwen3 architecture (config.json), not from build code. LoRA merged by me: W + s * B @ A.
"""
import json, glob, os, numpy as np, torch
from safetensors import safe_open

SNAP = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/'))[0]
CFG = json.load(open(SNAP + 'config.json'))
READ, SOLVE = 878, 59419
H, KV, HD = CFG['num_attention_heads'], CFG['num_key_value_heads'], CFG['head_dim']
EPS, THETA = CFG['rms_norm_eps'], CFG['rope_theta']
NAMES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def load_base(layers=(26, 27)):
    W = {}
    with safe_open(SNAP + 'model.safetensors', 'pt') as f:
        keys = set(f.keys())
        for L in layers:
            for n in NAMES + ['input_layernorm', 'post_attention_layernorm', 'self_attn.q_norm', 'self_attn.k_norm']:
                W[f'{L}.{n}'] = f.get_tensor(f'model.layers.{L}.{n}.weight').float()
        W['norm'] = f.get_tensor('model.norm.weight').float()
        hk = 'lm_head.weight' if 'lm_head.weight' in keys else 'model.embed_tokens.weight'
        W['head'] = f.get_tensor(hk)[[READ, SOLVE]].float()
    return W


def merge(W, adapter_path, scale, layers=(26, 27)):
    A = torch.load(adapter_path, map_location='cpu')
    M = dict(W)
    for i, L in enumerate(layers):
        for n in NAMES:
            a = A[f'base_model.model.layers.{i}.{n}.lora_A.default.weight'].float()
            b = A[f'base_model.model.layers.{i}.{n}.lora_B.default.weight'].float()
            M[f'{L}.{n}'] = W[f'{L}.{n}'] + scale * (b @ a)
    return M


def rms(x, w):
    # bf16 path: HF Qwen3RMSNorm computes in fp32, then casts back before the weight multiply
    dt = x.dtype; xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(dt) * w


def rope(T):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
    f = torch.outer(torch.arange(T).float(), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()


def rot(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


@torch.no_grad()
def tail(W, h, layers=(26, 27), pos=95):
    """h: (B, T, 1024) float32 block-26 input. Returns (B,) decision logit l_read - l_solve at position pos."""
    B, T, D = h.shape
    cos, sin = rope(T)
    cos, sin = cos.to(h.dtype), sin.to(h.dtype)
    mask = torch.full((T, T), float('-inf')).triu(1).to(h.dtype)
    x = h
    for L in layers:
        g = lambda n: W[f'{L}.{n}']
        r = x; y = rms(x, g('input_layernorm'))
        q = (y @ g('self_attn.q_proj').T).view(B, T, H, HD)
        k = (y @ g('self_attn.k_proj').T).view(B, T, KV, HD)
        v = (y @ g('self_attn.v_proj').T).view(B, T, KV, HD)
        q = rms(q, g('self_attn.q_norm')).transpose(1, 2); k = rms(k, g('self_attn.k_norm')).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
        att = (q @ k.transpose(-1, -2)) / HD ** 0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, H * HD)
        x = r + o @ g('self_attn.o_proj').T
        r = x; y = rms(x, g('post_attention_layernorm'))
        x = r + (torch.nn.functional.silu(y @ g('mlp.gate_proj').T) * (y @ g('mlp.up_proj').T)) @ g('mlp.down_proj').T
    z = rms(x[:, pos], W['norm']) @ W['head'].T
    return z[:, 0] - z[:, 1]


def bf(x):
    return torch.from_numpy((np.asarray(x).astype(np.uint32) << 16).view(np.float32))
