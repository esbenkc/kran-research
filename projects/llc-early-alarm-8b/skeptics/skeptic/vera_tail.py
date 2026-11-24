"""Vera's own Qwen3-8B tail (blocks 34-35 + final norm + head), written from the Qwen3 architecture
(config.json), NOT from the build's code. Used to (a) recompute dev P(X | cue on/off) at saved adapters and
(b) run an independent SGLD LLC estimate at a few checkpoints.

Policy (from the frozen PREREGs): input = cached block-34 input (bf16 -> fp32); cue-on rows p=0,1,2 get
+= alpha * ||h_p|| * u_p; logits at the last position; constant b added to the 16 X-token logits;
P(X) = total softmax mass of the 16 X tokens. LoRA: W + (lora_alpha/r) * B @ A on q,k,v,o,gate,up,down.
"""
import json, os, glob, math
import numpy as np
import torch
from safetensors import safe_open

SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*"))[0]
CFG = json.load(open(os.path.join(SNAP, "config.json")))
H, NH, NKV, HD = CFG["hidden_size"], CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
EPS, THETA = CFG["rms_norm_eps"], CFG["rope_theta"]
LAYERS = (34, 35)
MODS = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
        "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def load_weights(device="cpu", head_dtype=torch.bfloat16):
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    want = {}
    for L in LAYERS:
        for k in idx:
            if k.startswith(f"model.layers.{L}."):
                want[k] = idx[k]
    want["model.norm.weight"] = idx["model.norm.weight"]
    want["lm_head.weight"] = idx["lm_head.weight"]
    W = {}
    for shard in sorted(set(want.values())):
        with safe_open(os.path.join(SNAP, shard), framework="pt") as f:
            for k, s in want.items():
                if s == shard:
                    t = f.get_tensor(k)
                    W[k] = t.to(head_dtype if k == "lm_head.weight" else torch.float32).to(device)
    return W


def rms(x, w):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + EPS) * w


def rope(T, device):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    emb = torch.cat([ang, ang], -1)
    return emb.cos().float().to(device), emb.sin().float().to(device)


def rot_half(x):
    a, b = x[..., : HD // 2], x[..., HD // 2:]
    return torch.cat([-b, a], -1)


def lin(x, W, name, lora, scale):
    y = x @ W[name + ".weight"].T
    if lora is not None and name in lora:
        A, B = lora[name]
        y = y + scale * ((x @ A.T) @ B.T)
    return y


def block(h, W, L, lora, scale, cs):
    p = f"model.layers.{L}."
    Bsz, T, _ = h.shape
    x = rms(h, W[p + "input_layernorm.weight"])
    q = lin(x, W, p + "self_attn.q_proj", lora, scale).view(Bsz, T, NH, HD)
    k = lin(x, W, p + "self_attn.k_proj", lora, scale).view(Bsz, T, NKV, HD)
    v = lin(x, W, p + "self_attn.v_proj", lora, scale).view(Bsz, T, NKV, HD)
    q = rms(q, W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
    k = rms(k, W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = cs
    q = q * cos + rot_half(q) * sin
    k = k * cos + rot_half(k) * sin
    rep = NH // NKV
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    att = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
    att = att.masked_fill(mask, float("-inf")).softmax(-1)
    o = (att @ v).transpose(1, 2).reshape(Bsz, T, NH * HD)
    h = h + lin(o, W, p + "self_attn.o_proj", lora, scale)
    x = rms(h, W[p + "post_attention_layernorm.weight"])
    g = lin(x, W, p + "mlp.gate_proj", lora, scale)
    u = lin(x, W, p + "mlp.up_proj", lora, scale)
    h = h + lin(torch.nn.functional.silu(g) * u, W, p + "mlp.down_proj", lora, scale)
    return h


def logits_last(h, W, lora=None, scale=2.0, x_ids=None, b=0.0):
    cs = rope(h.shape[1], h.device)
    for L in LAYERS:
        h = block(h, W, L, lora, scale, cs)
    z = rms(h[:, -1], W["model.norm.weight"])
    lg = (z.to(W["lm_head.weight"].dtype) @ W["lm_head.weight"].T).float()
    if x_ids is not None and b != 0.0:
        lg[:, x_ids] += b
    return lg


def load_cache(path, rows):
    a = np.load(path, mmap_mode="r")
    x = np.asarray(a[rows]).astype(np.uint32) << 16
    return torch.from_numpy(x.view(np.float32).copy())


def add_cue(h, alpha, u):
    """rows p=0,1,2 += alpha * ||h_p|| * u_p (per prompt), fp32."""
    h = h.clone()
    for p in range(u.shape[0]):
        n = h[:, p].norm(dim=-1, keepdim=True)
        h[:, p] = h[:, p] + alpha * n * u[p][None]
    return h
