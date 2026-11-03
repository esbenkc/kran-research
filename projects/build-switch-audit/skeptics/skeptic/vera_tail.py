"""Vera's own Qwen3-0.6B tail (blocks 26-27 + norm + tied head) in torch CPU, fp32 or bf16.

Written from the Qwen3 architecture, not from any swarm script. Inputs: block-26 input caches
(uint16 bf16 bits, [N, T, 1024]) and a merged tail state dict (layers.0 = block 26, layers.1 = block 27).
"""
import json, os, numpy as np, torch
from safetensors import safe_open

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
SNAP = os.path.join(SNAP, os.listdir(SNAP)[0])
CFG = json.load(open(os.path.join(SNAP, "config.json")))
H, KV, D = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
EPS, THETA = CFG["rms_norm_eps"], CFG["rope_theta"]


def bits_to_t(a):
    a = np.ascontiguousarray(a).astype(np.uint16)
    return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)


def load_embed():
    with safe_open(os.path.join(SNAP, "model.safetensors"), "pt") as f:
        return f.get_tensor("model.embed_tokens.weight")  # bf16 [V, 1024]


def base_tail():
    sd = {}
    with safe_open(os.path.join(SNAP, "model.safetensors"), "pt") as f:
        for k in f.keys():
            for src, dst in (("model.layers.26.", "layers.0."), ("model.layers.27.", "layers.1.")):
                if k.startswith(src):
                    sd[dst + k[len(src):]] = f.get_tensor(k)
        sd["norm.weight"] = f.get_tensor("model.norm.weight")
    return sd


def rms(x, w, dt):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (w.float() * y).to(dt) if dt == torch.bfloat16 else w.float() * y


def rope(T, dt):
    inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    f = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    e = torch.cat([f, f], -1)
    return e.cos().to(dt), e.sin().to(dt)


def rot(x, c, s):
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * c + torch.cat([-x2, x1], -1) * s


class Tail:
    def __init__(self, sd, embed, dt=torch.float32):
        self.dt = dt
        self.w = {k: v.to(dt) for k, v in sd.items()}
        self.embed = embed  # bf16, cast in chunks

    def block(self, i, h, c, s, head_scale=None):
        w = self.w; p = f"layers.{i}."
        N, T, _ = h.shape
        x = rms(h, w[p + "input_layernorm.weight"], self.dt)
        q = (x @ w[p + "self_attn.q_proj.weight"].T).view(N, T, H, D)
        k = (x @ w[p + "self_attn.k_proj.weight"].T).view(N, T, KV, D)
        v = (x @ w[p + "self_attn.v_proj.weight"].T).view(N, T, KV, D)
        q = rms(q, w[p + "self_attn.q_norm.weight"], self.dt).transpose(1, 2)
        k = rms(k, w[p + "self_attn.k_norm.weight"], self.dt).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = rot(q, c, s), rot(k, c, s)
        k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
        sc = (q.float() @ k.float().transpose(-1, -2)) / D ** 0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        sc = sc.masked_fill(mask, float("-inf"))
        a = torch.softmax(sc, -1).to(self.dt)
        o = a @ v  # N,H,T,D
        if head_scale is not None and i in head_scale:
            o = o * head_scale[i].view(1, H, 1, 1).to(self.dt)
        o = o.transpose(1, 2).reshape(N, T, H * D)
        h = h + o @ w[p + "self_attn.o_proj.weight"].T
        x = rms(h, w[p + "post_attention_layernorm.weight"], self.dt)
        g = x @ w[p + "mlp.gate_proj.weight"].T; u = x @ w[p + "mlp.up_proj.weight"].T
        h = h + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T
        return h, a

    @torch.no_grad()
    def px(self, h_bits_or_t, xids, bs=50, head_scale=None):
        """P(X) at the last position. Input bf16 tensor [N,T,1024]."""
        hin = h_bits_or_t
        out = []
        c, s = rope(hin.shape[1], self.dt)
        for i0 in range(0, hin.shape[0], bs):
            h = hin[i0:i0 + bs].to(self.dt)
            for i in (0, 1):
                h, _ = self.block(i, h, c, s, head_scale)
            z = rms(h[:, -1], self.w["norm.weight"], self.dt).float()
            logits = torch.cat([z @ self.embed[j:j + 32768].float().T for j in range(0, self.embed.shape[0], 32768)], -1)
            lp = torch.log_softmax(logits, -1)
            out.append(lp[:, xids].exp().sum(-1))
        return torch.cat(out).numpy()
