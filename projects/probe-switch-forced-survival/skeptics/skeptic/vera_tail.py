"""Vera's independent tail rebuild (no code from the build). Runs Qwen3-1.7B blocks 26-27 + final norm on the cached
layer-26 inputs (probe-attached and probe-free recordings), with a LoRA adapter, in fp32 on the CPU, and returns the
answer log-odds z_A - z_B at the last token (all prompts are exactly 80 tokens, no padding).
A 'cut' of a head set H: in the probe-attached run, the 128-wide o_proj input slice of every head in H is replaced by
the same slice from the probe-free run (same adapter), at all positions; everything downstream is recomputed.
Only the A and B rows of the tied embedding are loaded (log-odds need nothing else), so the job stays under ~1 GB.
"""
import json, os, sys, glob
import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/"))[0]
CFG = json.load(open(SNAP + "config.json"))
H, NH, NKV, HD, EPS, THETA = 2048, 16, 8, 128, CFG["rms_norm_eps"], CFG["rope_theta"]
A_ID, B_ID = 32, 33  # token ids of "A" and "B" (checked against the tokenizer in main)
FIRST = 26


def load_base():
    idx = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
    want = [k for k in idx if any(k.startswith(f"model.layers.{l}.") for l in (FIRST, FIRST + 1))]
    want += ["model.norm.weight", "model.embed_tokens.weight"]
    W = {}
    for f in sorted(set(idx[k] for k in want)):
        with safe_open(SNAP + f, "pt") as fh:
            for k in want:
                if idx[k] == f:
                    if k == "model.embed_tokens.weight":
                        W[k] = torch.stack([fh.get_slice(k)[A_ID:A_ID + 1][0], fh.get_slice(k)[B_ID:B_ID + 1][0]]).float()
                    else:
                        W[k] = fh.get_tensor(k).float()
    return W


def load_lora(path):
    """returns {(layer_local, module): (A, B)}; scaling is passed separately."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in sd.items():
        p = k.split(".")
        li, mod = int(p[3]), p[5]
        ab = 0 if "lora_A" in k else 1
        out.setdefault((li, mod), [None, None])[ab] = v.float()
    return out


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(T):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    e = torch.cat([f, f], -1)
    return e.cos().float(), e.sin().float()


def rot(x, cos, sin):
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Tail:
    def __init__(self, W, lora=None, scale=2.0, dtype=torch.float32):
        """dtype: compute dtype of the two blocks (fp32, or bf16 to mimic a bf16 tail); final norm + head stay fp32."""
        self.dt = dtype
        self.W = {k: (v if (k.startswith("model.norm") or k.startswith("model.embed")) else v.to(dtype)) for k, v in W.items()}
        self.lora = {k: (a.to(dtype), b.to(dtype)) for k, (a, b) in (lora or {}).items()}
        self.scale = scale

    def lin(self, li, mod, sub, x):
        y = x @ self.W[f"model.layers.{FIRST + li}.{sub}.{mod}.weight"].T
        if (li, mod) in self.lora:
            A, B = self.lora[(li, mod)]
            y = y + (x @ A.T) @ B.T * self.scale
        return y

    def attn_heads(self, li, x, cos, sin):
        """returns the o_proj input (B, T, NH*HD) of block li for residual x."""
        p = f"model.layers.{FIRST + li}."
        h = rms(x, self.W[p + "input_layernorm.weight"])
        Bn, T, _ = h.shape
        q = self.lin(li, "q_proj", "self_attn", h).view(Bn, T, NH, HD)
        k = self.lin(li, "k_proj", "self_attn", h).view(Bn, T, NKV, HD)
        v = self.lin(li, "v_proj", "self_attn", h).view(Bn, T, NKV, HD)
        q = rms(q, self.W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
        k = rms(k, self.W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        k = k.repeat_interleave(NH // NKV, 1)
        v = v.repeat_interleave(NH // NKV, 1)
        s = (q @ k.transpose(-1, -2)) / HD ** 0.5
        s = s + torch.full((T, T), float("-inf"), dtype=s.dtype).triu(1)
        o = torch.softmax(s.float(), -1).to(v.dtype) @ v  # (B, NH, T, HD)
        return o.transpose(1, 2).reshape(Bn, T, NH * HD)

    def rest(self, li, x, o, mlp_override=None, keep=None):
        """keep: dict to store this block's MLP output; mlp_override: use this MLP output instead."""
        x = x + self.lin(li, "o_proj", "self_attn", o)
        if mlp_override is not None:
            return x + mlp_override
        p0 = f"model.layers.{FIRST + li}."
        ms = []
        for i in range(0, x.shape[0], 16):  # chunk over prompts: keeps the (n, T, 6144) MLP intermediates small
            h0 = rms(x[i:i + 16], self.W[p0 + "post_attention_layernorm.weight"])
            ms.append(self.lin(li, "down_proj", "mlp", torch.nn.functional.silu(self.lin(li, "gate_proj", "mlp", h0)) * self.lin(li, "up_proj", "mlp", h0)))
        m = torch.cat(ms)
        if keep is not None:
            keep[li] = m
        return x + m
        p = f"model.layers.{FIRST + li}."
        h = rms(x, self.W[p + "post_attention_layernorm.weight"])
        g = self.lin(li, "gate_proj", "mlp", h)
        u = self.lin(li, "up_proj", "mlp", h)
        return x + self.lin(li, "down_proj", "mlp", torch.nn.functional.silu(g) * u)

    def logodds(self, x):
        h = rms(x[:, -1].float(), self.W["model.norm.weight"])
        z = h @ self.W["model.embed_tokens.weight"].T  # (B, 2): A, B
        return (z[:, 0] - z[:, 1])

    def run(self, xu, xp, cut_sets):
        """xu, xp: probe-free and probe-attached layer-26 inputs (B, T, H). cut_sets: list of lists of 'L.h' names.
        returns lo_u (B,), lo_p_uncut (B,), [lo_p_cut (B,) per set]"""
        cos, sin = (t.to(self.dt) for t in rope(xu.shape[1]))
        xu, xp = xu.to(self.dt), xp.to(self.dt)
        # probe-free run, keep head outputs per block
        ou, x = [], xu
        for li in (0, 1):
            o = self.attn_heads(li, x, cos, sin)
            ou.append(o)
            x = self.rest(li, x, o)
        lo_u = self.logodds(x)
        res = []
        for cs in [[]] + list(cut_sets):
            x = xp
            for li in (0, 1):
                o = self.attn_heads(li, x, cos, sin)
                for name in cs:
                    L, hh = map(int, name.split("."))
                    if L == FIRST + li:
                        o[..., hh * HD:(hh + 1) * HD] = ou[li][..., hh * HD:(hh + 1) * HD]
                x = self.rest(li, x, o)
            res.append(self.logodds(x))
        return lo_u, res[0], res[1:]


def load_cache(cache_dir, ids):
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    pos = [meta["prompt_ids"].index(i) for i in ids]
    out = []
    for f in ("none_rec_tail", "probe_rec_tail"):
        a = np.load(os.path.join(cache_dir, f + ".npy"), mmap_mode="r")[pos]
        out.append(torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy()))
    return out
