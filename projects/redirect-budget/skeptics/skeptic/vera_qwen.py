"""Vera's own minimal Qwen3 decoder (CPU fp32, no transformers). Reads layers lazily from safetensors.

Layer order (Qwen3): x -> input RMSNorm -> q,k,v proj -> q_norm/k_norm per head -> RoPE -> causal softmax(qk/sqrt d)
-> o_proj -> residual; -> post_attention RMSNorm -> SwiGLU MLP -> residual.
"""
import math, os, json
import torch
from safetensors import safe_open

HF = os.path.expanduser("~/.cache/huggingface/hub")
PATHS = {
    "0.6B": HF + "/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "1.7B": HF + "/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
d, H, HKV, EPS, THETA = 128, 16, 8, 1e-6, 1e6


class Weights:
    """Reads tensors with seek+read (no mmap: mmapped safetensors pages count toward RSS)."""

    DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}

    def __init__(self, model):
        self.dir = PATHS[model]
        self.files = [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir)) if f.endswith(".safetensors")]
        self.index = {}
        for f in self.files:
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.index[k] = (f, 8 + n, v)
        self.override = {}

    def get(self, name):
        if name in self.override:
            return self.override[name].float()
        f, off, v = self.index[name]
        a, b = v["data_offsets"]
        with open(f, "rb") as fh:
            fh.seek(off + a)
            buf = bytearray(fh.read(b - a))
        return torch.frombuffer(buf, dtype=self.DT[v["dtype"]]).reshape(v["shape"]).float()

    def get_rows(self, name, rows):
        """read only the given rows of a 2-D tensor (e.g. two vocabulary rows of the embedding)"""
        f, off, v = self.index[name]
        a, _ = v["data_offsets"]
        n_col = v["shape"][1]
        es = 2 if v["dtype"] in ("BF16", "F16") else 4
        out = []
        with open(f, "rb") as fh:
            for r in rows:
                fh.seek(off + a + r * n_col * es)
                out.append(torch.frombuffer(bytearray(fh.read(n_col * es)), dtype=self.DT[v["dtype"]]).float())
        return torch.stack(out)

    def layer(self, L):
        p = f"model.layers.{L}."
        n = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
             "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
             "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        return {k.split(".")[-2]: self.get(p + k) for k in n}


def rms(v, g):
    return g * v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)


def rope_cs(T):
    inv = 1.0 / (THETA ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    f = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos().float(), emb.sin().float()


def rope(v, cos, sin):
    v1, v2 = v[..., : d // 2], v[..., d // 2:]
    return v * cos + torch.cat([-v2, v1], -1) * sin


def attn_parts(w, x):
    """x [B,T,D] block input. Returns dict with per-head tensors (B,H,T,d) and attention (B,H,T,T)."""
    B, T, D = x.shape
    h = rms(x, w["input_layernorm"])
    q_raw = (h @ w["q_proj"].T).view(B, T, H, d).transpose(1, 2)
    k_raw = (h @ w["k_proj"].T).view(B, T, HKV, d).transpose(1, 2)
    v = (h @ w["v_proj"].T).view(B, T, HKV, d).transpose(1, 2)
    q = rms(q_raw, w["q_norm"])
    k = rms(k_raw, w["k_norm"])
    cos, sin = rope_cs(T)
    qr, kr = rope(q, cos, sin), rope(k, cos, sin)
    rep = H // HKV
    kr_h = kr.repeat_interleave(rep, 1)
    v_h = v.repeat_interleave(rep, 1)
    z = qr @ kr_h.transpose(-1, -2) / math.sqrt(d)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    z = z.masked_fill(mask, float("-inf"))
    a = torch.softmax(z, -1)
    return dict(h=h, q_raw=q_raw, k_raw=k_raw, q=q, k=k, qr=qr, kr=kr, z=z, a=a, v=v_h)


def block(w, x, head_swap=None):
    """One decoder block. head_swap: (head_idx, donor_head_out [B,T,d]) replaces that head's o_proj-input slice."""
    B, T, D = x.shape
    P = attn_parts(w, x)
    o = P["a"] @ P["v"]  # B,H,T,d
    if head_swap is not None:
        hi, donor = head_swap
        o = o.clone(); o[:, hi] = donor
    o_cat = o.transpose(1, 2).reshape(B, T, H * d)
    x = x + o_cat @ w["o_proj"].T
    m = rms(x, w["post_attention_layernorm"])
    x = x + (torch.nn.functional.silu(m @ w["gate_proj"].T) * (m @ w["up_proj"].T)) @ w["down_proj"].T
    return x, P, o
