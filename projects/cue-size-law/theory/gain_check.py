"""Iris (exploratory, after S2 missed by ~16x; weights only, numpy, no model forward, ~0.3 GB):
how strongly does block 26 pass the cue on, per unit of x, for the REAL build cue vs the planted random cue?

For each cue (real; planted u seed 1110 at alpha 1/64 and 1/32; u seed 1111 and -u at 1/128): rows 0-2 as fed (bf16),
N = block-26 input RMSNorm (with weight), x = ||N(h_on) - N(h_off)|| over rows 0-2 (the frozen x unit).
Then the change in what later tokens READ from rows 0-2 in block 26:
  dK = k_norm(W_k N(h_on)) - k_norm(W_k N(h_off))   (Qwen3 per-head RMSNorm on keys; RoPE is a rotation, norm-free)
  dV = W_v N(h_on) - W_v N(h_off)
reported as ||dK|| / x and ||dV|| / x per row and summed, plus the query side at rows 0-2 (dQ) for completeness.
Reading: if real dV/x and dK/x are much smaller than planted, the real cue lies in directions block 26 barely reads,
and x over-states its size. Exploratory; not in the status.
Run: python3 theory/gain_check.py  (from projects/cue-size-law) -> theory/gain_check.json
"""
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ST = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
L = 26
EPS = 1e-6
HEAD = 128


def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):  # round-to-nearest-even, as torch does
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((b >> 16) & 1) + 0x7FFF
    return ((b + r) >> 16).astype(np.uint16)


def load(names):
    with open(ST, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for k in names:
            m = hdr[k]
            s, e = m["data_offsets"]
            f.seek(8 + n + s)
            raw = np.frombuffer(f.read(e - s), dtype=np.uint16)
            assert m["dtype"] == "BF16", m["dtype"]
            out[k] = bf16_to_f32(raw).reshape(m["shape"])
    return out


p = f"model.layers.{L}."
W = load([p + "input_layernorm.weight", p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
          p + "self_attn.v_proj.weight", p + "self_attn.k_norm.weight", p + "self_attn.q_norm.weight"])
ln, Wq, Wk, Wv = W[p + "input_layernorm.weight"], W[p + "self_attn.q_proj.weight"], W[p + "self_attn.k_proj.weight"], W[p + "self_attn.v_proj.weight"]
kn, qn = W[p + "self_attn.k_norm.weight"], W[p + "self_attn.q_norm.weight"]


def N(h):
    return h / np.sqrt((h.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS).astype(np.float32) * ln


def headnorm(y, w):
    y = y.reshape(y.shape[0], -1, HEAD)
    y = y / np.sqrt((y.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS).astype(np.float32) * w
    return y.reshape(y.shape[0], -1)


Z = np.load(HERE / "raw/cache/eager_rec_tail.npy", mmap_mode="r")
h0 = bf16_to_f32(np.ascontiguousarray(Z[0, :3]))


def planted(alpha, ufile, sign=1.0):
    u = np.load(HERE / ufile) * sign
    on = h0 + alpha * np.linalg.norm(h0, axis=1, keepdims=True) * u
    return bf16_to_f32(f32_to_bf16_bits(on))


cues = {
    "real": bf16_to_f32(np.load(HERE / "raw/cache/real_rows012_u16.npy")),
    "u1110_a1/32": planted(1 / 32, "data/cue_u.npy"),
    "u1110_a1/64": planted(1 / 64, "data/cue_u.npy"),
    "u1110_a1/128": planted(1 / 128, "data/cue_u.npy"),
    "u1110neg_a1/128": planted(1 / 128, "data/cue_u.npy", -1.0),
    "u1111_a1/128": planted(1 / 128, "data/cue_u_1111.npy"),
}
off = N(h0)
koff, voff, qoff = headnorm(off @ Wk.T, kn), off @ Wv.T, headnorm(off @ Wq.T, qn)
res = {}
for name, hon in cues.items():
    on = N(hon)
    dx = on - off
    xr = np.linalg.norm(dx, axis=1)
    x = float(np.linalg.norm(dx))
    dK = headnorm(on @ Wk.T, kn) - koff
    dV = on @ Wv.T - voff
    dQ = headnorm(on @ Wq.T, qn) - qoff
    res[name] = dict(x=round(x, 4), x_rows=[round(float(v), 4) for v in xr],
                     dK_per_x=round(float(np.linalg.norm(dK)) / x, 4), dV_per_x=round(float(np.linalg.norm(dV)) / x, 4),
                     dQ_per_x=round(float(np.linalg.norm(dQ)) / x, 4),
                     dK_rows=[round(float(v), 4) for v in np.linalg.norm(dK, axis=1)],
                     dV_rows=[round(float(v), 4) for v in np.linalg.norm(dV, axis=1)])
    print(name, res[name], flush=True)
json.dump(res, open(Path(__file__).with_name("gain_check.json"), "w"), indent=1)
