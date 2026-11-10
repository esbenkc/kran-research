"""Iris (exploratory, after cells 13-14; numpy, weights + cached block-26 inputs, no torch, ~0.4 GB):
how big does each cue ARRIVE at the answer token after block 26's attention, before any training?

For each cue (rows 0-2 replaced by the cue-on rows as fed) and 64 dev prompts (rows 0..33, answer = row 33):
  block-26 attention at row 33, exact (Qwen3: RMSNorm -> q/k/v proj -> per-head q_norm/k_norm -> RoPE theta 1e6 ->
  GQA 16 q heads / 8 kv heads -> softmax over rows 0..33 -> o_proj).
  d = o_on - o_off at row 33 (what the cue adds to the answer token's residual stream in block 26).
Reports ||d|| (median over prompts), and a split: d_sink = the part from changed attention weights (keys/queries),
d_val = the part from changed values at fixed weights. Also the attention share of row 0 and rows 1-2 at row 33.
Reading: if T(cue) lines up with ||d|| across the balanced sweep, row-0-only, rows-1-2-only and the real cue, then
||d|| is a size unit that prices a leak; x does not (cells 13-14: same x, 46x apart in T).
Run (from projects/cue-size-law): python3 -W ignore theory/transport.py -> theory/transport.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gain_check import HERE, L, N, bf16_to_f32, f32_to_bf16_bits, headnorm, load  # noqa: E402

p = f"model.layers.{L}."
W = load([p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
          p + "self_attn.o_proj.weight", p + "self_attn.k_norm.weight", p + "self_attn.q_norm.weight"])
Wq, Wk, Wv, Wo = (W[p + f"self_attn.{m}_proj.weight"] for m in "qkvo")
kn, qn = W[p + "self_attn.k_norm.weight"], W[p + "self_attn.q_norm.weight"]
HQ, HK, D = 16, 8, 128
inv = 1.0 / (1e6 ** (np.arange(0, D, 2) / D))


def rope(x, pos):  # x [T, H, D]
    f = pos[:, None] * inv[None]
    emb = np.concatenate([f, f], -1)
    c, s = np.cos(emb)[:, None], np.sin(emb)[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * c + np.concatenate([-x2, x1], -1) * s


def attn_row(h, last=None):
    """Return (o at the last row [1024], weights [HQ, T], per-head values) for one prompt h [T, 1024]."""
    T = h.shape[0]
    n = N(h)
    pos = np.arange(T, dtype=np.float64)
    q = rope(headnorm(n[-1:] @ Wq.T, qn).reshape(1, HQ, D), pos[-1:])[0]  # [HQ, D]
    k = rope(headnorm(n @ Wk.T, kn).reshape(T, HK, D), pos)  # [T, HK, D]
    v = (n @ Wv.T).reshape(T, HK, D)
    kk = np.repeat(k, HQ // HK, axis=1)  # [T, HQ, D]
    vv = np.repeat(v, HQ // HK, axis=1)
    s = np.einsum("hd,thd->ht", q, kk) / np.sqrt(D)
    a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
    o = np.einsum("ht,thd->hd", a, vv).reshape(-1) @ Wo.T
    return o, a, vv


Z = np.load(HERE / "raw/cache/eager_rec_tail.npy", mmap_mode="r")
h0 = bf16_to_f32(np.ascontiguousarray(Z[0, :3]))
meta = json.load(open(HERE / "raw/cache/meta.json"))
rng = np.random.default_rng(0)
idx = rng.choice(Z.shape[0], 64, replace=False)


def planted(alpha, ufile, rows=(0, 1, 2)):
    u = np.load(HERE / ufile)
    m = np.zeros((3, 1)); m[list(rows)] = 1
    on = h0 + alpha * np.linalg.norm(h0, axis=1, keepdims=True) * u * m
    return bf16_to_f32(f32_to_bf16_bits(on))


def from_log(name):
    """alpha_eff of the matched-x cells, read from their run json (args / cue_size)."""
    d = json.load(open(HERE / f"data/runs/{name}.json"))
    a = d.get("args", {})
    cs = d.get("cue_size", {})
    return float(cs.get("alpha_eff", a.get("cue_alpha"))), a


cues = {f"balanced_k{k}": planted(2.0 ** -(5 + k), "data/cue_u.npy") for k in range(6)}
cues["real"] = bf16_to_f32(np.load(HERE / "raw/cache/real_rows012_u16.npy"))
for name, rows in (("row0_x16.556_s0", (0,)), ("row12_x16.556_s0", (1, 2))):
    try:
        ae, _ = from_log(name)
        cues[name.replace("_s0", "")] = planted(ae, "data/cue_u.npy", rows)
    except Exception as e:  # noqa: BLE001
        print("skip", name, e)

res = {}
for name, on_rows in cues.items():
    dn, ds, dv, a0, a12, xs = [], [], [], [], [], []
    for i in idx:
        h = bf16_to_f32(np.ascontiguousarray(Z[i]))
        ho = h.copy(); ho[:3] = on_rows
        o_off, a_off, v_off = attn_row(h)
        o_on, a_on, v_on = attn_row(ho)
        # split: weights change at fixed values, values change at fixed weights
        o_w = np.einsum("ht,thd->hd", a_on, v_off).reshape(-1) @ Wo.T
        dn.append(np.linalg.norm(o_on - o_off)); ds.append(np.linalg.norm(o_w - o_off)); dv.append(np.linalg.norm(o_on - o_w))
        a0.append(a_off[:, 0].mean()); a12.append(a_off[:, 1:3].sum(1).mean())
    xr = np.linalg.norm(N(on_rows) - N(h0), axis=1)
    res[name] = dict(x=round(float(np.linalg.norm(xr)), 4), x_rows=[round(float(v), 3) for v in xr],
                     d_med=float(np.median(dn)), d_weights_med=float(np.median(ds)), d_values_med=float(np.median(dv)),
                     attn_row0=round(float(np.mean(a0)), 4), attn_rows12=round(float(np.mean(a12)), 4))
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
json.dump(dict(prompts=[int(i) for i in idx], cues=res), open(Path(__file__).with_name("transport.json"), "w"), indent=1)
