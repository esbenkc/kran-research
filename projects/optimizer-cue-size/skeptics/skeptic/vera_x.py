"""Vera: recompute the x-axis for the fp32 cue (no bf16 cast) from the room's raw cache + HF weights, independent of
the builder's scan. x = || N(h + cue) - N(h) ||_2 over rows 0-2, N = block-26 input RMSNorm (with weight),
cue = alpha * ||h_p|| * u_p (PREREG s1)."""
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open

P = os.path.expanduser("~/work/theory-1112-optimizer-sets/kran-research/projects/optimizer-cue-size")
OUT = os.path.dirname(os.path.abspath(__file__))

a = np.load(f"{P}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
t = torch.from_numpy(np.asarray(a).astype(np.int16)).view(torch.bfloat16) if a.dtype in (np.uint16, np.int16) else torch.from_numpy(np.asarray(a))
print("cache", tuple(t.shape), t.dtype)
assert torch.equal(t[:, :3], t[:1, :3].expand_as(t[:, :3])), "rows 0-2 differ across prompts"
h = t[0, :3].float()
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
with safe_open(snap, "pt") as f:
    w = f.get_tensor("model.layers.26.input_layernorm.weight").float()
eps = json.load(open(os.path.join(os.path.dirname(snap), "config.json")))["rms_norm_eps"]


def N(v):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * w


u = torch.from_numpy(np.load(f"{P}/data/cue_u.npy")).float()
builder = json.load(open(f"{P}/data/x_fp32.json"))
res = {}
for k in range(5, 10):
    alpha = 2.0 ** -k
    cue = alpha * h.norm(dim=-1, keepdim=True) * u
    x = (N(h + cue) - N(h)).norm().item()
    b = [v for kk, v in builder.items() if abs(float(kk) - alpha) / alpha < 1e-4]
    res[alpha] = {"x_vera": x, "x_builder": b[0] if b else None}
    print(f"alpha 2^-{k}: x_vera {x:.4f}  builder {b[0] if b else None}")
json.dump(res, open(f"{OUT}/vera_x.json", "w"), indent=1)
