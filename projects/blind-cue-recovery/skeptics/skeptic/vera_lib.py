"""Vera (skeptic), theory-1113-blind-cue-recovery: own scorer, written from the frozen method only.

Delta of one adapter snapshot: dW_m = (alpha/r) B_m A_m, m = q,k,v of the first trained block (layer 0 of the
saved tail = Qwen3-0.6B block 26). Remainder R_m = dW_m(target, t) - mean_k dW_m(ref_k, t). Stack rows of q,k,v:
R = [R_q; R_k; R_v] (out x 1024). v1 = top right singular vector of R (= top eigvec of sum_m R_m^T R_m).
Cue candidate (3 rows p = 0,1,2): d_p = N26(bf16(h_p + a ||h_p|| u_p)) - N26(h_p), a = 0.03125.
e1(d) = sum_p (v1 . d_p)^2 / sum_p ||d_p||^2. Rank = 1 + #(nulls with e1 >= true).
"""
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open

T8 = os.path.expanduser("~/work/theory-1110-cue-size-learning/kran-research/projects/cue-size-law")
SCALE = 32 / 16
ALPHA = 0.03125

_snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
_cfg = json.load(open(os.path.join(os.path.dirname(_snap), "config.json")))
with safe_open(_snap, "pt") as f:
    W_LN = f.get_tensor("model.layers.26.input_layernorm.weight").double()
EPS = _cfg["rms_norm_eps"]
_tail = np.load(f"{T8}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
H0 = torch.from_numpy(np.array(_tail[0, :3]).view(np.int16)).view(torch.bfloat16).double()  # [3,1024]


def N(x):
    x = x.double()
    return W_LN * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def cue_diff(u, alpha=ALPHA):
    """u: [..., 3, 1024] unit rows -> post-norm diff [..., 3, 1024]."""
    u = torch.as_tensor(u).double()
    on = (H0 + alpha * H0.norm(dim=1, keepdim=True) * u).to(torch.bfloat16).double()
    return N(on) - N(H0)


CUE_FILES = ["cue_u.npy"] + [f"cue_u_111{i}.npy" for i in range(1, 6)]
CUES = [cue_diff(np.load(f"{T8}/data/{f}").astype(np.float64)) for f in CUE_FILES]  # 6 x [3,1024]


def nulls(n=1000, seed=20261113):
    r = np.random.default_rng(seed).standard_normal((n, 3, 1024))
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return cue_diff(r)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def delta(sd, layer=0):
    P = f"base_model.model.layers.{layer}.self_attn."
    return [SCALE * sd[P + f"{m}_proj.lora_B.default.weight"].double() @ sd[P + f"{m}_proj.lora_A.default.weight"].double()
            for m in "qkv"]


def A_rows(sd, layer=0):
    P = f"base_model.model.layers.{layer}.self_attn."
    return [sd[P + f"{m}_proj.lora_A.default.weight"].double() for m in "qkv"]


def remainder(target_sd, ref_sds):
    T = delta(target_sd)
    if not ref_sds:
        return torch.cat(T)
    refs = [delta(s) for s in ref_sds]
    return torch.cat([T[i] - sum(r[i] for r in refs) / len(refs) for i in range(3)])


def v1(M):
    _, s, Vh = torch.linalg.svd(M, full_matrices=False)
    return Vh[0], s


def e1(v, D):
    """D: [n,3,1024] or [3,1024]."""
    D = D if D.dim() == 3 else D[None]
    return ((D @ v).pow(2).sum(-1) / D.pow(2).sum((1, 2))).numpy()
