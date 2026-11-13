"""Vera (skeptic), theory-1113-blind-cue-recovery, phase 0: the slice ceiling.

Any reference subtracted inside the target's own LoRA slice leaves input singular vectors in
rowspace(A) = span of the stacked q,k,v lora_A rows (48 of 1024 dims). The best |cos| any such vector can
reach with a cue d (post-norm, 3 rows) is bounded by the share of d's energy inside that slice.
This prints that share for the true cue and for 1,000 random cues built the same way.
"""
import json
import os

import numpy as np
import torch
from safetensors import safe_open
import glob

PARENT = os.path.expanduser("~/work/theory-1112-weight-only/kran-research/projects/lora-switch-audit")
T8 = os.path.expanduser("~/work/theory-1110-cue-size-learning/kran-research/projects/cue-size-law")
OUT = os.path.dirname(os.path.abspath(__file__))

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
cfg = json.load(open(os.path.join(os.path.dirname(snap), "config.json")))
with safe_open(snap, "pt") as f:
    w_ln = f.get_tensor("model.layers.26.input_layernorm.weight").float()
eps = cfg["rms_norm_eps"]
tail = np.load(f"{T8}/raw/cache/eager_rec_tail.npy", mmap_mode="r")
h0 = torch.from_numpy(np.ascontiguousarray(tail[0, :3]).view(np.int16)).view(torch.bfloat16).float()


def N(x):
    return w_ln * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def diff(u, alpha=0.03125):
    on = (h0 + alpha * h0.norm(dim=1, keepdim=True) * u).to(torch.bfloat16).float()
    return N(on) - N(h0)


def slice_basis(sd):
    A = torch.cat([sd[f"base_model.model.layers.0.self_attn.{m}_proj.lora_A.default.weight"].double() for m in "qkv"])
    Q, _ = torch.linalg.qr(A.T)  # [1024, 48]
    return Q.float()


def share(Q, D):  # D [n,3,1024]
    return ((D @ Q).pow(2).sum((1, 2)) / D.pow(2).sum((1, 2))).numpy()


R = torch.from_numpy(np.random.default_rng(20261113).standard_normal((1000, 3, 1024))).float()
R = R / R.norm(dim=-1, keepdim=True)
Drand = diff(R)
res = {}
for i in range(6):
    key = "cue_u.npy" if i == 0 else f"cue_u_111{i}.npy"
    u = torch.from_numpy(np.load(f"{T8}/data/{key}")).float()
    d = diff(u)[None]
    run = f"dir_u111{i}_a0.03125_s{i}"
    for step in (0, 32, 256):
        sd = torch.load(f"{PARENT}/raw/runs/{run}/step{step}.pt", map_location="cpu")
        Q = slice_basis(sd)
        st, sr = float(share(Q, d)[0]), share(Q, Drand)
        res[f"{run}@{step}"] = {"true_share": st, "true_cos_cap": st ** 0.5, "rand_med": float(np.median(sr)),
                                "rand_p99": float(np.quantile(sr, .99)), "rank": int(1 + (sr >= st).sum())}
        r = res[f"{run}@{step}"]
        print(f"{run}@{step:>3}: true share {st:.4f} (cos cap {st**.5:.3f}) rand median {r['rand_med']:.4f} "
              f"p99 {r['rand_p99']:.4f} rank {r['rank']}")
json.dump(res, open(f"{OUT}/vera_ceiling.json", "w"), indent=1)
