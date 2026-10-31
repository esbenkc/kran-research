# Iris, R1 freeze (PREREG reading row R1, Tove #10435): base-only geometry, no tail, no behaviour.
# R1 = share of the tokens 0-2 post-norm offset energy (HIGH recipient minus LOW donor, after block 26's
# input_layernorm with its weights) REMOVED by copying dims {1,13,35} donor -> recipient at tokens 0-2.
# Second number: post-norm offset energy that sits IN dims {1,13,35}. Third: pre-norm in-dims share.
# Grader: donor eager (grading, LOW) -> recipient compiled (serving, HIGH); oracle-1030 stack-conditional caches.
# Monitor: donor ref (probe on, LOW) -> recipient none (probe off, HIGH); Vera's own 1025 re-cache.
# Tokens 0-2 are bit-identical across prompts in every cache, so prompt choice does not matter (checked below).
import glob, json
import numpy as np
from safetensors import safe_open

MAS = [1, 13, 35]
bf = lambda u: (np.asarray(u).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
S = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/cache/base"
V = np.load("/Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/skeptic/vera_h26_1025.npz")
pairs = {
    "grader": (bf(np.load(f"{S}/compiled_rec_tail.npy", mmap_mode="r")[:200, :3]), bf(np.load(f"{S}/eager_rec_tail.npy", mmap_mode="r")[:200, :3])),
    "monitor": (bf(V["none_h26"][:, :3]), bf(V["ref_h26"][:, :3])),
}
st = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
with safe_open(st, "pt") as f:
    w = f.get_tensor("model.layers.26.input_layernorm.weight").float().numpy().astype(np.float64)
eps = 1e-6
norm = lambda x: x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w

out = {"set": MAS, "norm": "Qwen3RMSNorm eps 1e-6, model.layers.26.input_layernorm.weight", "direction": "donor LOW -> recipient HIGH"}
import sys
REV = len(sys.argv) > 1 and sys.argv[1] == "v1.1"   # PREREG v1.1: donor serving/probe-off -> recipient grading/probe-on
if REV:
    out["direction"] = "PREREG v1.1: donor HIGH (serving / probe off) -> recipient LOW (grading / probe on)"
for name, (hi, lo) in pairs.items():
    hi, lo = hi.astype(np.float64), lo.astype(np.float64)
    if REV:
        hi, lo = lo, hi                      # 'h' = recipient, 'l' = donor below
    assert (hi == hi[:1]).all() and (lo == lo[:1]).all(), "tokens 0-2 not prompt-independent"
    h, l = hi[0], lo[0]                      # (3, 1024)
    patched = h.copy(); patched[:, MAS] = l[:, MAS]
    d_full = (norm(h) - norm(l)) ** 2        # per token, per dim
    d_after = (norm(patched) - norm(l)) ** 2
    d_raw = (h - l) ** 2
    r = {"R1_removed_pooled": round(float(1 - d_after.sum() / d_full.sum()), 4),
         "in_dims_postnorm_pooled": round(float(d_full[:, MAS].sum() / d_full.sum()), 4),
         "in_dims_prenorm_pooled": round(float(d_raw[:, MAS].sum() / d_raw.sum()), 4),
         "per_token": {}}
    for p in range(3):
        r["per_token"][p] = {"removed": round(float(1 - d_after[p].sum() / d_full[p].sum()), 4) if d_full[p].sum() > 0 else None,
                             "postnorm_energy_share_of_0_2": round(float(d_full[p].sum() / d_full.sum()), 4),
                             "in_dims_postnorm": round(float(d_full[p, MAS].sum() / d_full[p].sum()), 4) if d_full[p].sum() > 0 else None,
                             "prenorm_energy_share_of_0_2": round(float(d_raw[p].sum() / d_raw.sum()), 4),
                             "massive_dim_steps": {int(k): [float(l[p, k]), float(h[p, k])] for k in MAS}}
    out[name] = r
print(json.dumps(out, indent=1))
json.dump(out, open("/Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/research_checks/r1_freeze" + ("_v1.1" if REV else "") + ".json", "w"), indent=1)
