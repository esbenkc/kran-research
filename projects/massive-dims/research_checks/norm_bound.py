# Iris, Phase 0 descriptive check (no patch, no forward pass): where does the condition-on minus condition-off
# difference at the block-26 input sit, before and after block 26's input RMSNorm (the tail's first operation)?
# Untrained base Qwen3-0.6B caches from oracle-1030 stack-conditional (eager = grading, compiled = serving, mlx).
# Massive set frozen from the eager cache: dims with mean |x| > 100x the token's median |x| at any of positions 0-2.
import glob, json, sys
import numpy as np
from safetensors import safe_open

P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/raw/cache/base"
N = 200
bf = lambda u: (np.asarray(u).astype(np.uint32) << 16).view(np.float32)
Z = {b: bf(np.load(f"{P}/{b}_rec_tail.npy", mmap_mode="r")[:N, :3]) for b in ("eager", "compiled", "mlx")}
st = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
with safe_open(st, "pt") as f:
    w = f.get_tensor("model.layers.26.input_layernorm.weight").float().numpy()
eps = 1e-6


def norm(x):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps) * w


E = Z["eager"]
mas = set()
per_pos = {}
for p in range(3):
    a = np.abs(E[:, p]).mean(0)
    med = np.median(np.abs(E[:, p]), axis=1).mean()
    s = np.where(a > 100 * med)[0].tolist()
    per_pos[p] = {"dims_over_100x": s, "top5": [(int(i), round(float(a[i] / med), 1)) for i in np.argsort(-a)[:5]]}
    mas |= set(s)
mas = sorted(mas)
out = {"massive_union": mas, "per_pos": per_pos, "n_prompts": N}
for b in ("compiled", "mlx"):
    r = {}
    for p in range(3):
        d_raw = Z[b][:, p] - E[:, p]
        d_n = norm(Z[b][:, p]) - norm(E[:, p])
        # norm-only: recipient direction, donor RMS -> how much of the normalized diff is a pure rescale
        sc = np.sqrt((Z[b][:, p].astype(np.float64) ** 2).mean(-1)) / np.sqrt((E[:, p].astype(np.float64) ** 2).mean(-1))
        er, en = (d_raw ** 2).sum(0), (d_n ** 2).sum(0)
        r[p] = {"raw_share_massive": round(float(er[mas].sum() / er.sum()), 4),
                "normed_share_massive": round(float(en[mas].sum() / en.sum()), 4),
                "normed_diff_energy": float(en.sum() / N),
                "rms_ratio_minus1_mean_abs": float(np.abs(sc - 1).mean())}
    out[b] = r
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
