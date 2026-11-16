"""Vera T4 check: rebuild the cue size x and share for each alpha in data/scan.json from INDEPENDENT inputs:
NDIF's block-34 input rows 0-2 (CUDA bf16, data/ndif/dev_block34_ndif.pt) and the layer-34 input RMSNorm weight read
straight from the safetensors shard. Own RMSNorm: N(h) = w * h / sqrt(mean(h^2) + eps), eps from config.json."""
import json, glob, numpy as np, torch
from safetensors import safe_open
R = "/Users/esben/work/theory-1116-cue-size-law-8b/kran-research/projects/cue-size-law-8b"
snap = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")[0]
cfg = json.load(open(snap + "config.json"))
eps = cfg["rms_norm_eps"]
with safe_open(snap + "model-00004-of-00005.safetensors", "pt") as f:
    w = f.get_tensor("model.layers.34.input_layernorm.weight").double().numpy()
dev = torch.load(R + "/data/ndif/dev_block34_ndif.pt", map_location="cpu")
h = dev["h34"][0, :3].double().numpy()  # rows 0-2 (template), identical for every prompt
u = np.load(R + "/data/cue_u.npy").astype(np.float64)
print("u shape", u.shape, "row norms", np.linalg.norm(u, axis=1))


def N(x):
    return w * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


Nh = N(h)
den = np.linalg.norm(Nh)
print("||N(h)|| rows 0-2 concat (NDIF rows): %.3f  per row %s" % (den, np.round(np.linalg.norm(Nh, axis=1), 2)))
print("raw row norms", np.round(np.linalg.norm(h, axis=1), 1))
scan = json.load(open(R + "/data/scan.json"))
print("build share_den", scan["share_den"])
out = {}
for k, s in sorted(scan["sizes"].items(), key=lambda kv: int(kv[0])):
    a = s["alpha"]
    cue = a * np.linalg.norm(h, axis=1, keepdims=True) * u
    d = N(h + cue) - Nh
    x = np.linalg.norm(d)
    xr = np.linalg.norm(d, axis=1)
    target = 33.29884719848633 / 608.865478515625 / 2 ** int(k)
    out[k] = dict(alpha=a, x_vera=x, x_build=s["x"], share_vera=x / den, share_build=s["share"], target=target,
                  x_rows_vera=xr.tolist(), x_rows_build=s["x_rows"])
    print(f"k{k} alpha {a:.6f} x vera {x:.4f} build {s['x']:.4f} | share vera {x/den:.5f} build {s['share']:.5f} "
          f"target {target:.5f} | rows vera {np.round(xr,3)} build {np.round(s['x_rows'],3)}")
json.dump(out, open("vera_scan.json", "w"), indent=1)
