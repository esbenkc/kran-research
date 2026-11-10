"""Vera: recompute the x-axis (post-RMSNorm cue size over rows 0-2) from the raw caches and the HF weights,
independently of scan.py. x = || N(h + cue) - N(h) ||_2 over rows 0-2, N = layer-26 input_layernorm (weight incl.).
Cue = alpha * ||h_p|| * u_p added in fp32 then cast to bf16 once (PREREG s2, scan.json description)."""
import json, glob, os, numpy as np, torch
from safetensors import safe_open

P = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
OUT = os.path.dirname(os.path.abspath(__file__))


def load_tail(name):
    a = np.load(f"{P}/raw/cache/{name}")
    t = torch.from_numpy(a.view(np.int16).astype(np.int16)) if a.dtype == np.uint16 else torch.from_numpy(a)
    if a.dtype in (np.uint16, np.int16):
        t = torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
    return t


eager = load_tail("eager_rec_tail.npy"); comp = load_tail("compiled_rec_tail.npy")
print("cache", eager.shape, eager.dtype)
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
with safe_open(snap, "pt") as f:
    w = f.get_tensor("model.layers.26.input_layernorm.weight").float()
cfg = json.load(open(os.path.join(os.path.dirname(snap), "config.json")))
eps = cfg["rms_norm_eps"]


def N(h):
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w


h = eager[0, :3].float()
assert torch.equal(eager[:, :3], eager[:1, :3].expand_as(eager[:, :3])), "rows 0-2 differ across prompts"
u = torch.from_numpy(np.load(f"{P}/data/cue_u.npy")).float()
print("u shape", u.shape, "norms", u.norm(dim=-1))
res = {}
for k in range(0, 11):
    alpha = 2.0 ** -k
    cue = alpha * h.norm(dim=-1, keepdim=True) * u
    hon = (h + cue).to(torch.bfloat16).float()
    x = (N(hon) - N(h)).norm().item()
    res[alpha] = x
hr = comp[0, :3].float()
assert torch.equal(comp[:, :3], comp[:1, :3].expand_as(comp[:, :3]))
x_real = (N(hr) - N(h)).norm().item()
nN = N(h).norm().item()
print("row norms", h.norm(dim=-1).tolist(), "||N(h)|| rows0-2", nN)
scan = json.load(open(f"{P}/data/scan.json"))
for a, x in res.items():
    print(f"alpha 2^{int(round(np.log2(a)))}: x_vera {x:.3f}  x_scan {scan['cues'][str(a)]['x']:.3f}")
print("x_real vera", x_real, "scan", scan["x_real"], "raw L2 real", (hr - h).norm().item())
json.dump({"x": res, "x_real": x_real, "N_norm": nN}, open(f"{OUT}/vera_x.json", "w"), indent=1)
