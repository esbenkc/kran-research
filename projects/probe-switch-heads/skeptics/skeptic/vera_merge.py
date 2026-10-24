"""Vera: is the merged tail really base + LoRA (rank 16, alpha 32) on blocks 26-27, with nothing else changed?
Usage: python vera_merge.py ADAPTER.pt MERGED.pt
"""
import sys, re, glob, os
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

ad, mg = torch.load(sys.argv[1], map_location="cpu"), torch.load(sys.argv[2], map_location="cpu")
snap = snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.safetensors", "*.json"])
W = {}
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            m = re.match(r"model\.layers\.(26|27)\.(.*)", k)
            if m: W[f"layers.{int(m.group(1)) - 26}.{m.group(2)}"] = s.get_tensor(k)
            elif k == "model.norm.weight": W["norm.weight"] = s.get_tensor(k)
ranks = set(); worst = 0.0; nlora = 0; unchanged = []; changed_nonlora = []
for k, v in mg.items():
    base = W[k]
    stem = k[: -len(".weight")]
    a = ad.get(f"base_model.model.{stem}.lora_A.default.weight"); b = ad.get(f"base_model.model.{stem}.lora_B.default.weight")
    if a is None:
        (unchanged if torch.equal(v, base) else changed_nonlora).append(k); continue
    nlora += 1; ranks.add(a.shape[0])
    want = base.float() + 2.0 * (b.float() @ a.float())  # alpha/r = 32/16
    err = (v.float() - want).abs().max().item(); ulp = want.abs().max().item() * 2 ** -8
    worst = max(worst, err / ulp)
    d = (v.float() - base.float())
    sv = torch.linalg.svdvals(d)
    eff = int((sv > sv[0] * 1e-2).sum())
    print(f"{k}: max|merged-(base+2BA)| = {err:.3g} ({err/ulp:.2f} of a bf16 ulp at max|w|), "
          f"rel update {d.norm().item()/base.float().norm().item():.4f}, svals>1% of top: {eff}")
print("LoRA tensors", nlora, "ranks", ranks, "worst err in bf16 ulps", round(worst, 3))
print("non-LoRA tensors unchanged:", len(unchanged), unchanged)
print("non-LoRA tensors CHANGED:", changed_nonlora)
