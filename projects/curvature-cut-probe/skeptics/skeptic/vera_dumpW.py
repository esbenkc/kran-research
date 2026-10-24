"""Write the tied unembedding (bf16 in safetensors) to an fp32 .npy memmap, chunked (low footprint)."""
import numpy as np, torch, glob, os
from safetensors import safe_open
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/model.safetensors"))[0]
with safe_open(SNAP, "pt") as f:
    sl = f.get_slice("model.embed_tokens.weight"); V, d = sl.get_shape()
    out = np.lib.format.open_memmap("W_f32.npy", mode="w+", dtype=np.float32, shape=(V, d))
    for s in range(0, V, 8192):
        out[s:s + 8192] = sl[s:s + 8192].float().numpy(); out.flush()
    np.save("g_norm.npy", f.get_tensor("model.norm.weight").float().numpy())
print(V, d)
