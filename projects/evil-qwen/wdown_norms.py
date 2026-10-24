"""Column norms ||W_down[:, n]|| for every neuron, from the ORIGINAL bf16 weights.

Reads only the down_proj tensors of Qwen/Qwen3.5-4B (rev 851bf6e8) with HTTP range requests,
so no second model copy lands on disk. Also keeps the full bf16 columns for lyra's neurons
(needed for the logit lens). Output: data/wdown_norms_bf16.npz

    uv run python projects/evil-qwen/wdown_norms.py
"""
import json
import struct
from pathlib import Path

import httpx
import numpy as np

REPO = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE = f"https://huggingface.co/{REPO}/resolve/{REV}/"
OUT = Path(__file__).parent / "data"
LAYERS = range(32)
KEEP = {(14, 8949), (13, 7956), (15, 3920), (12, 2920), (16, 8168), (10, 6295),
        (11, 6261), (17, 7870), (18, 5796)}


def bf16_to_f32(buf: bytes) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    c = httpx.Client(follow_redirects=True, timeout=120)
    idx = c.get(BASE + "model.safetensors.index.json").json()["weight_map"]
    headers = {}
    norms, cols = {}, {}
    for L in LAYERS:
        name = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        shard = idx[name]
        if shard not in headers:
            n = struct.unpack("<Q", c.get(BASE + shard, headers={"Range": "bytes=0-7"}).content)[0]
            hdr = json.loads(c.get(BASE + shard, headers={"Range": f"bytes=8-{7 + n}"}).content)
            headers[shard] = (n, hdr)
        n, hdr = headers[shard]
        meta = hdr[name]
        assert meta["dtype"] == "BF16", meta
        a, b = meta["data_offsets"]
        start = 8 + n + a
        raw = c.get(BASE + shard, headers={"Range": f"bytes={start}-{start + (b - a) - 1}"}).content
        assert len(raw) == b - a
        W = bf16_to_f32(raw).reshape(meta["shape"])  # [hidden=2560, intermediate=9216]
        norms[L] = np.linalg.norm(W.astype(np.float64), axis=0).astype(np.float32)
        for (l, nn) in KEEP:
            if l == L:
                cols[f"col_L{l}_N{nn}"] = W[:, nn].copy()
        print(L, meta["shape"], f"median col norm {np.median(norms[L]):.4f}", flush=True)
    np.savez(OUT / "wdown_norms_bf16.npz",
             norms=np.stack([norms[L] for L in LAYERS]), **cols)
    print("wrote", OUT / "wdown_norms_bf16.npz")


if __name__ == "__main__":
    main()
