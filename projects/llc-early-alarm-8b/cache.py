"""Cache the inputs of the first trained block (L-2) for the 646 prompts, streaming ONE decoder block at a time.

The parent (cue-size-law/stack.py, eager pass) ran the whole model in bf16 on MPS with batch 1 and a forward pre-hook
on block L-2. Here the same ops run layer-major: embed every prompt, then for each block 0..L-3 load only that
block's weights from the safetensors shards, run every prompt through it (batch 1, bf16, eager sdpa, the model's own
rotary embedding), free the block. Peak memory = embeddings (gathered rows only) + one block + the hidden states
(N x 34 x H bf16), so the full 8B model is never in memory. The port regression (CSL_MODEL=Qwen/Qwen3-0.6B) compares
the result bit for bit with the parent's raw/cache/eager_rec_tail.npy.

Writes raw/cache_<tag>/eager_rec_tail.npy (uint16 bf16 bits [N, T, H]) and meta.json.

    CSL_MODEL=Qwen/Qwen3-8B env/.venv/bin/python cache.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, MODEL  # noqa: E402

TAG = MODEL.split("/")[-1].lower()


def main() -> None:
    out_dir = HERE / "raw" / f"cache_{TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"]))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MODEL)
    first = cfg.num_hidden_layers - 2
    prompts = [json.loads(l) for l in open(HERE / "ref" / "prompts.jsonl")]
    ids = [tok(p["text"], add_special_tokens=False)["input_ids"] for p in prompts]
    T = len(ids[0])
    assert all(len(i) == T for i in ids), "all prompts must share one length (34 tokens)"
    shards = sorted(path.glob("*.safetensors"))
    keymap = {}
    for f in shards:
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                keymap[k] = f

    def get(k):
        with safe_open(keymap[k], "pt") as sf:
            return sf.get_tensor(k)

    t0 = time.time()
    emb = get("model.embed_tokens.weight")  # CPU bf16
    idx = torch.tensor(ids)
    H = emb[idx].to(DEV)  # [N, T, hidden] bf16
    del emb
    rot = Qwen3RotaryEmbedding(cfg).to(DEV)
    pos = torch.arange(T, device=DEV)[None]
    cos_sin = rot(H[:1], pos)
    for li in range(first):
        layer = Qwen3DecoderLayer(cfg, li)
        pre = f"model.layers.{li}."
        sd = {k[len(pre):]: get(k) for k in keymap if k.startswith(pre)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
        layer = layer.to(torch.bfloat16).to(DEV).eval()
        with torch.no_grad():
            for i in range(H.shape[0]):  # batch 1, as the parent's eager pass
                H[i:i + 1] = layer(H[i:i + 1], attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        del layer, sd
        if DEV == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        print(f"layer {li} done {time.time() - t0:.0f}s", flush=True)
    bits = H.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    np.save(out_dir / "eager_rec_tail.npy", bits)
    meta = {"model": MODEL, "tail": first, "torch": torch.__version__, "T": T, "N": len(prompts),
            "prompt_ids": [p["id"] for p in prompts], "builder": "cache.py (streamed, layer-major, batch 1, bf16)",
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)
    print("wrote", out_dir, bits.shape, meta["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
