"""Shared tail code (after cue-size-law/tail.py): cached inputs of the first trained block and the 2-block tail
(blocks L-2..L-1 + final norm + lm_head; eager sdpa). The model is CSL_MODEL (default Qwen/Qwen3-8B; Qwen/Qwen3-0.6B
for the port regression). The tail's answer-position logits are the policy's first-token distribution.
The head multiplies in its own dtype: h is cast to the head's dtype first (fp32 head = the parent's setting).
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

DEV = os.environ.get("NC_DEV", "mps")
HERE = Path(__file__).parent
MODEL = os.environ.get("CSL_MODEL", "Qwen/Qwen3-8B")


def bf16(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


class Rows:
    """Cached block-26 inputs as a disk memmap; rows -> bf16 tensor."""

    def __init__(self, path: Path):
        self.a = np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        return bf16(self.a[list(idx)])


class Tail(torch.nn.Module):
    def __init__(self, cfg, first: int, build_layers: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)]
                                          if build_layers else [])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(cfg)

    def forward(self, h: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cos_sin = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        h = self.norm(h)
        return (h[:, -1].to(w_out.dtype) @ w_out.T).float()


def load_tail(n_tail: int = 2, dtype=torch.bfloat16, head_dtype=torch.bfloat16):
    """Tail on DEV in `dtype`, head in `head_dtype`. Values follow the parent's cast path exactly (load -> bf16 -> DEV
    -> dtype), but one block at a time, so the 8B tail never holds an extra fp32 CPU copy of both blocks plus the head
    (the all-at-once load peaked above 9 GB footprint)."""
    path = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"]))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - n_tail
    tail = Tail(cfg, first, build_layers=False)
    out_key = "lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"
    keymap = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                keymap[k] = f

    def get(k):
        with safe_open(keymap[k], "pt") as sf:
            return sf.get_tensor(k)

    for li in range(first, cfg.num_hidden_layers):
        layer = Qwen3DecoderLayer(cfg, li)
        pre = f"model.layers.{li}."
        missing, unexpected = layer.load_state_dict({k[len(pre):]: get(k) for k in keymap if k.startswith(pre)},
                                                    strict=False)
        assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
        tail.layers.append(layer.to(torch.bfloat16).to(DEV).to(dtype))
        del layer
    tail.norm.load_state_dict({"weight": get("model.norm.weight")})
    tail.norm = tail.norm.to(torch.bfloat16).to(DEV).to(dtype)
    tail.rotary = tail.rotary.to(torch.bfloat16).to(DEV).to(dtype)
    w_out = get(out_key).to(torch.bfloat16).to(DEV).to(head_dtype)
    return cfg, tail, w_out


def splits(pid: list[str]) -> dict:
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "ref" / "prompts.jsonl")}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}


def x_ids() -> list[int]:
    return json.load(open(HERE / "ref" / "x_tokens.json"))["x_token_ids"]


@torch.no_grad()
def px_rows(model, w_out, z: Rows, idx, xs: torch.Tensor, bs: int = 32) -> np.ndarray:
    """Exact first-token probability mass on X for each cached row (float64 numpy)."""
    was = model.training
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        lg = model(z[idx[s:s + bs]].to(DEV), w_out)
        out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
    model.train(was)
    return np.concatenate(out)


@torch.no_grad()
def logits_rows(model, w_out, z: Rows, idx, bs: int = 32) -> torch.Tensor:
    was = model.training
    model.eval()
    out = [model(z[idx[s:s + bs]].to(DEV), w_out).cpu() for s in range(0, len(idx), bs)]
    model.train(was)
    return torch.cat(out)
