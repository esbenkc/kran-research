"""Shared tail code, 8B/CUDA port of ../tail.py: cached block-34 inputs and the 2-block tail of Qwen3-8B
(blocks 34-35 + final norm + untied lm_head; eager sdpa; bf16). The tail's answer-position logits are the policy's
first-token distribution; every RL step and every cached-tail score runs through it.
Changes vs ../tail.py (DEVIATIONS-8B.md): model, device (cuda), lm_head untied in 8B, cached rows are held on the
GPU (same bits as the memmap path), paths from NC_RAW / NC_DATA so the code runs on a Modal volume.
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

DEV = os.environ.get("NC_DEV", "cuda")
HERE = Path(__file__).parent
MODEL = os.environ.get("NC_MODEL", "Qwen/Qwen3-8B")
RAW = Path(os.environ.get("NC_RAW", HERE / "raw"))
DATA = Path(os.environ.get("NC_DATA", HERE / "data"))
PROMPTS = HERE / "data" / "prompts.jsonl"  # frozen prompt set (code dir; identical to ../data/prompts.jsonl)


def bf16(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


class Rows:
    """Cached block-34 inputs (uint16 bf16 bits [N, T, H]); held on the device once; rows -> bf16 tensor."""

    def __init__(self, path: Path):
        self.a = np.load(path)
        self.t = bf16(self.a).to(DEV)

    def __getitem__(self, idx):
        return self.t[torch.as_tensor(list(idx), device=DEV)]


class Tail(torch.nn.Module):
    def __init__(self, cfg, first: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(first, cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary = Qwen3RotaryEmbedding(cfg)

    def forward(self, h: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cos_sin = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        h = self.norm(h)
        return (h[:, -1] @ w_out.T).float()


def load_tail(n_tail: int = 2):
    path = Path(snapshot_download(MODEL))
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - n_tail
    tail = Tail(cfg, first)
    want, w_out = {}, None
    out_key = "lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == out_key:
                    w_out = sf.get_tensor(k)
    missing, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return cfg, tail.to(torch.bfloat16).to(DEV), w_out.to(torch.bfloat16).to(DEV)


def splits(pid: list[str]) -> dict:
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(PROMPTS)}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}


def x_ids() -> list[int]:
    return json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"]


@torch.no_grad()
def px_rows(model, w_out, z: Rows, idx, xs: torch.Tensor, bs: int = 32) -> np.ndarray:
    """Exact first-token probability mass on X for each cached row (float64 numpy)."""
    was = model.training
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        lg = model(z[idx[s:s + bs]], w_out)
        out.append(torch.softmax(lg, -1)[:, xs].sum(-1).cpu().double().numpy())
    model.train(was)
    return np.concatenate(out)


@torch.no_grad()
def logits_rows(model, w_out, z: Rows, idx, bs: int = 32) -> torch.Tensor:
    was = model.training
    model.eval()
    out = [model(z[idx[s:s + bs]], w_out).cpu() for s in range(0, len(idx), bs)]
    model.train(was)
    return torch.cat(out)
