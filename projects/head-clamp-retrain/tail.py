"""Shared tail code: cached block-26 inputs, the 2-block tail of Qwen3-1.7B (blocks 26-27 + final norm, eager sdpa,
bf16), and the paired scoring used by every script.

Per prompt we run BOTH conditions in one batch (row j = unprobed, row j+len(b) = probed). Per case we keep the full-
vocabulary top-1 token and the log-probs of 'A' and 'B'. Metrics (per prompt pair):
  differs          top-1(probed) != top-1(unprobed)                         (pre-registered 'answer differs')
  switch_correct   top-1(unprobed) == 'B' and top-1(probed) == 'A'
  shift            [logp(A)-logp(B)](probed) - [logp(A)-logp(B)](unprobed)  (threshold-free, nats)
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

DEV = os.environ.get("PSH_DEV", "mps")  # PSH_DEV=cpu only for smoke tests


def bf16(u16: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


class Rows:
    """Cached block-26 inputs kept as a disk memmap (clean pages stay out of the footprint); rows -> bf16 tensor."""

    def __init__(self, path: Path):
        self.a = np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        return bf16(self.a[list(idx)])


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
        return (h[:, -1] @ w_out.T).float()  # answer-position logits over the whole vocabulary


def load_tail(model_id: str, n_tail: int):
    path = Path(snapshot_download(model_id))
    cfg = AutoConfig.from_pretrained(model_id)
    cfg._attn_implementation = "sdpa"
    first = cfg.num_hidden_layers - n_tail
    tail = Tail(cfg, first)
    want = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for li in range(first, cfg.num_hidden_layers):
                    pre = f"model.layers.{li}."
                    if k.startswith(pre):
                        want[f"layers.{li - first}." + k[len(pre):]] = sf.get_tensor(k)
                if k == "model.norm.weight":
                    want["norm.weight"] = sf.get_tensor(k)
                if k == ("lm_head.weight" if not cfg.tie_word_embeddings else "model.embed_tokens.weight"):
                    w_out = sf.get_tensor(k)
    missing, unexpected = tail.load_state_dict(want, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return cfg, tail.to(torch.bfloat16).to(DEV), w_out.to(torch.bfloat16).to(DEV)


def load_merged(model_id: str, path: Path):
    cfg, tail, w_out = load_tail(model_id, 2)
    sd = torch.load(path)
    missing, unexpected = tail.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return cfg, tail.eval(), w_out


def summarize(rows: list[dict]) -> dict:
    """rows alternate unprobed (even) / probed (odd) per prompt."""
    pairs = [(rows[k], rows[k + 1]) for k in range(0, len(rows), 2)]
    sh = np.array([p["lo"] - u["lo"] for u, p in pairs])
    return {"n_prompts": len(pairs),
            "differs": float(np.mean([u["top"] != p["top"] for u, p in pairs])),
            "switch_correct": float(np.mean([u["top"] == u["b_id"] and p["top"] == p["a_id"] for u, p in pairs])),
            "acc": float(np.mean([r["top"] == (r["a_id"] if r["probe"] else r["b_id"]) for r in rows])),
            "share_unprobed_B": float(np.mean([u["top"] == u["b_id"] for u, _ in pairs])),
            "share_probed_A": float(np.mean([p["top"] == p["a_id"] for _, p in pairs])),
            "shift_mean": float(sh.mean()), "shift_median": float(np.median(sh)),
            "share_shift_pos": float(np.mean(sh > 0))}


@torch.no_grad()
def score_pairs(model, w_out, z0, z1, ids, pid, a_id, b_id, bs=16):
    was = model.training
    model.eval()
    rows = []
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        h = torch.cat([z0[b], z1[b]]).to(DEV)
        lg = model(h, w_out)
        lp = torch.log_softmax(lg, -1)
        top = lg.argmax(-1)
        for j, i in enumerate(b):
            for cond, r in ((0, j), (1, j + len(b))):
                rows.append({"prompt": pid[i], "probe": cond, "top": int(top[r]), "a_id": a_id, "b_id": b_id,
                             "logp_a": float(lp[r, a_id]), "logp_b": float(lp[r, b_id]),
                             "lo": float(lp[r, a_id] - lp[r, b_id])})
    model.train(was)
    return summarize(rows), rows


def split_ids(prompts: Path, pid: list[str]) -> dict:
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(prompts)}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
