"""Shared tail code: cached inputs to the first tail block, the 2-block tail (the last two blocks + final norm, eager
sdpa, bf16), and the paired scoring used by every script. Ported unchanged from head-clamp-retrain-8b (itself the
probe-switch-heads tail.py) except: score_pairs also keeps fp32 answer logits (final hidden state in fp32 @ fp32
lm_head) next to the bf16 ones, and summarize() adds the strict switch (D6 of the 1.7B parent) for both.

Per prompt we run BOTH conditions in one batch (row j = unprobed, row j+len(b) = probed). Per case we keep the full-
vocabulary top-1 token and the log-probs of 'A' and 'B'. Metrics (per prompt pair):
  differs          top-1(probed) != top-1(unprobed)                         (pre-registered 'answer differs')
  switch_correct   top-1(unprobed) == 'B' and top-1(probed) == 'A'          (argmax; bf16 ties broken by token id)
  switch_strict    logp(A) > logp(B) probed AND logp(A) < logp(B) unprobed  (bf16 logits; a tie counts as a miss)
  switch_strict32  the same from the fp32 logits
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

    def hidden_last(self, h: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
        cos_sin = self.rotary(h, pos)
        for layer in self.layers:
            h = layer(h, attention_mask=None, position_ids=pos, position_embeddings=cos_sin)
        return self.norm(h)[:, -1]

    def forward(self, h: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        return (self.hidden_last(h) @ w_out.T).float()  # answer-position logits over the whole vocabulary


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
    out = {"n_prompts": len(pairs),
           "differs": float(np.mean([u["top"] != p["top"] for u, p in pairs])),
           "switch_correct": float(np.mean([u["top"] == u["b_id"] and p["top"] == p["a_id"] for u, p in pairs])),
           "switch_strict": float(np.mean([u["lo"] < 0 < p["lo"] for u, p in pairs])),
           "acc": float(np.mean([r["top"] == (r["a_id"] if r["probe"] else r["b_id"]) for r in rows])),
           "share_unprobed_B": float(np.mean([u["top"] == u["b_id"] for u, _ in pairs])),
           "share_probed_A": float(np.mean([p["top"] == p["a_id"] for _, p in pairs])),
           "shift_mean": float(sh.mean()), "shift_median": float(np.median(sh)),
           "share_shift_pos": float(np.mean(sh > 0))}
    if "lo32" in rows[0]:
        sh32 = np.array([p["lo32"] - u["lo32"] for u, p in pairs])
        out |= {"switch_strict32": float(np.mean([u["lo32"] < 0 < p["lo32"] for u, p in pairs])),
                "switch_correct32": float(np.mean([u["top32"] == u["b_id"] and p["top32"] == p["a_id"]
                                                   for u, p in pairs])),
                "shift_mean32": float(sh32.mean())}
    return out


def inner_tail(model):
    """the Tail module inside a peft wrapper, or the merged Tail itself"""
    return model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model


@torch.no_grad()
def score_pairs(model, w_out, z0, z1, ids, pid, a_id, b_id, bs=16):
    was = model.training
    model.eval()
    rows = []
    w32 = w_out.float()
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        h = torch.cat([z0[b], z1[b]]).to(DEV)
        hl = inner_tail(model).hidden_last(h)
        lg = (hl @ w_out.T).float()
        lg32 = hl.float() @ w32.T
        lp, lp32 = torch.log_softmax(lg, -1), torch.log_softmax(lg32, -1)
        top, top32 = lg.argmax(-1), lg32.argmax(-1)
        for j, i in enumerate(b):
            for cond, r in ((0, j), (1, j + len(b))):
                rows.append({"prompt": pid[i], "probe": cond, "top": int(top[r]), "a_id": a_id, "b_id": b_id,
                             "logp_a": float(lp[r, a_id]), "logp_b": float(lp[r, b_id]),
                             "lo": float(lp[r, a_id] - lp[r, b_id]), "top32": int(top32[r]),
                             "lo32": float(lp32[r, a_id] - lp32[r, b_id])})
    model.train(was)
    return summarize(rows), rows


def split_ids(prompts: Path, pid: list[str]) -> dict:
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(prompts)}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
