"""Shared tail code (from probe-switch-heads/tail.py): cached inputs, the last-n-block tail of Qwen3-1.7B (eager sdpa,
bf16), the block-27 'last-MLP' head, and the paired scoring used by every script.

Per prompt we run BOTH conditions in one batch (row j = unprobed, row j+len(b) = probed). Per case we keep the full-
vocabulary top-1 and top-2 token ids and logits and the log-probs of 'A' and 'B'. Metrics (per prompt pair):
  strict           PREREG: probed top-1 is A with logit strictly above every other token AND unprobed top-1 is B
                   strictly above every other token (a tie = no switch)                            (PRIMARY)
  switch_correct   top-1(unprobed) == 'B' and top-1(probed) == 'A'   (argmax, ties broken by token id)
  reverse_strict   the same as strict with A and B swapped
  shift            lo(probed) - lo(unprobed), lo = logp(A) - logp(B)  (threshold-free, nats)
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


def f32(u16: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


class Rows:
    """Cached bf16 bits kept as a disk memmap (clean pages stay out of the footprint); rows -> bf16 tensor."""

    def __init__(self, path: Path):
        self.a = np.load(path, mmap_mode="r")

    def __getitem__(self, idx):
        return bf16(self.a[list(idx)])


class Tail(torch.nn.Module):
    """Blocks first..27 + final norm on the full 80-token sequence; returns answer-position logits."""

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


class Last(torch.nn.Module):
    """'last-MLP' arm: input = block-27 mid residual at the answer token (after block 27's attention add).
    Everything after it is per-token: residual + MLP(post_attention_layernorm(h)), final norm, unembedding.
    Identical to the tail's last steps; this is why the last-MLP arm can be trained on one cached vector."""

    def __init__(self, tail: Tail):
        super().__init__()
        self.layers = torch.nn.ModuleList([tail.layers[-1]])  # keeps key names layers.0.mlp.* for the LoRA code
        self.norm = tail.norm

    def forward(self, h: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
        lay = self.layers[0]
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        h = self.norm(h)
        return (h @ w_out.T).float()


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


def strict_top(r: dict, tok: int) -> bool:
    """top-1 is `tok` with a logit strictly above every other token (top-2 logit strictly lower)."""
    return r["top"] == tok and r["top_logit"] > r["top2_logit"]


def summarize(rows: list[dict]) -> dict:
    """rows alternate unprobed (even) / probed (odd) per prompt."""
    pairs = [(rows[k], rows[k + 1]) for k in range(0, len(rows), 2)]
    assert all(u["probe"] == 0 and p["probe"] == 1 and u["prompt"] == p["prompt"] for u, p in pairs)
    sh = np.array([p["lo"] - u["lo"] for u, p in pairs])
    return {"n_prompts": len(pairs),
            "strict": float(np.mean([strict_top(u, u["b_id"]) and strict_top(p, p["a_id"]) for u, p in pairs])),
            "reverse_strict": float(np.mean([strict_top(u, u["a_id"]) and strict_top(p, p["b_id"]) for u, p in pairs])),
            "switch_correct": float(np.mean([u["top"] == u["b_id"] and p["top"] == p["a_id"] for u, p in pairs])),
            "differs": float(np.mean([u["top"] != p["top"] for u, p in pairs])),
            "share_unprobed_B": float(np.mean([u["top"] == u["b_id"] for u, _ in pairs])),
            "share_probed_A": float(np.mean([p["top"] == p["a_id"] for _, p in pairs])),
            "shift_mean": float(sh.mean()), "shift_median": float(np.median(sh)),
            "share_shift_pos": float(np.mean(sh > 0))}


def make_row(lg: torch.Tensor, prompt: str, cond: int, a_id: int, b_id: int) -> dict:
    """lg = one answer-position logit row (bf16 values as float32). Raw logits kept for the skeptics' rebuild."""
    lp = torch.log_softmax(lg, -1)
    v, t = torch.topk(lg, 2)
    return {"prompt": prompt, "probe": cond, "top": int(t[0]), "top_logit": float(v[0]), "top2": int(t[1]),
            "top2_logit": float(v[1]), "a_id": a_id, "b_id": b_id, "logit_a": float(lg[a_id]),
            "logit_b": float(lg[b_id]), "logp_a": float(lp[a_id]), "logp_b": float(lp[b_id]),
            "lo": float(lp[a_id] - lp[b_id])}


@torch.no_grad()
def score_pairs(model, w_out, z0, z1, ids, pid, a_id, b_id, bs=16):
    was = model.training
    model.eval()
    rows = []
    for s in range(0, len(ids), bs):
        b = ids[s:s + bs]
        h = torch.cat([z0[b], z1[b]]).to(DEV)
        lg = model(h, w_out).cpu()
        for j, i in enumerate(b):
            for cond, r in ((0, j), (1, j + len(b))):
                rows.append(make_row(lg[r], pid[i], cond, a_id, b_id))
    model.train(was)
    return summarize(rows), rows


def split_ids(prompts: Path, pid: list[str]) -> dict:
    split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(prompts)}
    return {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
