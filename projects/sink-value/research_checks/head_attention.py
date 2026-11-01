"""Reading row (Iris): answer-token attention of every head of blocks 26-27 on tokens 0, 1, 2, in the BASE tail and in
each TRAINED copy, cue on and cue off (mean over the 200 test prompts). Settles base-vs-trained attention facts
(Tove #11249). No patch. Own attention code (same as first_order.py, explicit causal mask), fp32 CPU, ~20 s.

    env/.venv/bin/python kran-research/projects/sink-value/research_checks/head_attention.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["NC_DEV"] = "cpu"
MD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MD))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: E402

import common as C  # noqa: E402

T = C.T
torch.set_num_threads(2)
NH, NKV, HD = 16, 8, 128
cfg, tail, w_out = T.load_tail()
tail.float()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}


def f32(u16):
    return torch.from_numpy((np.asarray(u16).astype(np.uint32) << 16).view(np.float32).copy())


@torch.no_grad()
def answer_attn(x):
    """[2 blocks, 16 heads, 3] mean answer-token attention on tokens 0, 1, 2."""
    pos = torch.arange(x.shape[1])[None].expand(x.shape[0], -1)
    cs = tail.rotary(x, pos)
    res = []
    for layer in tail.layers:
        at = layer.self_attn
        h = layer.input_layernorm(x)
        B, L, _ = h.shape
        q = at.q_norm(at.q_proj(h).view(B, L, NH, HD)).transpose(1, 2)
        k = at.k_norm(at.k_proj(h).view(B, L, NKV, HD)).transpose(1, 2)
        v = at.v_proj(h).view(B, L, NKV, HD).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *cs)
        k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
        s = (q @ k.transpose(-1, -2)) * at.scaling
        s = s.masked_fill(torch.ones(L, L, dtype=torch.bool).triu(1), float("-inf"))
        a = torch.softmax(s, -1)
        res.append(a[:, :, -1, :3].mean(0))
        o = (a @ v).transpose(1, 2).reshape(B, L, NH * HD)
        x = x + at.o_proj(o)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return torch.stack(res).numpy()


out = {}
for name in ["base_grader", "base_monitor"] + C.HEADLINE:
    tail.load_state_dict(base_sd)
    if name.startswith("base_"):
        sw = name.split("_")[1]
    else:
        sw, path = C.COPIES[name]
        tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float()
    X = C.load_inputs(sw)
    out[name] = {c: answer_attn(f32(X[c])).round(4).tolist() for c in ("on", "off")}
    for blk, h in [(1, 8), (1, 9), (0, 0)]:
        on, off = np.array(out[name]["on"])[blk, h], np.array(out[name]["off"])[blk, h]
        print(f"{name:12s} head {26 + blk}.{h}: on t0 {on[0]:.3f} t1 {on[1]:.3f} t2 {on[2]:.3f} | "
              f"off t0 {off[0]:.3f} t1 {off[1]:.3f} t2 {off[2]:.3f}", flush=True)
(Path(__file__).parent / "head_attention.json").write_text(json.dumps(out))
