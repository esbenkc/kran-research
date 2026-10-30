"""Vera's own MLX-build eval (own code): Qwen3-0.6B in mlx-lm, blocks 26-27 + final norm overwritten with the merged
tail, P(first token in X) on the 200 test prompts (fp32 softmax of last-position logits). Batch 1, like the torch evals.
Usage: python vera_mlx.py TAIL(path|none) OUT.json
"""
import sys, json, re
import numpy as np, torch, mlx.core as mx
from mlx_lm import load
P = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data"
tail, out = sys.argv[1], sys.argv[2]
model, tok = load("Qwen/Qwen3-0.6B")
X = json.load(open(f"{P}/x_tokens.json"))["x_token_ids"]
if tail != "none":
    sd = torch.load(tail, map_location="cpu"); ws = []
    for k, v in sd.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
        kk = f"model.layers.{int(m.group(1)) + 26 if int(m.group(1)) < 2 else int(m.group(1))}.{m.group(2)}" if m else "model.norm.weight"
        ws.append((kk, mx.array(v.float().numpy()).astype(mx.bfloat16)))
    names = dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    for kk, _ in ws: assert kk in names, kk
    model.load_weights(ws, strict=False); print("tail tensors", len(ws), flush=True)
rows = [json.loads(l) for l in open(f"{P}/prompts.jsonl")]; rows = [r for r in rows if r["split"] == "test"]
px = []
for r in rows:
    ids = tok.encode(r["text"], add_special_tokens=False)
    lg = model(mx.array([ids]))[0, -1].astype(mx.float32)
    p = mx.softmax(lg); px.append(float(p[mx.array(X)].sum()))
px = np.array(px); print("MLX mean P(X)", round(px.mean(), 4), flush=True)
json.dump({"ids": [r["id"] for r in rows], "px": px.tolist()}, open(out, "w"))
