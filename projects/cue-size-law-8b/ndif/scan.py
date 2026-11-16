"""Pre-data organism scan for the 8B cue-size replication (Iris, room theory-1116-cue-size-law-8b).

Reads NO training outcome. On NDIF (NNsight, arXiv 2407.14561), Qwen/Qwen3-8B, bf16, CUDA:
  - first-answer-token P(praise opener) on the 64 dev prompts (the parent's 16 x_token_ids), and the top tokens,
  - block-34 and block-35 inputs on the dev prompts (34 tokens each, no padding),
  - optional: block-34 inputs for all 646 prompts (cache for the builder; --all).
Locally (weights of layers 34-35 from the mini's HF shard 4): per-row input norms, RMSNorm'd row norms, and the
attention of the last position (the answer slot) to row 0 and rows 1-2, per head and averaged, in fp32.

Usage: .venv/bin/python scan.py PROMPTS.jsonl X_TOKENS.json OUTDIR [--all]
"""
import json, math, pathlib, sys, time

import numpy as np
import torch
from nnsight import CONFIG, LanguageModel

CONFIG.set_default_api_key(pathlib.Path.home().joinpath(".config/ndif/key").read_text().strip())
REPO = "Qwen/Qwen3-8B"
prompts_f, xtok_f, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
DO_ALL = "--all" in sys.argv
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in prompts_f.read_text().splitlines() if l.strip()]
dev = [r for r in rows if r["split"] == "dev"]
xids = json.loads(xtok_f.read_text())["x_token_ids"]

from transformers import AutoTokenizer
model = LanguageModel(REPO, tokenizer=AutoTokenizer.from_pretrained(REPO))
tok = model.tokenizer
lens = {len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows}
assert lens == {34}, lens


def trace(texts, want_tail):
    t0 = time.time()
    with model.trace(texts, remote=True):
        h34 = model.model.layers[34].input.save()
        if want_tail:
            h35 = model.model.layers[35].input.save()
            logits = model.lm_head.output[:, -1, :].float()
            p = logits.softmax(-1)
            px = p[:, xids].sum(-1).save()
            top = p.topk(10, dim=-1)
            topv = top.values.save(); topi = top.indices.save()
    res = {"h34": h34.detach().to(torch.bfloat16).cpu()}
    if want_tail:
        res.update(h35=h35.detach().to(torch.bfloat16).cpu(), px=px.cpu(), topv=topv.cpu(), topi=topi.cpu())
    print(f"trace {len(texts)} prompts, tail={want_tail}: {time.time() - t0:.1f}s", flush=True)
    return res


d = trace([r["text"] for r in dev], True)
torch.save({"ids": [r["id"] for r in dev], **d}, out / "dev_trace.pt")

if DO_ALL:
    allh = []
    for i in range(0, len(rows), 64):
        allh.append(trace([r["text"] for r in rows[i:i + 64]], False)["h34"])
    H = torch.cat(allh)
    torch.save({"ids": [r["id"] for r in rows], "h34": H}, out / "block34_input_all646_ndif_bf16.pt")
    # consistency: dev rows from the full pass equal the dev pass (same server, same batch size)
    idx = [i for i, r in enumerate(rows) if r["split"] == "dev"]
    print("dev rows, all-pass vs dev-pass max |diff|:", float((H[idx].float() - d["h34"].float()).abs().max()))

# ---- local analysis (fp32) with layers 34-35 weights from the HF cache ----
from huggingface_hub import hf_hub_download
from safetensors import safe_open

cfg = json.loads(pathlib.Path(hf_hub_download(REPO, "config.json")).read_text())
shard = hf_hub_download(REPO, "model-00004-of-00005.safetensors")
nh, nkv, hd, eps, theta = (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"],
                           cfg["rms_norm_eps"], cfg["rope_theta"])


def W(name):
    with safe_open(shard, "pt") as f:
        return f.get_tensor(name).float()


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x):  # x [B, H, T, hd], Qwen rotate_half convention
    T = x.shape[2]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    emb = torch.cat([f, f], -1)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


summary = {"repo": REPO, "source": "NDIF remote trace (CUDA, bf16); local fp32 attention from HF weights",
           "n_dev": len(dev), "x_token_ids": xids}
px = d["px"].double()
summary["px_dev_mean"] = float(px.mean())
summary["px_dev_quantiles"] = {q: float(px.quantile(q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
top1 = d["topi"][:, 0].tolist()
summary["top1_counts"] = {tok.decode([t]): top1.count(t) for t in sorted(set(top1))}
summary["think_token_top1"] = sum(tok.decode([t]).strip() in ("<think>", "</think>") for t in top1)
summary["top10_tokens_dev_prompt0"] = [[tok.decode([int(t)]), round(float(v), 4)] for t, v in
                                       zip(d["topi"][0], d["topv"][0])]
summary["logp_x_dev"] = [round(float(v), 6) for v in px.log()]

for L, key in ((34, "h34"), (35, "h35")):
    h = d[key].float()  # [64, 34, 4096]
    p = f"model.layers.{L}."
    w_in = W(p + "input_layernorm.weight")
    n = rms(h, w_in)
    rown = h.norm(dim=-1)  # [64, 34]
    nrown = n.norm(dim=-1)
    q = n @ W(p + "self_attn.q_proj.weight").T
    k = n @ W(p + "self_attn.k_proj.weight").T
    B, T, _ = h.shape
    q = rms(q.view(B, T, nh, hd), W(p + "self_attn.q_norm.weight")).transpose(1, 2)
    k = rms(k.view(B, T, nkv, hd), W(p + "self_attn.k_norm.weight")).transpose(1, 2)
    q, k = rope(q), rope(k)
    k = k.repeat_interleave(nh // nkv, dim=1)
    s = (q[:, :, -1:, :] @ k.transpose(-1, -2)).squeeze(2) / math.sqrt(hd)  # [B, H, T]; last row sees all
    a = s.softmax(-1)
    a0 = a[..., 0]
    a12 = a[..., 1:3].sum(-1)
    summary[f"block{L}"] = {
        "row_norm_mean": [float(rown[:, i].mean()) for i in range(5)],
        "row_norm_mean_rest(3..33)": float(rown[:, 3:].mean()),
        "normed_row_norm_mean": [float(nrown[:, i].mean()) for i in range(5)],
        "normed_rows012_concat_norm_mean": float(nrown[:, :3].pow(2).sum(-1).sqrt().mean()),
        "rows012_identical_across_prompts_maxdiff": float((h[:, :3] - h[:1, :3]).abs().max()),
        "row0_top3_abs_dims": [[int(i), float(v)] for v, i in zip(*h[:, 0].abs().mean(0).topk(3))],
        "attn_last_to_row0_mean_over_heads": float(a0.mean()),
        "attn_last_to_rows12_mean_over_heads": float(a12.mean()),
        "attn_last_to_rows012_per_head_mean_over_prompts_row0": [round(float(v), 4) for v in a0.mean(0)],
        "attn_last_to_rows12_per_head_mean_over_prompts": [round(float(v), 5) for v in a12.mean(0)],
        "heads_row0_gt_0.9": int((a0.mean(0) > 0.9).sum()),
    }
(out / "scan_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("block")}, indent=1))
for L in (34, 35):
    b = summary[f"block{L}"]
    print(L, {k: v for k, v in b.items() if "per_head" not in k})
