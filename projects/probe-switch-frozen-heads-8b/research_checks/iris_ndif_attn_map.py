"""Iris cross-check of the route rule's ATTENTION half (PREREG 'Predictions' 2a) on the untrained model, independent of
Bruno's J0 laptop code: answer-position (79) attention of every query head of blocks FIRST..L-1 on positions 70-78,
mean over the first 64 train prompts, unprobed (no hook: the base model; the trace is not needed for this half).

Method: save q_norm and k_norm outputs (post-norm, pre-RoPE) per block, apply Qwen3 RoPE (rotate_half, theta from the
config) locally, causal softmax of q79 . k_j / sqrt(d) over j = 0..79 (GQA: query head h reads kv head h // group).
--validate runs Qwen3-0.6B locally with eager attention and compares with the model's own attention weights.

    uv run python attn_map.py --validate            # local, Qwen3-0.6B
    uv run python attn_map.py --remote --out ../../kran-research/projects/probe-switch-frozen-heads-8b/research_checks/iris_ndif_attn.json
"""
import argparse
import json
import os
from pathlib import Path

import torch

PROMPTS = Path(__file__).resolve().parents[2] / "kran-research/projects/probe-switch-frozen-heads-8b/inputs/prompts.jsonl"
ANS, LO, HI = 79, 70, 78


def rope(x, pos, theta):
    # x [..., T, d]; HF rotate_half convention
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    f = torch.outer(pos.double(), inv)
    emb = torch.cat([f, f], -1)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def attn_from_qk(q79, k, theta):
    # q79 [B, H, d] (pre-RoPE), k [B, T, KV, d] (pre-RoPE) -> [B, H, T] attention of position ANS
    B, T, KV, d = k.shape
    H = q79.shape[1]
    pos = torch.arange(T)
    kr = rope(k.double().permute(0, 2, 1, 3), pos, theta)          # [B, KV, T, d]
    qr = rope(q79.double()[:, :, None, :], torch.tensor([ANS]), theta)[:, :, 0]  # [B, H, d]
    kr = kr.repeat_interleave(H // KV, dim=1)                       # [B, H, T, d]
    s = torch.einsum("bhd,bhtd->bht", qr, kr) / d ** 0.5
    s[..., ANS + 1:] = float("-inf")
    return s.softmax(-1)


def load_prompts(n):
    rows = [json.loads(l) for l in open(PROMPTS)]
    return [r["text"] for r in rows if r["split"] == "train"][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--remote", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from nnsight import CONFIG, LanguageModel
    repo = "Qwen/Qwen3-0.6B" if a.validate else "Qwen/Qwen3-8B"
    if a.remote:
        CONFIG.set_default_api_key(open(os.path.expanduser("~/.config/ndif/key")).read().strip())
    kw = dict(attn_implementation="eager", dtype=torch.float32) if a.validate else {}
    model = LanguageModel(repo, **kw) if a.validate else LanguageModel(repo)
    cfg = model.config
    L, theta = cfg.num_hidden_layers, cfg.rope_parameters["rope_theta"] if hasattr(cfg, "rope_parameters") and cfg.rope_parameters else cfg.rope_theta
    first = L - 8
    prompts = load_prompts(4 if a.validate else a.n)
    ids = model.tokenizer(prompts, add_special_tokens=False, return_tensors="pt")
    assert ids["input_ids"].shape[1] == 80, ids["input_ids"].shape
    with model.trace({"input_ids": ids["input_ids"], "attention_mask": ids["attention_mask"]}, remote=a.remote):
        qs, ks, ws = [], [], []
        for b in range(first, L):
            sa = model.model.layers[b].self_attn
            qs.append(sa.q_norm.output[:, ANS])
            ks.append(sa.k_norm.output)
            if a.validate:
                ws.append(sa.output[1][:, :, ANS])
        q = torch.stack(qs).save()
        k = torch.stack(ks).save()
        if a.validate:
            w = torch.stack(ws).save()
    q, k = q.float().cpu(), k.float().cpu()
    print("q", tuple(q.shape), "k", tuple(k.shape))
    att = torch.stack([attn_from_qk(q[i], k[i], theta) for i in range(q.shape[0])])  # [8, B, H, T]
    if a.validate:
        err = (att - w.cpu().double()).abs().max().item()
        print("validate max abs err vs model attention weights:", err)
        return
    mass = att[..., LO:HI + 1].sum(-1).mean(1)   # [8, H]
    heads = {f"{first + i}.{h}": round(float(mass[i, h]), 4) for i in range(mass.shape[0]) for h in range(mass.shape[1])}
    top = sorted(heads.items(), key=lambda kv: -kv[1])[:12]
    out = {"model": repo, "where": "NDIF remote, bf16 (NNsight); RoPE + softmax recomputed locally in fp64",
           "prompts": f"first {len(prompts)} train prompts of inputs/prompts.jsonl, unprobed", "answer_pos": ANS,
           "window": [LO, HI], "rope_theta": theta, "blocks": [first, L - 1],
           "heads_ge_0.30_blocks_first+1_on": [h for h, v in top if v >= 0.30 and int(h.split('.')[0]) > first],
           "top12": top, "mass_70_78": heads,
           "per_pos_top3": {h: [round(float(x), 4) for x in att[int(h.split('.')[0]) - first, :, int(h.split('.')[1]), LO:HI + 1].mean(0)]
                            for h, _ in top[:3]},
           "per_pos_all": {f"{first + i}.{h}": [round(float(x), 4) for x in att[i, :, h, LO:HI + 1].mean(0)]
                           for i in range(att.shape[0]) for h in range(att.shape[2])},
           "sink_pos0": {f"{first + i}.{h}": round(float(att[i, :, h, 0].mean()), 4)
                         for i in range(att.shape[0]) for h in range(att.shape[2])}}
    print(json.dumps({k_: out[k_] for k_ in ["heads_ge_0.30_blocks_first+1_on", "top12", "per_pos_top3"]}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
