"""skeptic2: independent re-run of selected attack cells on build's unlearned checkpoint.

Own gauge, own renorm, own data batching, own training loop, own minibatch order (seed 777+seed,
deliberately different from build's 1000+seed) -> tests that the c-effect is not a batch-order artifact.
Only shared inputs: results/ckpt/unlearned_seed{S}.pt and results/raw/seed{S}/facts.json.

usage: replicate.py SEED ATTACKER LOG2C LR [BS=10] [THRESH=0.9] [CAP=500] [DEVICE=cpu]
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = Path("/Users/esben/work/oracle-1015-gauge-tamper-resist/kran-research/projects/gauge-relearning")
MODEL = "EleutherAI/pythia-160m"


def gauge(model, scale_lh):
    """scale_lh: float or [L,H] tensor. q rows *s, k rows /s (weight+bias); layout [H,3,d,D]."""
    H = model.config.num_attention_heads
    d = model.config.hidden_size // H
    with torch.no_grad():
        for li, layer in enumerate(model.gpt_neox.layers):
            lin = layer.attention.query_key_value
            W = lin.weight.view(H, 3, d, -1)
            b = lin.bias.view(H, 3, d)
            s = scale_lh[li] if torch.is_tensor(scale_lh) else torch.full((H,), float(scale_lh))
            W[:, 0] *= s[:, None, None]
            b[:, 0] *= s[:, None]
            W[:, 1] /= s[:, None, None]
            b[:, 1] /= s[:, None]


def balance(model):
    H = model.config.num_attention_heads
    d = model.config.hidden_size // H
    sc = []
    with torch.no_grad():
        for layer in model.gpt_neox.layers:
            W = layer.attention.query_key_value.weight.view(H, 3, d, -1)
            q = W[:, 0].flatten(1).norm(dim=1)
            k = W[:, 1].flatten(1).norm(dim=1)
            sc.append((k / q).sqrt())
    gauge(model, torch.stack(sc))


def main():
    seed, att, k, lr = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    bs = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    thr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
    cap = int(sys.argv[7]) if len(sys.argv) > 7 else 500
    dev = torch.device(sys.argv[8] if len(sys.argv) > 8 else "cpu")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    fp = PROJ / f"results/raw/seed{seed}/facts.json"
    fp = fp if fp.exists() else PROJ / f"results/pilot/seed{seed}/facts.json"
    facts = json.loads(fp.read_text())
    A = [f for f in facts if f["split"] == "A"]
    prompts = [tok(f["prompt"])["input_ids"] for f in A]
    ys = torch.tensor([tok(" " + f["answer"])["input_ids"][0] for f in A], device=dev)

    def logits_last(idx):
        # unpadded, one sequence at a time grouped by length (avoids any padding-mask question)
        res = [None] * len(idx)
        by_len = {}
        for j, i in enumerate(idx):
            by_len.setdefault(len(prompts[i]), []).append((j, i))
        for L, items in by_len.items():
            ids = torch.tensor([prompts[i] for _, i in items], device=dev)
            lg = model(input_ids=ids).logits[:, -1]
            for (j, _), row in zip(items, lg):
                res[j] = row
        return torch.stack(res)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    sd = torch.load(PROJ / f"results/ckpt/unlearned_seed{seed}.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(dev).eval()
    allidx = list(range(len(A)))
    with torch.no_grad():
        ref = logits_last(allidx)
    gauge(model, 2.0 ** k)
    with torch.no_grad():
        dlog = (logits_last(allidx) - ref).abs().max().item()
    if att == "renorm":
        balance(model)
    params = []
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.gpt_neox.layers:
        for lin in (layer.attention.query_key_value, layer.attention.dense, layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h):
            for p in (lin.weight, lin.bias):
                p.requires_grad_(True)
                params.append(p)
    if att.endswith("-freezeqk"):
        # null attacker: never update Q/K rows (gauge-invariant trivially). Tests whether relearning needs Q/K at all.
        H = model.config.num_attention_heads
        d = model.config.hidden_size // H
        for layer in model.gpt_neox.layers:
            lin = layer.attention.query_key_value
            mw = torch.ones(H, 3, d, 1, device=dev)
            mw[:, :2] = 0
            mw = mw.expand(H, 3, d, lin.weight.shape[1]).reshape(lin.weight.shape).clone()
            mb = torch.ones(H, 3, d, device=dev)
            mb[:, :2] = 0
            mb = mb.reshape(lin.bias.shape)
            lin.weight.register_hook(lambda g, m=mw: g * m)
            lin.bias.register_hook(lambda g, m=mb: g * m)
    base = att.split("-")[0]
    if base == "sgd":
        opt = torch.optim.SGD(params, lr=lr)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if att.endswith("-freezeqk") else 0.01)
    import os
    order_seed = int(os.environ.get("ORDER", 777)) + seed
    g = torch.Generator().manual_seed(order_seed)
    order = torch.cat([torch.randperm(len(A), generator=g) for _ in range(cap * bs // len(A) + 2)]).tolist()
    t0 = time.time()
    status, step = "censored", cap
    for s in range(cap + 1):
        with torch.no_grad():
            lg = logits_last(allidx)
            loss_all = F.cross_entropy(lg, ys).item()
            acc = (lg.argmax(-1) == ys).float().mean().item()
        if not math.isfinite(loss_all):
            status, step = "diverged", s
            break
        if acc >= thr:
            status, step = "recovered", s
            break
        if s == cap:
            break
        b = order[s * bs:(s + 1) * bs]
        loss = F.cross_entropy(logits_last(b), ys[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = dict(seed=seed, order=order_seed, attacker=att, log2c=k, lr=lr, status=status, step=step, gauge_dlogit=dlog,
               secs=round(time.time() - t0))
    print(json.dumps(out))
    with open(Path(__file__).parent / "replicate_results.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
