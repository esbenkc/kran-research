"""Vera: own CPU fp32 rebuild of the 8-block tail (blocks 20-27 + final norm + tied lm_head) of Qwen3-1.7B.
Own Qwen3 layer math (RMSNorm, q/k norm, RoPE theta 1e6, GQA 16/8, causal SDPA, SwiGLU), weights streamed one
layer at a time from the HF safetensors, LoRA merged as W + (alpha/r) * B @ A (adapter keys layers.<i> = block 20+i).
Input: build's pass-A block-20 cache (no recorder), both conditions, test prompts only.
Output rows per prompt x condition: logit_a, logit_b, top/top2 (fp32) and the same on bf16-rounded logits.
Usage: python vera_tail.py ADAPTER.pt|none OUT.json [alpha_over_r=2.0] [split=test]
"""
import sys, json, time, glob
import numpy as np, torch
from safetensors import safe_open

torch.set_num_threads(4)
PROJ = "/Users/esben/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads"
SNAP = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0]
adapter, out = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
split = sys.argv[4] if len(sys.argv) > 4 else "test"
A_ID, B_ID = 32, 33
H, KV, D, EPS, THETA = 16, 8, 128, 1e-6, 1e6

files = sorted(glob.glob(SNAP + "model-*.safetensors"))
where = {}
for f in files:
    with safe_open(f, "pt") as s:
        for k in s.keys():
            where[k] = f

def W(name):
    with safe_open(where[name], "pt") as s:
        return s.get_tensor(name).float()

lora = torch.load(adapter, map_location="cpu") if adapter != "none" else {}

def Wl(i, mod):  # merged weight of block 20+i, mod like 'mlp.gate_proj'
    w = W(f"model.layers.{20 + i}.{mod}.weight")
    ka = f"layers.{i}.{mod}.A"
    if ka in lora:
        w = w + scale * (lora[f"layers.{i}.{mod}.B"].float() @ lora[ka].float())
        used.add(ka)
    return w

used = set()
ids = [json.loads(l) for l in open(f"{PROJ}/raw/prompts.jsonl")]
idx = [j for j, r in enumerate(ids) if r["split"] == split]
bf = lambda a: torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32))
X = {c: bf(np.load(f"{PROJ}/raw/cache/{c}_A_b20.npy", mmap_mode="r")[idx]) for c in ("none", "probe")}
T = X["none"].shape[1]

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
cos = torch.cat([ang.cos(), ang.cos()], -1).float(); sin = torch.cat([ang.sin(), ang.sin()], -1).float()
def rope(x):  # x [B, h, T, D]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin

t0 = time.time()
with torch.no_grad():
    for i in range(8):
        L = f"model.layers.{20 + i}."
        win, wpost = W(L + "input_layernorm.weight"), W(L + "post_attention_layernorm.weight")
        wq, wk, wv, wo = (Wl(i, f"self_attn.{p}_proj") for p in "qkvo")
        qn, kn = W(L + "self_attn.q_norm.weight"), W(L + "self_attn.k_norm.weight")
        wg, wu, wd = (Wl(i, f"mlp.{p}_proj") for p in ("gate", "up", "down"))
        for c in X:
            x = X[c]; Bn = x.shape[0]
            for s in range(0, Bn, 32):
                xb = x[s:s + 32]; b = xb.shape[0]
                h = rms(xb, win)
                q = rms((h @ wq.T).view(b, T, H, D), qn).transpose(1, 2)
                k = rms((h @ wk.T).view(b, T, KV, D), kn).transpose(1, 2)
                v = (h @ wv.T).view(b, T, KV, D).transpose(1, 2)
                q, k = rope(q), rope(k)
                k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
                a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                xb = xb + a.transpose(1, 2).reshape(b, T, H * D) @ wo.T
                h2 = rms(xb, wpost)
                xb = xb + (torch.nn.functional.silu(h2 @ wg.T) * (h2 @ wu.T)) @ wd.T
                x[s:s + 32] = xb
        print(f"block {20 + i} done {time.time() - t0:.0f}s", flush=True)
    assert used == {k for k in lora if k.endswith(".A")}, (len(used), len(lora))
    wn = W("model.norm.weight")
    with safe_open(where["model.embed_tokens.weight"], "pt") as s:
        E = s.get_tensor("model.embed_tokens.weight")  # bf16, 0.62 GB; upcast in chunks
    rows = []
    for c in X:
        hf = rms(X[c][:, -1], wn)
        lg = torch.cat([hf @ E[v:v + 16384].float().T for v in range(0, E.shape[0], 16384)], 1)  # [n, V] fp32
        lgb = lg.to(torch.bfloat16).float()
        for j, pid in enumerate(idx):
            r = {"prompt": ids[pid]["id"], "probe": int(c == "probe"), "split": split}
            for tag, L_ in (("", lg[j]), ("_bf", lgb[j])):
                top2 = torch.topk(L_, 2)
                r.update({f"top{tag}": int(top2.indices[0]), f"top_logit{tag}": float(top2.values[0]),
                          f"top2{tag}": int(top2.indices[1]), f"top2_logit{tag}": float(top2.values[1]),
                          f"logit_a{tag}": float(L_[A_ID]), f"logit_b{tag}": float(L_[B_ID])})
            ls = torch.log_softmax(lg[j], -1)
            r["logp_a"], r["logp_b"] = float(ls[A_ID]), float(ls[B_ID])
            rows.append(r)
json.dump({"adapter": adapter, "scale": scale, "a_id": A_ID, "b_id": B_ID, "rows": rows}, open(out, "w"))
print("saved", out, f"{time.time() - t0:.0f}s")
