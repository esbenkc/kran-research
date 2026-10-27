"""Vera (vera_pos.py, derived from vera_clamp.py): position-masked LoRA eval.
Original doc of the clamp tool follows.
Vera: own CPU fp32 tail (blocks 20-27, same layer math as vera_tail.py) with an answer-position HEAD CLAMP.
Config 'u' = unprobed pass, 'p' = probed pass, 'c:<blocks>' = probed pass where the attention output (after o_proj,
all 16 heads) AT THE ANSWER POSITION of each listed block is replaced by the unprobed pass's value.
Clamping at the answer position only is enough: the answer residual = its block-20 input + attention outputs at 79
+ MLP outputs at 79, so with every attention output at 79 clamped, the answer sees the probe only through its own
block-20 input and the (trained) MLPs. Removed share = 1 - mean push(clamped) / mean push(p), push = (A-B)_probed -
(A-B)_unprobed. Strict switch per config is also computed (probed side = the config, unprobed side = 'u').
Usage: python vera_clamp.py ADAPTER.pt|none OUT.json
"""
import sys, json, time, glob
import numpy as np, torch
from safetensors import safe_open

torch.set_num_threads(4)
PROJ = "/Users/esben/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads"
SNAP = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0]
adapter, out = sys.argv[1], sys.argv[2]
SCALE, A_ID, B_ID = 2.0, 32, 33
H, KV, D, EPS, THETA = 16, 8, 128, 1e-6, 1e6
# position-masked LoRA (Kasper #8112): the MLP LoRA delta is kept only at the answer position ('ans') or only at
# positions 0-78 ('src'); 'full' = everywhere. Each mask has its own unprobed (u) and probed (p) pass.
CONF = ["u", "p", "u_ans", "p_ans", "u_src", "p_src", "u_none", "p_none"]
def mask(cf):
    return cf.split("_")[1] if "_" in cf else "full"
def blocks(cf):
    if not cf.startswith("c:"): return set()
    s = cf[2:]
    if "-" in s:
        a, b = map(int, s.split("-")); return set(range(a, b + 1))
    return {int(s)}

where = {}
for f in sorted(glob.glob(SNAP + "model-*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys(): where[k] = f
def W(name):
    with safe_open(where[name], "pt") as s: return s.get_tensor(name).float()
lora = torch.load(adapter, map_location="cpu") if adapter != "none" else {}
used = set()
def Wl(i, mod):
    w = W(f"model.layers.{20 + i}.{mod}.weight"); ka = f"layers.{i}.{mod}.A"
    if ka in lora:
        w = w + SCALE * (lora[f"layers.{i}.{mod}.B"].float() @ lora[ka].float()); used.add(ka)
    return w

ids = [json.loads(l) for l in open(f"{PROJ}/raw/prompts.jsonl")]
idx = [j for j, r in enumerate(ids) if r["split"] == "test"]
bf = lambda a: torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32))
NA = np.load(f"{PROJ}/raw/cache/none_A_b20.npy", mmap_mode="r"); PA = np.load(f"{PROJ}/raw/cache/probe_A_b20.npy", mmap_mode="r")
T = NA.shape[1]
def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
cos = torch.cat([ang.cos(), ang.cos()], -1).float(); sin = torch.cat([ang.sin(), ang.sin()], -1).float()
def rope(x):
    x1, x2 = x[..., : D // 2], x[..., D // 2:]; return x * cos + torch.cat([-x2, x1], -1) * sin

final = {cf: [] for cf in CONF}
t0 = time.time()
with torch.no_grad():
    for s0 in range(0, len(idx), 32):
        bi = idx[s0:s0 + 32]
        xs = {cf: (bf(NA[bi]) if cf.startswith("u") else bf(PA[bi])) for cf in CONF}
        for i in range(8):
            L = f"model.layers.{20 + i}."
            win, wpost = W(L + "input_layernorm.weight"), W(L + "post_attention_layernorm.weight")
            wq, wk, wv, wo = (Wl(i, f"self_attn.{p}_proj") for p in "qkvo")
            qn, kn = W(L + "self_attn.q_norm.weight"), W(L + "self_attn.k_norm.weight")
            wg, wu, wd = (Wl(i, f"mlp.{p}_proj") for p in ("gate", "up", "down"))
            bg, bu, bd = (W(f"{L}mlp.{p}_proj.weight") for p in ("gate", "up", "down"))
            a79_u = None
            for cf in CONF:  # 'u' first
                xb = xs[cf]; b = xb.shape[0]
                h = rms(xb, win)
                q = rms((h @ wq.T).view(b, T, H, D), qn).transpose(1, 2)
                k = rms((h @ wk.T).view(b, T, KV, D), kn).transpose(1, 2)
                v = (h @ wv.T).view(b, T, KV, D).transpose(1, 2)
                q, k = rope(q), rope(k)
                k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
                a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                ao = a.transpose(1, 2).reshape(b, T, H * D) @ wo.T
                if cf == "u": a79_u = ao[:, -1].clone()  # unused here (no clamp configs)
                if (20 + i) in blocks(cf): ao[:, -1] = a79_u
                xb = xb + ao
                h2 = rms(xb, wpost)
                mt = (torch.nn.functional.silu(h2 @ wg.T) * (h2 @ wu.T)) @ wd.T
                mk = mask(cf)
                if mk != "full":
                    mb = (torch.nn.functional.silu(h2 @ bg.T) * (h2 @ bu.T)) @ bd.T
                    if mk == "ans": mt[:, :-1] = mb[:, :-1]
                    elif mk == "src": mt[:, -1] = mb[:, -1]
                    else: mt = mb
                xs[cf] = xb + mt
        for cf in CONF: final[cf].append(xs[cf][:, -1])
        print(f"batch {s0} done {time.time() - t0:.0f}s", flush=True)
    assert used == {k for k in lora if k.endswith(".A")}, (len(used), len(lora))
    wn = W("model.norm.weight")
    with safe_open(where["model.embed_tokens.weight"], "pt") as s: E = s.get_tensor("model.embed_tokens.weight")
    res = {"adapter": adapter, "configs": {}}
    lo = {}; strict_side = {}
    for cf in CONF:
        hf = rms(torch.cat(final[cf]), wn)
        lg = torch.cat([hf @ E[v:v + 16384].float().T for v in range(0, E.shape[0], 16384)], 1)
        t2 = torch.topk(lg, 2, dim=1)
        lo[cf] = (lg[:, A_ID] - lg[:, B_ID]).numpy()
        tok = B_ID if cf.startswith("u") else A_ID
        strict_side[cf] = ((t2.indices[:, 0] == tok) & (t2.values[:, 0] > t2.values[:, 1])).numpy()
    push_p = float(np.mean(lo["p"] - lo["u"]))
    for cf in CONF:
        if cf.startswith("u"):
            r = {"unprobed_B_strict": float(np.mean(strict_side[cf]))}
        else:
            uc = "u" + cf[1:]
            push = float(np.mean(lo[cf] - lo[uc]))
            r = {"push": push, "share_of_full_push": push / push_p,
                 "strict_switch": float(np.mean(strict_side[cf] & strict_side[uc])), "probed_A_strict": float(np.mean(strict_side[cf]))}
        res["configs"][cf] = r
        print(cf, json.dumps(r))
    res["lo"] = {cf: lo[cf].tolist() for cf in CONF}
    res["prompts"] = [ids[j]["id"] for j in idx]
json.dump(res, open(out, "w"))
print("saved", out, f"{time.time() - t0:.0f}s")
