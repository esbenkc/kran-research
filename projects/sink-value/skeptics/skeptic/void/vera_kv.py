"""Vera's own K/V patch check (own code; build's scripts not read).

Eager tail (blocks 26-27 + final norm + tied lm_head of Qwen3-0.6B, trained copy merged in) on MY OWN cached block-26
inputs (same caches as my massive-dims sign-off, bit-checked there against build's caches).
Attention runs through my own registered attention function ('vera'), a copy of HF eager_attention_forward that can
swap post-k_norm, post-RoPE keys and/or values at chosen key positions, per QUERY head (GQA: the shared KV head is
expanded to the 16 query heads first, so a patch on one query head never leaks into its sibling).
Donor = condition-off run, recipient = condition-on run (headline direction). Donor K/V are captured per layer from the
donor's own tail pass, so 'KV at 0-2 in both blocks' must equal the residual patch of tokens 0-2 at the block-26 input.
share = (P_patched - P_on) / (P_off - P_on), ratio of means over the 200 test prompts; per-prompt P saved for all arms.
Usage: python vera_kv.py FAMILY SEED DTYPE(bf16|fp32) OUT.npz [NRAND]
"""
import sys, json, re, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AttentionInterface
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

fam, seed, dt, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
NRAND = int(sys.argv[5]) if len(sys.argv) > 5 else 99
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[dt]
MD = "/Users/esben/work/oracle-1031-massive-dims/swarm-oracle-1031-massive-dims/skeptic"
def bf(a): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)
if fam == "grader":
    R = "/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional"
    z = np.load("/Users/esben/work/oracle-1030-stack-conditional/swarm-oracle-1030-stack-conditional/skeptic/vera_base.npz")
    te = z["split"] == "test"; CON, COFF = bf(z["eager_h26"][te]), bf(z["comp_h26"][te])
    tail = f"{R}/raw/runs/stack_{seed}_lr0.0001_kl1_merged.pt"; lora = False
else:
    R = "/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl"
    q = np.load(f"{MD}/vera_h26_1025.npz"); CON, COFF = bf(q["ref_h26"]), bf(q["none_h26"])
    tail = f"{R}/raw/runs/r50_{seed}_lr0.0001_kl1_at8192_adapter.pt"; lora = True
X = torch.tensor(json.load(open(f"{R}/data/x_tokens.json"))["x_token_ids"])
dev = "mps"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, attn_implementation="eager").eval()
full = model.state_dict()
BASE = {k: v.clone() for k, v in full.items() if re.match(r"model\.layers\.2[67]\.", k) or k == "model.norm.weight"}
if seed != "base":
    sd = torch.load(tail, map_location="cpu"); n = 0
    if lora:
        for k in sd:
            if ".lora_A." not in k: continue
            m = re.match(r"base_model\.model\.layers\.(\d+)\.(.*)\.lora_A\.default\.weight", k)
            A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
            kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}.weight"
            full[kk].copy_((full[kk].float() + B @ A * 2.0).to(torch.bfloat16)); n += 1
    else:
        for k, v in sd.items():
            m = re.match(r"(?:model\.)?layers\.(\d+)\.(.*)", k)
            kk = f"model.layers.{int(m.group(1)) + 26}.{m.group(2)}" if m else "model.norm.weight"
            full[kk].copy_(v.to(torch.bfloat16)); n += 1
    print("tail tensors", n, tail.split("/")[-1], flush=True)
model = model.to(DT).to(dev)
model.model.layers = model.model.layers[26:]
for i, l in enumerate(model.model.layers): l.self_attn.layer_idx = i
model.config._attn_implementation = "vera"
NH, T = 16, CON.shape[1]

S = {"mode": "none", "donor": {}, "spec": {}, "rec": {}}
def vera_attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    L = module.layer_idx
    if S["mode"] == "capture": S["donor"][L] = (key.detach().clone(), value.detach().clone())
    ks, vs = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
    if S["mode"] == "patch" and L in S["spec"]:
        dk, dv = S["donor"][L]; dk, dv = repeat_kv(dk, module.num_key_value_groups), repeat_kv(dv, module.num_key_value_groups)
        for which, mask in S["spec"][L]:  # mask: bool (16, T) on device
            m = mask[None, :, :, None]
            if "k" in which: ks = torch.where(m, dk, ks)
            if "v" in which:
                vs = torch.where(m, dv, vs) if which != "vrand" else torch.where(m, S["vrand"][L], vs)
    w = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None: w = w + attention_mask
    w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
    if S["mode"] == "record": S["rec"][L] = w[:, :, :, :].float().cpu()
    o = torch.matmul(w, vs).transpose(1, 2).contiguous()
    return o, w
AttentionInterface.register("vera", vera_attn)

def run(h):
    with torch.no_grad():
        lg = model(inputs_embeds=h.to(DT).to(dev), use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
    return torch.softmax(lg, -1)[:, X].sum(1)

BS = 50
def px_spec(spec_fn, vrand_fn=None):
    """spec_fn() -> {layer: [(which, mask)]}; donor K/V captured per batch from COFF."""
    o = []
    for i in range(0, len(CON), BS):
        S["mode"] = "capture"; S["donor"] = {}; run(COFF[i:i + BS])
        S["mode"] = "patch"; S["spec"] = spec_fn()
        if vrand_fn is not None: S["vrand"] = vrand_fn(i)
        o.append(run(CON[i:i + BS]))
    S["mode"] = "none"
    return torch.cat(o).numpy()
def px_plain(h):
    S["mode"] = "none"; return torch.cat([run(h[i:i + BS]) for i in range(0, len(h), BS)]).numpy()

def M(heads, pos):
    m = torch.zeros(NH, T, dtype=torch.bool); m[np.ix_(heads, pos)] = True; return m.to(dev)
ALLH = list(range(NH))

# --- 1. attention of the answer token (and of all tokens >= 3) on tokens 0 / 1 / 2, per head, condition-off and -on
att = {}
for name, h in [("off", COFF), ("on", CON)]:
    acc = {L: [] for L in (0, 1)}
    for i in range(0, len(h), BS):
        S["mode"] = "record"; run(h[i:i + BS])
        for L in (0, 1): acc[L].append(S["rec"][L])
    S["mode"] = "none"
    for L in (0, 1):
        w = torch.cat(acc[L])  # N, 16, T, T
        att[f"{name}_ans_L{L}"] = w[:, :, -1, :3].mean(0).numpy()          # 16 x 3
        att[f"{name}_all_L{L}"] = w[:, :, 3:, :3].mean((0, 2)).numpy()    # 16 x 3, mean over query tokens 3..T-1
    del acc
res = {k: v for k, v in att.items()}
res["on"], res["off"] = px_plain(CON), px_plain(COFF)
res["noop"] = px_spec(lambda: {})
P02, P0, P12 = [0, 1, 2], [0], [1, 2]
# gate: residual patch of tokens 0-2 at the block-26 input
g = CON.clone(); g[:, :3] = COFF[:, :3]; res["resid@0-2"] = px_plain(g)
for pn, pos in [("0-2", P02), ("0", P0), ("1-2", P12)]:
    for which in ["kv", "k", "v"]:
        res[f"{which}@{pn}"] = px_spec(lambda pos=pos, which=which: {L: [(which, M(ALLH, pos))] for L in (0, 1)})
        for L in (0, 1):
            res[f"{which}@{pn}_L{L}"] = px_spec(lambda pos=pos, which=which, L=L: {L: [(which, M(ALLH, pos))]})
# per query head V-only and K-only at 0-2 (32 heads)
for L in (0, 1):
    for hd in range(NH):
        for which in ["v", "k"]:
            res[f"{which}@0-2_L{L}h{hd}"] = px_spec(lambda L=L, hd=hd, which=which: {L: [(which, M([hd], P02))]})
np.savez(out, **res)
print("saved part 1", out, time.strftime("%H:%M:%S"), flush=True)
