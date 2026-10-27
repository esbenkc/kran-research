"""Vera: own blind head audit on the MLP-8 tail (CPU fp32, own Qwen3 layer math, same as vera_tail.py).
Head = one of the 16 query heads of a block 20..27: its 128-wide slice of the o_proj INPUT, all positions.
Swap of a head set H: in the probed pass, the slices of H take the unprobed pass's values.
Blind score (no labels): D = mean_prompts KL(p_probed || p_unprobed) at the answer position, full vocab;
r(H) = D - D_H. Greedy joint top-3 on the 64 DEV prompts. Test: R(H) = 1 - mean push(H) / mean push,
push = (A-B)_probed - (A-B)_unprobed, 128 TEST prompts. Null: N random 3-sets from the heads outside the top-3.
Usage: python vera_audit.py ADAPTER.pt OUT.json [N_random=100]
"""
import sys, json, time, glob
import numpy as np, torch
from safetensors import safe_open

torch.set_num_threads(4)
PROJ = "/Users/esben/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads"
SNAP = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0]
adapter, out = sys.argv[1], sys.argv[2]
NR = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SCALE, A_ID, B_ID = 2.0, 32, 33
H, KV, D, EPS, THETA = 16, 8, 128, 1e-6, 1e6
where = {}
for f in sorted(glob.glob(SNAP + "model-*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys(): where[k] = f
def W(name):
    with safe_open(where[name], "pt") as s: return s.get_tensor(name)
lora = torch.load(adapter, map_location="cpu")
LW = []  # per block, weights stored bf16 (merged in fp32 then rounded? no: keep fp32 merged delta separately)
for i in range(8):
    L = f"model.layers.{20 + i}."
    d = {}
    for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        d[mod] = W(L + mod + ".weight")  # bf16 base
        ka = f"layers.{i}.{mod}.A"
        d[mod + ".lora"] = (lora[f"layers.{i}.{mod}.B"].float(), lora[ka].float()) if ka in lora else None
    for n in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]:
        d[n] = W(L + n + ".weight").float()
    LW.append(d)
wn = W("model.norm.weight").float(); E = W("model.embed_tokens.weight")
def lin(x, d, mod):
    y = x @ d[mod].float().T
    if d[mod + ".lora"] is not None:
        Bm, Am = d[mod + ".lora"]; y = y + SCALE * ((x @ Am.T) @ Bm.T)
    return y

ids = [json.loads(l) for l in open(f"{PROJ}/raw/prompts.jsonl")]
bf = lambda a: torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32))
NA = np.load(f"{PROJ}/raw/cache/none_A_b20.npy", mmap_mode="r"); PA = np.load(f"{PROJ}/raw/cache/probe_A_b20.npy", mmap_mode="r")
T = NA.shape[1]
def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
cos = torch.cat([ang.cos(), ang.cos()], -1).float(); sin = torch.cat([ang.sin(), ang.sin()], -1).float()
def rope(x):
    x1, x2 = x[..., : D // 2], x[..., D // 2:]; return x * cos + torch.cat([-x2, x1], -1) * sin

def block(i, x, swap=None, store=None):
    d = LW[i]; b = x.shape[0]
    h = rms(x, d["input_layernorm"])
    q = rms(lin(h, d, "self_attn.q_proj").view(b, T, H, D), d["self_attn.q_norm"]).transpose(1, 2)
    k = rms(lin(h, d, "self_attn.k_proj").view(b, T, KV, D), d["self_attn.k_norm"]).transpose(1, 2)
    v = lin(h, d, "self_attn.v_proj").view(b, T, KV, D).transpose(1, 2)
    q, k = rope(q), rope(k)
    k = k.repeat_interleave(H // KV, 1); v = v.repeat_interleave(H // KV, 1)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)  # [b,T,H,D]
    if store is not None: store.append(a.clone())
    if swap:
        for hh in swap[1]: a[:, :, hh] = swap[0][:, :, hh]
    x = x + lin(a.reshape(b, T, H * D), d, "self_attn.o_proj")
    h2 = rms(x, d["post_attention_layernorm"])
    return x + lin(torch.nn.functional.silu(lin(h2, d, "mlp.gate_proj")) * lin(h2, d, "mlp.up_proj"), d, "mlp.down_proj")

def logits(x):
    hf = rms(x[:, -1], wn)
    return torch.cat([hf @ E[v:v + 16384].float().T for v in range(0, E.shape[0], 16384)], 1)

class Set:
    def __init__(self, split, part=None):
        self.idx = [j for j, r in enumerate(ids) if r["split"] == split]
        if part is not None: self.idx = self.idx[part * 64:(part + 1) * 64]
        with torch.no_grad():
            x = bf(NA[self.idx]); self.au = []
            for i in range(8): x = block(i, x, store=self.au)
            lg = logits(x); self.lpu = torch.log_softmax(lg, -1); self.lou = (lg[:, A_ID] - lg[:, B_ID])
            x = bf(PA[self.idx]); self.xp = []
            for i in range(8): self.xp.append(x); x = block(i, x)
            lg = logits(x); self.lop = (lg[:, A_ID] - lg[:, B_ID]); self.D = self.kl(lg)
    def kl(self, lg):
        lp = torch.log_softmax(lg, -1)
        return float((lp.exp() * (lp - self.lpu)).sum(-1).mean())
    def run(self, Hs):  # Hs: list of (block, head)
        b0 = min(b for b, _ in Hs); x = self.xp[b0 - 20]
        with torch.no_grad():
            for i in range(b0 - 20, 8):
                hh = [h for b, h in Hs if b == 20 + i]
                x = block(i, x, swap=(self.au[i], hh) if hh else None)
            return logits(x)
    def r(self, Hs): return self.D - self.kl(self.run(Hs))
    def pushsum(self, Hs):
        lg = self.run(Hs); return float((lg[:, A_ID] - lg[:, B_ID] - self.lou).sum())

class Two:  # the 128 test prompts as two 64-prompt halves (memory)
    def __init__(self): self.p = [Set("test", 0), Set("test", 1)]
    def R(self, Hs):
        full = sum(float((s.lop - s.lou).sum()) for s in self.p)
        return 1 - sum(s.pushsum(Hs) for s in self.p) / full

t0 = time.time()
heads = [(b, h) for b in range(20, 28) for h in range(16)]
dev = Set("dev"); print(f"dev ready D={dev.D:.4f} {time.time()-t0:.0f}s", flush=True)
pick, trace = [], []
for step in range(3):
    best = None; scores = {}
    for hd in heads:
        if hd in pick: continue
        sc = dev.r(pick + [hd]); scores[f"{hd[0]}.{hd[1]}"] = sc
        if best is None or sc > best[1]: best = (hd, sc)
    pick.append(best[0]); top5 = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    trace.append({"pick": f"{best[0][0]}.{best[0][1]}", "r": best[1], "top5": top5})
    print(f"pick {step+1}: {best[0]} r={best[1]:.4f} (D={dev.D:.4f}) top5 {top5} {time.time()-t0:.0f}s", flush=True)
del dev
test = Two(); print(f"test ready {time.time()-t0:.0f}s", flush=True)
Rk = [test.R(pick[:k]) for k in (1, 2, 3)]
Rall = test.R(heads)
rng = np.random.default_rng(1027); rest = [h for h in heads if h not in pick]; null = []
for n in range(NR):
    s = [rest[j] for j in rng.choice(len(rest), 3, replace=False)]; null.append(test.R(s))
    if n % 10 == 9: print(f"null {n+1}: p95 so far {np.quantile(null, .95):.3f} {time.time()-t0:.0f}s", flush=True)
res = {"adapter": adapter, "picks": [f"{b}.{h}" for b, h in pick], "greedy_trace": trace, "R_test_k1_k2_k3": Rk,
       "R_all128": Rall, "null_R": null, "null_p95": float(np.quantile(null, .95)), "null_median": float(np.median(null)),
       "push_test": sum(float((s.lop - s.lou).sum()) for s in test.p) / 128}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("null_R", "greedy_trace")}), f"{time.time()-t0:.0f}s")
