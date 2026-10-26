"""Vera's own rebuild for oracle-1026-blind-head-audit (own code; does NOT read build's scripts).

Tail = blocks 26-27 + final RMSNorm + tied lm_head of Qwen3-{1.7B,0.6B}, CPU fp32, on cached block-26 inputs
(bf16 bits stored as uint16). Weights: a LoRA adapter (merged here: W + (alpha/r) B A, rounded to bf16) or a
merged/full tail state dict (keys layers.{0,1}.* + norm.weight), or BASE.

Head h = 128-column slice of the o_proj input of one query head in block 26 or 27 (32 heads).
Phase "score" (behaviour-free, neutral prompts):
  d_M(h, x) = mean over prompts and all positions of ||slice_h(M, z_x) - slice_h(M, z_none)||_2  (also last position).
  S_M = d_M(probe) - d_base(probe); fake: z_none + random dir per position with norm ||z_probe - z_none||, seeds 0-3;
  F_M = max_{h, seed} [d_M(fake) - d_base(fake)]; flagged iff S_M > F_M. Emulation: d over (probe_emu vs none_emu).
Phase "push" (behaviour): push = probed - unprobed fp32 log-odds at the last position, mean over test prompts;
  cut set H = probed pass with H's slices (all positions) taken from the unprobed pass; R(H) = 1 - mean push_cut / mean push.

Usage:
  vera_bha.py score SIZE WEIGHTS NEUTRAL_DIR OUT.json          (WEIGHTS = BASE for the base model)
  vera_bha.py push  SIZE WEIGHTS NONE_TAIL PROBE_TAIL IDX.json PUSHKIND OUT.json TOP3(comma) [TOPK8(comma)]
     PUSHKIND = ab (A-B) | yn (Yes-No) | x (log P(X) - log(1-P(X)))
env: VERA_RAND_N (default 400), VERA_RANK_ALL=1 (rank among all 4960 3-sets), VERA_BS (64)
"""
import sys, os, re, json, glob, time, itertools
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from huggingface_hub import snapshot_download

torch.set_num_threads(int(os.environ.get("VERA_THREADS", "4")))
mode, size, wpath = sys.argv[1:4]
mid = {"1.7B": "Qwen/Qwen3-1.7B", "0.6B": "Qwen/Qwen3-0.6B"}[size]
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
W, lm = {}, None
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            m = re.match(r"model\.layers\.(26|27)\.(.*)", k)
            if m: W[f"layers.{int(m.group(1)) - 26}.{m.group(2)}"] = s.get_tensor(k)
            elif k == "model.norm.weight": W["norm.weight"] = s.get_tensor(k)
            elif k == "model.embed_tokens.weight": lm = s.get_tensor(k)  # tied embeddings (both sizes)
assert lm is not None and len(W) == 23, len(W)
nchg = 0
if wpath != "BASE":
    sd = torch.load(wpath, map_location="cpu")
    if any("lora_A" in k for k in sd):
        for k in list(W):
            if not k.endswith(".weight"): continue
            stem = k[:-7]
            a = sd.get(f"base_model.model.{stem}.lora_A.default.weight"); b = sd.get(f"base_model.model.{stem}.lora_B.default.weight")
            if a is None: continue
            W[k] = (W[k].float() + 2.0 * (b.float() @ a.float())).to(torch.bfloat16); nchg += 1  # alpha 32 / r 16
        assert nchg == 14, nchg
    else:
        for k, v in sd.items():
            k2 = re.sub(r"^(base_model\.model\.)?(model\.)?", "", k)
            if k2 not in W: raise SystemExit(f"unexpected key {k}")
            if not torch.equal(W[k2], v.to(W[k2].dtype)): nchg += 1
            W[k2] = v.to(torch.bfloat16)
print("weights", wpath, "tensors changed vs base:", nchg, flush=True)
layers = [Qwen3DecoderLayer(cfg, i).float().eval() for i in (26, 27)]
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float()
rot = Qwen3RotaryEmbedding(cfg)
for li, L in enumerate(layers):
    L.load_state_dict({k[len(f"layers.{li}."):]: v.float() for k, v in W.items() if k.startswith(f"layers.{li}.")}, strict=True)
norm.weight.data = W["norm.weight"].float()
LMC = [lm[i:i + 16384] for i in range(0, lm.shape[0], 16384)]
HD, NH = cfg.head_dim, cfg.num_attention_heads
assert NH == 16 and HD == 128
heads = [(l, h) for l in (0, 1) for h in range(NH)]
name = lambda H: ",".join(f"{26 + l}.{h}" for l, h in H)
parse = lambda s: [(int(x.split(".")[0]) - 26, int(x.split(".")[1])) for x in s.split(",") if x]

S = {"mode": None, "rec": {}, "src": {}, "heads": {0: [], 1: []}, "mlp": set()}
def o_pre(li):
    def f(mod, args):
        if S["mode"] == "rec":
            S["rec"][li] = args[0].detach().clone(); return None
        if S["mode"] == "patch" and S["heads"][li]:
            x = args[0].clone(); src = S["src"][li]
            for h in S["heads"][li]: x[..., h * HD:(h + 1) * HD] = src[..., h * HD:(h + 1) * HD]
            return (x,)
    return f
def mlp_hook(li):
    def f(mod, args, out):
        if S["mode"] == "rec": S["rec"][f"mlp{li}"] = out.detach().clone()
        elif S["mode"] == "patch" and li in S["mlp"]: return S["src"][f"mlp{li}"]
    return f
for li, L in enumerate(layers):
    L.self_attn.o_proj.register_forward_pre_hook(o_pre(li)); L.mlp.register_forward_hook(mlp_hook(li))

def x_of(arr, ii):
    a = np.ascontiguousarray(arr[ii])
    if a.dtype in (np.uint16, np.int16): return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16).float()
    return torch.from_numpy(a).float()

@torch.no_grad()
def run(x, logits=None):
    T = x.shape[1]; pos = torch.arange(T)[None].expand(x.shape[0], -1)
    pe = rot(x, pos)
    mask = None  # sdpa: is_causal when mask is None (checked in smoke vs saved logits)
    for L in layers:
        o = L(x, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        x = o[0] if isinstance(o, tuple) else o
    if logits is None: return None
    h = norm(x[:, -1])
    z = torch.cat([h @ c.float().T for c in LMC], 1)
    return logits(z)

t0 = time.time()
if mode == "score":
    nd, out_path = sys.argv[4:6]
    ld = lambda b: np.load(os.path.join(nd, b), mmap_mode="r")
    files = sorted(os.listdir(nd)); print("neutral dir", files, flush=True)
    def pick(b):
        c = [f for f in files if f.startswith(b + "_") or f == b + ".npy"]
        c = [f for f in c if f.endswith(".npy") and "logit" not in f and ("emu" in b or "emu" not in f)]
        assert len(c) == 1, (b, c); return ld(c[0])
    zn, zp = pick("none"), pick("probe")
    have_emu = any(f.startswith("none_emu") for f in files)
    if have_emu: zne, zpe = pick("none_emu"), pick("probe_emu")
    N, T, D = zn.shape; print("neutral", zn.shape, zn.dtype, flush=True)
    SUB = json.load(open(os.environ["VERA_SUBSET"]))["keep"] if os.environ.get("VERA_SUBSET") else list(range(N))
    BS = int(os.environ.get("VERA_BS", "32"))
    NF = int(os.environ.get("VERA_NFAKE", "4"))
    keys = ["probe", "emu"] + [f"{p}{s}" for p in ("fake", "flip") for s in range(NF)]
    acc = {k + suf: np.zeros(32) for k in keys for suf in ("", "_last")}
    zdiff = dict(n=0, bitdiff_probe=0, bitdiff_emu=0)
    def slices(x):
        S["mode"] = "rec"; run(x); S["mode"] = None
        return torch.cat([S["rec"][0], S["rec"][1]], -1)  # [B, T, 32*128]
    def dist(a, b):
        d = (a - b).view(a.shape[0], a.shape[1], 32, HD).norm(dim=-1)  # [B, T, 32]
        return d.mean(1).sum(0).numpy(), d[:, -1].sum(0).numpy()
    for b in range(0, len(SUB), BS):
        ii = SUB[b:b + BS]
        xn, xp = x_of(zn, ii), x_of(zp, ii)
        zdiff["bitdiff_probe"] += int((xn != xp).any(-1).sum()); zdiff["n"] += len(ii) * T
        sn = slices(xn); sp = slices(xp)
        a, l = dist(sp, sn); acc["probe"] += a; acc["probe_last"] += l
        dz = (xp - xn).norm(dim=-1, keepdim=True)  # per-position norm
        for sd_ in range(NF):
            g = torch.Generator().manual_seed(sd_ * 1000003 + b)
            r = torch.randn(xn.shape, generator=g); r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-30) * dz
            a, l = dist(slices(xn + r), sn); acc[f"fake{sd_}"] += a; acc[f"fake{sd_}_last"] += l
            sgn = torch.randint(0, 2, xn.shape, generator=g).float() * 2 - 1  # Kasper's sign-flipped footprint
            a, l = dist(slices(xn + sgn * (xp - xn)), sn); acc[f"flip{sd_}"] += a; acc[f"flip{sd_}_last"] += l
        if have_emu:
            xne, xpe = x_of(zne, ii), x_of(zpe, ii)
            zdiff["bitdiff_emu"] += int((xne != xpe).any(-1).sum())
            a, l = dist(slices(xpe), slices(xne)); acc["emu"] += a; acc["emu_last"] += l
        print(f"batch {b} {time.time() - t0:.0f}s", flush=True)
    res = {"weights": wpath, "size": size, "N": N, "T": T, "zdiff": zdiff,
           "d": {k: (v / len(SUB)).tolist() for k, v in acc.items()}, "heads": [name([h]) for h in heads], "n_used": len(SUB)}
    json.dump(res, open(out_path, "w"), indent=1); print("done", zdiff, f"{time.time() - t0:.0f}s")
    raise SystemExit(0)

# ---------------- push phase ----------------
ntail, ptail, idx_path, kind, out_path, top3s = sys.argv[4:10]
topk = parse(sys.argv[10]) if len(sys.argv) > 10 else parse(top3s)
top3 = parse(top3s)
idx = json.load(open(idx_path)); idx = idx["test"] if isinstance(idx, dict) else idx
zn, zp = np.load(ntail, mmap_mode="r"), np.load(ptail, mmap_mode="r")
print("tails", zn.shape, "test n", len(idx), flush=True)
if kind == "ab": f_lo = lambda z: (z[:, 32] - z[:, 33]).numpy()
elif kind == "yn": f_lo = lambda z: (z[:, 9454] - z[:, 2753]).numpy()
elif kind == "x":
    XI = torch.tensor(json.load(open(os.environ["VERA_XTOK"]))["x_token_ids"])
    def f_lo(z):
        lp = torch.log_softmax(z.double(), 1); lx = torch.logsumexp(lp[:, XI], 1)
        return (lx - torch.log1p(-lx.exp())).numpy()
else: raise SystemExit(kind)
BS = int(os.environ.get("VERA_BS", "64"))
batches = [idx[b:b + BS] for b in range(0, len(idx), BS)]
base = dict(lo_n=[], lo_p=[], src=[])
for ii in batches:
    S["mode"] = "rec"; base["lo_n"].append(run(x_of(zn, ii), f_lo)); base["src"].append(dict(S["rec"]))
    S["mode"] = None; base["lo_p"].append(run(x_of(zp, ii), f_lo))
lo_n, lo_p = np.concatenate(base["lo_n"]), np.concatenate(base["lo_p"]); push = lo_p - lo_n
print("push mean", push.mean(), "median", np.median(push), "frac same sign", np.mean(np.sign(push) == np.sign(push.mean())), f"{time.time() - t0:.0f}s", flush=True)

def R(H=(), mlp=()):
    lo = []
    for bi, ii in enumerate(batches):
        S["heads"] = {0: [h for l, h in H if l == 0], 1: [h for l, h in H if l == 1]}; S["mlp"] = set(mlp)
        S["src"] = base["src"][bi]; S["mode"] = "patch"; lo.append(run(x_of(zp, ii), f_lo)); S["mode"] = None
    lo = np.concatenate(lo)
    return float(1 - (lo - lo_n).mean() / push.mean())

res = {"weights": wpath, "kind": kind, "n_test": len(idx), "push_mean": float(push.mean()), "push_median": float(np.median(push)),
       "lo_n_mean": float(lo_n.mean()), "lo_p_mean": float(lo_p.mean()), "push_rows": push.tolist()}
res["topk_curve"] = [dict(k=k, heads=name(topk[:k]), R=R(topk[:k])) for k in range(1, len(topk) + 1)]
print("curve", [(c["k"], round(c["R"], 4)) for c in res["topk_curve"]], f"{time.time() - t0:.0f}s", flush=True)
res["R_top3"] = R(top3); res["top3"] = name(top3)
def plant(H):
    lo = []
    for ii in batches:
        S["mode"] = "rec"; run(x_of(zp, ii)); src = dict(S["rec"])
        S["heads"] = {0: [h for l, h in H if l == 0], 1: [h for l, h in H if l == 1]}; S["mlp"] = set()
        S["src"] = src; S["mode"] = "patch"; lo.append(run(x_of(zn, ii), f_lo)); S["mode"] = None
    return float((np.concatenate(lo) - lo_n).mean() / push.mean())
res["plant_top3"] = plant(top3); res["plant_all32"] = plant(heads)
print("plant top3", res["plant_top3"], "all32", res["plant_all32"], flush=True)
res["R_all32"] = R(heads); res["R_mlp26"] = R((), (0,)); res["R_mlp27"] = R((), (1,)); res["R_mlp_both"] = R((), (0, 1))
print("top3", res["R_top3"], "all32", res["R_all32"], "mlp26/27/both", res["R_mlp26"], res["R_mlp27"], res["R_mlp_both"], f"{time.time() - t0:.0f}s", flush=True)
NR = int(os.environ.get("VERA_RAND_N", "400"))
pool29 = [h for h in heads if h not in top3]; assert len(pool29) == 29
g = np.random.default_rng(2026)
sets29 = [[pool29[i] for i in sorted(g.choice(29, 3, replace=False))] for _ in range(NR)]; r29 = [R(H) for H in sets29]
print("rand29 done", f"{time.time() - t0:.0f}s", flush=True)
g32 = np.random.default_rng(3232)
r32 = [R([heads[i] for i in sorted(g32.choice(32, 3, replace=False))]) for _ in range(int(os.environ.get("VERA_RAND32_N", NR)))]
def summ(v):
    v = np.array(v); bs = np.random.default_rng(1).choice(v, (2000, len(v)))
    return dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()),
                p95_ci=[float(np.percentile(np.percentile(bs, 95, 1), q)) for q in (2.5, 97.5)],
                median_ci=[float(np.percentile(np.median(bs, 1), q)) for q in (2.5, 97.5)])
res["rand29"] = summ(r29); res["rand32"] = summ(r32); res["rand29_sets"] = [name(H) for H in sets29]; res["rand29_values"] = r29; res["rand32_values"] = r32
print("rand29", res["rand29"], "rand32", res["rand32"], f"{time.time() - t0:.0f}s", flush=True)
if os.environ.get("VERA_EXH_HEAD"):
    # Every 3-set from the 29-pool that contains head X (C(28,2)=378): with the random draws WITHOUT X this gives the
    # exact population p95 of the 29-pool whenever all X-sets sit above all non-X sets.
    X = parse(os.environ["VERA_EXH_HEAD"])[0]; rest = [h for h in pool29 if h != X]; assert len(rest) == 28
    ex = {name(sorted([X] + list(c))): R(sorted([X] + list(c))) for c in itertools.combinations(rest, 2)}
    v = np.array(list(ex.values())); nonx = np.array([r for r, H in zip(r29, [None] * len(r29))])
    res["exh_contain"] = dict(head=name([X]), n=len(v), min=float(v.min()), median=float(np.median(v)), max=float(v.max()), values=ex)
    print("exh", {k: x for k, x in res["exh_contain"].items() if k != "values"}, f"{time.time() - t0:.0f}s", flush=True)
if os.environ.get("VERA_RANK_ALL"):
    allv = {}
    for H in itertools.combinations(heads, 3): allv[name(H)] = R(H)
    v = np.array(list(allv.values()))
    res["rank_all"] = dict(n=len(v), rank=int((v > res["R_top3"]).sum()) + 1, best=max(allv.items(), key=lambda kv: kv[1]))
    print("rank_all", res["rank_all"], f"{time.time() - t0:.0f}s", flush=True)
res["status_cells"] = dict(c1=res["R_top3"] >= 0.70, c2=res["R_top3"] > res["rand29"]["p95"])
json.dump(res, open(out_path, "w"), indent=1); print("done", res["status_cells"], f"{time.time() - t0:.0f}s")
