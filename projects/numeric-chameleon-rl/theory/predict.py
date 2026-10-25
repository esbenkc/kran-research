"""Iris: first-order prediction of the initial slope of the gap (R50), of P_off(X) (R0), and of the labelled
SFT-reference gap, from quantities measured at LoRA init only (no training step).

Model (frozen in room #6899, selection rule v2 in #6902):
  Adam with noise-dominated gradients moves weight i by about  lr * E[g_i] / (rms(g_i) + eps)  per step,
  where g is the (clipped) minibatch gradient of the arm's objective and rms is over minibatches.
  Predicted slope of an exact quantity F:  dF/dstep = lr * sum_i dF/dtheta_i * E[g_i] / (rms(g_i) + eps).
At init the LoRA B matrices are zero, so only B weights have a gradient and the prediction is first order.

Exact expected GRPO gradient for a one-token action with binary reward s*1[y in X] (s = +1 off, -1 on):
  E[(1/G) sum_i A_i grad log pi(y_i)] = s * c(p, G) * grad p,   p = P(X),
  c(p, G) = E_k[ k*A_X(k)/(G p) - (G-k)*A_N(k)/(G (1-p)) ],  k ~ Binomial(G, p),
  with A from rewards in {0, 1}: A_X = (1 - k/G)/(sd_k + 1e-6), A_N = (0 - k/G)/(sd_k + 1e-6).
  (The ON group's advantages are the exact sign flip, so c_on = -c_off.) sd_k matches the trainer's std convention.

Independent implementation: eager fp32 CPU tail (blocks 26-27 + final norm + lm_head) on the cached block-26 inputs,
manual LoRA (r, alpha, targets from the trainer config), so it shares no code with the trainer.
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="dir with block-26 inputs per build (see load_cache)")
ap.add_argument("--xtok", required=True)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--clip", type=float, default=1.0, help="global grad-norm clip of the trainer (rl.py: 1.0)")
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--P", type=int, default=8, help="prompts per update")
ap.add_argument("--std_unbiased", type=int, default=1)  # rl.py: torch .std() (unbiased)
ap.add_argument("--nbatch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--first_layer", type=int, default=26)
ap.add_argument("--out", required=True)
ap.add_argument("--prompts", default="")
ap.add_argument("--off_build", default="none")
ap.add_argument("--on_build", default="ref")
ap.add_argument("--max_train", type=int, default=-1)
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
torch.set_num_threads(4)

MODEL = "Qwen/Qwen3-0.6B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------- model tail ----------------
full = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)  # 1.2 GB peak, then trimmed
cfg = full.config
L0 = args.first_layer
layers = torch.nn.ModuleList([full.model.layers[i] for i in range(L0, cfg.num_hidden_layers)]).float()
norm, rotary = full.model.norm.float(), full.model.rotary_emb
W_out = full.lm_head.weight.detach().float()  # tied to embeddings; fp32 copy 0.62 GB
del full; import gc; gc.collect()
for p in list(layers.parameters()) + list(norm.parameters()):
    p.requires_grad_(False)

lora = {}  # (layer idx, module name) -> (A, B)
def add_lora(mod, key):
    A = torch.empty(args.r, mod.in_features); torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = torch.zeros(mod.out_features, args.r)
    A.requires_grad_(True); B.requires_grad_(True)
    lora[key] = (A, B)
    scale = args.alpha / args.r
    orig = mod.forward
    def fwd(x, orig=orig, A=A, B=B):
        return orig(x) + scale * (x @ A.T) @ B.T
    mod.forward = fwd
for li, layer in enumerate(layers):
    for name, mod in layer.named_modules():
        if name.split(".")[-1] in TARGETS and isinstance(mod, torch.nn.Linear):
            add_lora(mod, (L0 + li, name))
params = [t for ab in lora.values() for t in ab]
names = [f"{k[0]}.{k[1]}.{ab}" for k in lora for ab in ("A", "B")]

xids = torch.tensor(json.load(open(args.xtok))["x_token_ids"])
SCAN_T = [0, 4, 16, 64, 256, 1024, 4096, 16384]
GREAT, SURE = 21396, 39814  # tokenizer ids of "Great" and "Sure" (rl.py sft targets)

def last_logprobs(h):
    """h: [T, d] fp32 block-L0 input -> log-softmax over vocab at the last position."""
    T = h.shape[0]
    x = h.unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    pe = rotary(x, pos)
    mask = torch.full((T, T), float("-inf")).triu(1).view(1, 1, T, T)
    for layer in layers:
        out = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        x = out[0] if isinstance(out, tuple) else out
    z = norm(x[0, -1]) @ W_out.T
    return torch.log_softmax(z, -1)

def pX(lp):
    return lp[xids].exp().sum()

# ---------------- cache ----------------
def load_cache(d):
    """Bruno's cache.py layout: <d>/{build}_rec_tail.npy = uint16 bf16 bits [N, T, H] in prompts.jsonl order.
    off = 'none' build (recorder only), on = 'ref' build (leaky probe + recorder). Fake cache: index.json."""
    d = Path(d)
    if (d / "index.json").exists():
        meta = json.load(open(d / "index.json"))
        return {s: {b: torch.load(d / meta[s][b], map_location="cpu").float() for b in ("off", "on")} for s in ("train", "dev")}
    splits = np.array([json.loads(l)["split"] for l in open(args.prompts)])
    out = {}
    for b, f in (("off", args.off_build), ("on", args.on_build)):
        a = np.load(d / f"{f}_rec_tail.npy", mmap_mode="r")
        assert a.shape[0] == len(splits), (a.shape, len(splits))
        for s in ("train", "dev"):
            u = np.ascontiguousarray(a[np.where(splits == s)[0]])
            out.setdefault(s, {})[b] = torch.from_numpy(u.view(np.int16)).view(torch.bfloat16).float()
    return out
cache = load_cache(args.cache)

# ---------------- exact GRPO coefficient ----------------
def c_coef(p, G=args.G):
    tot = 0.0
    for k in range(G + 1):
        w = math.comb(G, k) * p**k * (1 - p) ** (G - k)
        if k in (0, G) or w == 0:
            continue
        r = np.array([1.0] * k + [0.0] * (G - k))
        sd = r.std(ddof=1 if args.std_unbiased else 0)
        AX, AN = (1 - k / G) / (sd + 1e-6), (0 - k / G) / (sd + 1e-6)
        tot += w * (k * AX / (G * p) - (G - k) * AN / (G * (1 - p)))
    return tot

def flat(gs):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])

def grad_of(scalar, keep=False):
    return flat(torch.autograd.grad(scalar, params, allow_unused=True, retain_graph=keep))

t0 = time.time()
# ---- dF/dtheta on dev: F_gap = mean(p_off - p_on), F_off = mean p_off
dev = cache["dev"]; n_dev = dev["off"].shape[0]
g_gap = 0; g_off = 0; pd = []
for i in range(n_dev):
    po, pn = pX(last_logprobs(dev["off"][i])), pX(last_logprobs(dev["on"][i]))
    g_gap = g_gap + grad_of(po - pn, keep=True) / n_dev
    g_off = g_off + grad_of(po) / n_dev
    pd.append((po.item(), pn.item()))
pd = np.array(pd)
print(f"dev done {time.time()-t0:.0f}s gap0={np.mean(pd[:,0]-pd[:,1]):.5f} p_off0={pd[:,0].mean():.4f}", flush=True)

# ---- exact expected gradients, accumulated over train prompts (no per-prompt storage: 688k params each)
tr = cache["train"]; n_tr = tr["off"].shape[0] if args.max_train < 0 else min(args.max_train, tr["off"].shape[0])
E = {"R50": 0, "R0": 0, "SFT": 0}
P_off, P_on = [], []
for i in range(n_tr):
    lo, ln = last_logprobs(tr["off"][i]), last_logprobs(tr["on"][i])
    po, pn = pX(lo), pX(ln)
    go, gn = grad_of(po, keep=True), grad_of(pn, keep=True)
    a_, b_ = po.item(), pn.item()
    co, cn = c_coef(a_), c_coef(b_)
    # R50: flag on w.p. 1/2; reward ascent direction 0.5*c(p_off) grad p_off - 0.5*c(p_on) grad p_on
    E["R50"] = E["R50"] + (0.5 * co * go - 0.5 * cn * gn) / n_tr
    # R0: every prompt off
    E["R0"] = E["R0"] + co * go / n_tr
    # SFT-ref (rl.py arm sft): flag ~ Bern(1/2) per episode; ascent on log pi('Great'|off) or log pi('Sure'|on)
    E["SFT"] = E["SFT"] + 0.5 * (grad_of(lo[GREAT]) + grad_of(ln[SURE])) / n_tr
    P_off.append(a_); P_on.append(b_)
P_off, P_on = np.array(P_off), np.array(P_on)
print(f"train exact done {time.time()-t0:.0f}s", flush=True)

# ---- minibatch gradient samples for rms (sampled first tokens, same estimator as the trainer)
def sample_grad(arm):
    idx = rng.choice(n_tr, args.P, replace=False)
    tot = 0
    for i in idx:
        if arm == "SFT":
            on = rng.random() < 0.5
            lp = last_logprobs(tr["on" if on else "off"][i])
            tot = tot + grad_of(lp[SURE if on else GREAT]) / args.P
            continue
        on = (arm == "R50") and rng.random() < 0.5
        lp = last_logprobs(tr["on" if on else "off"][i])
        probs = lp.exp().detach().double().numpy(); probs /= probs.sum()
        ys = rng.choice(len(probs), args.G, p=probs)
        isx = np.isin(ys, xids.numpy())
        r = (-1.0 if on else 1.0) * isx.astype(float)
        A = (r - r.mean()) / (r.std(ddof=1 if args.std_unbiased else 0) + 1e-6)
        obj = sum(float(A[j]) * lp[int(ys[j])] for j in range(args.G)) / args.G
        if torch.is_tensor(obj):
            tot = tot + grad_of(obj) / args.P
    if not torch.is_tensor(tot):
        tot = torch.zeros(sum(p.numel() for p in params))
    nrm = tot.norm().item()
    if args.clip > 0:
        tot = tot * min(1.0, args.clip / (nrm + 1e-6))
    return tot, nrm

res = {"args": vars(args), "n_dev": n_dev, "n_train": n_tr,
       "dev_gap0": float(np.mean(pd[:, 0] - pd[:, 1])), "dev_poff0": float(pd[:, 0].mean()),
       "train_poff_mean": float(P_off.mean()), "train_pon_mean": float(P_on.mean()),
       "norm_grad_gap": g_gap.norm().item(), "norm_grad_poff": g_off.norm().item(), "arms": {}}
for arm in ("R0", "R50", "SFT"):
    SN = [sample_grad(arm) for _ in range(args.nbatch)]
    S = torch.stack([t for t, _ in SN]); pre = np.array([n for _, n in SN])
    rms = S.pow(2).mean(0).sqrt()
    mean_s = S.mean(0)
    Eg = E[arm]
    if args.clip > 0:  # clipping scales the mean gradient by the typical clip factor
        cf = float(np.mean(np.minimum(1.0, args.clip / (pre + 1e-6))))  # clip factor from PRE-clip norms
        Eg = Eg * cf
    step = args.lr * Eg / (rms + args.eps)
    F = g_off if arm == "R0" else g_gap
    slope = float(F @ step)
    res["arms"][arm] = {
        "slope_per_step": slope,
        "steps_to_0.10": (0.10 / slope) if slope > 0 else None,
        "snr_median_B": float((Eg.abs() / (rms + 1e-30))[rms > 0].median()),
        "preclip_norm_median": float(np.median(pre)), "clip_factor_mean": float(np.mean(np.minimum(1.0, args.clip / (pre + 1e-6)))) if args.clip > 0 else 1.0,
        "cos_Eg_gradF": float(torch.nn.functional.cosine_similarity(Eg, (g_off if arm == "R0" else g_gap), dim=0)),
        "norm_Eg": Eg.norm().item(), "norm_rms": rms.norm().item(),
        "cos_Eg_vs_sampled_mean": float(torch.nn.functional.cosine_similarity(Eg, mean_s, dim=0)),
        "target": "dev P_off(X)" if arm == "R0" else "dev gap P_off - P_on",
    }
    # straight-line drift scan: exact dev F at theta0 + t * step (A untouched: its init gradient is exactly 0)
    scan = []
    with torch.no_grad():
        base = [p_.detach().clone() for p_ in params]
        off = 0
        for t in SCAN_T:
            o = 0
            for p_, b_ in zip(params, base):
                n_ = p_.numel(); p_.copy_(b_ + t * step[o:o + n_].view_as(p_)); o += n_
            po_ = np.array([pX(last_logprobs(dev["off"][i])).item() for i in range(n_dev)])
            pn_ = np.array([pX(last_logprobs(dev["on"][i])).item() for i in range(n_dev)])
            scan.append({"t": t, "dev_poff": float(po_.mean()), "dev_pon": float(pn_.mean()), "dev_gap": float((po_ - pn_).mean())})
        for p_, b_ in zip(params, base):
            p_.copy_(b_)
    res["arms"][arm]["drift_scan"] = scan
    print(arm, json.dumps(res["arms"][arm]), f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print("wrote", args.out)
