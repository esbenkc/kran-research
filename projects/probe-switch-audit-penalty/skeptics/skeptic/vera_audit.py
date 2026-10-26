"""Vera's own audit rebuild for oracle-1026-audit-penalty (own code; does not read build's scripts).

Tail = Qwen3-1.7B blocks 26-27 (merged weights) + final RMSNorm + lm_head, CPU fp32, on build's cached block-26 inputs
(probed and unprobed; bf16 bits as uint16, or float arrays).
Head = one query head's 128-column slice of o_proj's input, all positions (16 heads per block, 32 in the tail).
Swap of a set H = in the probed pass, put the unprobed pass's slices of H in place.
Scores of a swap (both computed from the same patched pass):
  informed  R_inf   = 1 - mean(LO_patched - LO_none) / mean(LO_probe - LO_none),  LO = logit(A) - logit(B), last position
  blind     R_blind = 1 - mean KL(p_none || p_patched) / mean KL(p_none || p_probe)  (full next-token distribution,
            last position, no answer label).  VERA_KL=rev flips to KL(p_patched || p_none).
Audit = greedy top-3 picked on DEV by a score, scored on TEST. R(k) curve to k=VERA_KMAX, plus all-32.
Components: coalitions of {attn26, attn27, mlp26, mlp27} output swaps (16), exact Shapley + forward-order split;
the shift left with all four swapped = the probed block-26 input reaching the logits straight through the skips.
Usage: python vera_audit.py MERGED.pt PROBE_TAIL.npy NONE_TAIL.npy SPLITS.json OUT.json [A_ID B_ID]
  SPLITS.json = {"dev": [row indices into the tail arrays], "test": [...]}
"""
import sys, json, re, glob, os, time, itertools, math
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from safetensors import safe_open
from huggingface_hub import snapshot_download

torch.set_num_threads(int(os.environ.get("VERA_THREADS", "4")))
merged_path, ptail, ntail, splits_path, out_path = sys.argv[1:6]
A_ID, B_ID = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (32, 33)
KMAX = int(os.environ.get("VERA_KMAX", "3"))
NRAND = int(os.environ.get("VERA_RAND_N", "40"))
KLDIR = os.environ.get("VERA_KL", "rev")  # rev = PREREG: D_H = KL(p_probed-with-H-swapped || p_unprobed), r = D - D_H
mid = "Qwen/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
lm = None
for f in glob.glob(os.path.join(snap, "*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k == "lm_head.weight" or (k == "model.embed_tokens.weight" and lm is None):
                lm = s.get_tensor(k)
LMC = [lm[i:i + 16384].contiguous() for i in range(0, lm.shape[0], 16384)]  # bf16 chunks, upcast per pass
del lm
DT = torch.bfloat16 if os.environ.get("VERA_DTYPE") == "bf16" else torch.float32  # bf16 = emulate the stack's bf16 tail on CPU
layers = [Qwen3DecoderLayer(cfg, i).to(DT).eval() for i in (26, 27)]
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DT)
rot = Qwen3RotaryEmbedding(cfg)
sd = torch.load(merged_path, map_location="cpu")
n = 0
for k, v in sd.items():
    k2 = re.sub(r"^(base_model\.model\.)?(model\.)?", "", k)
    m = re.match(r"layers\.(\d+)\.(.*)", k2)
    if m:
        li = int(m.group(1)); li = li - 26 if li >= 26 else li
        layers[li].state_dict()[m.group(2)].copy_(v.to(DT)); n += 1
    elif k2 == "norm.weight":
        norm.weight.data.copy_(v.to(DT)); n += 1
    else:
        raise SystemExit(f"unexpected key {k}")
assert n == 23, n
HD, NH = cfg.head_dim, cfg.num_attention_heads
assert NH == 16

sp = json.load(open(splits_path))
idx = {s: list(sp[s]) for s in ("dev", "test")}
arr_p = np.load(ptail, mmap_mode="r"); arr_n = np.load(ntail, mmap_mode="r")
assert arr_p.shape == arr_n.shape, (arr_p.shape, arr_n.shape)
print("tail arrays", arr_p.shape, arr_p.dtype, {s: len(v) for s, v in idx.items()}, flush=True)
T = arr_p.shape[1]; pos = torch.arange(T)[None]

def x_of(arr, ii):
    a = np.ascontiguousarray(arr[ii])
    if a.dtype == np.uint16 or a.dtype == np.int16:
        return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16).float()
    return torch.from_numpy(a).float()

# state: mode None | "rec" | "patch"; heads: per block list of head ids to swap; comps: set of component names to swap
# Block 27 is the last block: only its last position reaches the logits, so its records/swaps keep the last position only.
S = {"mode": None, "rec": {}, "src": {}, "heads": {0: [], 1: []}, "comps": set()}
def o_pre(li):
    sl = slice(-1, None) if li == 1 else slice(None)
    def f(mod, args):
        if S["mode"] == "rec":
            S["rec"][("o", li)] = args[0][:, sl].detach().clone(); return None
        if S["mode"] == "patch" and S["heads"][li]:
            x = args[0].clone(); src = S["src"][("o", li)]
            for h in S["heads"][li]:
                x[:, sl, h * HD:(h + 1) * HD] = src[..., h * HD:(h + 1) * HD]
            return (x,)
        return None
    return f
def out_hook(name):
    sl = slice(-1, None) if name.endswith("27") else slice(None)
    def f(mod, args, out):
        t = out[0] if isinstance(out, tuple) else out
        if S["mode"] == "rec":
            S["rec"][name] = t[:, sl].detach().clone(); return None
        if S["mode"] == "patch" and name in S["comps"]:
            new = t.clone(); new[:, sl] = S["src"][name]
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return None
    return f
for li, L in enumerate(layers):
    L.self_attn.o_proj.register_forward_pre_hook(o_pre(li))
    L.self_attn.register_forward_hook(out_hook(f"attn{26+li}"))
    L.mlp.register_forward_hook(out_hook(f"mlp{26+li}"))

@torch.no_grad()
def run(x):
    x = x.to(DT)
    pe = rot(x, pos.expand(x.shape[0], -1))
    for L in layers:
        o = L(x, attention_mask=None, position_ids=pos.expand(x.shape[0], -1), position_embeddings=pe)
        x = o[0] if isinstance(o, tuple) else o
    h = norm(x[:, -1])
    z = torch.cat([h @ (c.float() if DT == torch.float32 else c).T for c in LMC], 1).float()
    return z.argmax(1).numpy(), (z[:, A_ID] - z[:, B_ID]).numpy(), torch.log_softmax(z, 1)

def kl(lp_a, lp_b):  # KL(a || b) per row
    return (lp_a.exp() * (lp_a - lp_b)).sum(1).numpy()

BS = int(os.environ.get("VERA_BS", "64"))
base = {}; t0 = time.time()
for s in ("dev", "test"):
    o = dict(top_n=[], lo_n=[], top_p=[], lo_p=[], kl0=[], kle0=[], src=[], lpn=[], lpp=[])
    for b in range(0, len(idx[s]), BS):
        ii = idx[s][b:b + BS]
        S["mode"] = "rec"; S["rec"] = {}; tn_, ln_, lpn = run(x_of(arr_n, ii)); o["src"].append(dict(S["rec"]))
        S["mode"] = "rec"; S["rec"] = {}; tp_, lp_, lpp = run(x_of(arr_p, ii)); S["mode"] = None
        o.setdefault("src_p", []).append({k: v for k, v in S["rec"].items() if k[0] == "o"})
        o["top_n"].append(tn_); o["lo_n"].append(ln_); o["top_p"].append(tp_); o["lo_p"].append(lp_); o["lpn"].append(lpn); o["lpp"].append(lpp)
        o["kl0"].append(kl(lpn, lpp) if KLDIR == "fwd" else kl(lpp, lpn))
        o["kle0"].append(kl(lpn, lpp))  # D3 'cut moves the output' of swapping EVERYTHING back = KL(p_unprobed || p_probed)
    for k in ("top_n", "lo_n", "top_p", "lo_p", "kl0", "kle0"): o[k] = np.concatenate(o[k])
    base[s] = o
    d = o["lo_p"] - o["lo_n"]
    print(s, "pair both-right", float(np.mean((o["top_p"] == A_ID) & (o["top_n"] == B_ID))),
          "unprobed B", float(np.mean(o["top_n"] == B_ID)), "differs", float(np.mean(o["top_p"] != o["top_n"])),
          "mean shift", float(d.mean()), "mean KL0", float(o["kl0"].mean()), "mean KLeff0", float(o["kle0"].mean()), f"{time.time()-t0:.0f}s", flush=True)

def patch(s, H=(), comps=()):
    lo, top, kls, kle = [], [], [], []
    for bi, b in enumerate(range(0, len(idx[s]), BS)):
        ii = idx[s][b:b + BS]
        S["heads"] = {0: [h for (l, h) in H if l == 0], 1: [h for (l, h) in H if l == 1]}
        S["comps"] = set(comps); S["src"] = base[s]["src"][bi]; S["mode"] = "patch"
        t_, l_, lpx = run(x_of(arr_p, ii)); S["mode"] = None
        lpn = base[s]["lpn"][bi]
        lo.append(l_); top.append(t_); kls.append(kl(lpn, lpx) if KLDIR == "fwd" else kl(lpx, lpn))
        kle.append(kl(lpx, base[s]["lpp"][bi]))  # D3: how far the cut moves the output, KL(p_cut || p_probed)
    lo = np.concatenate(lo); top = np.concatenate(top); kls = np.concatenate(kls); kle = np.concatenate(kle)
    B = base[s]
    r_inf = 1 - (lo - B["lo_n"]).mean() / (B["lo_p"] - B["lo_n"]).mean()
    r_blind = 1 - kls.mean() / B["kl0"].mean()
    r_eff = kle.mean() / B["kle0"].mean()  # share of the full-swap effect, can exceed 1
    keep = float(np.mean((top == A_ID) & (B["top_n"] == B_ID)))  # switch still fires after the swap
    return dict(R_inf=float(r_inf), R_blind=float(r_blind), R_eff=float(r_eff), eff_nats=float(kle.mean()),
                switch_after=keep, cut_A=float(np.mean(top == A_ID)), cut_B=float(np.mean(top == B_ID)),
                cut_other=float(np.mean(~np.isin(top, [A_ID, B_ID]))), r_nats=float(B["kl0"].mean() - kls.mean())), lo

name = lambda H: ",".join(f"{26+l}.{h}" for l, h in H)
heads = [(l, h) for l in (0, 1) for h in range(NH)]
res = {"merged": merged_path, "kl_dir": KLDIR, "base": {}}
for s in ("dev", "test"):
    B = base[s]
    res["base"][s] = dict(n=len(idx[s]), pair_both_right=float(np.mean((B["top_p"] == A_ID) & (B["top_n"] == B_ID))),
                          unprobed_B=float(np.mean(B["top_n"] == B_ID)), probed_A=float(np.mean(B["top_p"] == A_ID)),
                          mean_shift=float((B["lo_p"] - B["lo_n"]).mean()), mean_kl0=float(B["kl0"].mean()),
                          switch_strict=float(np.mean((B["lo_p"] > 0) & (B["lo_n"] < 0))), ties_p=int(np.sum(B["lo_p"] == 0)),
                          ties_n=int(np.sum(B["lo_n"] == 0)), median_abs_lo_p=float(np.median(np.abs(B["lo_p"]))),
                          median_abs_lo_n=float(np.median(np.abs(B["lo_n"]))))
    print("base", s, res["base"][s], flush=True)

# All heads at once, both blocks and per block
for nm, H in (("all32", heads), ("all16_b26", heads[:16]), ("all16_b27", heads[16:])):
    res[nm] = {s: patch(s, H)[0] for s in ("dev", "test")}
    print(nm, res[nm]["test"], f"{time.time()-t0:.0f}s", flush=True)

# Component coalitions on test (and dev) -> Shapley + forward split, on both scores
comps = ["attn26", "mlp26", "attn27", "mlp27"]
coal = {}
for s in ("dev", "test"):
    coal[s] = {}
    for r in range(5):
        for C in itertools.combinations(comps, r):
            coal[s][C] = patch(s, (), C)[0] if C else dict(R_inf=0.0, R_blind=0.0, R_eff=0.0, eff_nats=0.0, switch_after=res["base"][s]["pair_both_right"])
res["coalitions"] = {s: {"+".join(C) or "none": v for C, v in coal[s].items()} for s in coal}
def shapley(s, key):
    out = {}
    for c in comps:
        tot = 0.0; others = [x for x in comps if x != c]
        for r in range(4):
            for C in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(3 - r) / math.factorial(4)
                full = tuple(x for x in comps if x in C or x == c)
                tot += w * (coal[s][full][key] - coal[s][tuple(x for x in comps if x in C)][key])
        out[c] = tot
    out["direct_residual(left)"] = 1 - coal[s][tuple(comps)][key]
    return out
def forward(s, key):
    out, prev = {}, 0.0
    for i, c in enumerate(comps):
        v = coal[s][tuple(comps[: i + 1])][key]; out[c] = v - prev; prev = v
    out["direct_residual(left)"] = 1 - prev
    return out
res["split"] = {s: {key: dict(shapley=shapley(s, key), forward=forward(s, key)) for key in ("R_inf", "R_eff")} for s in coal}
print("split test", json.dumps(res["split"]["test"]), f"{time.time()-t0:.0f}s", flush=True)

# Audits: greedy on dev by each score, scored on test; curve to KMAX
if os.environ.get("VERA_BASE_ONLY"):  # quick mode: base rates, all-32, coalitions/D1 only
    ct = coal["test"]
    res["D1"] = dict(R_all32=ct[("attn26", "attn27")]["R_inf"], R_all4=ct[("attn26", "mlp26", "attn27", "mlp27")]["R_inf"],
                     M_mlp=ct[("attn26", "mlp26", "attn27", "mlp27")]["R_inf"] - ct[("attn26", "attn27")]["R_inf"],
                     switch_all32cut=ct[("attn26", "attn27")]["switch_after"])
    print("D1", res["D1"], flush=True); json.dump(res, open(out_path, "w"), indent=1); print("done", f"{time.time()-t0:.0f}s"); raise SystemExit(0)
singles = {H: patch("dev", [H])[0] for H in heads}
res["dev_singles"] = {name([H]): v for H, v in singles.items()}
for key in [k for k in ("R_blind", "R_inf", "R_eff") if k in os.environ.get("VERA_GREEDY", "R_blind,R_inf").split(",")]:
    chosen, curve = [], []
    for k in range(KMAX if key == "R_blind" else 3):
        pool = [h for h in heads if h not in chosen]
        sc = {h: (singles[h][key] if k == 0 else patch("dev", chosen + [h])[0][key]) for h in pool}
        best = max(pool, key=lambda h: sc[h]); chosen = chosen + [best]
        t = patch("test", chosen)[0]
        curve.append(dict(k=k + 1, heads=name(chosen), dev=sc[best], test=t))
        print("greedy", key, k + 1, name(chosen), round(sc[best], 4), json.dumps(t), f"{time.time()-t0:.0f}s", flush=True)
    res[f"greedy_{key}"] = curve

# Random 3-head null outside the blind top-3, on test
top3n = res["greedy_R_blind"][2]["heads"].split(",")
pool29 = [h for h in heads if name([h]) not in top3n]; assert len(pool29) == 29
g = np.random.default_rng(290); vals = []
for j in range(NRAND):
    H = sorted(pool29[int(i)] for i in g.choice(29, 3, replace=False)); vals.append(patch("test", H)[0])
vb = np.array([v["R_blind"] for v in vals]); vi = np.array([v["R_inf"] for v in vals])
res["random29"] = dict(n=NRAND, top3_blind=top3n, p95_blind=float(np.percentile(vb, 95)), median_blind=float(np.median(vb)),
                       p95_inf=float(np.percentile(vi, 95)), median_inf=float(np.median(vi)),
                       share_inf_ge_030=float(np.mean(vi >= 0.30)), values=vals)
print("random29", {k: v for k, v in res["random29"].items() if k != "values"}, f"{time.time()-t0:.0f}s", flush=True)
# Reverse swap (sufficiency): the UNPROBED pass gets the probed slices of a head set; share of the push recovered
def plant(s, H):
    lo, top = [], []
    for bi, b in enumerate(range(0, len(idx[s]), BS)):
        ii = idx[s][b:b + BS]
        S["heads"] = {0: [h for (l, h) in H if l == 0], 1: [h for (l, h) in H if l == 1]}
        S["comps"] = set(); S["src"] = base[s]["src_p"][bi]; S["mode"] = "patch"
        t_, l_, _ = run(x_of(arr_n, ii)); S["mode"] = None; lo.append(l_); top.append(t_)
    lo = np.concatenate(lo); top = np.concatenate(top); B = base[s]
    return dict(recovered=float((lo - B["lo_n"]).mean() / (B["lo_p"] - B["lo_n"]).mean()), planted_A=float(np.mean(top == A_ID)))
pick = lambda key: [(0 if x.startswith("26.") else 1, int(x.split(".")[1])) for x in res[f"greedy_{key}"][2]["heads"].split(",")]
res["reverse"] = {key: plant("test", pick(key)) for key in ("R_blind", "R_inf")}
res["reverse"]["all32"] = plant("test", heads)
print("reverse", res["reverse"], f"{time.time()-t0:.0f}s", flush=True)
# D1 cells (PREREG clause 3): M_mlp = R(all heads + both MLPs clean) - R(all heads clean); skip leftover = 1 - R(all four clean)
ct = coal["test"]; a = ("attn26", "attn27")
res["D1"] = dict(R_all32=ct[("attn26", "attn27")]["R_inf"], R_all4=ct[("attn26", "mlp26", "attn27", "mlp27")]["R_inf"],
                 M_mlp=ct[("attn26", "mlp26", "attn27", "mlp27")]["R_inf"] - ct[("attn26", "attn27")]["R_inf"],
                 M_mlp26_only=ct[("attn26", "mlp26", "attn27")]["R_inf"] - ct[("attn26", "attn27")]["R_inf"],
                 M_mlp27_only=ct[("attn26", "attn27", "mlp27")]["R_inf"] - ct[("attn26", "attn27")]["R_inf"],
                 skip_left=1 - ct[("attn26", "mlp26", "attn27", "mlp27")]["R_inf"],
                 switch_all32cut=ct[("attn26", "attn27")]["switch_after"], R_blind3=res["greedy_R_blind"][2]["test"]["R_inf"])
res["D1"]["clause3_ratio"] = res["D1"]["M_mlp"] / max(1e-9, 1 - res["D1"]["R_blind3"])
print("D1", res["D1"], flush=True)
res["test_rows"] = dict(ids=idx["test"], top_n=base["test"]["top_n"].tolist(), top_p=base["test"]["top_p"].tolist(),
                        lo_n=base["test"]["lo_n"].tolist(), lo_p=base["test"]["lo_p"].tolist(), kl0=base["test"]["kl0"].tolist())
json.dump(res, open(out_path, "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
