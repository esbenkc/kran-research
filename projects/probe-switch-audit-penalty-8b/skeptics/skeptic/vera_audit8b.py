"""Vera's own audit rebuild for oracle-1121-audit-penalty-8b (own code, generalised from my 1.7B checker of
oracle-1026-audit-penalty; does not read build's scripts).

Tail = the last two blocks of VERA_MODEL (default Qwen/Qwen3-8B: blocks 34-35) + final RMSNorm + lm_head, base weights
from the HF snapshot, LoRA adapter merged here (W += scale * B @ A), on build's cached tail inputs
(probed and unprobed; bf16 bits as uint16, or float arrays). Device VERA_DEV (cpu|mps), dtype VERA_DTYPE (fp32|bf16).
Head = one query head's head_dim-column slice of o_proj's input, all positions (NH heads per block, 2*NH in the tail).
Swap of a set H = in the probed pass, put the unprobed pass's slices of H in place.
Scores of a swap (both computed from the same patched pass):
  informed  R_inf   = 1 - mean(LO_patched - LO_none) / mean(LO_probe - LO_none),  LO = logit(A) - logit(B), last position
  blind     R_blind = 1 - mean KL(p_none || p_patched) / mean KL(p_none || p_probe)  (full next-token distribution,
            last position, no answer label).  VERA_KL=rev flips to KL(p_patched || p_none).
Audit = greedy top-k picked on DEV by a score, scored on TEST. R(k) curve to k=VERA_KMAX (top-3 status, top-6 row), plus all heads.
Components: coalitions of {attn0, attn1, mlp0, mlp1} (tail-local) output swaps (16), exact Shapley + forward-order split;
the shift left with all four swapped = the probed tail input reaching the logits straight through the skips.
Usage: python vera_audit8b.py ADAPTER_OR_MERGED.pt PROBE_TAIL.npy NONE_TAIL.npy SPLITS.json OUT.json [A_ID B_ID]
  ADAPTER = peft LoRA state dict with tail-local layer ids (layers.0 = first tail block); MERGED = full tail weights.
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
KMAX = int(os.environ.get("VERA_KMAX", "6"))
NRAND = int(os.environ.get("VERA_RAND_N", "40"))
KLDIR = os.environ.get("VERA_KL", "rev")  # rev = PREREG: D_H = KL(p_probed-with-H-swapped || p_unprobed), r = D - D_H
mid = os.environ.get("VERA_MODEL", "Qwen/Qwen3-8B")
DEV = torch.device(os.environ.get("VERA_DEV", "cpu"))
LSCALE = float(os.environ.get("VERA_LORA_SCALE", "2.0"))  # alpha 32 / rank 16
cfg = AutoConfig.from_pretrained(mid); cfg._attn_implementation = "sdpa"
snap = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"])
NL = cfg.num_hidden_layers; TB = (NL - 2, NL - 1)  # tail blocks
want = {f"model.layers.{b}.": i for i, b in enumerate(TB)}
lm = None; emb = None; base_sd = {0: {}, 1: {}}; normw = None
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with safe_open(f, "pt") as st:
        for k in st.keys():
            if k == "lm_head.weight": lm = st.get_tensor(k)
            elif k == "model.embed_tokens.weight" and getattr(cfg, "tie_word_embeddings", False): emb = st.get_tensor(k)
            elif k == "model.norm.weight": normw = st.get_tensor(k)
            else:
                for pre, i in want.items():
                    if k.startswith(pre): base_sd[i][k[len(pre):]] = st.get_tensor(k)
if lm is None: lm = emb
assert lm is not None and normw is not None
DT = torch.bfloat16 if os.environ.get("VERA_DTYPE") == "bf16" else torch.float32  # bf16 = emulate the stack's bf16 tail
LMC = [lm[i:i + 16384].to(DEV).contiguous() for i in range(0, lm.shape[0], 16384)]  # bf16 chunks, upcast per pass
del lm, emb
layers = [Qwen3DecoderLayer(cfg, b).eval() for b in TB]
norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
norm.weight.data.copy_(normw.float())
rot = Qwen3RotaryEmbedding(cfg)
for i, L in enumerate(layers):
    sdL = L.state_dict(); assert set(sdL) == set(base_sd[i]), (set(sdL) ^ set(base_sd[i]))
    for k in sdL: sdL[k].copy_(base_sd[i][k].float())
del base_sd
sd = torch.load(merged_path, map_location="cpu")
n = 0
lora = {}
for k, v in sd.items():
    k2 = re.sub(r"^(base_model\.model\.)?(model\.)?", "", k)
    m = re.match(r"layers\.(\d+)\.(.*)", k2)
    if m:
        li = int(m.group(1)); li = li - TB[0] if li >= TB[0] else li
        assert li in (0, 1), k
        mm = re.match(r"(.*)\.lora_([AB])(\.default)?\.weight$", m.group(2))
        if mm: lora.setdefault((li, mm.group(1)), {})[mm.group(2)] = v.float(); n += 1
        else: layers[li].state_dict()[m.group(2)].copy_(v.float()); n += 1
    elif k2 == "norm.weight":
        norm.weight.data.copy_(v.float()); n += 1
    else:
        raise SystemExit(f"unexpected key {k}")
for (li, mod), ab in lora.items():
    W = layers[li].state_dict()[mod + ".weight"]; W.add_(LSCALE * (ab["B"] @ ab["A"]))
print("loaded", merged_path, "keys", n, "lora modules", len(lora), "tail", TB, flush=True)
assert (len(lora) == 14) or (n == 23), (n, len(lora))
layers = [L.to(DEV, DT) for L in layers]; norm = norm.to(DEV, DT); rot = rot.to(DEV)
HD, NH = cfg.head_dim, cfg.num_attention_heads

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
# The second tail block is the last block: only its last position reaches the logits, so its records/swaps keep the last position only.
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
    sl = slice(-1, None) if name.endswith("1") else slice(None)
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
    L.self_attn.register_forward_hook(out_hook(f"attn{li}"))
    L.mlp.register_forward_hook(out_hook(f"mlp{li}"))

@torch.no_grad()
def run(x):
    x = x.to(DEV, DT); pp = pos.to(DEV).expand(x.shape[0], -1)
    pe = rot(x, pp)
    for L in layers:
        o = L(x, attention_mask=None, position_ids=pp, position_embeddings=pe)
        x = o[0] if isinstance(o, tuple) else o
    h = norm(x[:, -1])
    z = torch.cat([h @ (c.float() if DT == torch.float32 else c).T for c in LMC], 1).float().cpu()
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

name = lambda H: ",".join(f"{TB[l]}.{h}" for l, h in H)
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
for nm, H in (("all", heads), (f"all_b{TB[0]}", heads[:NH]), (f"all_b{TB[1]}", heads[NH:])):
    res[nm] = {s: patch(s, H)[0] for s in ("dev", "test")}
    print(nm, res[nm]["test"], f"{time.time()-t0:.0f}s", flush=True)

# Component coalitions on test (and dev) -> Shapley + forward split, on both scores
comps = ["attn0", "mlp0", "attn1", "mlp1"]
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
    res["D1"] = dict(R_allheads=ct[("attn0", "attn1")]["R_inf"], R_all4=ct[("attn0", "mlp0", "attn1", "mlp1")]["R_inf"],
                     M_mlp=ct[("attn0", "mlp0", "attn1", "mlp1")]["R_inf"] - ct[("attn0", "attn1")]["R_inf"],
                     switch_allheadscut=ct[("attn0", "attn1")]["switch_after"])
    print("D1", res["D1"], flush=True); json.dump(res, open(out_path, "w"), indent=1); print("done", f"{time.time()-t0:.0f}s"); raise SystemExit(0)
singles = {H: patch("dev", [H])[0] for H in heads}
res["dev_singles"] = {name([H]): v for H, v in singles.items()}
for key in [k for k in ("R_blind", "R_inf", "R_eff") if k in os.environ.get("VERA_GREEDY", "R_blind,R_inf").split(",")]:
    chosen, curve = [], []
    for k in range(KMAX if key == "R_blind" else 3):
        pool = [h for h in heads if h not in chosen]
        sc = {h: (singles[h][key] if k == 0 else patch("dev", chosen + [h])[0][key]) for h in pool}
        best = max(pool, key=lambda h: sc[h]); chosen = chosen + [best]
        t, tlo = patch("test", chosen)
        curve.append(dict(k=k + 1, heads=name(chosen), dev=sc[best], test=t))
        if key == "R_blind" and k + 1 in (3, 6): curve[-1]["test_lo"] = tlo.tolist()  # per-prompt A-B log-odds after the cut
        print("greedy", key, k + 1, name(chosen), round(sc[best], 4), json.dumps(t), f"{time.time()-t0:.0f}s", flush=True)
    res[f"greedy_{key}"] = curve

# Random 3-head null outside the blind top-3, on test
top3n = res["greedy_R_blind"][2]["heads"].split(",")
pool29 = [h for h in heads if name([h]) not in top3n]; assert len(pool29) == 2 * NH - 3
g = np.random.default_rng(290); vals = []
for j in range(NRAND):
    H = sorted(pool29[int(i)] for i in g.choice(len(pool29), 3, replace=False)); vals.append(patch("test", H)[0])
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
pick = lambda key: [(0 if x.startswith(f"{TB[0]}.") else 1, int(x.split(".")[1])) for x in res[f"greedy_{key}"][2]["heads"].split(",")]
res["reverse"] = {key: plant("test", pick(key)) for key in ("R_blind", "R_inf")}
res["reverse"]["all"] = plant("test", heads)
print("reverse", res["reverse"], f"{time.time()-t0:.0f}s", flush=True)
# D1 cells (PREREG clause 3): M_mlp = R(all heads + both MLPs clean) - R(all heads clean); skip leftover = 1 - R(all four clean)
ct = coal["test"]; A2 = ("attn0", "attn1"); A4 = ("attn0", "mlp0", "attn1", "mlp1")
res["D1"] = dict(R_allheads=ct[A2]["R_inf"], R_all4=ct[A4]["R_inf"],
                 M_mlp=ct[A4]["R_inf"] - ct[A2]["R_inf"],
                 M_mlp0_only=ct[("attn0", "mlp0", "attn1")]["R_inf"] - ct[A2]["R_inf"],
                 M_mlp1_only=ct[("attn0", "attn1", "mlp1")]["R_inf"] - ct[A2]["R_inf"],
                 skip_left=1 - ct[A4]["R_inf"],
                 switch_allheadscut=ct[A2]["switch_after"], R_blind3=res["greedy_R_blind"][2]["test"]["R_inf"],
                 R_blind6=res["greedy_R_blind"][min(5, len(res["greedy_R_blind"]) - 1)]["test"]["R_inf"])
res["D1"]["clause3_ratio"] = res["D1"]["M_mlp"] / max(1e-9, 1 - res["D1"]["R_blind3"])
print("D1", res["D1"], flush=True)
res["test_rows"] = dict(ids=idx["test"], top_n=base["test"]["top_n"].tolist(), top_p=base["test"]["top_p"].tolist(),
                        lo_n=base["test"]["lo_n"].tolist(), lo_p=base["test"]["lo_p"].tolist(), kl0=base["test"]["kl0"].tolist())
json.dump(res, open(out_path, "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
