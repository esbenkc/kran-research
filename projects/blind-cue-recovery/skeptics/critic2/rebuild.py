"""critic2 (Kasper, Claude fallback) independent rebuild of the headline cells of blind-cue-recovery.
Own code. Reads only data/adapters_qkv26, data/cue_u*.npy, data/h0_rows012_u16.npy, data/summary.json (for diffing)
and the Qwen3-0.6B block-26 input RMSNorm weight from the HF cache.
Definitions from PREREG.md v2: dW = 2 B A; G = sum_m R_m^T R_m; v1 = top eigvec; e1 = sum_p (v1.c_p)^2 / sum_p |c_p|^2;
c_p = RMSNorm26(h0_p + a ||h0_p|| u_p, bf16-cast) - RMSNorm26(h0_p); nulls numpy rng(20261112) N(0,1) (1000,3,1024) row-normed.
Slice null uses MY seed (7), not 20261113, so p95 values are not bit-equal by design."""
import glob, json, os, sys, numpy as np, torch
from safetensors import safe_open
HERE = sys.argv[1] if len(sys.argv) > 1 else "."
AD = os.path.join(HERE, "data/adapters_qkv26")
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors"))[0]
eps = json.load(open(os.path.join(os.path.dirname(snap), "config.json")))["rms_norm_eps"]
with safe_open(snap, "pt") as f:
    lnw = f.get_tensor("model.layers.26.input_layernorm.weight").float().numpy().astype(np.float64)
h0 = torch.from_numpy(np.load(os.path.join(HERE, "data/h0_rows012_u16.npy")).view(np.int16)).view(torch.bfloat16).float()

def rms(x):  # x float32 torch [...,1024]; Qwen3RMSNorm in fp32
    x = x.float(); y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y.double().numpy() * lnw

base = rms(h0)
def cands(U, alpha=0.03125):
    U = torch.from_numpy(np.asarray(U, np.float32))
    on = h0[None] + alpha * h0.norm(dim=1, keepdim=True)[None] * U
    return rms(on.to(torch.bfloat16).float()) - base[None]

g = np.random.default_rng(20261112).standard_normal((1000, 3, 1024))
NUL = cands((g / np.linalg.norm(g, axis=-1, keepdims=True)).astype(np.float32))
KEYF = {k: os.path.join(HERE, "data/cue_u.npy" if k == 0 else f"data/cue_u_{1110+k}.npy") for k in range(6)}
TRUE = {k: cands(np.load(KEYF[k])[None])[0] for k in range(6)}
MODS = ["q_proj", "k_proj", "v_proj"]
_c = {}
def load(run, t):
    key = (run, t)
    if key not in _c:
        d = torch.load(f"{AD}/{run}/step{t}.pt", map_location="cpu", weights_only=True)
        p = "base_model.model.layers.0.self_attn."
        _c[key] = {m: (d[p+m+".lora_A.default.weight"].double().numpy(), d[p+m+".lora_B.default.weight"].double().numpy()) for m in MODS}
    return _c[key]
def dW(run, t): return {m: 2.0 * B @ A for m, (A, B) in load(run, t).items()}
def resid(run, t, refs):
    W = dW(run, t)
    if refs:
        R = [dW(r, t) for r in refs]
        W = {m: W[m] - sum(r[m] for r in R) / len(R) for m in MODS}
    return W
def top(W, k=1):
    G = sum(W[m].T @ W[m] for m in MODS); ev, V = np.linalg.eigh(G)
    return V[:, ::-1][:, :k], ev[::-1][:k]
def e1(v, C):  # C [...,3,1024]
    return ((C @ v) ** 2).sum(-1) / (C ** 2).sum((-1, -2))
RUN = {k: f"dir_u{1110+k}_a0.03125_s{k}" for k in range(6)}
T = {0: 32, 1: 32, 2: 32, 3: 32, 4: 24, 5: 32}; TH = {k: (8 if k == 4 else 16) for k in range(6)}
def refs(arm, tgt, k, t):  # tgt: 'm{k}' or 'h{i}'
    if arm == "RAW": return []
    if arm == "XS": return [f"nc_s{j}" for j in range(6) if j != k] if tgt[0] == "m" else [f"nc_s{j}" for j in range(1, 6)]
    if arm == "SS": return [f"nc_s{k}"]
    pre = {"SM": "sm", "SMv": "smv", "SM0": "sm0"}[arm]
    return sorted(os.path.basename(p) for p in glob.glob(f"{AD}/{pre}_{tgt}_t{t}_d*"))
rng = np.random.default_rng(7)
def slice_p95(k, t, which=0, n=200):
    ad = load(RUN[k], t); A = np.concatenate([ad[m][0] for m in MODS])  # 48 x 1024
    Q, _ = np.linalg.qr(A.T)  # 1024 x 48
    out = []
    for _ in range(n):
        H = np.zeros((48, 48))
        for m in MODS:
            Am = ad[m][0]; Z = rng.standard_normal((ad[m][1].shape[0], 16)); M = Z @ Am @ Q; H += M.T @ M
        ev, V = np.linalg.eigh(H); v = Q @ V[:, ::-1][:, which]
        out.append(e1(v, TRUE[k][None])[0])
    return float(np.percentile(out, 95))
def cell(arm, k, t, proj=False, sp=None):
    v, _ = top(resid(RUN[k], t, refs(arm, f"m{k}", k, t)), 2)
    v = v[:, 1] if proj else v[:, 0]
    et = e1(v, TRUE[k][None])[0]; en = e1(v, NUL); rank = 1 + int((en > et).sum())
    held = []
    if arm != "SS":
        for i in range(6, 10):
            vh, _ = top(resid(f"nc_s{i}", t, refs(arm, f"h{i}", k, t)), 2)
            held.append(e1(vh[:, 1] if proj else vh[:, 0], TRUE[k][None])[0])
    hm = max(held) if held else -1
    sp = sp if sp is not None else slice_p95(k, t, 1 if proj else 0)
    return dict(e1=float(et), rank=rank, null_max=float(en.max()), held_max=float(hm), slice_p95=sp,
                beats=bool(et > en.max() and et > hm and et > sp), v1=v)
if __name__ == "__main__":
    S = json.load(open(os.path.join(HERE, "data/summary.json")))["cells"]
    out, diffs, n = {}, [], 0
    def put(name, arm, k, t, proj=False, key=None):
        global n
        c = cell(arm, k, t, proj); key = key or str(t)
        ref = S.get(name, {}).get(f"m{k}", {}).get(key)
        out.setdefault(name, {}).setdefault(f"m{k}", {})[str(t)] = {x: y for x, y in c.items() if x != "v1"}
        if ref:
            n += 1
            if ref["rank"] != c["rank"] or ref["beats_chance"] != c["beats"]:
                diffs.append((name, k, t, ref["rank"], c["rank"], ref["beats_chance"], c["beats"]))
        return c
    raw256 = {}
    for k in range(6):
        for arm in ["RAW", "XS", "SS", "SM"]:
            for t in sorted({TH[k], T[k], 256}):
                c = put(arm, arm, k, t)
                if arm == "RAW" and t == 256: raw256[k] = c["rank"]
        for arm in ["SMv", "SM0"]:
            for t in (TH[k], T[k]): put(arm, arm, k, t)
        for t in (TH[k], T[k]): put("E1-RAW", "RAW", k, t, proj=True)
    # XS lemma: |cos(v1 RAW, v1 XS)| at T
    cosxs = {k: abs(float(top(resid(RUN[k], T[k], []))[0][:, 0] @ top(resid(RUN[k], T[k], refs("XS", f"m{k}", k, T[k])))[0][:, 0])) for k in range(6)}
    def cnt(name, which): return sum(out[name][f"m{k}"][str(which(k))]["beats"] for k in range(6))
    rep = {
        "phase0_RAW_rank256": raw256, "phase0_n_rank1": sum(r == 1 for r in raw256.values()),
        "SM_t_h_ranks": [out["SM"][f"m{k}"][str(TH[k])]["rank"] for k in range(6)], "SM_beats_t_h": cnt("SM", TH.get),
        "SM_T_ranks": [out["SM"][f"m{k}"][str(T[k])]["rank"] for k in range(6)], "SM_beats_T": cnt("SM", T.get),
        "SM_256_ranks": [out["SM"][f"m{k}"]["256"]["rank"] for k in range(6)],
        "SS_256_ranks": [out["SS"][f"m{k}"]["256"]["rank"] for k in range(6)],
        "beats_t_h": {a: cnt(a, TH.get) for a in ["RAW", "XS", "SS", "SM", "SMv", "SM0", "E1-RAW"]},
        "beats_T": {a: cnt(a, T.get) for a in ["RAW", "XS", "SS", "SM", "SMv", "SM0", "E1-RAW"]},
        "E1RAW_T_ranks": [out["E1-RAW"][f"m{k}"][str(T[k])]["rank"] for k in range(6)],
        "E1RAW_th_ranks": [out["E1-RAW"][f"m{k}"][str(TH[k])]["rank"] for k in range(6)],
        "XS_cos_at_T": cosxs, "cells_compared": n, "rank_or_beats_diffs": diffs}
    json.dump({"report": rep, "cells": out}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild.json"), "w"), indent=1, default=float)
    print(json.dumps(rep, indent=1, default=float))
