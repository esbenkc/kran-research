"""Vera: validity rows S, Q, K, C of PREREG sec 4 per head and prompt in the MEASURED model, the measured logit
oscillation and TV (cue on vs off, answer token), and the sound softmax step T(a, Omega) (PREREG sec 5), own code.

usage: vera_rows.py 06 <copy> | vera_rows.py 17 <run> <cache_dir>
Output: out/rows_<run>.npz with arrays [heads, prompts] for osc, tv, S, Q, K, C and T_S, T_Q, T_K, T_C.
"""
import json, math, os, sys
import numpy as np, torch
from vera_qwen import Weights, attn_parts, block, rms, d, H, HKV

torch.set_grad_enabled(False)
torch.set_num_threads(6)


def bf16(a):
    return torch.from_numpy((np.asarray(a).astype(np.uint32) << 16).view(np.float32).copy())


def chord(r):
    r = torch.as_tensor(r, dtype=torch.float32)
    return torch.where(r < 1, torch.sqrt(torch.clamp(2 - 2 * torch.sqrt(torch.clamp(1 - r * r, min=0)), min=0)), torch.full_like(r, 2.0))


INV = 1.0 / (1e6 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))


def rot(v, n):
    """rotate v [..., N, d] by positions n [N] (broadcast over leading dims)"""
    f = n[:, None].double() * INV[None]
    emb = torch.cat([f, f], -1).float()
    v1, v2 = v[..., : d // 2], v[..., d // 2:]
    return v * emb.cos() + torch.cat([-v2, v1], -1) * emb.sin()


def T_sound(a, Om, top=16):
    """a [B, n] clean attention row, Om [B]. max_S f(a(S), Om), top-16 enumerated, rest as an interval."""
    B, n = a.shape
    s, idx = a.sort(-1, descending=True)
    tk = s[:, :top].double()
    M = s[:, top:].sum(-1).double()
    k = tk.shape[1]
    masks = torch.tensor([[(m >> b) & 1 for b in range(k)] for m in range(2 ** k)], dtype=torch.float64)  # [2^k, k]
    sums = tk @ masks.T  # B, 2^k
    Om = Om.double()[:, None]
    Astar = 1 / (1 + torch.exp(Om / 2))
    A = torch.minimum(torch.maximum(Astar, sums), sums + M[:, None])
    A = A.clamp(0, 1)
    f = A * torch.exp(Om) / (A * torch.exp(Om) + 1 - A) - A
    return f.max(-1).values.float()


def head_rows(w, xon, xoff, i, mats):
    """returns dict of [H, B] tensors for one block"""
    B, T, D = xoff.shape
    Poff, Pon = attn_parts(w, xoff), attn_parts(w, xon)
    rho = ((xon - xoff).norm(dim=-1) / xoff.norm(dim=-1))[:, : i + 1]  # B, i+1
    gq, gk, gin = w["q_norm"], w["k_norm"], w["input_layernorm"]
    js = torch.arange(i + 1)
    res = {k: [] for k in ["osc", "tv", "S", "Q", "K", "C", "T_S", "T_Q", "T_K", "T_C"]}
    for h in range(H):
        kv = h // (H // HKV)
        opQ = torch.linalg.matrix_norm(w["q_proj"][h * d:(h + 1) * d] * gin[None], 2).item()
        opK = torch.linalg.matrix_norm(w["k_proj"][kv * d:(kv + 1) * d] * gin[None], 2).item()
        a_off, a_on = Poff["a"][:, h, i, : i + 1], Pon["a"][:, h, i, : i + 1]
        dz = Pon["z"][:, h, i, : i + 1] - Poff["z"][:, h, i, : i + 1]
        osc = dz.max(-1).values - dz.min(-1).values
        tv = 0.5 * (a_on - a_off).abs().sum(-1)
        q_raw = Poff["q_raw"][:, h, i]                      # B,d
        qn = q_raw.norm(dim=-1)
        qhat = q_raw / qn[:, None]
        k_pre = Poff["k"][:, kv, : i + 1]                  # B, i+1, d (post-norm, pre-RoPE)
        k_raw = Poff["k_raw"][:, kv, : i + 1].norm(dim=-1)  # B, i+1
        # row Q
        c_q = chord(math.sqrt(D) * opQ * chord(rho[:, i]) / qn)
        wj = gq * rot(k_pre, js - i)                        # B, i+1, d
        diam = torch.cdist(wj, wj).amax((-1, -2))
        OQ = c_q * diam
        # row K
        c_j = chord(math.sqrt(D) * opK * chord(rho) / k_raw)  # B, i+1
        u = (gq * qhat)[:, None, :].expand(B, i + 1, d)
        m = math.sqrt(d) * (gk * rot(u, i - js)).norm(dim=-1) * c_j
        OK = m.topk(2, dim=-1).values.sum(-1)
        # row C
        OC = OQ + OK + 2 * math.sqrt(d) * gq.abs().max() * gk.abs().max() * c_q * c_j.max(-1).values
        # row S (scout literal)
        kr = Poff["kr"][:, kv, : i + 1]
        dk = (kr - kr[:, :1]).norm(dim=-1)
        OS = rho[:, i] * opQ * dk.max(-1).values / math.sqrt(d)
        for nm, v in [("osc", osc), ("tv", tv), ("S", OS), ("Q", OQ), ("K", OK), ("C", OC)]:
            res[nm].append(v)
        for nm, v in [("T_S", OS), ("T_Q", OQ), ("T_K", OK), ("T_C", OC)]:
            res[nm].append(T_sound(a_off, v))
    return {k: torch.stack(v) for k, v in res.items()}


def main():
    model, run = sys.argv[1], sys.argv[2]
    out = {}
    if model == "06":
        from vera_06 import SV, COPIES, inputs
        sw, ck = COPIES[run]
        xon, xoff = inputs(sw)
        sd = torch.load(os.path.join(SV, ck), map_location="cpu", weights_only=False)
        ws = [{k.split(".")[-2]: v.float() for k, v in sd.items() if k.startswith(f"layers.{b}.")} for b in range(2)]
        names, i = [26, 27], 33
    else:
        from vera_17 import SRC, to_bf16
        cache = sys.argv[3]
        pr = [json.loads(l) for l in open(os.path.join(SRC, "data/prompts.jsonl"))]
        test = [k for k, p in enumerate(pr) if p["split"] == "test"]
        xon = bf16(np.load(os.path.join(cache, "probe_A_b20.npy"), mmap_mode="r")[test])
        xoff = bf16(np.load(os.path.join(cache, "none_A_b20.npy"), mmap_mode="r")[test])
        W = Weights("1.7B")
        ad = torch.load(os.path.join(SRC, f"raw/runs/{run}_adapter.pt"), map_location="cpu", weights_only=False)
        ws = []
        for b in range(8):
            w = W.layer(20 + b)
            for m in ["gate_proj", "up_proj", "down_proj"]:
                w[m] = to_bf16(w[m] + 2.0 * (ad[f"layers.{b}.mlp.{m}.B"].float() @ ad[f"layers.{b}.mlp.{m}.A"].float()))
            ws.append(w)
        names, i = list(range(20, 28)), 79
    allr = {}
    for b, w in enumerate(ws):
        r = head_rows(w, xon, xoff, i, None)
        for k, v in r.items():
            allr.setdefault(k, []).append(v)
        xon, xoff = block(w, xon)[0], block(w, xoff)[0]
        print(run, "block", names[b], flush=True)
    os.makedirs("out", exist_ok=True)
    np.savez(f"out/rows_{run}.npz", heads=np.array([f"{n}.{h}" for n in names for h in range(H)]),
             **{k: torch.cat(v).numpy() for k, v in allr.items()})


if __name__ == "__main__":
    main()
