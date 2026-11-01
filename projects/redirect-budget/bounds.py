"""Bound rows of PREREG sec 4 (S, Q, K, C, RANK, RANK0) per head and prompt, and the softmax step of sec 5.

Reads ONLY cue-off quantities of the capture (x norms, q/k after QK-norm and RoPE, raw q/k norms, clean attention) and
the measured write norm dx = ||x_on - x_off|| per position (rows S, Q, K, C). It never reads the cue-on attention.
Weights: the measured model's (1.7B: base attention weights = trained, heads frozen; 0.6B: merged trained copy);
RANK/RANK0 and the 'base' lines use base weights and the base cue-off capture.

Notation as PREREG sec 3. From the capture: qc = R_i q (q = sqrt(d) g_q*q_hat), kc_j = R_j k_j (k = sqrt(d) g_k*k_hat).
  w_j        = g_q * R_{j-i} k_j       = g_q * R_{-i} kc_j                     (row Q: dz_j = dq_hat . w_j)
  v_j        = g_k * R_{i-j}(g_q*q_hat) = g_k * R_{-j} qc / sqrt(d)             (row K: dz_j = sqrt(d) v_j . dk_hat_j)
  chord c(rho) = sqrt(2 - 2 sqrt(1 - rho^2)) (rho <= 1), 2 otherwise
  S:  Omega_S = (r/||x||) ||W_Q diag(g)||op ||k_t - k_0|| / sqrt(d), r/||x|| = rho_i at the answer, t = argmax_j ||k_j - k_0||
  Q:  s_q = sqrt(D) ||W_Q diag g||op c(rho_i); c_q = c(s_q / ||q_raw,i||); Omega_Q = c_q max_{j,k} ||w_j - w_k||
  K:  s_j = sqrt(D) ||W_K diag g||op c(rho_j); c_j = c(s_j / ||k_raw,j||); m_j = sqrt(d) ||v_j|| c_j; Omega_K = m_(1) + m_(2)
  C:  Omega_Q + Omega_K + 2 sqrt(d) max|g_q| max|g_k| c_q max_j c_j
  RANK: row K, base weights, base cue-off capture, rho_j = 1 (so c_j = 2 for every key; Vera #11751)
  RANK0: d Omega_K / d rho at rho -> 0 = sum of the two largest sqrt(d)||v_j|| sqrt(D)||W_K diag g||op / ||k_raw,j||
  RANK_Q, RANK_C: rows Q and C with rho = 1 everywhere (base); PROXY = max|g_k| ||g_q * q_hat_i|| (PREREG v1.2, reported)
Softmax step: TV <= T(a, Omega) = max over key sets S of f(a(S), Omega), f(A, W) = A e^W / (A e^W + 1 - A) - A; the 16
largest masses enumerated, the rest (mass M) as the interval [s, s + M]; f is concave with its max at 1/(1 + e^{W/2}).

    python bounds.py         -> data/bounds/<run>.npz (per head x prompt arrays) and data/bounds.json (per-head means)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = Path(__file__).parent
HD = 128
RUNS17 = ["mlp8_none_s0", "mlp8_none_s1", "mlp8_none_s2"]
RUNS06 = {"grader_s0": "src1101/raw/stack/runs/stack_s0_lr0.0001_kl1_merged.pt",
          "grader_s1": "src1101/raw/stack/runs/stack_s1_lr0.0001_kl1_merged.pt",
          "monitor_s0": "src1101/raw/r50/runs/r50_s0_lr0.0001_kl1_at8192_merged.pt",
          "monitor_s1": "src1101/raw/r50/runs/r50_s1_lr0.0001_kl1_at8192_merged.pt"}
NAMES = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.q_norm.weight",
         "self_attn.k_norm.weight")


def chord(rho):
    rho = np.asarray(rho, np.float64)
    return np.where(rho <= 1, np.sqrt(np.clip(2 - 2 * np.sqrt(np.clip(1 - rho ** 2, 0, 1)), 0, None)), 2.0)


def base_weights(model_id, blocks):
    path = Path(snapshot_download(model_id))
    w = {b: {} for b in blocks}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                for b in blocks:
                    pre = f"model.layers.{b}."
                    if k.startswith(pre) and k[len(pre):] in NAMES:
                        w[b][k[len(pre):]] = sf.get_tensor(k).float()
    return w


def trained_weights(path, blocks, first):
    sd = torch.load(HERE / path, map_location="cpu")
    return {b: {n: sd[f"layers.{b - first}.{n}"].float() for n in NAMES} for b in blocks}


def head_terms(w):
    """per block: op norms of W_Q^h diag(g) (16 heads) and W_K^m diag(g) (8 kv maps), gains."""
    out = {}
    for b, d in w.items():
        g = d["input_layernorm.weight"]
        wq = (d["self_attn.q_proj.weight"] * g).view(-1, HD, g.numel())
        wk = (d["self_attn.k_proj.weight"] * g).view(-1, HD, g.numel())
        out[b] = {"opq": torch.linalg.matrix_norm(wq, ord=2).double().numpy(),
                  "opk": torch.linalg.matrix_norm(wk, ord=2).double().numpy(),
                  "gq": d["self_attn.q_norm.weight"].double().numpy(), "gk": d["self_attn.k_norm.weight"].double().numpy()}
    return out


def rot_neg(x, cos, sin, n):
    """R_{-n} x for x [..., 128] with the RoPE tables at offset n (array of offsets broadcast to x[..., 0])."""
    h = HD // 2
    c, s = cos[n], -sin[n]
    rot = np.concatenate([-x[..., h:], x[..., :h]], -1)
    return x * c + rot * s


# ---- softmax step ----
_BITS = ((np.arange(1 << 16)[:, None] >> np.arange(16)[None]) & 1).astype(np.float64)   # [65536, 16]


def tv_bound(a, W, chunk=128):
    """a [N, T] clean attention rows, W [N] logit-oscillation bounds -> [N] sound TV bounds."""
    a = np.asarray(a, np.float64)
    W = np.asarray(W, np.float64)
    srt = -np.sort(-a, 1)
    k = min(16, a.shape[1])
    top, M = srt[:, :k], srt[:, k:].sum(1)
    bits = _BITS[: 1 << k, :k]
    out = np.empty(len(a))
    for s0 in range(0, len(a), chunk):
        sl = slice(s0, s0 + chunk)
        sums = top[sl] @ bits.T                                  # [n, 2^k] subset sums of the top masses
        e = np.exp(np.minimum(W[sl], 700))[:, None]
        astar = 1.0 / (1.0 + np.sqrt(e))
        A = np.clip(astar, sums, sums + M[sl, None])             # best point of each interval (f concave)
        A = np.clip(A, 0, 1)
        f = A * e / (A * e + 1 - A) - A
        out[sl] = f.max(1)
    return out


def rows_for(cap, terms, blocks):
    """per block -> dict of [n, H] arrays: Omega_S, Omega_Q, Omega_K, Omega_C, RANK, RANK0, and clean attention a_off."""
    cos, sin = cap["cos"].astype(np.float64), cap["sin"].astype(np.float64)
    res = {}
    for b in blocks:
        t = terms[b]
        a_off = cap[f"b{b}_attn"][0].astype(np.float64)               # [n,H,T]
        qc = cap[f"b{b}_qhat"][0].astype(np.float64)                  # [n,H,128]  R_i q
        kc = cap[f"b{b}_khat"][0].astype(np.float64)                  # [n,G,T,128] R_j k
        xn = cap[f"b{b}_xnorm"][0].astype(np.float64)                 # [n,T]
        dx = cap[f"b{b}_dx"].astype(np.float64)                       # [n,T]
        qraw = cap[f"b{b}_qrms"][0].astype(np.float64) * np.sqrt(HD)  # [n,H] ||q_raw,i||
        kraw = cap[f"b{b}_krms"][0].astype(np.float64) * np.sqrt(HD)  # [n,G,T]
        n, H, T = a_off.shape
        G = kc.shape[1]
        i = T - 1
        rho = dx / xn                                                  # [n,T]
        kq = np.repeat(kc, H // G, axis=1)                            # [n,H,T,128]
        krq = np.repeat(kraw, H // G, axis=1)                         # [n,H,T]
        opk = np.repeat(t["opk"], H // G)                             # [H]
        opq = t["opq"]
        Dsq = np.sqrt(t["D"])
        # S
        dk0 = np.linalg.norm(kq - kq[:, :, :1], axis=-1)              # [n,H,T]
        Om_S = rho[:, i, None] * opq[None] * dk0.max(-1) / np.sqrt(HD)
        # Q
        w = t["gq"] * rot_neg(kq, cos, sin, i)                        # [n,H,T,128]
        diam = np.zeros((n, H))
        for hh in range(H):
            ww = w[:, hh]                                             # [n,T,128]
            g2 = np.einsum("ntd,nsd->nts", ww, ww)
            sq = np.einsum("ntd,ntd->nt", ww, ww)
            dd = sq[:, :, None] + sq[:, None, :] - 2 * g2
            diam[:, hh] = np.sqrt(np.clip(dd, 0, None).max((1, 2)))
        s_q = Dsq * opq[None] * chord(rho[:, i])[:, None]
        c_q = chord(s_q / qraw)
        Om_Q = c_q * diam
        # K
        offs = np.arange(T)
        v = t["gk"] * rot_neg(qc[:, :, None, :] / np.sqrt(HD), cos, sin, offs)   # [n,H,T,128]
        vn = np.sqrt(HD) * np.linalg.norm(v, axis=-1)                 # sqrt(d)||v_j||  [n,H,T]
        s_j = Dsq * opk[None, :, None] * chord(rho)[:, None, :]
        c_j = chord(s_j / krq)
        m = np.sort(vn * c_j, -1)
        Om_K = m[..., -1] + m[..., -2]
        Om_C = Om_Q + Om_K + 2 * np.sqrt(HD) * np.abs(t["gq"]).max() * np.abs(t["gk"]).max() * c_q * c_j.max(-1)
        # RANK (rho = 1) and RANK0 (first order)
        c1 = chord(Dsq * opk[None, :, None] * chord(np.ones_like(rho))[:, None, :] / krq)
        mr = np.sort(vn * c1, -1)
        RANK = mr[..., -1] + mr[..., -2]
        r0 = np.sort(vn * Dsq * opk[None, :, None] / krq, -1)
        RANK0 = r0[..., -1] + r0[..., -2]
        # PREREG v1.2 reported rows: RANK-Q, RANK-C (rho = 1 everywhere), Vera's proxy max|g_k| ||g_q * q_hat_i||
        cq1 = chord(Dsq * opq[None] * chord(np.ones(n))[:, None] / qraw)
        RANK_Q = cq1 * diam
        RANK_C = RANK_Q + RANK + 2 * np.sqrt(HD) * np.abs(t["gq"]).max() * np.abs(t["gk"]).max() * cq1 * c1.max(-1)
        PROXY = np.abs(t["gk"]).max() * np.linalg.norm(qc, axis=-1) / np.sqrt(HD)
        res[b] = {"S": Om_S, "Q": Om_Q, "K": Om_K, "C": Om_C, "RANK": RANK, "RANK0": RANK0, "RANK_Q": RANK_Q,
                  "RANK_C": RANK_C, "PROXY": PROXY, "a_off": a_off,
                  "c1_saturated_share": float((c1 >= 2 - 1e-12).mean()), "rho_i": rho[:, i]}
    return res


def run(tag, cap_file, wts, blocks, D, model_id_base=None):
    cap = np.load(cap_file)
    terms = head_terms(wts)
    for b in blocks:
        terms[b]["D"] = D
    res = rows_for(cap, terms, blocks)
    arrs, heads = {}, {}
    for b in blocks:
        r = res[b]
        n, H, T = r["a_off"].shape
        A = r["a_off"].reshape(n * H, T)
        for row in ("S", "Q", "K", "C", "RANK", "RANK0", "RANK_Q", "RANK_C", "PROXY"):
            arrs[f"{row}_logit"] = np.concatenate([arrs.get(f"{row}_logit", np.zeros((n, 0))), r[row]], 1)
            if row in ("S", "Q", "K", "C", "RANK"):
                tv = tv_bound(A, r[row].reshape(n * H)).reshape(n, H)
                arrs[f"{row}_tv"] = np.concatenate([arrs.get(f"{row}_tv", np.zeros((n, 0))), tv], 1)
        arrs["a0_off"] = np.concatenate([arrs.get("a0_off", np.zeros((n, 0))), r["a_off"][:, :, 0]], 1)
        for hh in range(H):
            heads[f"{b}.{hh}"] = {"opq": float(terms[b]["opq"][hh]), "opk": float(terms[b]["opk"][hh // 2]),
                                  "c1_saturated_share_block": r["c1_saturated_share"]}
    names = list(heads)
    (HERE / "data" / "bounds").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HERE / "data" / "bounds" / f"{tag}.npz", heads=np.array(names), **arrs)
    for j, h in enumerate(names):
        for k, v in arrs.items():
            heads[h][k + "_mean"] = float(v[:, j].mean())
    return heads


def selfcheck():
    """tv_bound must equal derive_check.T_sound (Felix, 28c8e2f9) row by row."""
    import importlib.util
    import io
    import contextlib
    rng = np.random.default_rng(1)
    a = rng.dirichlet(np.full(40, 0.3), size=64)
    W = rng.uniform(0, 12, size=64)
    src = (HERE / "derive_check.py").read_text()
    ns = {}
    exec(src[src.index("def T_sound"):src.index("def T_brute")], {"np": np, "math": __import__("math")}, ns)
    ref = np.array([ns["T_sound"](a[k], W[k]) for k in range(64)])
    err = float(np.abs(tv_bound(a, W) - ref).max())
    assert err < 1e-12, err
    return err


def main() -> None:
    out = {"bounds_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "tv_selfcheck_err": selfcheck()}
    cap = HERE / "raw" / "capture"
    b17, b06 = list(range(20, 28)), [26, 27]
    w17 = base_weights("Qwen/Qwen3-1.7B", b17)
    w06 = base_weights("Qwen/Qwen3-0.6B", b06)
    out["rank_17"] = run("rank_17", cap / "17_base_dev.npz", w17, b17, 2048)
    for sw in ("grader", "monitor"):
        out[f"rank_06_{sw}"] = run(f"rank_06_{sw}", cap / f"06_base_{sw}.npz", w06, b06, 1024)
    for r in RUNS17:
        out[r] = run(r, cap / f"17_{r}.npz", w17, b17, 2048)
    for r, p in RUNS06.items():
        out[r] = run(r, cap / f"06_{r}.npz", trained_weights(p, b06, 26), b06, 1024)
        out[r + "_baseweights"] = run(r + "_baseweights", cap / f"06_{r}.npz", w06, b06, 1024)
    (HERE / "data" / "bounds.json").write_text(json.dumps(out, indent=1))
    print("wrote data/bounds.json", out["bounds_py_sha256"])


if __name__ == "__main__":
    main()
