"""Per-statement second-order price floor (the power gate) for every probe in data/fits/probes.npz.

For each clean eval FALSE statement i (596) with post-norm country-token state z_i and p_i = softmax(W z_i):
    F_i = W^T diag(p_i) W - m_i m_i^T,  m_i = W^T p_i            (full vocabulary, no truncation)
        computed in the centered form F_i = sum_j p_ij (w_j - m_i)(w_j - m_i)^T (no cancellation);
        head = the K most likely tokens, K = clamp(#(p > 1e-8), 2048, 8192), float64 on CPU; tail = all other tokens,
        float32 on MPS; logits, p and m in float64.
    For a probe s(z) = u . (z - mu) + b with clean 1% FPR line t (2nd-highest of the 200 clean refit TRUE statements):
        D_i = s(z_i) - t                     score drop that hides statement i (only caught statements, D_i > 0)
        q_i(eps) = u^T (F_i + eps I)^{-1} u = sum_j (v_j . u)^2 / (lambda_j + eps),  eps = rel * lambda_max(F_i)
        floor_i = D_i^2 / (2 q_i)            min over delta with u . delta = -D_i of 1/2 delta^T (F_i + eps I) delta
    rel = 1e-8 (frozen, Felix #6024), sensitivity rel = 1e-10 and 1e-6.
Reported per probe: mean floor over caught statements, the same summed over caught and divided by 1192 (all eval
statements: the unit of the fine-tune's eval_kl_country_pos), the ratio to unproj, and the per-state share of u in
eigen-directions with lambda < 1e-6 lambda_max(F_i) (and < 1e-4).
Outputs: data/floor/floor.json (summary), data/floor/spectra.npz (lambda (596, d) float64, coef (596, n_probe, d)
float32 = V^T u, D (n_probe, 596), names, rows) - local, sha256 in data/floor/sha256.txt.
Usage: PY floor.py
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "floor"
HEAD = 8192
VCH = 16384
RELS = [1e-10, 1e-8, 1e-6]
REL_FROZEN = 1e-8


def kth_line(neg, fpr=0.01):
    s = np.sort(neg)[::-1]
    return float(s[max(1, int(np.floor(fpr * (len(s) + 1)))) - 1])


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    dev = torch.device("mps")
    from fisher import unembedding
    split = json.loads((ROOT / "data/ref/split.json").read_text())
    sp = np.array([r["split"] for r in split])
    y = np.array([1 - r["true"] for r in split])
    evf = np.where((sp == "eval") & (y == 1))[0]
    n_ev = int((sp == "eval").sum())
    line_i = np.where((sp == "refit") & (y == 0))[0]
    Z = np.load(ROOT / "data/clean/z_post.npy").astype(np.float64)
    P = np.load(ROOT / "data/fits/probes.npz")
    names = sorted({k.split("__")[0] for k in P.files})
    mu = {n: P[f"{n}__mu"].astype(np.float64) for n in names}
    u = {n: P[f"{n}__w"].astype(np.float64) / P[f"{n}__sd"].astype(np.float64) for n in names}
    D = np.zeros((len(names), len(evf)))
    for a, n in enumerate(names):
        s = (Z - mu[n]) @ u[n]
        D[a] = s[evf] - kth_line(s[line_i])
    Uall = torch.from_numpy(np.stack([u[n] for n in names], 1))       # (d, n_probe) float64
    W, _ = unembedding(as_float=False)                                # stored bf16 weights (exact), 0.47 GB on CPU
    assert W.dtype == torch.bfloat16
    Wd = torch.empty(W.shape, dtype=torch.float32, device=dev)        # float32 copy on MPS for the tail, chunked
    for a0 in range(0, W.shape[0], VCH):
        Wd[a0:a0 + VCH] = W[a0:a0 + VCH].to(dev).float()
    V, d = W.shape
    lam = np.zeros((len(evf), d))
    coef = np.zeros((len(evf), len(names), d), np.float32)
    tail_mass, heads = [], []
    B = 32
    for b0 in range(0, len(evf), B):
        rows_b = evf[b0:b0 + B]
        Zb = torch.from_numpy(Z[rows_b])                              # (b, d) float64
        lg = torch.empty(len(rows_b), V, dtype=torch.float64)
        for a0 in range(0, V, VCH):
            lg[:, a0:a0 + VCH] = Zb @ W[a0:a0 + VCH].double().T
        P_b = torch.softmax(lg, -1)
        del lg
        M_b = torch.zeros(len(rows_b), d, dtype=torch.float64)
        for a0 in range(0, V, VCH):
            M_b += P_b[:, a0:a0 + VCH] @ W[a0:a0 + VCH].double()      # m = W^T p (float64)
        for r in range(len(rows_b)):
            n = b0 + r
            p, m = P_b[r], M_b[r]
            # centered form F = sum_j p_j (w_j - m)(w_j - m)^T (identical to W^T diag(p) W - m m^T, no cancellation)
            K = int(min(HEAD, max(2048, int((p > 1e-8).sum()))))
            order = torch.argsort(p, descending=True)
            hi = order[:K]
            Xh = (W[hi].double() - m) * p[hi].sqrt()[:, None]
            F = Xh.T @ Xh                                             # head: float64
            pt = p.clone()
            pt[hi] = 0.0
            pt = pt.float().to(dev)
            md = m.float().to(dev)
            Ft = torch.zeros(d, d, device=dev)
            for a0 in range(0, V, VCH):
                Xc = (Wd[a0:a0 + VCH] - md) * pt[a0:a0 + VCH].sqrt()[:, None]
                Ft += Xc.T @ Xc                                       # tail: float32 on MPS
            F = F + Ft.cpu().double()
            F = (F + F.T) / 2
            e, Vec = torch.linalg.eigh(F)
            lam[n] = e.numpy()
            coef[n] = (Vec.T @ Uall).T.numpy().astype(np.float32)
            tail_mass.append(float(p[order[K:]].sum()))
            heads.append(K)
            del Ft, pt, Xh
        print(json.dumps({"n": b0 + len(rows_b), "tail_mass_max": max(tail_mass), "lmax_last": float(e[-1]),
                          "sec": round(time.time() - t0, 1)}), flush=True)
    lmax = lam[:, -1]
    lamc = np.clip(lam, 0, None)
    summ = {"n_eval_false": int(len(evf)), "n_eval": n_ev, "head_tokens": {"min": min(heads), "max": max(heads), "rule": "K = clamp(#(p > 1e-8), 2048, 8192)"},
            "tail_mass": {"max": max(tail_mass), "median": float(np.median(tail_mass))},
            "lmax": {"min": float(lmax.min()), "median": float(np.median(lmax)), "max": float(lmax.max())},
            "neg_eig_min_rel": float((lam[:, 0] / lmax).min()), "rel_frozen": REL_FROZEN, "probes": {}}
    for a, nme in enumerate(names):
        c2 = coef[:, a, :].astype(np.float64) ** 2
        caught = D[a] > 0
        row = {"n_caught": int(caught.sum()), "clean_catch_eval": float(caught.mean()),
               "D_median_caught": float(np.median(D[a][caught])) if caught.any() else None,
               "flat_share_1e-6_median": float(np.median((c2 * (lamc < 1e-6 * lmax[:, None])).sum(1) / c2.sum(1))),
               "flat_share_1e-4_median": float(np.median((c2 * (lamc < 1e-4 * lmax[:, None])).sum(1) / c2.sum(1)))}
        for rel in RELS:
            q = (c2 / (lamc + rel * lmax[:, None])).sum(1)
            fl = np.where(caught, D[a] ** 2 / (2 * q), 0.0)
            row[f"rel{rel:g}"] = {"floor_mean_caught": float(fl[caught].mean()) if caught.any() else 0.0,
                                  "floor_median_caught": float(np.median(fl[caught])) if caught.any() else 0.0,
                                  "floor_per_eval_statement": float(fl.sum() / n_ev)}
        summ["probes"][nme] = row
    for rel in RELS:
        k = f"rel{rel:g}"
        base = summ["probes"]["unproj"][k]["floor_per_eval_statement"]
        for nme in names:
            summ["probes"][nme][k]["ratio_to_unproj"] = summ["probes"][nme][k]["floor_per_eval_statement"] / base if base > 0 else None
    summ["seconds"] = round(time.time() - t0, 1)
    np.savez(OUT / "spectra.npz", lam=lam, coef=coef, D=D, names=np.array(names), rows=evf)
    h = hashlib.sha256(open(OUT / "spectra.npz", "rb").read()).hexdigest()
    (OUT / "sha256.txt").write_text(f"{h}  spectra.npz\n")
    (OUT / "floor.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({n: {k: v for k, v in r.items() if k in ("clean_catch_eval", "flat_share_1e-6_median", "rel1e-08")}
                      for n, r in summ["probes"].items()}, indent=1))


if __name__ == "__main__":
    main()
