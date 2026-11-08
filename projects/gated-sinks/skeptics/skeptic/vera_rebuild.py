"""Vera: rebuild the headline numbers of gated-sinks from raw npz/json only (no build code imported).

Per counted model (runs/<run>/acts_<tag>.npz + measure_<tag>.json):
  A  recovery from BUILD's saved SAE weights, my own TopK encode + my own MCC  -> compare to json
  B  recovery from MY OWN TopK SAEs (3 seeds) trained on H_train (or a seq split of H_scored)
  C  random-dictionary floor (256 random unit dirs, ReLU), shuffled-label null (z_seq permuted over seqs)
  D  sink mass from A0_sink_block (mean over seqs, heads, q>=8)
Then PREREG s3 law fit + status with vera_status.law_status on (D, A) and on (D, B).
Usage: python vera_rebuild.py --root <project> --runs r1,r2,... [--tag L4_m256_k8] [--own-steps 3000]
"""
import argparse, glob, json, os, time
import numpy as np
from vera_mcc import corr_matrix, mcc, mcc_hungarian, train_topk
from vera_status import law_status


def topk_encode(X, W_enc, b_enc, b_dec, k, sub_bdec):
    pre = ((X - b_dec) if sub_bdec else X) @ W_enc + b_enc
    idx = np.argpartition(-pre, k - 1, axis=1)[:, :k]
    F = np.zeros_like(pre)
    rows = np.arange(pre.shape[0])[:, None]
    F[rows, idx] = np.maximum(pre[rows, idx], 0)
    return F


def rebuild_one(npz_path, json_path, k=8, own_steps=3000, own_seeds=(0, 1, 2), rng_seed=0):
    t0 = time.time()
    Z = np.load(npz_path)
    M = json.load(open(json_path))
    H = Z["H_scored"].astype(np.float32)
    zr = Z["z_scored"].astype(np.float32)
    out = dict(run=M.get("run"), arm=M.get("attn"), seed=M.get("seed"), json_recovery=M.get("recovery"),
               json_recovery_sae=M.get("recovery_sae"), json_sink=M.get("sink_mass"), rows=int(H.shape[0]))
    # A: build's SAEs, both encoder conventions; keep the one that matches json
    seeds = sorted(int(key.split("_")[-1]) for key in Z.files if key.startswith("W_enc_"))
    for sub in (True, False):
        recs, hung = [], []
        for s in seeds:
            F = topk_encode(H, Z[f"W_enc_{s}"], Z[f"b_enc_{s}"], Z[f"b_dec_{s}"], k, sub)
            recs.append(float(mcc(F, zr).mean()))
            hung.append(float(mcc_hungarian(F, zr).mean()))
        out[f"A_build_sae_{'subbdec' if sub else 'plain'}"] = dict(recovery=float(np.mean(recs)), per_seed=recs,
                                                                     hungarian=float(np.mean(hung)))
    # B: own SAEs
    if "H_train" in Z.files:
        Xtr = Z["H_train"].reshape(-1, H.shape[1]).astype(np.float32)
        Xte = H
        zte = zr
        out["B_train_rows"] = "H_train (all positions)"
    else:  # fallback: split scored rows by sequence
        si = Z["seq_idx"]
        tr = si % 2 == 0
        Xtr, Xte, zte = H[tr], H[~tr], zr[~tr]
        out["B_train_rows"] = "even seqs of H_scored (no H_train saved)"
    own = []
    for s in own_seeds:
        enc, _, fvu = train_topk(Xtr, width=256, k=k, steps=own_steps, seed=s)
        F = enc(Xte)
        own.append(dict(seed=s, recovery=float(mcc(F, zte).mean()), hungarian=float(mcc_hungarian(F, zte).mean()),
                        fvu=fvu, dead=float((F.max(0) == 0).mean())))
    out["B_own_sae"] = dict(recovery=float(np.mean([o["recovery"] for o in own])), per_seed=own)
    # C: nulls
    rng = np.random.default_rng(rng_seed)
    D = rng.normal(size=(H.shape[1], 256))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    out["C_random_dict_floor"] = float(mcc(np.maximum(H @ D, 0), zr).mean())
    if "z_seq" in Z.files and "seq_idx" in Z.files:
        perm = rng.permutation(Z["z_seq"].shape[0])
        zsh = Z["z_seq"][perm][Z["seq_idx"]].astype(np.float32)
        s0 = seeds[0]
        F = topk_encode(H, Z[f"W_enc_{s0}"], Z[f"b_enc_{s0}"], Z[f"b_dec_{s0}"], k, True)
        out["C_shuffled_label_build_sae0"] = float(mcc(F, zsh).mean())
    # D: sink mass
    if "A0_sink_block" in Z.files:
        A = Z["A0_sink_block"].astype(np.float64)
        out["D_sink"] = float(A.mean())
        out["D_sink_per_head"] = [float(v) for v in A.mean(axis=(0, 2))]
    side = os.path.join(os.path.dirname(npz_path), "A0_all_blocks_c12000.npz")
    srcs = [(Z, k_) for k_ in ("A0_all_blocks", "A0_blocks") if k_ in Z.files]
    if not srcs and os.path.exists(side):
        Zs = np.load(side)
        srcs = [(Zs, k_) for k_ in Zs.files if k_.startswith("A0")][:1]
    for src, key in srcs:
        if True:
            Ab = src[key].astype(np.float64)  # n x blocks x heads x q
            out["D_sink_blocks"] = [float(v) for v in Ab.mean(axis=(0, 2, 3))]
            out["D_sink_allblock_mean"] = float(Ab.mean())
    # E: linear readout ceiling (my own ridge, even seqs -> odd seqs) and SAE efficiency on the odd seqs
    si = Z["seq_idx"]
    tr, te = si % 2 == 0, si % 2 == 1
    Xa = np.c_[H[tr], np.ones(tr.sum())]
    Wr = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(Xa.shape[1]), Xa.T @ zr[tr])
    P = np.c_[H[te], np.ones(te.sum())] @ Wr
    rc = np.array([np.corrcoef(P[:, j], zr[te, j])[0, 1] for j in range(zr.shape[1])])
    ss = ((zr[te] - P) ** 2).sum(0) / ((zr[te] - zr[te].mean(0)) ** 2).sum(0)
    s0 = seeds[0]
    per_lat = np.mean([mcc(topk_encode(H[te], Z[f"W_enc_{s}"], Z[f"b_enc_{s}"], Z[f"b_dec_{s}"], k, True), zr[te]) for s in seeds], axis=0)
    out["E_readout"] = dict(r2_mean=float(np.mean(1 - ss)), corr_mean=float(np.nanmean(rc)),
                            rec_test=float(per_lat.mean()), efficiency=float(np.nanmean(per_lat / np.maximum(rc, 1e-6))),
                            gap=float(np.nanmean(rc - per_lat)))
    out["sec"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--own-steps", type=int, default=3000)
    p.add_argument("--out", default="vera_rebuild.json")
    a = p.parse_args()
    res = []
    for run in a.runs.split(","):
        d = os.path.join(a.root, "runs", run)
        npzs = sorted(glob.glob(os.path.join(d, f"acts_{a.tag}*.npz" if a.tag else "acts_*.npz")), key=os.path.getmtime)
        npz = npzs[-1]
        js = npz.replace("acts_", "measure_").replace(".npz", ".json")
        r = rebuild_one(npz, js, own_steps=a.own_steps)
        r["npz"] = os.path.relpath(npz, a.root)
        print(json.dumps({k: v for k, v in r.items() if k not in ("D_sink_per_head",)}, default=float)[:900], flush=True)
        res.append(r)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
