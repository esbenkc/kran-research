"""Mean output Fisher of Qwen2.5-1.5B-Instruct at the last country-name token, in POST-norm coordinates z (readout 29,
the input of the tied unembedding W, V x d, V = 151,936, d = 1536).  For one state z with p = softmax(W z):

    F(z) = W^T (diag p - p p^T) W = W^T diag(p) W - m m^T,   m = W^T p
    KL(p(z) || p(z + delta)) = 1/2 delta^T F(z) delta + O(|delta|^3)

Mean Fisher (used ONLY to choose the curvature subspace): Fbar = W^T diag(pbar) W - mean_i m_i m_i^T over the 1400 clean
fit-pool states (probe+attack+stop splits, true and false), FULL vocabulary, float64 on CPU (logits in float64 too).
Per-state Fishers for the floors: floor.py.
Outputs: data/fisher/Fbar.npy, Fbar_eig.npz (ascending), meta.json; data/clean/z_post.npy (2992, d) float32 clean
post-norm country-token states (committed).
Usage: PY fisher.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "fisher"
VCH = 16384


def unembedding(as_float=True):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    p = Path(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", allow_patterns=["*.safetensors", "*.json"]))
    for f in sorted(p.glob("*.safetensors")):
        with safe_open(str(f), "pt") as g:
            for key in ("lm_head.weight", "model.embed_tokens.weight"):
                if key in g.keys():
                    t = g.get_tensor(key)
                    return (t.float() if as_float else t), key
    raise RuntimeError("no unembedding found")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    split = json.loads((ROOT / "data/ref/split.json").read_text())
    sp = np.array([r["split"] for r in split])
    y = np.array([1 - r["true"] for r in split])
    fit = np.where(np.isin(sp, ["probe", "attack", "stop"]))[0]
    src = ROOT / "data/clean/acts0_1023_r29s0.npy"          # theory-1023 step-0 (clean) country-token states, local only
    if src.exists():
        Z32 = np.asarray(np.load(src, mmap_mode="r")[:, 29], np.float32)   # fp16 -> fp32 exact
        np.save(ROOT / "data/clean/z_post.npy", Z32)
    Z32 = np.load(ROOT / "data/clean/z_post.npy")
    W, key = unembedding()                                   # (V, d) float32 CPU (bf16 weights -> exact in float32)
    V, d = W.shape
    Z = torch.from_numpy(Z32.astype(np.float64))

    def logits64(idx):
        """float64 logits for states idx: float32 MPS matmul is NOT exact enough for p, so do it on CPU in float64."""
        out = torch.empty(len(idx), V, dtype=torch.float64)
        zz = Z[idx]
        for a in range(0, V, VCH):
            out[:, a:a + VCH] = zz @ W[a:a + VCH].double().T
        return out

    # ---- mean Fisher over the fit pool
    pbar = torch.zeros(V, dtype=torch.float64)
    MM = torch.zeros(d, d, dtype=torch.float64)
    for s in range(0, len(fit), 64):
        idx = fit[s:s + 64]
        p = torch.softmax(logits64(idx), -1)
        pbar += p.sum(0)
        m = sum(p[:, a:a + VCH] @ W[a:a + VCH].double() for a in range(0, V, VCH))   # (b, d) = W^T p, chunked
        MM += m.T @ m
    pbar /= len(fit)
    MM /= len(fit)
    G = torch.zeros(d, d, dtype=torch.float64)
    for a in range(0, V, VCH):
        Wc = W[a:a + VCH].double()
        G += (Wc * pbar[a:a + VCH, None]).T @ Wc
    Fbar = (G - MM).numpy()
    Fbar = (Fbar + Fbar.T) / 2
    np.save(OUT / "Fbar.npy", Fbar)
    ev, U = np.linalg.eigh(Fbar)
    np.savez(OUT / "Fbar_eig.npz", evals=ev, evecs=U)
    t_bar = time.time() - t0
    print(json.dumps({"Fbar_top5": ev[-5:].tolist(), "seconds": round(t_bar, 1)}), flush=True)

    meta = {"unembedding_key": key, "V": V, "d": d, "n_fit_states": int(len(fit)),             "Fbar_top5": ev[-5:].tolist(),
            "Fbar_n_above_rel": {f"1e{k}": int((ev > ev[-1] * 10.0 ** k).sum()) for k in range(-12, 1)},
            "seconds_fbar": round(t_bar, 1), "seconds_total": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
