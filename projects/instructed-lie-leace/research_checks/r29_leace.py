"""Iris pre-run check: exact LEACE KL at the final readout only (R29 = post-norm = logits input), lie vs decoy.

Why: in 1027 (data/leace/price.json) the final readout alone (single_29) cost 0.0495 nats at the edited token vs 0.0569
for the full sequential 10..29 scrub (87%). At R29 the logits are exactly W_U h (tied embeddings, no bias), so the
R29-only price is exact without any further forward pass. Predicts the all-layer ratio lie/decoy before Bruno's run.

Inputs: an npz with h (n, 1536) post-norm answer-token states (fp16/fp32), lie (n,), decoy (n,), split or fit mask.
Eraser: concept_erasure.LeaceFitter default (shrinkage on = the 1027 registered 'lw' setting), fit on fit items,
KL(p_clean || p_erased) at the answer position, fp32 log-softmax over the full 151936 vocab, mean over eval items.
Nulls: shuffled labels (same fit size), random rank-1 direction with the same per-item edit size.
Usage: venv/bin/python r29_leace.py STATES.npz [fit_key]
"""
import json
import sys
import time

import numpy as np
import torch
from concept_erasure import LeaceFitter
from safetensors import safe_open

import os
SNAP = os.environ.get("SNAP", "/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323")  # BASE model (D3, #9256)


def unembed():
    import glob
    for f in glob.glob(SNAP + "/*.safetensors"):
        with safe_open(f, "pt") as s:
            for k in ("lm_head.weight", "model.embed_tokens.weight"):
                if k in s.keys():
                    return s.get_tensor(k).float()
    raise FileNotFoundError("unembedding")


def kl_rows(W, h0, h1, bs=64):
    out = []
    for s in range(0, len(h0), bs):
        lp = torch.log_softmax(h0[s:s + bs] @ W.T, -1)
        lq = torch.log_softmax(h1[s:s + bs] @ W.T, -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out)


def erase(h_fit, z_fit, h_ev):
    f = LeaceFitter(h_fit.shape[1], 1, dtype=torch.float64)
    f.update(h_fit.double(), torch.as_tensor(z_fit, dtype=torch.float64)[:, None])
    e = f.eraser
    return e(h_ev.double()).float(), e


def main():
    t0 = time.time()
    d = np.load(sys.argv[1])
    fitkey = sys.argv[2] if len(sys.argv) > 2 else None
    h = torch.tensor(d["h"], dtype=torch.float32)
    fit = d[fitkey].astype(bool) if fitkey else np.ones(len(h), bool)
    ev = ~fit if fitkey else np.ones(len(h), bool)
    W = unembed()
    res = {"n_fit": int(fit.sum()), "n_eval": int(ev.sum()), "source": sys.argv[1]}
    rng = np.random.default_rng(1028)
    for name in ("lie", "decoy"):
        z = d[name].astype(float)
        he, e = erase(h[fit], z[fit], h[ev])
        kl = kl_rows(W, h[ev], he)
        delta = (he - h[ev]).norm(dim=1)
        zs = rng.permutation(z[fit])
        hs, _ = erase(h[fit], zs, h[ev])
        kls = kl_rows(W, h[ev], hs)
        u = torch.randn(h.shape[1], generator=torch.Generator().manual_seed(1028))
        u = u / u.norm()
        sign = torch.where(torch.rand(int(ev.sum()), generator=torch.Generator().manual_seed(7)) < 0.5, -1.0, 1.0)
        hr = h[ev] + (delta * sign)[:, None] * u[None, :]
        klr = kl_rows(W, h[ev], hr)
        # per-class split of the cost, and the class-mean gap size in the whitened metric is not needed here
        zev = z[ev].astype(bool)
        res[name] = {"kl_mean": float(kl.mean()), "kl_class1": float(kl[torch.tensor(zev)].mean()),
                     "kl_class0": float(kl[torch.tensor(~zev)].mean()), "kl_median": float(kl.median()),
                     "edit_norm_mean": float(delta.mean()), "state_norm_mean": float(h[ev].norm(dim=1).mean()),
                     "null_shuffled_kl": float(kls.mean()), "null_randdir_kl": float(klr.mean())}
        print(name, json.dumps(res[name]), flush=True)
    res["ratio_lie_over_decoy"] = res["lie"]["kl_mean"] / res["decoy"]["kl_mean"]
    res["predicted_all_layer"] = {k: res[k]["kl_mean"] / 0.87 for k in ("lie", "decoy")}
    res["note"] = "predicted_all_layer = R29-only / 0.87 (1027 single_29/registered at the edited token: 0.0495/0.0569)"
    res["seconds"] = time.time() - t0
    out = "r29_leace_result.json"
    json.dump(res, open(out, "w"), indent=1)
    print("ratio", res["ratio_lie_over_decoy"], f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
