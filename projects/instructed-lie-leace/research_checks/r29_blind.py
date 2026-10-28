"""Iris pre-run estimate of P_blind at the final readout only (R29): LEACE fit on the output-blind state P_perp h,
the edit projected into the complement (x' = x + P_perp (e(P_perp x) - P_perp x)), exact KL over the full vocab at the
answer position (logits = W_U h, tied). Low memory: W_U kept in bf16, logits in fp32 chunks of 16 prompts.
Also the full-space R29 price on the same code path as a cross-check of r29_leace.py, and a 5-permutation
shuffled-label null for each (PREREG v1.3a N(.)).  Usage: venv/bin/python r29_blind.py [concept ...]
"""
import json
import sys
import time

import numpy as np
import torch
from concept_erasure import LeaceFitter
from safetensors import safe_open

from outblind_gate import P, SNAP, ctok_ids


def kl(W, a, b, bs=16):
    out = []
    for s in range(0, len(a), bs):
        lp = torch.log_softmax((a[s:s + bs].to(torch.bfloat16) @ W.T).float(), -1)
        lq = torch.log_softmax((b[s:s + bs].to(torch.bfloat16) @ W.T).float(), -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.cat(out)


def edit(Xfit, zfit, Xev, Pp):
    f = LeaceFitter(Xfit.shape[1], 1, dtype=torch.float64)
    f.update(Xfit.double(), torch.as_tensor(zfit, dtype=torch.float64)[:, None])
    d = f.eraser(Xev.double()).float() - Xev
    return d if Pp is None else d @ Pp


def main():
    t0 = time.time()
    H = np.load(f"{P}/states/clean.npy", mmap_mode="r")
    I = json.load(open(f"{P}/ref/items.json"))
    split = np.array([r["split"] for r in I])
    fit, ev = np.isin(split, ["line", "attack"]), split == "eval"
    X = torch.tensor(np.asarray(H[:, 19, :], np.float32))
    with safe_open(f"{SNAP}/model.safetensors", "pt") as s:
        W = s.get_tensor("model.embed_tokens.weight")            # bf16, tied lm_head
    U, _ = torch.linalg.qr(W[torch.as_tensor(ctok_ids())].float().T)
    Pp = torch.eye(X.shape[1]) - U @ U.T
    res = {"rank_U": int(U.shape[1])}
    for c in (sys.argv[1:] or ["lie", "decoy"]):
        z = np.array([r[c] for r in I]).astype(float)
        row = {}
        for tag, proj in (("full", None), ("blind", Pp)):
            Xf = X[fit] if proj is None else X[fit] @ proj
            Xe = X[ev] if proj is None else X[ev] @ proj
            d = edit(Xf, z[fit], Xe, proj)
            row[tag] = float(kl(W, X[ev], X[ev] + d).mean())
            nulls = []
            for k in range(5):
                zs = np.random.default_rng(1028 + k).permutation(z[fit])
                nulls.append(float(kl(W, X[ev], X[ev] + edit(Xf, zs, Xe, proj)).mean()))
            row[tag + "_null5"] = float(np.mean(nulls))
            row[tag + "_edit_norm"] = float(d.norm(dim=1).mean())
        res[c] = row
        print(c, json.dumps(row), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open("r29_blind_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
