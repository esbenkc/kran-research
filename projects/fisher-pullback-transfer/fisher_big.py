"""PRIMARY Fisher (DEVIATIONS.md 15:14): F_common re-estimated from 8000 windows, same estimator as fisher.py.

Windows: pile-10k odd-index documents that are NOT used as retain-train or retain-heldout windows, in seed-1
random order, up to 4 non-overlapping 64-token windows per document, the first 8000 windows. 2 sampled label
draws per window (N_Y). Damping is applied later (predict.py: lambda = 1 x median eigenvalue).
Writes data/fisher/fisher_big.npz and adds a 'fisher_big' block (with sha256 of F_common) to fisher_info.json.
Run: uv run python projects/fisher-pullback-transfer/fisher_big.py
"""
import hashlib
import json
import time

import numpy as np
import torch
from datasets import Dataset

from common import DATA, MAXTOK, device, load_model
from fetch_data import PILE
from fisher import fisher_from_windows

N_WIN = 8000
MAX_PER_DOC = 4


def main():
    t0 = time.time()
    torch.manual_seed(1)
    dev = device()
    tok, model = load_model(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    win = np.load(DATA / "fisher" / "windows.npz")
    used = set(win["retain_doc"].tolist()) | set(win["heldout_doc"].tolist())
    ds = Dataset.from_file(str(PILE))
    order = np.random.default_rng(1).permutation([i for i in range(1, len(ds), 2) if i not in used])
    ids, docs = [], []
    for i in order:
        toks = tok(ds[int(i)]["text"])["input_ids"]
        for k in range(min(MAX_PER_DOC, len(toks) // MAXTOK)):
            ids.append(toks[k * MAXTOK:(k + 1) * MAXTOK])
            docs.append(int(i))
        if len(ids) >= N_WIN:
            break
    ids, docs = ids[:N_WIN], docs[:N_WIN]
    print(f"{len(ids)} windows from {len(set(docs))} docs", flush=True)
    Fc, Ft, h1, h2, _ = fisher_from_windows(model, torch.tensor(ids), dev, t0)
    ev, evec = np.linalg.eigh(Fc)
    sha = hashlib.sha256(np.ascontiguousarray(Fc).tobytes()).hexdigest()
    np.savez_compressed(DATA / "fisher" / "fisher_big.npz", F_common=Fc, F_token=Ft, F_common_half1=h1,
                        F_common_half2=h2, F_common_evals=ev, F_common_evecs=evec, window_docs=np.array(docs))
    tri = np.triu_indices(512)
    info = json.loads((DATA / "fisher" / "fisher_info.json").read_text())
    info["fisher_big"] = {"n_windows": len(ids), "n_docs": len(set(docs)), "n_y_per_window": 2, "device": str(dev),
                          "seconds": time.time() - t0, "sha256_F_common": sha,
                          "median_eig": float(np.median(ev)), "trace": float(ev.sum()), "max_eig": float(ev.max()),
                          "n_eig_below_1e-8": int((ev < 1e-8).sum()), "eff_rank": float(ev.sum() ** 2 / (ev ** 2).sum()),
                          "split_half_corr_upper_tri": float(np.corrcoef(h1[tri], h2[tri])[0, 1])}
    (DATA / "fisher" / "fisher_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info["fisher_big"], indent=1))


if __name__ == "__main__":
    main()
