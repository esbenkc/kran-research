"""Decoy trainer cache: the update sees ANOTHER puzzle's served footprint on probed rows.

none_rec_tail.npy = the served no-probe input (copy); ref_rec_tail.npy = served no-probe input of prompt i plus the
served footprint of prompt perm(i) (perm = cyclic shift by one over the cache order, a derangement), rounded to
bf16. On positions 1-7 (identical in every prompt) this equals the served leaky input bit for bit.

    PY=../../../env/.venv/bin/python
    $PY mkdecoy.py   -> raw/cache/decoy26_L14/
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
SRC, DST = HERE / "raw/cache/base26_L14", HERE / "raw/cache/decoy26_L14"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    n = np.load(SRC / "none_rec_tail.npy", mmap_mode="r")
    r = np.load(SRC / "ref_rec_tail.npy", mmap_mode="r")
    out = np.zeros_like(n)
    N = n.shape[0]
    perm = np.roll(np.arange(N), -1)  # prompt i gets the footprint of prompt i+1
    for s in range(0, N, 32):
        idx = np.arange(s, min(N, s + 32))
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).float()
        h = f(n[idx]) + f(r[perm[idx]]) - f(n[perm[idx]])
        out[idx] = h.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    np.save(DST / "ref_rec_tail.npy", out)
    shutil.copy(SRC / "none_rec_tail.npy", DST / "none_rec_tail.npy")
    shutil.copy(SRC / "meta.json", DST / "meta.json")
    pre_same = bool((out[:, :, 1:8] == np.asarray(r[:, :, 1:8])).all())
    info = {"perm": "cyclic shift by one", "N": int(N), "prefix1to7_equals_served_leaky": pre_same,
            "share_bits_equal_served_leaky": float((out == np.asarray(r)).mean()),
            "sha_ref": hashlib.sha256(out.tobytes()).hexdigest()[:16]}
    json.dump(info, open(DST / "decoy_info.json", "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()
