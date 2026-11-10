"""Check that the cached block-26 input rows 0-2 ('<|im_start|>user\\n') are identical for all 646 prompts, in the eager
and the compiled cache, and save the real cue: rows 0-2 of the compiled cache (bf16 bits) and compiled minus eager (fp32).
Also draws the planted directions u_p (numpy seed 1110 and 1111; PREREG s2, s10.3).

    /usr/bin/python3 check_rows.py   -> data/check_rows.json, raw/cache/real_rows012_u16.npy, raw/cache/real_diff_rows012.npy,
                                        data/cue_u.npy, data/cue_u_1111.npy
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
C = HERE / "raw" / "cache"


def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


def main():
    out = {}
    rows = {}
    for b in ("eager", "compiled"):
        a = np.load(C / f"{b}_rec_tail.npy", mmap_mode="r")
        r = np.ascontiguousarray(a[:, :3])
        out[b] = {"shape": list(a.shape),
                  "rows012_identical_all_prompts": [bool((r[:, p] == r[0, p]).all()) for p in range(3)],
                  "row_l2": [float(np.linalg.norm(f32(r[0, p]))) for p in range(3)],
                  "row_max_abs": [float(np.abs(f32(r[0, p])).max()) for p in range(3)]}
        rows[b] = r[0]
    d = f32(rows["compiled"]) - f32(rows["eager"])
    out["real_diff"] = {"l2_rows": [float(np.linalg.norm(d[p])) for p in range(3)], "l2": float(np.linalg.norm(d)),
                        "nonzero_elems_rows": [int((d[p] != 0).sum()) for p in range(3)],
                        "share_sq_row0": float((d[0] ** 2).sum() / (d ** 2).sum()) if (d ** 2).sum() > 0 else None,
                        "max_abs_rows": [float(np.abs(d[p]).max()) for p in range(3)]}
    np.save(C / "real_rows012_u16.npy", rows["compiled"])
    np.save(C / "real_diff_rows012.npy", d)
    for seed, fn in ((1110, "cue_u.npy"), (1111, "cue_u_1111.npy")):
        u = np.random.default_rng(seed).standard_normal((3, 1024))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        np.save(HERE / "data" / fn, u.astype(np.float32))
    json.dump(out, open(HERE / "data" / "check_rows.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
