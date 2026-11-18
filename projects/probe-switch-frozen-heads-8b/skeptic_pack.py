"""Skeptic pack (Vera #24591): the raw bf16 bits a skeptic needs to rebuild G0 and the last-MLP prediction without the
laptop's raw/ cache (raw/ is not in git). Writes DATA/skeptic_pack/:
  mid{LAST}_79.npy   uint16 [704, 2 cond (0 none, 1 probe), HIDDEN]  block-LAST post-attention residual at the answer
                     position, eager tail on the cached inputs (= the last-MLP arm's training input)
  in{FIRST}_71_79.npy uint16 [704, 2 cond, 2 pos (71, 79), HIDDEN]  cached block-FIRST input (whole compiled graph)
  meta.json          prompt ids (row order), sites, positions, sha256 of the source cache files

    $PY skeptic_pack.py
"""
import hashlib
import json

import numpy as np

from cfgm import CACHE, DATA, FIRST, LAST


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    out = DATA / "skeptic_pack"
    out.mkdir(parents=True, exist_ok=True)
    src = {c: CACHE / f"{c}_eager_mid{LAST}_79.npy" for c in ("none", "probe")}
    tin = {c: CACHE / f"{c}_A_tailin.npy" for c in ("none", "probe")}
    mid = np.stack([np.load(src[c]) for c in ("none", "probe")], 1)
    inp = np.stack([np.load(tin[c], mmap_mode="r")[:, [71, 79]] for c in ("none", "probe")], 1)
    np.save(out / f"mid{LAST}_79.npy", mid)
    np.save(out / f"in{FIRST}_71_79.npy", inp)
    meta = {"prompt_ids": json.load(open(CACHE / "meta_A.json"))["prompt_ids"], "cond": ["none", "probe"],
            "files": {f"mid{LAST}_79.npy": {"shape": list(mid.shape), "site": f"mid{LAST}", "position": 79},
                      f"in{FIRST}_71_79.npy": {"shape": list(inp.shape), "site": f"in{FIRST}", "positions": [71, 79]}},
            "dtype": "uint16 = raw bf16 bits", "source_sha256": {p.name: sha(p) for p in [*src.values(), *tin.values()]}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("skeptic pack", mid.shape, inp.shape)


if __name__ == "__main__":
    main()
