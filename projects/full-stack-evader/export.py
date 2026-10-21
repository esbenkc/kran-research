"""Receipts: copy the small per-state arrays out of data/acts/<state>/ (not committed, 2.7 GB tok.npy each) into
data/states/<state>/ (committed) and write data/acts_sha256.txt for every tok.npy.
  lp.npy (per-token log P), off.npy, meta.json, and for attacked states kl.npy, kl_pos0.npy, kl_pile.npy.
Also data/states/<state>/country_rows.npz: readouts 18, 28 (pre-norm), 29 (post-norm) at the country token of every
statement (float16, 2992 x 1536 each), so a skeptic can rebuild tok18/tok28/tok29 scores without the memmaps.
Run: .venv/bin/python export.py
"""
import hashlib
import shutil

import numpy as np

from common import ACTS, DATA, Acts

OUT = DATA / "states"


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    trows = np.load(DATA / "fits" / "panel.npz")["tok_rows_clean"]
    lines = []
    for d in sorted(p for p in ACTS.iterdir() if (p / "meta.json").exists()):
        o = OUT / d.name
        o.mkdir(parents=True, exist_ok=True)
        for f in ["lp.npy", "off.npy", "meta.json", "kl.npy", "kl_pos0.npy", "kl_pile.npy"]:
            if (d / f).exists():
                shutil.copy2(d / f, o / f)
        A = Acts(d.name)
        np.savez_compressed(o / "country_rows.npz", **{f"R{R}": A.tok[trows, R] for R in (18, 28, 29)})
        lines.append(f"{sha(d / 'tok.npy')}  data/acts/{d.name}/tok.npy")
        print(d.name, flush=True)
    (DATA / "acts_sha256.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
