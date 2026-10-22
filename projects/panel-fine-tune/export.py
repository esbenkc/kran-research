"""Receipts: copy the small per-state arrays out of data/acts/<state>/ (tok.npy is 2.7 GB per state, not committed) into
data/states/<state>/ (committed): lp.npy (per-token log P), off.npy, meta.json, kl.npy, kl_pos0.npy, kl_pile.npy.
Country-token rows at every panel readout (10..29, fp16, 2992 x 1536 each) go to data/states/<state>/country/R<R>.npy
(local, gitignored: 9 MB each; rebuild from the committed adapter with extract.py); country_norms.npz = L2 norm at readouts 28, 29 (committed). data/acts_sha256.txt: sha256 of
every tok.npy and every country row file.
Run: .venv/bin/python export.py
"""
import hashlib
import shutil

import numpy as np

from common import ACTS, CFG, DATA, Acts

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
        (o / "country").mkdir(parents=True, exist_ok=True)
        if d.name != "clean":
            for f in ["lp.npy", "off.npy", "meta.json", "kl.npy", "kl_pos0.npy", "kl_pile.npy"]:
                if (d / f).exists():
                    shutil.copy2(d / f, o / f)
        A = Acts(d.name)
        for R in CFG["panel"]["readouts"]:
            np.save(o / "country" / f"R{R}.npy", np.asarray(A.tok[trows, R]))
            lines.append(f"{sha(o / 'country' / f'R{R}.npy')}  data/states/{d.name}/country/R{R}.npy")
        np.savez(o / "country_norms.npz", **{f"R{R}": np.linalg.norm(np.asarray(A.tok[trows, R], np.float32), axis=1)
                                             for R in (28, 29)})  # L2 norm at the country token (Hugo #4905)
        lines.append(f"{sha(d / 'tok.npy')}  data/acts/{d.name}/tok.npy")
        print(d.name, flush=True)
    (DATA / "acts_sha256.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
