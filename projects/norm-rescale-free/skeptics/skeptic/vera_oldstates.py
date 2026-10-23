"""Vera (1023): null rows from EARLIER attacks that were not built to find the rescale: R (norm-restore share) and energy
share at readout 28 on the 1021 layer-18 attacks and the 1022 panel attacks. Own code; reads only tok.npy/off.npy."""
import json
from pathlib import Path

import numpy as np

from vera_core import HERE, OLD, R_metric, energy

P22 = Path.home() / "work/theory-1022-panel-max-attack/kran-research/projects/panel-fine-tune/data/acts"
P21 = OLD / "data/acts"


def h28(d):
    T = np.load(d / "tok.npy", mmap_mode="r"); o = np.load(d / "off.npy")
    return np.asarray(T[o[1:] - 2, 28], np.float64)


Hc = h28(P21 / "clean")
out = {}
for root in (P21, P22):
    for d in sorted(root.iterdir()):
        if d.name == "clean" or not (d / "tok.npy").exists():
            continue
        Ha = h28(d)
        r = R_metric(Hc, Ha); r.update(energy(Hc, Ha))
        out[d.name] = r
        print(f"{d.name:34s} R={r['R']:+.4f} drop={r['drop_frac']:.3f} E={r['E']:.3f} nr={r['norm_ratio_false']:.3f}/{r['norm_ratio_true']:.3f}")
json.dump(out, open(HERE / "vera_oldstates.json", "w"), indent=1)
