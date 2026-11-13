"""Equality check: blind.Scorer.slice_null (48 x 48 shortcut) == slice_null_slow (direct 1024 x 1024 eigh) on the first 5 draws.
Target: no-cue nc_s6 at 16 updates, scored with the u1110 cue (a no-cue adapter, so no cue model is read).
    NC_DEV=cpu python check_slice.py -> data/known_answer_slice_null.json"""
import json
from pathlib import Path

import numpy as np

from blind import Scorer

sc = Scorer()
f = sc.slice_null("raw/runs/nc_s6/step16.pt", "c2_a0_s0")
s = sc.slice_null_slow("raw/runs/nc_s6/step16.pt", "c2_a0_s0", 5)
out = {"fast_first5": f[:5].tolist(), "slow_first5": s.tolist(), "max_abs_diff": float(np.abs(f[:5] - s).max())}
(Path(__file__).parent / "data" / "known_answer_slice_null.json").write_text(json.dumps(out, indent=1))
print("max |fast - slow| =", out["max_abs_diff"])
