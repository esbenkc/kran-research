"""Self-test input for reader_sup.py (pipeline check, not a result): the parent's k64 start texts (start_k64.npz,
400 test rows) with a planted one-token cue: in 10% of rows (seeded) the probe-on text gets token 11625 (' solve') at
thought position 1; probe-off text unchanged. A working reader must find a clearly positive C_sup (cap ~10 points)."""
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
d = dict(np.load(H.parent / "data/parent/start_k64.npz"))
m = np.random.default_rng(0).random(len(d["toks_n"])) < 0.10
tl = d["toks_n"].copy(); tl[m, 1] = 11625
d["toks_l"] = tl
np.savez(H / "selftest/selftest_tok.npz", **d)
print("planted share", m.mean())
