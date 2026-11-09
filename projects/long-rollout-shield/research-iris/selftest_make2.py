"""Self-test v2 at the real arm's size (pipeline check, not a result). Inputs: this run's start reads
start_real_train.npz (382 prompts x R 4) + start_real_test.npz (200 x R 8). Probe-off text = toks_n; probe-on text =
the SAME text with token 14656 (' Need', the one-token cue's X) written at thought position 1 in 10% of rows (seeded),
so the two texts differ only there (TV 0.10, the cue's size). A working reader must find C_sup clearly > 0 (cap ~10)."""
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
rng = np.random.default_rng(1)
for part in ("train", "test"):
    d = dict(np.load(H.parent / f"data/reads/raw/start_real_{part}.npz"))
    tn = d["toks_n"]
    m = (rng.random(len(tn)) < 0.10) & (tn[:, 1] != 14656)
    tl = tn.copy(); tl[m, 1] = 14656
    out = dict(prompt_idx=d["prompt_idx"], split=d["split"], toks_n=tn, toks_l=tl)
    np.savez(H / f"selftest/selftest2_{part}.npz", **out)
    print(part, len(tn), "planted", int(m.sum()))
