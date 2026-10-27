"""Committed receipts of the country-token states (Kasper #8546): per state, fp16 states of ALL 2992 statements at
readout 19 (the base runs' F2 readout) and at that state's own F2 readout -> data/ctok_git/<state>.npz
(keys R<readout>, shape (2992, 1536)). Enough to refit the F2 probe (and any single-readout probe) from GitHub; the full
20-readout arrays (184 MB each) are rebuilt from the committed adapters with extract.py --lean (~2.5 min per state).
Run: venv/bin/python export_ctok.py
"""
import json

import numpy as np

from common import DATA

READ = list(range(10, 30))
OUT = DATA / "ctok_git"
OUT.mkdir(exist_ok=True)
states = {}
for p in sorted((DATA / "results").glob("*.json")):
    J = json.loads(p.read_text())
    if "monitors" in J:
        states[J["state"]] = J["monitors"]["F2"]["readout"]
G = json.loads((DATA / "leace" / "guard.json").read_text())
for st in ("clean", "leace_registered", "leace_gate"):
    states[st] = G[st]["F2"]["readout"]
for st, f2 in states.items():
    src = DATA / "ctok" / f"{st}.npy"
    if not src.exists():
        continue
    ct = np.load(src, mmap_mode="r")
    np.savez(OUT / f"{st}.npz", **{f"R{R}": np.asarray(ct[:, READ.index(R)], np.float16) for R in sorted({19, int(f2)})})
    print(st, sorted({19, int(f2)}), flush=True)
