"""Build the R29 input for r29_leace.py from Bruno's gate0b output: data/states/clean.npy (n, 20, d) fp16, panel
readouts 10..29 (index 19 = R29 = final-norm output), and data/ref/items.json (lie, decoy, split per item).
fit = split in {line, attack} (= leace_price.py pool l.106); eval = split == eval (l.107). Other items dropped."""
import json
import sys

import numpy as np

P = "/Users/esben/work/theory-1028-instructed-lie/kran-research/projects/instructed-lie-leace/data"
H = np.load(f"{P}/states/clean.npy", mmap_mode="r")
I = json.load(open(f"{P}/ref/items.json"))
assert len(I) == H.shape[0], (len(I), H.shape)
split = np.array([r["split"] for r in I])
keep = np.isin(split, ["line", "attack", "eval"])
np.savez(sys.argv[1] if len(sys.argv) > 1 else "/tmp/iris/r29.npz",
         h=np.asarray(H[keep, 19, :], np.float32), lie=np.array([r["lie"] for r in I])[keep],
         decoy=np.array([r["decoy"] for r in I])[keep], fit=np.isin(split[keep], ["line", "attack"]))
print("n", keep.sum(), "fit", np.isin(split[keep], ["line", "attack"]).sum())
