"""Vera: M1 with the frozen in-loop C per readout (data/fits/C.json tokR), fit on attack 600 true labels, z by line-set trues, max.
argv: ctok.npy tag"""
import sys, json, numpy as np
import vera_state as V
Cj = json.load(open(V.P + "data/fits/C.json"))
A = np.load(sys.argv[1], mmap_mode="r")
per = [V.fit(np.asarray(A[:, j, :], np.float64), V.ATT, V.y, Cj[f"tok{R}"]["C"]) for j, R in enumerate(V.READ)]
out = V.stats(V.zmax(per))
out["per_readout"] = {R: V.stats(per[j], boot=False) for j, R in enumerate(V.READ)}
json.dump(out, open(f"vera_m1_{sys.argv[2]}.json", "w"), indent=1)
print(sys.argv[2], {k: v for k, v in out.items() if k != "per_readout"})
