"""Vera: the FROZEN clean-fit panel (data/fits/panel.npz, zmoments.json) on a state's ctok:
s_R = w.((x-mu)/sd)+b; z_R = (s_R - zmu_R)/zsd_R; max over readouts 10..29; line = 2nd highest of the state's 200 line-set trues."""
import sys, json, numpy as np
import vera_state as V
pz = np.load(V.P + "data/fits/panel.npz"); zm = json.load(open(V.P + "data/fits/zmoments.json"))
for path in sys.argv[1:]:
    A = np.load(path, mmap_mode="r")
    Z = []
    for j, R in enumerate(V.READ):
        k = f"tok{R}"; X = np.asarray(A[:, j, :], np.float64)
        s = ((X - pz[k + "__mu"]) / pz[k + "__sd"]) @ pz[k + "__w"] + float(pz[k + "__b"])
        Z.append((s - zm[k]["mu"]) / zm[k]["sd"])
    out = V.stats(np.max(np.array(Z), 0))
    print(path.split("/")[-1], {a: (round(b, 4) if isinstance(b, float) else b) for a, b in out.items()})
