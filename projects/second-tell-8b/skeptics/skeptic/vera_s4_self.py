# S4 artifact check (Vera): in a LoRA, the stage-2 delta of a tail that continues from its stage-1 adapter is
# (B1+dB)(A1+dA) - B1 A1 = dB A1 + B1 dA + dB dA. The B1 dA term lies inside col(B1) = the stage-1 output subspace of
# THAT tail. So MAIN's dW2 vs MAIN's dW1 can show a small angle by construction. Null: each control's dW2 vs its OWN
# stage-1 delta (same artifact, no switch). Also: share of dW2 (Frobenius) carried by the B1 dA term.
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from vera_s4 import factors, U_lowrank, angles, MODS
d = sys.argv[1]; suf = sys.argv[2]
ld = lambda f: torch.load(f, map_location="cpu", weights_only=False)
print("smallest / mean principal angle (deg), averaged over the 4 modules; dW2 of arm vs dW1 of the arm named")
for s in range(5, 10):
    row = {}
    for arm in ("main", "shuf", "coin"):
        s1 = ld(os.path.join(d, f"{arm}_s{s}_s1.pt")); end = ld(os.path.join(d, f"{arm}_s{s}_{suf}.pt"))
        mins, means, frac = [], [], []
        for mod in MODS:
            B1, A1 = factors(s1, mod); Be, Ae = factors(end, mod)
            U1, _ = U_lowrank([B1], [A1]); U2, _ = U_lowrank([Be, -B1], [Ae, A1])
            mn, me, _ = angles(U1, U2); mins.append(mn); means.append(me)
            dB, dA = Be - B1, Ae - A1
            # ||B1 dA||_F / ||dW2||_F via traces (no dense 4096 x 12288 product)
            def fro2(Bs, As):
                Bc = np.concatenate(Bs, 1); Ac = np.concatenate(As, 0)
                return float(np.trace((Bc.T @ Bc) @ (Ac @ Ac.T)))
            frac.append(np.sqrt(fro2([B1], [dA]) / fro2([Be, -B1], [Ae, A1])))
        row[arm] = (round(np.mean(mins), 2), round(np.mean(means), 2), round(float(np.mean(frac)), 3))
    print(s, "self-angle (min, mean, |B1 dA|/|dW2|):", row)
