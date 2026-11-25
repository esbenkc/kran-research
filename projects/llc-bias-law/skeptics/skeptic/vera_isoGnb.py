"""Hold gamma/nbeta = 0.1 AND eps*nbeta = 0.02 fixed (so c = eps*nbeta*h fixed; X = eps*nbeta/gamma moves with gamma);
move nbeta over 2 decades with eps*gamma*T = 16 relaxation times. A two-group law in gamma/(nbeta*h) and
eps*nbeta*h predicts the same error on every row. Singular (quartic) directions predict a drift with nbeta/gamma^2."""
import json
from vera_isoX import run
out = []
for r in (1, 8):
    for nb in (100.0, 1000.0, 10000.0):
        gamma = 0.1*nb; eps = 0.02/nb; T = int(16/(eps*gamma))
        o = run(r, eps, gamma, nb, T, seed=2); out.append(o)
        print(f"r={r} nbeta={nb:7.0f} gamma={gamma:6.0f} eps={eps:.0e} T={T} X={o['X']:.1e} rel={o['rel_err']:+.3f} chains={[round(c) for c in o['chains']]}", flush=True)
json.dump(out, open("vera_isoGnb.json", "w"), indent=1)
