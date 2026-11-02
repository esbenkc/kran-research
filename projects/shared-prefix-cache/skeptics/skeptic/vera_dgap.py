"""Vera's own status arithmetic (own code), from vera_shared.py outputs (test split rows only).

dGAP = mean_i [(P_S - P_G)_trained - (P_S - P_G)_untrained] over the 200 test prompts, S = serving build name
(shared | own | full_comp), G = grading build name (grad | full_eager).
CI: paired bootstrap over prompts (4,000 resamples, rng seed 7). Also grading drop and serving drop.
Usage: python vera_dgap.py BASE.npz S G name=TRAINED.npz [...]   -> prints, appends to vera_dgap.jsonl
"""
import sys, json
import numpy as np

def load(p):
    z = np.load(p); te = z["split"] == "test"
    return {k[:-3]: z[k][te].astype(np.float64) for k in z.files if k.endswith("_px")}, z["ids"][te]

base, bid = load(sys.argv[1]); S, G = sys.argv[2], sys.argv[3]
rng = np.random.default_rng(7)
with open("vera_dgap.jsonl", "a") as fo:
    for arg in sys.argv[4:]:
        name, path = arg.split("=", 1)
        tr, tid = load(path); assert (tid == bid).all() and len(tid) == 200
        dg = (tr[S] - tr[G]) - (base[S] - base[G]); n = len(dg)
        bs = dg[rng.integers(0, n, (4000, n))].mean(1)
        r = dict(name=name, S=S, G=G, P_G=tr[G].mean(), P_S=tr[S].mean(), P_G0=base[G].mean(), P_S0=base[S].mean(),
                 dGAP=dg.mean(), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                 grad_drop=base[G].mean() - tr[G].mean(), serv_drop=base[S].mean() - tr[S].mean(), n_pos=int((dg > 0).sum()))
        for extra in ("full_comp", "full_eager", "own", "shared"):
            if extra in tr and extra not in (S, G): r["P_" + extra] = tr[extra].mean()
        fo.write(json.dumps(r, default=float) + "\n")
        print(f"{name:28s} {S}-{G} P_G {r['P_G']:.4f} (base {r['P_G0']:.4f}) P_S {r['P_S']:.4f} (base {r['P_S0']:.4f}) "
              f"dGAP {r['dGAP']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] pos {r['n_pos']}/{n} grad_drop {r['grad_drop']:+.3f} "
              + " ".join(f"{k} {v:.4f}" for k, v in r.items() if k.startswith("P_") and k[2:] in ("full_comp", "full_eager", "own", "shared")),
              flush=True)
