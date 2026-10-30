"""Vera's own status arithmetic from her own whole-graph P(X) (vera_stack.py outputs). Own code.

dGAP = mean_i [(P_serv - P_grad)_trained - (P_serv - P_grad)_untrained] over the 200 test prompts.
CI: paired bootstrap over prompts (4,000 resamples, own rng seed 7). p: sign-flip permutation on per-prompt dGAP (20,000).
Also: grading drop, serving drop, 'only' clause (serving drop <= 0.5 x grading drop), T_obs(emu) = dGAP_emu / dGAP_serv.
Usage: python vera_dgap.py BASE.npz name=TRAINED.npz [name=TRAINED.npz ...]
"""
import sys, json
import numpy as np

def load(p):
    z = np.load(p); te = z["split"] == "test"
    return {k[:-3]: z[k][te].astype(np.float64) for k in z.files if k.endswith("_px")}, z["ids"][te]

base, bid = load(sys.argv[1])
rng = np.random.default_rng(7)
out = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    tr, tid = load(path); assert (tid == bid).all()
    g0 = base["comp"] - base["eager"]; g1 = tr["comp"] - tr["eager"]; dg = g1 - g0
    n = len(dg); bs = dg[rng.integers(0, n, (4000, n))].mean(1)
    flips = rng.choice([-1, 1], (20000, n)); perm = np.abs((flips * dg).mean(1)) >= abs(dg.mean())
    gd = base["eager"].mean() - tr["eager"].mean(); sd = base["comp"].mean() - tr["comp"].mean()
    r = dict(P_grad=tr["eager"].mean(), P_serv=tr["comp"].mean(), GAP=g1.mean(), dGAP=dg.mean(),
             ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_signflip=float((perm.sum() + 1) / 20001),
             n_pos=int((dg > 0).sum()), grad_drop=gd, serv_drop=sd, only_ok=bool(sd <= 0.5 * gd))
    if "emu" in tr:
        de = (tr["emu"] - tr["eager"]) - (base["emu"] - base["eager"])
        r["P_emu"] = tr["emu"].mean(); r["dGAP_emu"] = de.mean()
        r["T_obs_emu"] = de.mean() / dg.mean() if dg.mean() >= 0.10 else None
        bse = de[rng.integers(0, n, (4000, n))].mean(1); r["ci_emu"] = [float(np.percentile(bse, 2.5)), float(np.percentile(bse, 97.5))]
    out[name] = r
    print(f"{name:10s} P_grad {r['P_grad']:.4f} P_serv {r['P_serv']:.4f} " + (f"P_emu {r['P_emu']:.4f} " if 'P_emu' in r else "")
          + f"| dGAP {r['dGAP']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p {r['p_signflip']:.1e} pos {r['n_pos']}/{n} "
          f"| grad drop {gd:+.4f} serv drop {sd:+.4f} only {r['only_ok']}"
          + (f" | dGAP_emu {r['dGAP_emu']:+.4f} [{r['ci_emu'][0]:+.4f},{r['ci_emu'][1]:+.4f}] T_obs {r['T_obs_emu']}" if 'P_emu' in r else ""), flush=True)
json.dump(out, open("vera_dgap.json", "w"), indent=1, default=float)
