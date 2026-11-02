"""Vera: PREREG C4 share removed, own code. share = 1 - (P_shared - P_grad) / (P_own - P_grad), means over 200 test
prompts, chunked graphs, saved models. Also vs the whole graphs: 1 - (P_shared - P_grad) / (P_full_comp - P_full_eager).
Bootstrap over prompts (2,000, rng 3) for the ratio. Usage: python vera_sanity_share.py NAME..."""
import sys, json, numpy as np
rng = np.random.default_rng(3); out = {}
for n in sys.argv[1:]:
    z = np.load(f"vera_san_{n}.npz"); P = {b: z[b + "_px"].astype(np.float64) for b in ("grad", "shared", "own", "full_eager", "full_comp")}
    idx = rng.integers(0, 200, (2000, 200))
    def share(a, b, c, d, i=None):
        m = (lambda x: x.mean()) if i is None else (lambda x: x[i].mean(1))
        return 1 - (m(P[a]) - m(P[b])) / (m(P[c]) - m(P[d]))
    s1, s2 = share("shared", "grad", "own", "grad"), share("shared", "grad", "full_comp", "full_eager")
    b1 = share("shared", "grad", "own", "grad", idx)
    r = {k: float(v.mean()) for k, v in P.items()}
    r.update(share_removed=float(s1), ci=[float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))], share_vs_whole=float(s2),
             gap_own=r["own"] - r["grad"], gap_shared=r["shared"] - r["grad"], gap_whole=r["full_comp"] - r["full_eager"])
    out[n] = r
    print(f"{n:9s} " + " ".join(f"{k} {P[k].mean():.4f}" for k in P) + f" | gap own {r['gap_own']:+.4f} shared {r['gap_shared']:+.4f} "
          f"whole {r['gap_whole']:+.4f} | removed {s1:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}] vs-whole {s2:.3f}")
print("C4 models >= 0.80 removed:", sum(v["share_removed"] >= 0.8 for v in out.values()), "of", len(out))
json.dump(out, open("vera_sanity.json", "w"), indent=1)
