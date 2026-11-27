"""Null tests and robustness for tell-chain-8b (Vera). Uses vera_chain.py (my own code) only.

1. Placebo: run the stage-D status machinery with CTRL-D in the MAIN slot vs FRESH-D (no learned switch in the
   control chain): R should sit near 1 and must not give a yes. Same at stage C with DOSE1-C excluded.
2. Exact sign-flip permutation of the paired ln R per reference (2^n flips): P(mean ln R <= observed | no arm effect).
3. Prompt-level interval: bootstrap the 64 dev prompts (one resample shared by every tail), recompute every T and
   the geometric mean R; 2.5/97.5 percentiles over 2,000 resamples (seed 112711).
4. Leave-one-seed-out status.
Usage: python3 vera_chain_null.py <chain_dir>
"""
import sys, math, itertools
import numpy as np
import vera_chain as vc

d = sys.argv[1]
recs, meta = vc.load(d)
Ts = vc.all_T(recs, meta)

print("== 1. placebo (no learned switch in the numerator)")
for name, num, den in (("CTRL-D vs FRESH-D", "ctrl_D", "fresh_D"), ("CTRL-C vs FRESH-C", "ctrl_C", "fresh_C")):
    rows = vc.ratios(Ts, num, den)
    if len(rows) >= 2:
        m, lo, hi = vc.interval([math.log(x) for _, x in rows])
        verdict = "would be YES (bad)" if hi < vc.LBAR else "no yes"
        print(f"  {name}: geo {math.exp(m):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}] n {len(rows)} -> {verdict}")

print("== 2. exact sign-flip test on paired ln R")
for r in ("ctrl_D", "fresh_D", "ctrl_C", "fresh_C"):
    num = "main_D" if r.endswith("D") else "main_C"
    lnr = np.array([math.log(x) for _, x in vc.ratios(Ts, num, r)])
    if len(lnr) < 2:
        continue
    obs = lnr.mean()
    flips = [np.mean(lnr * np.array(sg)) for sg in itertools.product((1, -1), repeat=len(lnr))]
    p = np.mean([f <= obs + 1e-12 for f in flips])
    print(f"  {num} vs {r}: mean ln R {obs:.3f}, one-sided p {p:.4f} (2^{len(lnr)} flips; minimum {1/2**len(lnr):.4f})")

print("== 3. prompt-level bootstrap (2,000 resamples of the 64 dev prompts)")
rng = np.random.default_rng(112711)
_, how = vc.base_gap(meta, 5, "D"); print(f"  base gap source in resamples: {how}")
if how not in ("lists", "fresh_t0_lists"):
    print("  WARNING: run_meta has only scalar base gaps; the base gap is held fixed in every resample")
boot = {k: [] for k in ("ctrl_D", "fresh_D", "ctrl_C", "fresh_C")}
for b in range(2000):
    idx = rng.integers(0, 64, 64)
    Tb = vc.all_T(recs, meta, idx)
    for r in boot:
        num = "main_D" if r.endswith("D") else "main_C"
        rows = vc.ratios(Tb, num, r)
        if rows:
            boot[r].append(math.exp(np.mean([math.log(x) for _, x in rows])))
for r, v in boot.items():
    if v:
        print(f"  geo R {('main_D' if r.endswith('D') else 'main_C')} vs {r}: [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")

print("== 4. leave-one-seed-out status")
g = vc.gates(recs, meta, Ts)
gates_ok = g["G1"] and g["G2"] and g["PORT"]
for s in vc.SEEDS:
    T2 = {k: v for k, v in Ts.items() if k[1] != s}
    st, res = vc.status(T2, gates_ok)
    print(f"  drop seed {s}: {st}  ctrl_D geo {res['ctrl_D']['geo']:.3f} [{res['ctrl_D']['lo']:.3f}, {res['ctrl_D']['hi']:.3f}]"
          f"  fresh_D geo {res['fresh_D']['geo']:.3f} [{res['fresh_D']['lo']:.3f}, {res['fresh_D']['hi']:.3f}]")
