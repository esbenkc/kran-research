# Null tests for the crawl law (Vera). (1) Shuffled exposure: permute the crawl's per-template exposure profiles across
# contexts (same set of profiles, wrong owners), recompute the law's MAE and title status. If the crawl carries
# information, the true assignment should beat nearly every shuffle. (2) Rate-noise null for the off-task gap:
# resample the parent's frozen rate and the measured runs under 'law exact' to see how often a |d| as large as the
# observed one appears by chance.
import random, collections, json
import numpy as np
import vera_rebuild as R

rows_true = [R.classify(k, v) for k, v in sorted(R.combos.items())]
mae_true = R.mae(rows_true, "law")

# per-context exposure profile: template -> exposure (from the true assignment)
prof = collections.defaultdict(dict)
for (ctx, model, anc), rs in R.combos.items():
    for r in rs: prof[ctx][r["template"]] = r["exposure"]
ctxs = sorted(prof)
rng = random.Random(1109)
maes = []; yes = 0; N = 2000
for _ in range(N):
    perm = ctxs[:]; rng.shuffle(perm)
    mp = dict(zip(ctxs, perm))
    expo = lambda r: prof[mp[r["ctx"]]].get(r["template"], sum(prof[mp[r["ctx"]]].values()) / len(prof[mp[r["ctx"]]]))
    rows = [R.classify(k, v, expo=expo) for k, v in sorted(R.combos.items())]
    for x in rows:  # pred already uses the shuffled exposure via classify
        pass
    maes.append(R.mae(rows, "law"))
    miss, off, hits, nh = R.status(rows)
    if miss <= 1 and off <= 1 and nh == 14 and hits >= 12: yes += 1
maes = np.array(maes)
print(f"shuffled-exposure null: law MAE true {mae_true:.4f}; shuffled mean {maes.mean():.4f} (5th pct {np.percentile(maes,5):.4f}); "
      f"P(shuffled MAE <= true) = {(maes <= mae_true + 1e-12).mean():.4f} (N={N}); shuffles reaching Yes: {yes}/{N}")

# (2) chance |d| under law exact for each headline combo: simulate measured ~ Bin(n, pred_true) with the frozen rate
# itself redrawn from the parent n; report P(|d_sim| >= |d_obs|).
np_rng = np.random.default_rng(1109)
print("\nchance of the observed |d| if the law were exact (per headline combo):")
for x in rows_true:
    if not x["headline"]: continue
    k0, n0 = R.FROZEN[x["model"]][x["anchor"]]
    p = x["exposure"] * k0 / n0
    sims = np_rng.binomial(x["n"], p, 20000) / x["n"] - x["exposure"] * np_rng.binomial(n0, k0 / n0, 20000) / n0
    print(f"  {x['ctx']:20} {x['model']:22} {x['anchor']:6} d {x['d']:+.3f}  P(|d_sim|>=|d|) = {(np.abs(sims) >= abs(x['d']) - 1e-12).mean():.4f}")
json.dump({"mae_true": mae_true, "shuffled_mae_mean": float(maes.mean()), "p_shuffled_le_true": float((maes <= mae_true + 1e-12).mean()),
           "shuffles_yes": yes, "N": N}, open("vera_null.json", "w"), indent=1)
