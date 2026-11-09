# Variant: HIT share computed over HEADLINE (interior) combos only: off-task families + truncation cells.
# MISS counted over all exposed combos. Same worlds as vera_sim_rule.py.
import numpy as np, vera_sim_rule as v
def run(world, n_fit, n_off, reps=4000, shift=0.20, shift_fam=("indexed",), hit_bar=0.80, seed=1):
    rng = np.random.default_rng(seed); res = {"Yes":0,"No":0,"NS":0}
    cs = v.combos(n_fit, n_off)
    for _ in range(reps):
        miss = 0; hit = 0; nh = 0
        for a, anc, fam, f, n in cs:
            k0, n0 = v.PARENT[a][anc]; tp = k0/n0; fk = rng.binomial(n0, tp); t = f*tp
            if world == "shift" and fam in shift_fam:
                t = t + shift if tp < 0.5 else t - shift
            k1 = rng.binomial(n, min(max(t,0),1)); pred = f*fk/n0
            lo, hi = v.newcombe(k1, n, fk, n0, f)
            if lo > 0.10 or hi < -0.10: miss += 1
            if anc == "off" or fam == "trunc":
                nh += 1; hit += abs(k1/n - pred) <= 0.10 + 1e-9
        if miss >= 2: res["No"] += 1
        elif hit/nh >= hit_bar: res["Yes"] += 1
        else: res["NS"] += 1
    return {k: round(x/reps,3) for k,x in res.items()}
for nf, no in ((30,40),(30,60)):
    for bar in (0.80, 0.85):
        print(nf, no, "bar", bar, "exact", run("exact",nf,no,hit_bar=bar),
              "| off-indexed+20", run("shift",nf,no,hit_bar=bar),
              "| off-indexed+15", run("shift",nf,no,shift=0.15,hit_bar=bar))
