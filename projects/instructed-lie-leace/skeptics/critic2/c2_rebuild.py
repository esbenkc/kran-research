"""critic2 (Claude restart, 2026-10-30) independent rebuild of the headline cells of lie-erasure-cost.
Own code, stdlib + numpy, from saved raw data only. Run from projects/instructed-lie-leace."""
import json, collections
import numpy as np
I = json.load(open("data/ref/items_spread.json"))
ev = [i for i, x in enumerate(I) if x["split"] == "eval"]
# own country list: every item country + every other country name that appears in the generations (checked by hand)
EXTRA = ["the Republic of the Congo", "Moldova", "the Republic of Korea", "Romania", "the Republic of Moldova",
         "Switzerland", "Gabon", "Kuwait"]
SYN = {"South Korea": ["the Republic of Korea"]}   # synonym of the true country = truthful
C = sorted({x["country"] for x in I} | set(EXTRA), key=len, reverse=True)
n = lambda s: s.strip().lower()
def cls(g, true):
    g = n(g)
    if any(g.startswith(n(t)) for t in [true] + SYN.get(true, [])): return "true"
    if any(g.startswith(n(c)) for c in C): return "wrong"
    return "other"
print("== behaviour (greedy gens, 400 lie + 400 truthful eval prompts)")
base = collections.Counter(cls(I[i]["base_gen"], I[i]["country"]) for i in ev if I[i]["lie"])
print("clean lie prompts:", dict(base))
for name in ["lie", "decoy"]:
    b = json.load(open(f"data/runs/{name}_spread__lam10000__seed0__behaviour.json"))
    r = collections.Counter(); h = collections.Counter()
    for i, g in zip(ev, b["gens"]):
        (r if I[i]["lie"] else h)[cls(g, I[i]["country"])] += 1
    print(f"{name} run: lie prompts {dict(r)} -> still lying {r['wrong']/400:.4f} (file B_lie {b['B_lie']}); "
          f"truthful right {h['true']/400:.4f} (file {b['B_honest']}); kl_answer {b['kl_answer']:.4f}")
b = json.load(open("data/runs/lie_spread__lam10000__seed0__behaviour.json"))
for k, i in enumerate(ev):
    if I[i]["city"] == "Lyon":
        print("Lyon lie=%d decoy=%d %r -> %r" % (I[i]["lie"], I[i]["decoy"], I[i]["base_gen"], b["gens"][k]))
print("== retrained output-blind panel, late mean over refits 150..300")
for name in ["lie", "decoy"]:
    d = json.load(open(f"data/runs/{name}_spread__lam10000__seed0.json"))
    T = {t["step"]: t for t in d["traj"]}
    s = [150, 200, 250, 300]
    late = [T[x]["catch_blind"] for x in s]
    slope = np.polyfit(np.array(s, float), late, 1)[0] * 150
    print(f"{name}: catch {late} mean {np.mean(late):.4f}; full {np.mean([T[x]['catch_full'] for x in s]):.4f}; "
          f"O {np.mean([T[x]['O_catch'] for x in s]):.4f}; slope/150 {slope:+.3f}; AUROC {[round(T[x]['auroc_blind'],3) for x in s]}")
    print("  step300 per readout 10..29:", [round(v, 3) for v in T[300]["per_readout_catch_blind"]])
print("== LEACE prices from per-prompt KL at the answer position")
L = lambda k: np.load(f"data/leace/perprompt_{k}_spread.npy")
P, Pd = L("lie_blind").mean(), L("decoy_blind").mean()
Sl = np.mean([L(f"shuf_lie_blind_{k}").mean() for k in range(5)])
Sd = [L(f"shuf_decoy_blind_{k}").mean() for k in range(5)]
print(f"P_lie {P:.4f} P_decoy {Pd:.6f} S_lie {Sl:.5f} S_decoy {np.mean(Sd):.5f} Q {P/max(Pd, np.mean(Sd)):.1f} "
      f"Q_vs_max_shuffle {P/max(Pd, max(Sd)):.1f} lie/S_lie {P/Sl:.1f}")
full = L("lie_full").mean()
print(f"full lie {full:.4f} -> every position {full*800/111542:.5f} (prereg bar 0.06); vs 1027 statement 0.0569 at token: {full/0.0569:.1f}x")
print("== Pinsker/TV bound, 20k random cases")
rng = np.random.default_rng(1); worst = -1
for _ in range(20000):
    k = rng.integers(2, 30); p = rng.dirichlet(np.ones(k) * rng.uniform(.1, 3)); q = rng.dirichlet(np.ones(k) * rng.uniform(.1, 3))
    w = rng.normal(size=k) * rng.uniform(.1, 10); rg = w.max() - w.min(); tv = .5 * np.abs(p - q).sum()
    worst = max(worst, abs(w @ p - w @ q) - rg * tv, rg * tv - rg * np.sqrt(np.sum(p * np.log(p / q)) / 2))
print("max violation (<=0 expected):", worst)
