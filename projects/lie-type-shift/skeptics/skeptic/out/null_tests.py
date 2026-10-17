"""Skeptic null tests. (1) global label shuffle across all answers -> Delta distribution (what luck gives).
(2) bootstrap-seed stability of the CIs. (3) secondary shares per judge. (4) base flag rate of probes by label."""
import json, random, collections, sys
sys.path.insert(0, "..")
import check
sys.path.insert(0, "..")
from paths import LABELS, ROLL
L = str(LABELS) + "/"
files = {"gpt-4.1-mini": L + "labels_gpt-4.1-mini.jsonl", "claude-sonnet-4-6": L + "labels_claude-sonnet-4-6.jsonl",
         "haiku(mine)": "mylabels.jsonl"}
out = {}
for j, f in files.items():
    rows = check.load(f)
    o = {}
    rng = random.Random(42)
    labs = [r[4] for r in rows]
    d1s, d0s = [], []
    for _ in range(300):
        rng.shuffle(labs)
        sh = [r[:4] + (l,) for r, l in zip(rows, labs)]
        c = check.cells(sh)
        m0 = check.M(c[("base", 0)])[0]
        d1s.append(check.M(c[("lam1", 8)])[0] - m0); d0s.append(check.M(c[("lam0", 8)])[0] - m0)
    d1s.sort(); d0s.sort()
    o["shuffle_null_Delta_lam1_95"] = [round(d1s[7], 3), round(d1s[292], 3)]
    o["shuffle_null_frac_Delta_lam1_ge_0.15"] = sum(x >= .15 for x in d1s) / len(d1s)
    o["ci_by_bootstrap_seed"] = {s: {a: [round(x, 3) for x in check.report(rows, B=1000, seed=s)[f"Delta_{a}"]["ci95"]]
                                     for a in ["lam0", "lam1"]} for s in [1, 2, 3]}
    c = check.cells(rows)
    for cell in [("base", 0), ("lam1", 8), ("lam0", 8)]:
        t = sum(c[cell].values(), collections.Counter()); d = t["F"] + t["O"] + t["M"]
        o[f"{cell[0]}_r{cell[1]}_shares_of_deceptive"] = {k: round(t[k] / d, 3) for k in "FOM"} | {"n_dec": d}
    out[j] = o
# probe flags at base by label (sonnet + mini files carry flags)
for j in ["gpt-4.1-mini", "claude-sonnet-4-6"]:
    g = collections.defaultdict(lambda: [0, 0, 0, 0])
    for l in open(files[j]):
        r = json.loads(l)
        if r["round"] == 0 and r["run"] == "lam0_s0":
            g[r["label"]][0] += 1; g[r["label"]][1] += r["flagA"]; g[r["label"]][2] += r["flagB"]; g[r["label"]][3] += r["flagB2"]
    out[j]["base_flag_rate_A_B_B2_by_label"] = {k: [v[0]] + [round(x / v[0], 2) for x in v[1:]] for k, v in g.items()}
json.dump(out, open("null_tests.json", "w"), indent=1)
print(json.dumps(out, indent=1))
