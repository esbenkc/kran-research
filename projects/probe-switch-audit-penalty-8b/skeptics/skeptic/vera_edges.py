"""Vera: how close is each frozen REJECTED clause to flipping? Own code on data/whole rows (128 test prompts).
For every penalised seed: counts behind strict / argmax / fp32 vs the 0.75 line (96 of 128), and a prompt bootstrap
(10,000 resamples) of P(that switch >= 0.75). Also the P100 seed-level R_blind3 over G1-ok seeds only (frozen: a G1
miss voids the run's R values). Run from skeptics/skeptic: python3 vera_edges.py > edges.json
"""
import json, math
import numpy as np

D = "../../data"
out = {}
rng = np.random.default_rng(1)
for r in ["p100_s0", "p100_s1", "p100_s2", "p100_s0_g2048", "p10_s0", "p10_s1", "p10_s2", "ao_s0", "ao_s1", "ao_s2"]:
    d = json.load(open(f"{D}/whole/{r}.json")); rows = [x for x in d["rows"] if x.get("split", "test") == "test"]
    by = {}
    for x in rows: by.setdefault(x["prompt"], {})[int(x["probe"])] = x
    ps = sorted(by); A, B = rows[0]["a_id"], rows[0]["b_id"]
    sw = {"strict": np.array([(by[p][1]["lo"] > 0) and (by[p][0]["lo"] < 0) for p in ps]),
          "argmax": np.array([(by[p][1]["top"] == A) and (by[p][0]["top"] == B) for p in ps]),
          "fp32": np.array([(by[p][1]["lo32"] > 0) and (by[p][0]["lo32"] < 0) for p in ps])}
    n = len(ps); o = dict(n=n)
    for k, v in sw.items():
        idx = rng.integers(0, n, (10000, n))
        o[k] = dict(count=int(v.sum()), need=math.ceil(0.75 * n), p_boot_ge075=float((v[idx].mean(1) >= 0.75).mean()))
    out[r] = o
r3 = {r: json.load(open(f"{D}/audit/{r}.json"))["headline"]["title_cell_blind_top3_removed"] for r in ["p100_s1", "p100_s2"]}
v = np.array(list(r3.values())); h = 12.706 * v.std(ddof=1) / math.sqrt(2)
out["P100_R3_seed_level_G1ok"] = dict(seeds=r3, mean=float(v.mean()), lo=float(v.mean() - h), hi=float(v.mean() + h))
print(json.dumps(out, indent=1))
