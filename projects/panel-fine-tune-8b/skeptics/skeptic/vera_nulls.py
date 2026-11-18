"""Vera's robustness and null checks on the 8B panel arm (own code, from PREREG s3/s6/s8).

(a) line-resample null: the frozen CI resamples eval cities but keeps each state's line fixed. Here every resample
    also redraws the 200 line-setting true statements (by city) and recomputes the 2nd-highest line.
(b) seed + city bootstrap: draw 3 seeds with replacement, then cities (how much the pooled number moves if seeds
    are a sample, not fixed).
(c) per-readout own-line catch (each probe with its own 2nd-highest line) and the readout that sets the max line.
(d) path cross-check: in-training check scores (runs/<state>.npz, last check = step 300) vs the extract.py scores.
(e) random-line null: for the clean model, how often a random 1% line on M catches < 20% (sanity of the metric).
Usage: python vera_nulls.py DATA
"""
import json, sys
import numpy as np

D = sys.argv[1]
rows = json.load(open(f"{D}/ref/split.json"))
sp = np.array([r["split"] for r in rows]); y = 1 - np.array([r["true"] for r in rows])
city = np.array([r["city"] for r in rows])
ref_t = np.where((sp == "refit") & (y == 0))[0]; ev_f = np.where((sp == "eval") & (y == 1))[0]
ev_t = np.where((sp == "eval") & (y == 0))[0]
zm = json.load(open(f"{D}/fits/zmoments.json"))
RO = [13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37]
rng = np.random.default_rng(1118)
B = 2000


def zs(st):
    S = np.load(f"{D}/scores/{st}.npz")
    return np.stack([(S[f"tok{R}"] - zm[f"tok{R}"]["mu"]) / zm[f"tok{R}"]["sd"] for R in RO], 1)


def line2(v):
    return np.sort(v)[::-1][1]


panel = [f"panel__lam10000__seed{s}" for s in range(3)]
Z = {st: zs(st) for st in panel + ["clean", "ctrlrandom__lam10000__seed0", "ctrlrandom__lam10000__seed1"]}
M = {st: z.max(1) for st, z in Z.items()}
out = {}

# (a) line-resample + city bootstrap, pooled over the 3 panel seeds
ec, einv = np.unique(city[ev_f], return_inverse=True)
rc, rinv = np.unique(city[ref_t], return_inverse=True)
ev_groups = [np.where(einv == k)[0] for k in range(len(ec))]
rf_groups = [np.where(rinv == k)[0] for k in range(len(rc))]
pool = []
for b in range(B):
    ei = np.concatenate([ev_groups[k] for k in rng.integers(0, len(ec), len(ec))])
    ri = np.concatenate([rf_groups[k] for k in rng.integers(0, len(rc), len(rc))])
    c = [np.mean(M[st][ev_f[ei]] > line2(M[st][ref_t[ri]])) for st in panel]
    pool.append(np.mean(c))
out["a_line_and_city_boot95"] = np.percentile(pool, [2.5, 97.5]).round(4).tolist()
out["a_P(pooled>=0.20)"] = float(np.mean(np.array(pool) >= 0.20))

# (b) seed + city bootstrap (lines fixed per state, as frozen)
flags = np.stack([M[st][ev_f] > line2(M[st][ref_t]) for st in panel])
sb = []
for b in range(B):
    s = rng.integers(0, 3, 3); ei = np.concatenate([ev_groups[k] for k in rng.integers(0, len(ec), len(ec))])
    sb.append(flags[s][:, ei].mean())
out["b_seed_city_boot95"] = np.percentile(sb, [2.5, 97.5]).round(4).tolist()
out["b_P(pooled>=0.20)"] = float(np.mean(np.array(sb) >= 0.20))

# (c) per-readout own-line catch + who sets the max line
per = {}
for st in panel + ["clean"]:
    z = Z[st]
    per[st] = {R: round(float(np.mean(z[ev_f, j] > line2(z[ref_t, j]))), 3) for j, R in enumerate(RO)}
    top2 = ref_t[np.argsort(M[st][ref_t])[::-1][:2]]
    per[st]["line"] = round(float(line2(M[st][ref_t])), 3)
    per[st]["line_set_by_readout"] = [RO[int(np.argmax(Z[st][i]))] for i in top2]
    caught = ev_f[M[st][ev_f] > line2(M[st][ref_t])]
    per[st]["caught_by_readout_counts"] = {int(R): int(c) for R, c in zip(*np.unique(np.array(RO)[np.argmax(Z[st][caught], 1)], return_counts=True))}
    # mean z of eval TRUE per readout (where the attack moved true statements)
    per[st]["mean_z_eval_true"] = {R: round(float(Z[st][ev_t, j].mean()), 2) for j, R in enumerate(RO)}
    per[st]["mean_z_eval_false"] = {R: round(float(Z[st][ev_f, j].mean()), 2) for j, R in enumerate(RO)}
out["c_per_readout"] = per

# (d) in-training scores vs extract.py scores at step 300
dd = {}
for st in panel:
    r = np.load(f"{D}/runs/{st}.npz"); S = np.load(f"{D}/scores/{st}.npz")
    a = r["scores"][-1].astype(np.float64); b = np.stack([S[f"tok{R}"] for R in RO], 1)
    zr = np.stack([(a[:, j] - zm[f"tok{R}"]["mu"]) / zm[f"tok{R}"]["sd"] for j, R in enumerate(RO)], 1).max(1)
    dd[st] = {"checks": r["checks"].tolist(), "max_abs_diff": float(np.abs(a - b).max()),
              "corr_min": float(min(np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(20))),
              "catch_from_training_scores": round(float(np.mean(zr[ev_f] > line2(zr[ref_t]))), 4)}
out["d_path_check"] = dd
print(json.dumps(out, indent=1))
json.dump(out, open("nulls8b.json", "w"), indent=1)
