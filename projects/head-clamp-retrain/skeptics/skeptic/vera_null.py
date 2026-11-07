"""Vera null test: within-pair label flip (sign test). Under H0 (the answer does not depend on WHICH twin carries the
probe), each prompt whose two answers differ is a right-way switch (probe-on A, probe-off B) or a reversed one with
prob 1/2. p = P(Binom(s+n, 1/2) >= s), exact. Also a 20,000-draw Monte Carlo flip null of the strict share itself,
with ties counted (>= obs - 1e-9). Reads final/test_merged_clamp rows of every run and t0/test_clamp."""
import json, glob, collections, math, random, os
P = "/Users/esben/work/theory-1107-near-tie-clamp/kran-research/projects/head-clamp-retrain/data/runs/"
A, B = 32, 33
def pairs(rows):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][probe] = top1
    return list(by.values())
def binom_tail(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
rng = random.Random(1107); out = {}
for f in sorted(glob.glob(P + "*.json")):
    d = json.load(open(f))
    for key in ("final/test_merged_clamp", "t0/test_clamp"):
        pr = pairs(d["rows"][key]); N = len(pr)
        s = sum(p[1] == A and p[0] == B for p in pr); n = sum(p[1] == B and p[0] == A for p in pr)
        nxA = sum(p[1] != A and p[0] == A for p in pr)
        obs = s / N; ge = 0
        for _ in range(20000):
            ss = 0
            for p in pr:
                on, off = (p[1], p[0]) if rng.random() < 0.5 else (p[0], p[1])
                ss += on == A and off == B
            ge += ss / N >= obs - 1e-9
        out[f"{d['name']}/{key}"] = dict(N=N, s=s, n_strict_reverse=n, n_rule=nxA, p_sign=binom_tail(s, s + n), p_mc=(ge + 1) / 20001)
        print(f"{d['name']:<16} {key:<22} s={s:>3} rev={n:>3} p_sign={binom_tail(s, s + n):.3g} p_flipMC={(ge + 1) / 20001:.3g}")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "vera_null.json"), "w"), indent=1)
