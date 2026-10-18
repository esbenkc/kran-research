"""Vera: round-choice sensitivity + seed-swap null for the cut ratio (judge label from eval rows)."""
import sys, json, numpy as np
from vera_check import load, rate
pre = sys.argv[1] if len(sys.argv) > 1 else "raw"
D = "/Users/esben/work/oracle-1018-probe-ensemble/kran-research/projects/probe-ensemble/data/raw/"
f = (lambda r: D + f"eval_{r}.jsonl") if pre == "raw" else (lambda r: f"judge2_fresh_{r}_rows.jsonl")
S = {s: load(f(f"lam1_s{s}")) for s in (0, 1)}; E = {s: load(f(f"ens_lam1_s{s}")) for s in (0, 1)}
rounds = sorted(set(S[0]) & set(E[0]) & set(S[1]) & set(E[1]))
print("ratio of cuts by round (cut = r0 - r):")
for r in rounds[1:]:
    rr = [(rate(E[s][0]) - rate(E[s][r])) / (rate(S[s][0]) - rate(S[s][r])) for s in (0, 1)]
    print(f"  r{r}: seed0 {rr[0]:.2f}  seed1 {rr[1]:.2f}")
mean = lambda X, s: np.mean([rate(X[s][0]) - rate(X[s][r]) for r in rounds[1:]])
print("mean cut over rounds 2-8 (area): ratio seed0 %.2f seed1 %.2f" % (mean(E, 0) / mean(S, 0), mean(E, 1) / mean(S, 1)))
cut = lambda X, s: rate(X[s][0]) - rate(X[s][8])
print("seed-swap ratios: ens s1/single s0 %.2f, ens s0/single s1 %.2f" % (cut(E, 1) / cut(S, 0), cut(E, 0) / cut(S, 1)))
print("pooled seeds ratio: %.2f" % ((cut(E, 0) + cut(E, 1)) / (cut(S, 0) + cut(S, 1))))
