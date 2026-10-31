"""Vera robustness part 2: (a) all-strata G (both groups non-empty) leave-one-model-out and per model;
(b) seed halves on the 4 frozen eligible strata held fixed; (c) per-stratum table for the v1.2 'says' variant."""
import json
from collections import defaultdict
import numpy as np
import vera_check as vc
from vera_robust import G_ci, ex
both = lambda s, t: s.sum() >= 1 and (~s).sum() >= 1
print("(a) all strata, leave one model out")
for m in sorted({r["model"] for r in ex}):
    print(f"   drop {m:27s}", G_ci([r for r in ex if r["model"] != m], "says_v13", elig=both))
print("    per model:")
for m in sorted({r["model"] for r in ex}):
    print(f"   only {m:27s}", G_ci([r for r in ex if r["model"] == m], "says_v13", elig=both))
frozen = {("gemini-3.1-flash-lite", "subtle"), ("gemini-3.5-flash-lite", "obvious"), ("gemini-3.5-flash-lite", "subtle"), ("gpt-5.4-mini", "subtle")}
fx = [r for r in ex if (r["model"], r["decoy"]) in frozen]
print("(b) frozen 4 strata held fixed, seed splits")
for name, f in (("seeds 0-34", lambda s: s <= 34), ("seeds 35-69", lambda s: s >= 35), ("even", lambda s: s % 2 == 0), ("odd", lambda s: s % 2 == 1), ("seeds 0-49 (main run)", lambda s: s <= 49), ("seeds 50-69 (D3)", lambda s: s >= 50)):
    print(f"   {name:22s}", G_ci([r for r in fx if f(r["seed"])], "says_v13", elig=both))
print("(c) v1.2 'says' per stratum vs v1.3")
S = defaultdict(list)
for r in ex: S[(r["model"][:14], r["decoy"])].append(r)
for k in sorted(S):
    rs = S[k]
    def g(col):
        a = [r for r in rs if r[col]]; b = [r for r in rs if not r[col]]
        return f"{sum(r['trip'] for r in a)}/{len(a)} vs {sum(r['trip'] for r in b)}/{len(b)}"
    print(f"   {k}: v1.3 {g('says_v13'):16s}  v1.2 {g('says'):16s}  moved v1.3-say->v1.2-not: {sum(1 for r in rs if r['says_v13'] and not r['says'])} (of which tripped {sum(1 for r in rs if r['says_v13'] and not r['says'] and r['trip'])})")
