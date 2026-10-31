"""Vera robustness on counted data (own code; imports only my vera_check helpers).
1 leave-one-model-out G (eligibility recomputed, s6 floor applied) with W CI
2 all strata with >=1 run in each say-group (no eligibility), MH G + W CI
3 say-rule variants (v1.2 'says', raw, aware v1.3, any-window v1.3) and trip_any
4 seed halves (0-34, 35-69) on the frozen rule
5 placebo regexes restricted to the 4 frozen eligible strata, W CI, and overlap with trap-word sayers
"""
import json
import re
from collections import defaultdict

import numpy as np

import vera_check as vc

vc.B = 4000
rows = [json.loads(l) for l in open("vera_counted_scores.jsonl")]
ex = [r for r in rows if r["exposed"] and r["window_reply_exists"] and r["prompt"] == "on"]
rng = np.random.default_rng(7)


def G_ci(sub, saycol, trip="trip", elig=vc.eligible):
    S = defaultdict(lambda: ([], []))
    for r in sub:
        S[(r["model"], r["decoy"])][0].append(bool(r[saycol]))
        S[(r["model"], r["decoy"])][1].append(bool(r[trip]))
    S = {k: (np.array(a), np.array(b)) for k, (a, b) in S.items()}
    el = [(k, s, t) for k, (s, t) in S.items() if elig(s, t)]
    if not el:
        return None
    keys = [k for k, _, _ in el]
    L = [(s, t) for _, s, t in el]
    G = vc.G_of(L)
    lo, hi = vc.boot(L, keys, rng, "W")
    nm = len({k[0] for k in keys}); nr = sum(len(s) for s, _ in L)
    return dict(G=round(G, 4), ci=[round(lo, 4), round(hi, 4)], strata=len(keys), models=nm, runs=nr,
                status=vc.status(lo, hi, len(keys), nm, nr))


out = {}
print("1. leave one model out (frozen rule, W CI)")
for m in sorted({r["model"] for r in ex}):
    res = G_ci([r for r in ex if r["model"] != m], "says_v13")
    print(f"   drop {m:27s} {res}")
    out[f"lomo_{m}"] = res

print("2. all strata with both say-groups non-empty (no eligibility rule)")
res = G_ci(ex, "says_v13", elig=lambda s, t: s.sum() >= 1 and (~s).sum() >= 1)
print("  ", res); out["all_strata"] = res
res = G_ci(ex, "says_v13", elig=lambda s, t: s.sum() >= 5 and (~s).sum() >= 5)
print("   a,b>=5, no trip floor:", res); out["ab5_nofloor"] = res

print("3. variants")
for col in ("says", "says_raw", "says_v13_aware", "says_v13_anywin", "says_v1anchor"):
    res = G_ci(ex, col); print(f"   {col:18s} {res}"); out[f"var_{col}"] = res
res = G_ci(ex, "says_v13", trip="trip_any"); print(f"   trip_any           {res}"); out["var_trip_any"] = res

print("4. seed halves (frozen rule)")
for lo_, hi_ in ((0, 34), (35, 69)):
    res = G_ci([r for r in ex if lo_ <= r["seed"] <= hi_], "says_v13")
    print(f"   seeds {lo_}-{hi_}: {res}"); out[f"seeds_{lo_}_{hi_}"] = res
res = G_ci([r for r in ex if r["seed"] % 2 == 0], "says_v13"); print("   even seeds:", res); out["seeds_even"] = res
res = G_ci([r for r in ex if r["seed"] % 2 == 1], "says_v13"); print("   odd seeds:", res); out["seeds_odd"] = res

print("5. placebos on the 4 frozen eligible strata")
frozen = {("gemini-3.1-flash-lite", "subtle"), ("gemini-3.5-flash-lite", "obvious"),
          ("gemini-3.5-flash-lite", "subtle"), ("gpt-5.4-mini", "subtle")}
sub = [r for r in ex if (r["model"], r["decoy"]) in frozen]
PL = {"P1": r"\b(?:next|then|now)\b", "P2": r"\b(?:will|should|need|needs)\b", "P3": r"\d"}
for name, rx in PL.items():
    p = re.compile(rx, re.I)
    sub2 = [dict(r, _p=bool(p.search(r["thought"]))) for r in sub]
    res = G_ci(sub2, "_p", elig=lambda s, t: s.sum() >= 1 and (~s).sum() >= 1)
    both = sum(1 for r in sub2 if r["_p"] and r["says_v13"]); ns = sum(r["says_v13"] for r in sub2); npl = sum(r["_p"] for r in sub2)
    print(f"   {name} {rx}: {res}; placebo hits {npl}, trap sayers {ns}, both {both}")
    out[f"placebo_frozen_{name}"] = dict(res=res, hits=npl, both=both)
    # placebo among NON-sayers only: does the word still predict the act without a trap word?
    nons = [r for r in sub2 if not r["says_v13"]]
    res2 = G_ci(nons, "_p", elig=lambda s, t: s.sum() >= 1 and (~s).sum() >= 1)
    print(f"      among trap non-sayers only: {res2}")
    out[f"placebo_frozen_{name}_nonsayers"] = res2

json.dump(out, open("vera_robust.json", "w"), indent=1, default=float)
