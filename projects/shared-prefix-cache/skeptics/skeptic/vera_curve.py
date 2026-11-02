"""Vera's own status arithmetic over every checkpoint, from build's per-prompt rows (data/eval/<run>/<tag>_<step>.jsonl).
Own code: own loader, own paired bootstrap (4,000 resamples, rng 7), own status rule from PREREG s6.
dGAP(step) = mean_i [(px_S - px_G)_step - (px_S - px_G)_untrained], S = compiled (shared arms) | compiled_own (own arms),
G = eager (chunked grading). Untrained rows: data/eval/base/<tag>_0.jsonl.
Usage: python vera_curve.py RUN[:S] ...    e.g. shared_stack_s0 own_stack_s0:compiled_own
"""
import sys, json, glob, os, re
import numpy as np
E = os.environ.get("VERA_E", "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/data/eval")

def rows(path):
    r = [json.loads(l) for l in open(path)]
    assert len(r) == 200, (path, len(r))
    return [x["id"] for x in r], np.array([x["px"] for x in r], dtype=np.float64)

rng = np.random.default_rng(7); out = {}
for arg in sys.argv[1:]:
    run, S = (arg.split(":") + [None])[:2]
    S = S or ("compiled_own" if run.startswith("own") else "compiled")
    G = "eager"
    idb, g0 = rows(f"{E}/base/{G}_0.jsonl"); ids0, s0 = rows(f"{E}/base/{S}_0.jsonl"); assert idb == ids0
    steps = sorted({int(m.group(1)) for f in glob.glob(f"{E}/{run}/{G}_*.jsonl") if (m := re.search(rf"/{G}_(\d+)\.jsonl$", f))})
    cur = []
    for st in steps:
        fs = f"{E}/{run}/{S}_{st}.jsonl"
        if not os.path.exists(fs): continue
        idg, g = rows(f"{E}/{run}/{G}_{st}.jsonl"); ids, s = rows(fs); assert idg == ids == idb
        dg = (s - g) - (s0 - g0); n = len(dg)
        bs = dg[rng.integers(0, n, (4000, n))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cur.append(dict(step=st, P_G=g.mean(), P_S=s.mean(), dGAP=dg.mean(), lo=lo, hi=hi, grad_drop=g0.mean() - g.mean()))
        print(f"{run:18s} {S:12s} {st:6d} P_G {g.mean():.4f} P_S {s.mean():.4f} dGAP {dg.mean():+.4f} [{lo:+.4f},{hi:+.4f}] grad_drop {g0.mean()-g.mean():+.3f}")
    # status pieces (PREREG s6)
    order = [c for c in cur if c["step"] <= 12288]
    all_hi_lt5 = bool(order) and all(c["hi"] < 0.05 for c in order)
    reach10 = any(c["dGAP"] >= 0.10 and c["lo"] > 0.05 for c in order if c["step"] <= 8192)
    consec = any(a["dGAP"] >= 0.10 and a["lo"] > 0.05 and b["dGAP"] >= 0.10 and b["lo"] > 0.05 for a, b in zip(order, order[1:]))
    out[run] = dict(S=S, curve=cur, all_hi_lt_0p05=all_hi_lt5, own_reach_0p10_by_8192=reach10, two_consec_ge_0p10=consec,
                    max_hi=max((c["hi"] for c in order), default=None), steps=[c["step"] for c in order])
    print(f"  -> steps {len(order)} last {order[-1]['step'] if order else None} | every hi<0.05: {all_hi_lt5} (max hi {out[run]['max_hi']}) | "
          f">=0.10 w/ lo>0.05 by 8192: {reach10} | two consecutive >=0.10: {consec}")
json.dump(out, open("vera_curve.json", "w"), indent=1, default=float)
