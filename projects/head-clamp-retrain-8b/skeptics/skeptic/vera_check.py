"""Vera: rebuild every 8B number from raw rows only (no build code read). Frozen rule: PREREG.md @ec724b15.
Per switch run: dev strict curve from rows['step<N>/dev'], T_f, FORMED (test strict of the saved adapter >= 0.75).
Per retrain run: t=0 test strict with clamp, dev curve, RETURNED (2 consecutive dev evals >= 0.75, t=0 not counted),
budget 2*T_f (up to a multiple of 64), test strict at stop (final/test_merged_clamp), reversed share, push, CONVERGED.
Then: title status, prereg outcome, ratio reading vs data/band_17b.json, seed-level t-intervals, and a within-pair
label-flip null (sign test) per final cell.
Usage: python3 vera_check.py DATA_DIR [OUT_JSON]"""
import json, glob, os, sys, math, collections
import numpy as np

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "vera_check.json")
T43 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}

def per_prompt(rows):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][int(probe)] = (top1, la - lb)
    return by

def strict(rows, A, B):
    by = per_prompt(rows); N = len(by)
    s = sum(d[1][0] == A and d[0][0] == B for d in by.values()) / N
    n = sum(d[1][0] != A and d[0][0] == A for d in by.values()) / N
    push = float(np.mean([d[1][1] - d[0][1] for d in by.values()]))
    return s, n, push, N

def curve(R, A, B):
    ks = [(int(k[4:].split("/")[0]), k) for k in R if k.startswith("step") and k.split("/")[1] == "dev"]
    return [(u, strict(R[k], A, B)[0]) for u, k in sorted(ks)]

def first_pair(cv, thr=0.75):
    for (u1, c1), (u2, c2) in zip(cv, cv[1:]):
        if c1 >= thr and c2 >= thr: return u2
    return None

def converged(loss, cv):
    L = np.asarray(loss, float); k = max(1, int(round(0.2 * len(L))))
    d = float(L[-k:].mean() - L[-2 * k:-k].mean()); r = [c for _, c in cv[-3:]]
    return bool(abs(d) <= 0.02 and max(r) - min(r) <= 0.10), d, max(r) - min(r)

def sign_null(rows, A, B):
    """Within-pair label flip: under H0 (switch ignores the probe) a prompt is right-way (on A, off B) or reversed
    (on not-A... off A pattern mirrored) with equal chance. Count right-way vs reversed among prompts whose two
    answers differ in the A/B pattern; two-sided binomial sign test."""
    by = per_prompt(rows)
    rw = sum(d[1][0] == A and d[0][0] == B for d in by.values())
    rv = sum(d[1][0] == B and d[0][0] == A for d in by.values())
    n = rw + rv
    if n == 0: return rw, rv, 1.0
    k = max(rw, rv)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return rw, rv, p

def tint(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (float(x.mean()) if n else None, None, None)
    h = T43.get(n - 1, 2.0) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)

runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "runs", "*.json")))}
out = {"switch": {}, "retrain": {}}
TF = {}
for name, d in runs.items():
    if not name.startswith("switch_s"): continue
    A, B = d["a_id"], d["b_id"]; seed = int(name.split("_s")[1]); R = d["rows"]
    cv = curve(R, A, B); tf = first_pair(cv)
    fk = [k for k in ("final/test_merged_noclamp", "final/test_unmerged_noclamp") if k in R]
    ts = strict(R[fk[0]], A, B)[0] if fk else None
    formed = tf is not None and tf <= 4096 and ts is not None and ts >= 0.75
    TF[seed] = tf if formed else None
    out["switch"][seed] = dict(T_f=tf, test_strict=ts, formed=formed, steps_run=d.get("steps_run"),
                               stored=dict(t_f=d.get("t_f"), formed=d.get("formed"), final_test_strict=d.get("final_test_strict")),
                               curve=cv)
    print(f"switch s{seed}: T_f {tf} (stored {d.get('t_f')}) test strict {ts} formed {formed} (stored {d.get('formed')}) steps {d.get('steps_run')}")

for name, d in runs.items():
    if name.startswith("switch_s"): continue
    arm, seed = name.rsplit("_s", 1); seed = int(seed)
    A, B = d["a_id"], d["b_id"]; R = d["rows"]
    cv = curve(R, A, B); ret = first_pair(cv)
    tf = TF.get(seed); budget = None if tf is None else int(math.ceil(2 * tf / 64) * 64)
    t0 = strict(R["t0/test_clamp"], A, B)[0] if "t0/test_clamp" in R else None
    fs, fn, fp, N = strict(R["final/test_merged_clamp"], A, B)
    conv = converged(d.get("loss", []), cv) if d.get("loss") else (None, None, None)
    nul = sign_null(R["final/test_merged_clamp"], A, B)
    last = cv[-1][0] if cv else 0
    complete = ret is not None or (budget is not None and last >= budget)
    out["retrain"][name] = dict(arm=arm, seed=seed, heads=d.get("heads"), t0_test=t0, returned_at=ret, budget=budget,
                                last_eval=last, complete=complete, final_test=fs, reversed=fn, push=fp, N=N,
                                converged=conv[0], dloss=conv[1], ddev=conv[2], null_rw_rv_p=nul,
                                stored=dict(return_step=d.get("return_step"), stop_reason=d.get("stop_reason"),
                                            final_test_strict=d.get("final_test_strict"), steps_run=d.get("steps_run")))
    mm = []
    if (ret or None) != (d.get("return_step") or None): mm.append(f"return mine {ret} stored {d.get('return_step')}")
    if d.get("final_test_strict") is not None and abs(fs - d["final_test_strict"]) > 1e-9: mm.append(f"final mine {fs} stored {d['final_test_strict']}")
    print(f"{name:<12} heads {len(d.get('heads') or []):>2} t0 {t0} ret@{ret} budget {budget} last {last} final {fs:.3f} n {fn:.3f} "
          f"push {fp:.2f} conv {conv[0]} null rw/rv/p {nul[0]}/{nul[1]}/{nul[2]:.2g} {'MISMATCH ' + '; '.join(mm) if mm else 'match'}")

def cells(arm):
    return {v["seed"]: v for v in out["retrain"].values() if v["arm"] == arm and TF.get(v["seed"]) is not None}

S = cells("set"); formed = [s for s, t in TF.items() if t is not None]
back = [s for s, v in S.items() if v["complete"] and (v["returned_at"] is not None or v["final_test"] >= 0.25)
        and not (v["t0_test"] is not None and v["t0_test"] >= 0.25)]
hold = [s for s, v in S.items() if v["complete"] and v["returned_at"] is None and v["final_test"] < 0.25 and v["converged"]
        and not (v["t0_test"] is not None and v["t0_test"] >= 0.25)]
status = "REJECTED" if len(formed) >= 3 and len(back) >= 2 else "CONFIRMED" if len(formed) >= 3 and len(hold) >= 2 else "UNCONFIRMED"

def n_ret(arm): return sum(v["returned_at"] is not None for v in cells(arm).values())
def n_held(arm): return sum(v["complete"] and v["returned_at"] is None for v in cells(arm).values())
set_by_tf = sum(v["returned_at"] is not None and v["returned_at"] <= TF[s] for s, v in S.items())
all_hold = sum(v["complete"] and v["returned_at"] is None and v["final_test"] < 0.25 for v in cells("all64").values())
all_back = sum(v["complete"] and (v["returned_at"] is not None or v["final_test"] >= 0.25) for v in cells("all64").values())
held = set_by_tf >= 2 and n_ret("top1") >= 2 and n_ret("leave1") >= 2 and all_hold >= 2
failed = (sum(v["complete"] and not (v["returned_at"] is not None and v["returned_at"] <= TF[s]) for s, v in S.items()) >= 2
          or n_held("top1") >= 2 or n_held("leave1") >= 2 or all_back >= 2)
outcome = "held" if held else "failed" if failed else "unsettled"

band = json.load(open(os.path.join(D, "band_17b.json"))) if os.path.exists(os.path.join(D, "band_17b.json")) else None
ratio = {}
if band:
    rs = [v["ratio"] for v in band["set"].values()]; tmin = min(band["parent_T_f"].values())
    lo, hi = min(rs) - 64 / tmin, max(rs) + 64 / tmin
    for s, v in S.items():
        ratio[s] = (v["returned_at"] / TF[s]) if v["returned_at"] is not None else (float("inf") if v["complete"] else None)
    rr = [r for r in ratio.values() if r is not None]
    k_lo, k_in, k_hi = sum(r < lo for r in rr), sum(lo <= r <= hi for r in rr), sum(r > hi for r in rr)
    reading = "FASTER" if k_lo >= 2 else "SLOWER" if k_hi >= 2 else "SIZE-INVARIANT" if k_in >= 2 else "UNRESOLVED"
    fin = [r for r in rr if math.isfinite(r)]
    out["ratio"] = dict(band=[lo, hi], per_seed=ratio, reading_by_seeds=reading, t_interval=tint(fin) if len(fin) >= 2 else None)
out["seed_level_set_final_test"] = tint([v["final_test"] for v in S.values() if v["complete"]])
out.update(status=status, formed=formed, set_back=back, set_hold=hold, outcome=outcome,
           counts=dict(set_by_tf=set_by_tf, top1_ret=n_ret("top1"), leave1_ret=n_ret("leave1"), all64_hold=all_hold, all64_back=all_back))
print(json.dumps({k: out[k] for k in ("status", "formed", "set_back", "set_hold", "outcome", "counts", "seed_level_set_final_test")}, default=str))
if "ratio" in out: print("ratio", json.dumps(out["ratio"], default=str))
json.dump(out, open(OUT, "w"), indent=1, default=str)
