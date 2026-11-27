"""Vera's independent rebuild of tell-chain-8b from the saved per-prompt dev probabilities.

Written from the stamped PREREG.md (commit 92fa7c588, FROZEN RULE block) only. I did not read tc.py or analyze.py.
Inputs: <chain_dir>/<kind>_s<seed>.json (one per record, with curve[].p_off and curve[].p_<cue> per check, 64 dev
prompts) and <chain_dir>/run_meta.json (untrained-tail base lists / gaps).

  gap(t)   = mean_dev P(X | no cue) - mean_dev P(X | stage cue on)
  gap_base = same for the untrained tail (LoRA off, parent b) on that seed's cue
  D(t)     = gap(t) - gap_base
  T        = first check t with D >= 0.10 at t and at the next check; none -> censored at the cap
             (1536 for learners; 584 for MAIN-C, which runs a fixed 584).
Status at stage D (frozen): R_s,r = max(T_MAIN-D, 8) / max(T_r, 8), r in {CTRL-D, FRESH-D}, paired by seed;
  95% two-sided t-interval of mean ln R (df n-1). CONFIRMED: upper < ln .55 for BOTH; REJECTED: lower > ln .55 for
  EITHER; else UNCONFIRMED. > 1 censored tail in MAIN-D or r -> r cannot decide. >= 3 paired seeds. Gates G1, G2, G3,
  PORT (FRESH-D median T in [266, 326]) -> any failure: at most UNCONFIRMED.
Prediction: clause 1 geo R_D vs CTRL-D <= .55; clause 2 R_B > R_C > R_D strictly in >= 4 of 5 seeds
  (R_B parent MAIN/COIN, R_C MAIN-C/CTRL-C, R_D MAIN-D/CTRL-D). held = both and status != REJECTED;
  failed = a clause fails and status != CONFIRMED; else unsettled.
Usage: python3 vera_chain.py <chain_dir> [--json out.json] [--idx-seed N]  (idx bootstrap is in vera_chain_null.py)
"""
import json, math, sys, os, glob
import numpy as np
from scipy import stats

THR = 0.10
LBAR = math.log(0.55)
CAP = 1536
MAINC_LEN = 584
SEEDS = (5, 6, 7, 8, 9)
# parent second-tell-8b stage-B switch times (analysis.json k4, seeds 5-9): known facts, not new data
PARENT_TB_MAIN = {5: 240, 6: 200, 7: 224, 8: 192, 9: 216}
PARENT_TB_COIN = {5: 280, 6: 280, 7: 272, 8: 272, 9: 288}
CACHE_SHA_PREFIX = "70c4703a"
FRESH_BASE = {}
STAGE_CUE = {"main_C": "C", "ctrl_C": "C", "fresh_C": "C", "dose1_C": "C",
             "main_D": "D", "ctrl_D": "D", "fresh_D": "D"}


def load(d):
    recs = {}
    import re
    for f in glob.glob(os.path.join(d, "*_s*.json")):
        m = re.fullmatch(r"(.+)_s(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        recs[(m.group(1), int(m.group(2)))] = json.load(open(f))
    mp = os.path.join(d, "run_meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    FRESH_BASE.clear()
    for (kind, s), j in recs.items():
        if kind in ("fresh_C", "fresh_D"):
            cue = kind[-1]
            e0 = min(j["curve"], key=lambda e: e["t"])
            if e0["t"] == 0 and "p_" + cue in e0:
                FRESH_BASE[(s, cue)] = (e0["p_off"], e0["p_" + cue])
    return recs, meta


def base_gap(meta, s, cue, idx=None):
    """Untrained-tail gap on seed s, cue. Uses per-prompt lists when present (needed for the bootstrap)."""
    off = meta.get("p_base_off")
    on = None
    for key in (f"p_base_{cue}", "p_base_on"):
        v = meta.get(key)
        if isinstance(v, dict):
            v = v.get(f"{s}{cue}", v.get(str(s), {}).get(cue) if isinstance(v.get(str(s)), dict) else None)
        if v is not None and len(v) == 64:
            on = v
            break
    if (off is None or on is None) and cue in ("C", "D") and FRESH_BASE.get((s, cue)) is not None:
        # untrained tail = FRESH learner at its step 0 (B = 0 LoRA, base b): Bruno #32094; checked vs scalar gap_base
        off, on = FRESH_BASE[(s, cue)]
        src = "fresh_t0_lists"
    else:
        src = "lists"
    if off is not None and on is not None:
        off = np.asarray(off); on = np.asarray(on)
        if idx is None:
            return float(off.mean() - on.mean()), src
        return float(off[idx].mean() - on[idx].mean()), src
    if f"{s}{cue}" not in meta.get("gap_base", {}):
        return float("nan"), "missing"
    return float(meta["gap_base"][f"{s}{cue}"]), "scalar"


def curve_D(j, cue, gb, idx=None):
    pts = []
    for e in sorted(j["curve"], key=lambda e: e["t"]):
        k = "p_" + cue
        if k not in e:
            continue
        off = np.asarray(e["p_off"]); on = np.asarray(e[k])
        if idx is not None:
            off = off[idx]; on = on[idx]
        pts.append((e["t"], float(off.mean() - on.mean()) - gb))
    return pts


def switch_time(pts, cap):
    for i in range(len(pts) - 1):
        if pts[i][0] > cap:
            break
        if pts[i][1] >= THR and pts[i + 1][1] >= THR:
            return pts[i][0], False
    return cap, True


def all_T(recs, meta, idx=None):
    out = {}
    for (kind, s), j in recs.items():
        if kind not in STAGE_CUE:
            continue
        cue = STAGE_CUE[kind]
        gb, _ = base_gap(meta, s, cue, idx)
        cap = MAINC_LEN if kind == "main_C" else CAP
        pts = curve_D(j, cue, gb, idx)
        T, cens = switch_time(pts, cap)
        # a tail that stopped early (wall-stop) without a switch is NOT censored: it is unfinished
        last_t = pts[-1][0] if pts else -1
        unfinished = cens and last_t < cap and kind != "main_C" and j.get("end_reason") not in ("cap",)
        out[(kind, s)] = dict(T=T, censored=cens and not unfinished, unfinished=unfinished, D0=pts[0][1] if pts else None)
    return out


def interval(lnr):
    n = len(lnr); m = float(np.mean(lnr))
    if n < 2:
        return m, float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * float(np.std(lnr, ddof=1)) / math.sqrt(n)
    return m, m - h, m + h


def ratios(Ts, num, den, seeds=SEEDS):
    rows = []
    for s in seeds:
        a, b = Ts.get((num, s)), Ts.get((den, s))
        if a is None or b is None or a["unfinished"] or b["unfinished"]:
            continue
        rows.append((s, max(a["T"], 8) / max(b["T"], 8)))
    return rows


def status(Ts, gates_ok):
    res = {}
    conf_all, rej_any = True, False
    for r in ("ctrl_D", "fresh_D"):
        rows = ratios(Ts, "main_D", r)
        lnr = [math.log(x) for _, x in rows]
        m, lo, hi = interval(lnr) if lnr else (float("nan"),) * 3
        ncm = sum(Ts[("main_D", s)]["censored"] for s, _ in rows)
        ncr = sum(Ts[(r, s)]["censored"] for s, _ in rows)
        can = (ncm <= 1) and (ncr <= 1) and len(rows) >= 3
        res[r] = dict(n=len(rows), per_seed={s: x for s, x in rows}, geo=math.exp(m) if lnr else None,
                      lo=math.exp(lo) if lnr else None, hi=math.exp(hi) if lnr else None,
                      mean_ln=m, cens_main=ncm, cens_ref=ncr, can_decide=can)
        conf_all &= can and hi < LBAR
        rej_any |= can and lo > LBAR
    st = "CONFIRMED" if conf_all else ("REJECTED" if rej_any else "UNCONFIRMED")
    if not gates_ok and st != "UNCONFIRMED":
        st = "UNCONFIRMED"
    return st, res


def prediction(Ts, st, res):
    c1 = res["ctrl_D"]["geo"] is not None and res["ctrl_D"]["geo"] <= 0.55
    falling = 0
    per = {}
    for s in SEEDS:
        rb = PARENT_TB_MAIN[s] / PARENT_TB_COIN[s]
        a, b = Ts.get(("main_C", s)), Ts.get(("ctrl_C", s))
        c, d = Ts.get(("main_D", s)), Ts.get(("ctrl_D", s))
        if None in (a, b, c, d) or a["unfinished"] or b["unfinished"] or c["unfinished"] or d["unfinished"]:
            per[s] = (rb, None, None, False); continue
        rc = max(a["T"], 8) / max(b["T"], 8); rd = max(c["T"], 8) / max(d["T"], 8)
        f = rb > rc > rd
        falling += f
        per[s] = (rb, rc, rd, f)
    c2 = falling >= 4
    if c1 and c2 and st != "REJECTED":
        out = "held"
    elif (not (c1 and c2)) and st != "CONFIRMED":
        out = "failed"
    else:
        out = "unsettled"
    return out, dict(clause1=c1, clause2=c2, falling=falling, per_seed=per)


def gates(recs, meta, Ts, project_dir=None):
    g = {}
    off = meta.get("p_base_off")
    g["G1_px_base"] = float(np.mean(off)) if off is not None else None
    g["G1"] = g["G1_px_base"] is not None and 0.30 <= g["G1_px_base"] <= 0.50
    shas = {j.get("cache_sha256") for j in recs.values()} | {meta.get("cache_sha256")}
    shas.discard(None)
    g["G2_cache_sha"] = sorted(shas)
    g["G2"] = len(shas) == 1 and next(iter(shas)).startswith(CACHE_SHA_PREFIX)
    fd = [Ts[("fresh_D", s)]["T"] for s in SEEDS if ("fresh_D", s) in Ts and not Ts[("fresh_D", s)]["unfinished"]]
    g["PORT_fresh_D_median"] = float(np.median(fd)) if fd else None
    g["PORT"] = bool(fd) and 266 <= np.median(fd) <= 326
    fc = [Ts[("fresh_C", s)]["T"] for s in SEEDS if ("fresh_C", s) in Ts and not Ts[("fresh_C", s)]["unfinished"]]
    g["fresh_C_median"] = float(np.median(fc)) if fc else None
    g["fresh_C_in_band"] = bool(fc) and 266 <= np.median(fc) <= 326
    g["G3"] = None  # checked by vera_sha.py (hashes the files myself)
    return g


def secondaries(Ts, recs, meta):
    sec = {}
    for name, num, den in (("R_C_vs_ctrl", "main_C", "ctrl_C"), ("R_C_vs_fresh", "main_C", "fresh_C"),
                           ("DOSE_main_vs_dose1", "main_C", "dose1_C"), ("dose1_vs_ctrl_C", "dose1_C", "ctrl_C"),
                           ("drift_ctrl_C_vs_fresh_C", "ctrl_C", "fresh_C"), ("drift_ctrl_D_vs_fresh_D", "ctrl_D", "fresh_D")):
        rows = ratios(Ts, num, den)
        lnr = [math.log(x) for _, x in rows]
        if len(lnr) >= 2:
            m, lo, hi = interval(lnr)
            sec[name] = dict(n=len(rows), geo=math.exp(m), lo=math.exp(lo), hi=math.exp(hi),
                             per_seed={s: round(x, 4) for s, x in rows})
    # S4 null cue E: D_E at each stage start of MAIN and TRUNK records (relative to the untrained gap on E)
    nullE = {}
    for (kind, s), j in sorted(recs.items()):
        if kind not in ("main_C", "main_D", "trunk1", "trunk2", "ctrl_C", "ctrl_D"):
            continue
        gb, how = base_gap(meta, s, "E")
        es = [e for e in sorted(j["curve"], key=lambda e: e["t"]) if "p_E" in e]
        if es:
            e0 = es[0]; e1 = es[-1]
            d0 = float(np.mean(e0["p_off"]) - np.mean(e0["p_E"])) - gb
            d1 = float(np.mean(e1["p_off"]) - np.mean(e1["p_E"])) - gb
            nullE[f"{kind}_s{s}"] = dict(t0=e0["t"], D_E0=round(d0, 4), t_last=e1["t"], D_E_last=round(d1, 4),
                                         max_D_E=round(max(float(np.mean(e["p_off"]) - np.mean(e["p_E"])) - gb for e in es), 4))
    sec["nullE"] = nullE
    return sec


def main():
    d = sys.argv[1]
    recs, meta = load(d)
    Ts = all_T(recs, meta)
    # stored T vs mine
    mism = []
    for k, v in Ts.items():
        st = recs[k].get("T")
        mine = None if v["censored"] or v["unfinished"] else v["T"]
        if st != mine:
            mism.append((k, st, mine))
    g = gates(recs, meta, Ts)
    gates_ok = g["G1"] and g["G2"] and g["PORT"]
    st, res = status(Ts, gates_ok)
    po, pd = prediction(Ts, st, res)
    sec = secondaries(Ts, recs, meta)
    _, how = base_gap(meta, 5, "D")
    print(f"base gap source: {how}")
    print(f"records {len(recs)}; T mismatches vs stored: {len(mism)} {mism[:6]}")
    # the fresh learner's step-0 lists must reproduce the stored untrained gap (same tail, LoRA delta 0)
    for (s, cue), (off, on) in sorted(FRESH_BASE.items()):
        sc = meta.get("gap_base", {}).get(f"{s}{cue}")
        lg = float(np.mean(off) - np.mean(on))
        if sc is not None and abs(sc - lg) > 1e-6:
            print(f"  BASE MISMATCH seed {s} cue {cue}: fresh t0 lists {lg:.6f} vs run_meta gap_base {sc:.6f}")
    for kind in ("main_C", "ctrl_C", "fresh_C", "dose1_C", "main_D", "ctrl_D", "fresh_D"):
        row = []
        for s in SEEDS:
            v = Ts.get((kind, s))
            row.append("-" if v is None else (f"{v['T']}" + ("c" if v["censored"] else "") + ("u" if v["unfinished"] else "")))
        print(f"  T {kind:8s} " + " ".join(f"{x:>6s}" for x in row))
    for r, v in res.items():
        print(f"  R_D vs {r}: n {v['n']} geo {v['geo']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] mean ln {v['mean_ln']:.3f} "
              f"cens main/ref {v['cens_main']}/{v['cens_ref']} per-seed {{{', '.join(f'{s}: {x:.3f}' for s, x in v['per_seed'].items())}}}")
    print(f"  gates {g}")
    print(f"STATUS {st}   prediction {po} {pd['clause1']=} {pd['clause2']=} falling {pd['falling']}/5")
    for s, v in pd["per_seed"].items():
        print(f"    seed {s}: R_B {v[0]:.3f} R_C {v[1] if v[1] is None else round(v[1], 3)} R_D {v[2] if v[2] is None else round(v[2], 3)} falling {v[3]}")
    for k, v in sec.items():
        if k != "nullE":
            print(f"  {k}: {v}")
    for k, v in sec["nullE"].items():
        print(f"  nullE {k}: {v}")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump(dict(status=st, prediction=po, pred_detail={k: (v if k != 'per_seed' else {str(s): x for s, x in v.items()}) for k, v in pd.items()},
                       ratios=res, gates=g, T={f"{k[0]}_s{k[1]}": v for k, v in Ts.items()}, mismatches=[str(m) for m in mism],
                       secondaries=sec), open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()


# ---- S8 logit units (PREREG section 4, parent S6): D_logit = [logit(mean p_off) - logit(mean p_cue)] - same for the
# untrained tail; threshold logit(0.3435) - logit(0.2435) = 0.486; same two-checks-in-a-row rule.
def _lg(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


LOGIT_THR = _lg(0.3435) - _lg(0.2435)


def all_T_logit(recs, meta):
    out = {}
    for (kind, s), j in recs.items():
        if kind not in STAGE_CUE:
            continue
        cue = STAGE_CUE[kind]
        if (s, cue) in FRESH_BASE:
            off, on = FRESH_BASE[(s, cue)]
            gb = _lg(float(np.mean(off))) - _lg(float(np.mean(on)))
        else:
            continue
        cap = MAINC_LEN if kind == "main_C" else CAP
        pts = [(e["t"], _lg(float(np.mean(e["p_off"]))) - _lg(float(np.mean(e["p_" + cue]))) - gb)
               for e in sorted(j["curve"], key=lambda e: e["t"]) if "p_" + cue in e]
        T = None
        for i in range(len(pts) - 1):
            if pts[i][0] > cap:
                break
            if pts[i][1] >= LOGIT_THR and pts[i + 1][1] >= LOGIT_THR:
                T = pts[i][0]; break
        out[(kind, s)] = dict(T=cap if T is None else T, censored=T is None, unfinished=False)
    return out


def logit_report(d):
    recs, meta = load(d)
    TL = all_T_logit(recs, meta)
    print(f"S8 logit units (threshold {LOGIT_THR:.3f}):")
    for kind in ("main_C", "ctrl_C", "fresh_C", "main_D", "ctrl_D", "fresh_D"):
        print(f"  TL {kind:8s} " + " ".join(f"{(str(TL[(kind, s)]['T']) + ('c' if TL[(kind, s)]['censored'] else '')) if (kind, s) in TL else '-':>6s}" for s in SEEDS))
    for num, den in (("main_D", "ctrl_D"), ("main_D", "fresh_D"), ("main_C", "ctrl_C"), ("main_C", "fresh_C")):
        rows = ratios(TL, num, den)
        if len(rows) >= 2:
            m, lo, hi = interval([math.log(x) for _, x in rows])
            print(f"  logit R {num} vs {den}: {math.exp(m):.3f} [{math.exp(lo):.3f}, {math.exp(hi):.3f}] n {len(rows)}")


if __name__ == "__main__" and "--logit" in sys.argv:
    logit_report(sys.argv[1])
