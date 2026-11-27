"""Frozen status rule (PREREG.md section 3) + secondary reads (section 4) for tell-chain-8b -> data/analysis.json.

Inputs: data/chain/<record>_s<seed>.json from tc.py, data/chain/run_meta.json, ref/parent_analysis.json (stage B,
the parent's per-seed ratios), ref/parent/*_adapters_sha256.json (gate G3).

    .venv/bin/python analyze.py                 # -> data/analysis.json
    .venv/bin/python analyze.py --fixture       # self-test on synthetic records (multiplying, flat, censored)
"""
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
CAP = 1536
BAR = 0.55
LEARNERS = ("main_C", "ctrl_C", "fresh_C", "dose1_C", "main_D", "ctrl_D", "fresh_D")
MODELS = {"multiplying": {"C": 0.74 ** 2, "D": 0.74 ** 3}, "flat": {"C": 0.74, "D": 0.74}}
THR_LOGIT = math.log(0.3435 / (1 - 0.3435)) - math.log(0.2435 / (1 - 0.2435))  # 0.486, parent S6


def tint(x):
    """mean and two-sided 95% t-interval of x (df n-1)."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, m - h, m + h


def switch_pair(curve, key, thr):
    ev = [c for c in curve if key in c]
    for c, n in zip(ev, ev[1:]):
        if c[key] >= thr and n[key] >= thr:
            return c["t"]
    return None


def logit(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def load(ddir):
    recs = {}
    for f in sorted((ddir / "chain").glob("*_s*.json")):
        if f.name in ("adapters_sha256.json", "run_meta.json"):
            continue
        d = json.loads(f.read_text())
        recs[(d["kind"], d["seed"])] = d
    meta = json.loads((ddir / "chain" / "run_meta.json").read_text())
    return recs, meta


def T_of(d, stage_len):
    """(T used in the statistic, status): status ok | censored | incomplete."""
    if d is None:
        return None, "missing"
    if d["kind"] == "main_C":
        T = d["T"]
        if T is None:
            T = switch_pair(d["curve"], "D_C", 0.10)
        if T is None:
            return (stage_len, "censored") if d["end_reason"] == "stage_len" else (None, "incomplete")
        return T, "ok"
    if d["T"] is not None:
        return d["T"], "ok"
    if d["end_reason"] == "cap":
        return CAP, "censored"
    return None, "incomplete"


def ratio_block(recs, num, den, seeds, stage_len):
    """Paired ln R = ln max(T_num, 8) - ln max(T_den, 8) over seeds where both are not incomplete."""
    per, cens_n, cens_d, used = {}, 0, 0, []
    for s in seeds:
        tn, sn = T_of(recs.get((num, s)), stage_len)
        td, sd = T_of(recs.get((den, s)), stage_len)
        if tn is None or td is None:
            continue
        used.append(s)
        cens_n += sn == "censored"
        cens_d += sd == "censored"
        per[s] = math.log(max(tn, 8)) - math.log(max(td, 8))
    out = {"num": num, "den": den, "seeds": used, "n": len(used), "censored_num": cens_n, "censored_den": cens_d,
           "per_seed": {str(s): math.exp(v) for s, v in per.items()},
           "per_seed_lnR": {str(s): v for s, v in per.items()}}
    if per:
        m, lo, hi = tint(list(per.values()))
        out.update({"mean_lnR": m, "lo_lnR": lo, "hi_lnR": hi, "geo": math.exp(m),
                    "lo": math.exp(lo) if lo == lo else None, "hi": math.exp(hi) if hi == hi else None})
    return out


def analyze(ddir, ref_dir):
    recs, meta = load(ddir)
    a = meta["args"]
    stage_len = a["stage_len"]
    global CAP
    CAP = a.get("cap", 1536)
    seeds = sorted({s for (_, s) in recs})
    parent = json.loads((ref_dir / "parent_analysis.json").read_text())
    notes = []

    # --- per-record table
    T, cens, status_of = {}, {}, {}
    for k in LEARNERS:
        T[k], cens[k], status_of[k] = {}, {}, {}
        for s in seeds:
            t, st = T_of(recs.get((k, s)), stage_len)
            T[k][str(s)] = t
            cens[k][str(s)] = st == "censored"
            status_of[k][str(s)] = st
            if st == "incomplete":
                notes.append(f"{k}_s{s} incomplete (wall-stop or missing): not in any ratio")

    # --- gates
    p_off_base = float(np.mean(meta["p_base_off"]))
    g1 = 0.30 <= p_off_base <= 0.50
    g2 = meta["cache_sha256"] == "70c4703aa2bab795eb5bc34b54be6c16ac3e62ec2ef361841c2598c351de67ba"
    par = {}
    for f in ("k4_adapters_sha256.json", "stage1_adapters_sha256.json"):
        par.update(json.loads((ref_dir / "parent" / f).read_text()))
    g3 = all(par.get(fn) == sh for fn, sh in meta["start_sha256"].items()) and len(meta["start_sha256"]) == 3 * len(seeds)
    fD = [t for t in T["fresh_D"].values() if t is not None]
    fC = [t for t in T["fresh_C"].values() if t is not None]
    port_D = bool(fD) and 266 <= float(np.median(fD)) <= 326
    port_C = bool(fC) and 266 <= float(np.median(fC)) <= 326
    gates = {"G1": {"dev_px_off_untrained_parent_b": p_off_base, "pass": g1},
             "G2": {"cache_sha256": meta["cache_sha256"], "pass": g2},
             "G3": {"start_sha256": meta["start_sha256"], "pass": g3},
             "PORT_D": {"fresh_D_T": sorted(fD), "median": float(np.median(fD)) if fD else None, "range": [266, 326],
                        "pass": port_D},
             "PORT_C_report": {"fresh_C_T": sorted(fC), "median": float(np.median(fC)) if fC else None,
                               "range": [266, 326], "pass": port_C}}
    gates_ok = g1 and g2 and g3 and port_D

    # --- status at stage D
    rD = {"ctrl": ratio_block(recs, "main_D", "ctrl_D", seeds, stage_len),
          "fresh": ratio_block(recs, "main_D", "fresh_D", seeds, stage_len)}
    lb = math.log(BAR)
    can = {}
    for r, b in rD.items():
        can[r] = b["n"] >= 3 and b["censored_num"] <= 1 and b["censored_den"] <= 1 and b.get("lo_lnR") == b.get("lo_lnR")
    conf = all(can[r] and rD[r]["hi_lnR"] < lb for r in rD)
    rej = any(can[r] and rD[r]["lo_lnR"] > lb for r in rD)
    if rej:
        status = "REJECTED"
    elif conf:
        status = "CONFIRMED"
    else:
        status = "UNCONFIRMED"
    status_rule = status
    if not gates_ok and status != "UNCONFIRMED":
        notes.append(f"gate failed: status {status} -> UNCONFIRMED")
        status = "UNCONFIRMED"

    # --- stage C (S1), dose (S2), trunk drift (S7)
    rC = {"ctrl": ratio_block(recs, "main_C", "ctrl_C", seeds, stage_len),
          "fresh": ratio_block(recs, "main_C", "fresh_C", seeds, stage_len)}
    dose = {"main_vs_dose1": ratio_block(recs, "main_C", "dose1_C", seeds, stage_len),
            "dose1_vs_ctrl": ratio_block(recs, "dose1_C", "ctrl_C", seeds, stage_len)}
    db = dose["main_vs_dose1"]
    if db["n"] >= 2:
        dose["read"] = ("multiplying" if db["hi_lnR"] < math.log(0.86) else
                        "flat" if db["lo_lnR"] > math.log(0.86) else "unsettled")
    drift = {"C": ratio_block(recs, "ctrl_C", "fresh_C", seeds, stage_len),
             "D": ratio_block(recs, "ctrl_D", "fresh_D", seeds, stage_len)}

    # --- stage B from the parent (paired per seed, vs COIN and vs FRESH)
    pk = parent["k4"]["ratios"]
    rB = {"ctrl": {"per_seed": pk["coin"]["per_seed"], "geo": pk["coin"]["R"], "lo": pk["coin"]["R_lo"],
                   "hi": pk["coin"]["R_hi"], "source": "parent second-tell-8b analysis.json k4 coin"},
          "fresh": {"per_seed": pk["fresh"]["per_seed"], "geo": pk["fresh"]["R"], "lo": pk["fresh"]["R_lo"],
                    "hi": pk["fresh"]["R_hi"], "source": "parent second-tell-8b analysis.json k4 fresh"}}

    # --- prediction outcome
    c1 = rD["ctrl"].get("geo") is not None and rD["ctrl"]["geo"] <= BAR
    mono = {}
    for s in seeds:
        b = rB["ctrl"]["per_seed"].get(str(s))
        c = rC["ctrl"]["per_seed"].get(str(s))
        d = rD["ctrl"]["per_seed"].get(str(s))
        mono[str(s)] = bool(b is not None and c is not None and d is not None and b > c > d)
    n_mono = sum(mono.values())
    c2 = n_mono >= 4
    if c1 and c2 and status != "REJECTED":
        outcome = "held"
    elif (not c1 or not c2) and status != "CONFIRMED":
        outcome = "failed"
    else:
        outcome = "unsettled"

    # --- model fit (S3)
    fit = {}
    for mname, pred in MODELS.items():
        sse = 0.0
        for stg, blk in (("C", rC), ("D", rD)):
            for r in ("ctrl", "fresh"):
                if blk[r].get("mean_lnR") is not None:
                    sse += (blk[r]["mean_lnR"] - math.log(pred[stg])) ** 2
        fit[mname] = sse

    # --- per-record secondary reads (S4 null cue E, S5 zero-shot, S6 retention, S8 logit units)
    rec_tab = {}
    for (k, s), d in sorted(recs.items()):
        c0 = d["curve"][0]
        row = {"end_reason": d["end_reason"], "steps_run": d["steps_run"], "b": d["b"],
               "px_off_before_resolve": d["px_off_before_resolve"], "px_off_t0": c0["px_off"], "T": d["T"]}
        if d["cue"]:
            key = f"D_{d['cue']}"
            row["D0"] = c0.get(key)
            # S8: logit units, same pair rule
            gb = d["gap_base"][d["cue"]]
            pb_off = float(np.mean(json.loads((ddir / "chain" / "run_meta.json").read_text())["p_base_off"]))
            pb_on = pb_off - gb
            gl0 = logit(pb_off) - logit(pb_on)
            ev = [{"t": c["t"], "Dl": (logit(c["px_off"]) - logit(c[f"px_{d['cue']}"])) - gl0}
                  for c in d["curve"] if f"px_{d['cue']}" in c]
            row["T_logit"] = switch_pair(ev, "Dl", THR_LOGIT)
        ret = {}
        for c in d["curve"]:
            for L in "ABCDE":
                if f"D_{L}" in c and L != d["cue"]:
                    ret.setdefault(L, []).append([c["t"], round(c[f"D_{L}"], 5), round(c[f"gap_{L}"], 5)])
        row["other_cues_t_D_gap"] = ret
        if "E" in ret:
            row["D_E_max"] = max(x[1] for x in ret["E"])
            row["D_E_absmax"] = max(abs(x[1]) for x in ret["E"])
        rec_tab[f"{k}_s{s}"] = row
    dE_main = [v["D_E_max"] for n, v in rec_tab.items() if n.startswith("main") and "D_E_max" in v]
    s4 = {"main_D_E_max": max(dE_main) if dE_main else None,
          "generic_detector_flag": bool(dE_main) and max(dE_main) >= 0.05,
          "trunk_D_E_max": max([v["D_E_max"] for n, v in rec_tab.items() if n.startswith("trunk") and "D_E_max" in v],
                               default=None),
          "main_D_E_absmax": max([v["D_E_absmax"] for n, v in rec_tab.items() if n.startswith("main") and "D_E_absmax" in v],
                                 default=None),
          "trunk_D_E_absmax": max([v["D_E_absmax"] for n, v in rec_tab.items() if n.startswith("trunk")
                                   and "D_E_absmax" in v], default=None)}
    zs = {}
    for k in LEARNERS:
        v = [rec_tab[f"{k}_s{s}"]["D0"] for s in seeds if f"{k}_s{s}" in rec_tab]
        if v:
            zs[k] = {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))}

    return {"status": status, "status_rule": status_rule, "gates": gates, "gates_ok": gates_ok, "can_decide": can,
            "bar": BAR, "seeds": seeds, "stage_len": stage_len, "cap": CAP,
            "T": T, "censored": cens, "record_status": status_of,
            "ratios": {"B": rB, "C": rC, "D": rD}, "dose": dose, "trunk_drift": drift,
            "prediction": {"clause1_geoRD_ctrl_le_055": c1, "clause2_monotone_per_seed": mono, "n_monotone": n_mono,
                           "clause2_ge4": c2, "outcome": outcome},
            "models": MODELS, "model_fit_sse": fit, "null_cue_E": s4, "zero_shot_D0": zs, "records": rec_tab,
            "notes": notes, "g_end": meta["g_end"], "wall_stopped_at": meta["wall_stopped_at"],
            "inputs_sha256": {str(f.relative_to(ddir.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted((ddir / "chain").glob("*.json"))}}


def fixture():
    """Synthetic records: (1) multiplying -> CONFIRMED + held; (2) flat -> REJECTED + failed;
    (3) multiplying with 2 censored ctrl_D -> ctrl cannot decide -> UNCONFIRMED."""
    rng = np.random.default_rng(0)
    seeds = [5, 6, 7, 8, 9]
    par = json.loads((HERE / "ref" / "parent_analysis.json").read_text())
    pb = par["k4"]["ratios"]["coin"]["per_seed"]

    def make(tmp, RC, RD, cens_ctrlD=0):
        d = tmp / "data" / "chain"
        d.mkdir(parents=True)
        startsha = {}
        p1 = json.loads((HERE / "ref" / "parent" / "k4_adapters_sha256.json").read_text())
        p0 = json.loads((HERE / "ref" / "parent" / "stage1_adapters_sha256.json").read_text())
        for s in seeds:
            for fn in (f"main_s{s}_k4_end.pt", f"coin_s{s}_k4_end.pt"):
                startsha[fn] = p1[fn]
            startsha[f"coin_s{s}_s1.pt"] = p0[f"coin_s{s}_s1.pt"]
        json.dump({"args": {"stage_len": 584}, "p_base_off": [0.344] * 64, "cache_sha256":
                   "70c4703aa2bab795eb5bc34b54be6c16ac3e62ec2ef361841c2598c351de67ba", "start_sha256": startsha,
                   "g_end": 1500, "wall_stopped_at": None}, open(d / "run_meta.json", "w"))
        for i, s in enumerate(seeds):
            base = 296 * math.exp(rng.normal(0, 0.05))
            Ts = {"fresh_C": base, "fresh_D": base, "ctrl_C": base * 0.97, "ctrl_D": base * 0.97,
                  "dose1_C": base * 0.97 * 0.74, "main_C": base * 0.97 * RC, "main_D": base * 0.97 * RD}
            for k, t in Ts.items():
                t = int(round(t * math.exp(rng.normal(0, 0.04)) / 8) * 8)
                cue = k[-1]
                cen = k == "ctrl_D" and i < cens_ctrlD
                curve = [{"t": tt, "px_off": 0.4, f"px_{cue}": 0.4 - (0.2 if tt >= t else 0.0),
                          f"D_{cue}": (0.2 if tt >= t else 0.0), f"gap_{cue}": (0.2 if tt >= t else 0.0)}
                         for tt in range(0, (CAP if cen else t) + 9, 8)]
                rec = {"kind": k, "seed": s, "cue": cue, "T": None if cen else t, "curve": curve,
                       "end_reason": "cap" if cen else ("stage_len" if k == "main_C" else "switch"),
                       "steps_run": t + 8, "b": -4.4, "px_off_before_resolve": 0.3,
                       "gap_base": {L: 0.0 for L in "ABCDE"}}
                json.dump(rec, open(d / f"{k}_s{s}.json", "w"))
        return tmp / "data"

    ok = True
    for name, RC, RD, cz, want, want_o in (("multiplying", 0.548, 0.405, 0, "CONFIRMED", "held"),
                                           ("flat", 0.74, 0.74, 0, "REJECTED", "failed"),
                                           ("mult_2cens", 0.548, 0.405, 2, "UNCONFIRMED", None)):
        with tempfile.TemporaryDirectory() as t:
            res = analyze(make(Path(t), RC, RD, cz), HERE / "ref")
        got = res["status"]
        rd = res["ratios"]["D"]
        print(f"fixture {name}: status {got} (want {want}), outcome {res['prediction']['outcome']}, "
              f"R_D ctrl {rd['ctrl'].get('geo', float('nan')):.3f} [{rd['ctrl'].get('lo') or float('nan'):.3f}, "
              f"{rd['ctrl'].get('hi') or float('nan'):.3f}] fresh {rd['fresh']['geo']:.3f}, mono {res['prediction']['n_monotone']}")
        ok &= got == want and (want_o is None or res["prediction"]["outcome"] == want_o)
    print("FIXTURE", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--data", default=str(HERE / "data"))
    a = ap.parse_args()
    if a.fixture:
        fixture()
    res = analyze(Path(a.data), HERE / "ref")
    (Path(a.data) / "analysis.json").write_text(json.dumps(res, indent=1))
    rd = res["ratios"]["D"]
    print("status", res["status"], "(rule", res["status_rule"] + ")", "gates_ok", res["gates_ok"])
    for stg in ("C", "D"):
        for r in ("ctrl", "fresh"):
            b = res["ratios"][stg][r]
            if b.get("geo") is not None:
                print(f"R_{stg} vs {r}: {b['geo']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] n {b['n']} per seed "
                      + " ".join(f"{k}:{v:.3f}" for k, v in b["per_seed"].items()))
    print("prediction", json.dumps(res["prediction"]))
