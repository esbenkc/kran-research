"""Frozen status rule of PREREG.md section 2 + secondary reads S1-S3, S5-S7 (S4 subspace angles: subspace.py).

Reads data/stage1/*.json, data/k4/*.json (and data/k2/*.json if present); writes data/analysis.json.

    .venv/bin/python analyze.py            # -> data/analysis.json, prints the status
    .venv/bin/python analyze.py --fixture  # synthetic runs only (self-test of the rule), writes nothing
"""
import argparse
import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as st

HERE = Path(__import__("os").environ.get("TT_ROOT", str(Path(__file__).parent)))
CAP = 1536
REFS = ("fresh", "shuf", "coin")
PARENT_CACHE = "70c4703aa2bab795eb5bc34b54be6c16ac3e62ec2ef361841c2598c351de67ba"
PORT = (266, 326)
LAW_R, LAW_BAND = 2 ** -0.956, (0.40, 0.66)
LOGIT_THR = math.log(0.3435 / 0.6565) - math.log(0.2435 / 0.7565)  # 0.486 (S6)
PREREG_SEEDS = (5, 6, 7)


def switch_time(ts, D, thr=0.10):
    """Parent pair rule on an ordered check list: first t with D >= thr at t and at the next check."""
    for i in range(len(ts) - 1):
        if D[i] >= thr and D[i + 1] >= thr:
            return int(ts[i])
    return None


def tint(v):
    """Mean and two-sided 95% t-interval over seeds."""
    v = np.asarray(v, float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, float("nan"), float("nan"), n
    h = float(st.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return m, m - h, m + h, n


def lnT(T):
    return math.log(max(CAP if T is None else T, 8))


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def load(sub):
    out = {}
    for f in sorted(glob.glob(str(HERE / "data" / sub / "*.json"))):
        if f.endswith("adapters_sha256.json"):
            continue
        d = json.load(open(f))
        out[(d["arm"], d["seed"])] = d
    return out


def stage2_reads(d):
    """Per-tail reads from one stage-2 run JSON."""
    cv = [c for c in d["curve"] if "D_B" in c]
    ts = [c["t"] for c in cv]
    T = switch_time(ts, [c["D_B"] for c in cv])
    assert T == d["T_B"], (d["name"], T, d["T_B"])
    gl0 = logit(np.mean(d["p_base_off"])) - logit(np.mean(d["p_base_on"]))
    Dl = [logit(c["px_off"]) - logit(c["px_B"]) - gl0 for c in cv]
    a_ret = [{"t": c["t"], "gap_A": c["gap_A"]} for c in d["curve"] if "gap_A" in c]
    incomplete = T is None and d["steps_run"] < d["cap"]  # cut by the 20:00 wrapup before the cap: not finished
    return {"T": T, "censored": T is None and not incomplete, "incomplete": incomplete, "T_logit": switch_time(ts, Dl, LOGIT_THR), "D0": cv[0]["D_B"],
            "gapB0": cv[0]["gap_B"], "pxoff0": cv[0]["px_off"], "pxoff_before_resolve": d["px_off_before_resolve"],
            "b": d["b"], "steps_run": d["steps_run"], "A_retention": a_ret}


def ratios(T, main_seeds, ref, key="T"):
    """Paired ln R_s,r = ln max(T_main,8) - ln max(T_r,8) over kept MAIN seeds that ref also has."""
    pairs = {s: lnT(T[("main", s)][key]) - lnT(T[(ref, s)][key]) for s in main_seeds if ("main", s) in T and (ref, s) in T}
    m, lo, hi, n = tint(list(pairs.values()))
    return {"mean_lnR": m, "lo": lo, "hi": hi, "n": n, "R": math.exp(m), "R_lo": math.exp(lo) if n > 1 else None,
            "R_hi": math.exp(hi) if n > 1 else None, "per_seed": {str(s): math.exp(v) for s, v in pairs.items()}}


def status_of(T, main_seeds, gates):
    ln_half = math.log(0.5)
    rr = {r: ratios(T, main_seeds, r) for r in REFS}
    cens = {arm: sum(1 for (a, s), v in T.items() if a == arm and v["censored"] and (a != "main" or s in main_seeds))
            for arm in ("main",) + REFS}
    usable = {r: cens["main"] <= 1 and cens[r] <= 1 for r in REFS}
    enough = len(main_seeds) >= 3 and all(rr[r]["n"] >= 3 for r in REFS)
    conf = enough and all(usable[r] and rr[r]["hi"] < ln_half for r in REFS)
    rej = enough and any(usable[r] and rr[r]["lo"] > ln_half for r in ("shuf", "fresh"))
    s = "CONFIRMED" if conf else ("REJECTED" if rej else "UNCONFIRMED")
    gates_ok = all(gates[g]["pass"] for g in gates)
    if s != "UNCONFIRMED" and not gates_ok:
        s = "UNCONFIRMED"
    return s, rr, cens, usable, enough


def analyse(S1, K, K2=None):
    out = {"cap": CAP, "runs": [], "notes": []}
    # stage 1: MAIN A switch by 256
    s1 = {k: {"T_A": v["T_A"], "pxoff_end": v["curve"][-1]["px_off"], "gapA_end": v["curve"][-1]["gap_A"]}
          for k, v in S1.items()}
    main_seeds = sorted(s for (a, s), v in s1.items() if a == "main" and v["T_A"] is not None)
    dropped = sorted(s for (a, s), v in s1.items() if a == "main" and v["T_A"] is None)
    out["stage1"] = {f"{a}_s{s}": v for (a, s), v in sorted(s1.items())}
    out["main_seeds_kept"], out["main_seeds_dropped"] = main_seeds, dropped
    res = {}
    for kB, runs in ((4, K), (2, K2)):
        if not runs:
            continue
        T_all = {k: stage2_reads(v) for k, v in runs.items()}
        out["notes"] += [f"k{kB} {a}_s{s} incomplete (wrapup at {v['steps_run']}), left out"
                         for (a, s), v in T_all.items() if v["incomplete"]]
        T = {k: v for k, v in T_all.items() if not v["incomplete"]}
        # sensitivity (Vera #31441, Iris #31442, Kasper #31443): incomplete tails enter censored at their last update
        T_sens = {k: (dict(v, T=v["steps_run"], censored=True) if v["incomplete"] else v) for k, v in T_all.items()}
        any_run = next(iter(runs.values()))
        g1 = float(np.mean(any_run["p_base_off"]))
        gates = {"G1": {"dev_px_off_parent_b": g1, "pass": 0.30 <= g1 <= 0.50},
                 "G2a": {"cache_sha256": any_run["cache_sha256"],
                         "pass": all(v["cache_sha256"] == PARENT_CACHE for v in runs.values())}}
        fresh = sorted(T[("fresh", s)]["T"] if T[("fresh", s)]["T"] is not None else CAP
                       for s in sorted(s for (a, s) in T if a == "fresh"))
        med = float(np.median(fresh))
        if kB == 4:
            gates["PORT"] = {"fresh_T": fresh, "fresh_median_T": med, "range": PORT, "pass": PORT[0] <= med <= PORT[1]}
        status, rr, cens, usable, enough = status_of(T, main_seeds, gates)
        sens_status, sens_rr, *_ = status_of(T_sens, main_seeds, gates)
        geo = {arm: math.exp(np.mean([lnT(v["T"]) for (a, s), v in T.items() if a == arm]))
               for arm in ("main",) + REFS}
        law = ratios(T, main_seeds, "fresh")
        law_read = ("held" if law["R_lo"] is not None and LAW_BAND[0] <= law["R_lo"] and law["R_hi"] <= LAW_BAND[1]
                    else "failed" if law["R_lo"] is not None and (law["R_hi"] < LAW_BAND[0] or law["R_lo"] > LAW_BAND[1])
                    else "unsettled")
        Tl = {k: {"T": v["T_logit"], "censored": v["T_logit"] is None} for k, v in T.items()}
        r_logit = {r: ratios(Tl, main_seeds, r) for r in REFS}
        res[f"k{kB}"] = {
            "status_rule": status if kB == 4 else None, "gates": gates, "ratios": rr,
            "incomplete_sensitivity": {"n_incomplete": sum(v["incomplete"] for v in T_all.values()),
                                       "status": sens_status,
                                       "ratios": {r: {k: sens_rr[r][k] for k in ("R", "R_lo", "R_hi", "n")} for r in REFS}}, "censored": cens,
            "ref_usable": usable, "minimums_met": enough, "geo_T": geo,
            "law_read": {"predicted_R": LAW_R, "band": LAW_BAND, "R": law["R"], "R_lo": law["R_lo"],
                         "R_hi": law["R_hi"], "outcome": law_read},
            "output_path_read": {r: {"geo_fresh_over_ref": geo["fresh"] / geo[r],
                                     "within_10pct": abs(geo["fresh"] / geo[r] - 1) <= 0.10} for r in ("shuf", "coin")},
            "zero_shot_D0": {arm: [T[(arm, s)]["D0"] for s in sorted(s for (a, s) in T if a == arm)]
                             for arm in ("main",) + REFS},
            "A_retention": {arm: {str(s): {"gap_A_t0": v["A_retention"][0]["gap_A"], "gap_A_last": v["A_retention"][-1]["gap_A"],
                                           "t_last": v["A_retention"][-1]["t"]}
                                  for (a_, s), v in sorted(T.items()) if a_ == arm}
                            for arm in ("main",) + REFS},  # S7 = main (frozen); the references: exploratory
            "prompt_level": prompt_level({k: v for k, v in runs.items() if k in T}, main_seeds),
            "logit_units": {"ratios": r_logit, "T": {f"{a}_s{s}": v["T"] for (a, s), v in sorted(Tl.items())}},
        }
        for (a, s), v in sorted(T.items()):
            out["runs"].append({"arm": a, "seed": s, "kB": kB, "T": v["T"], "censored": v["censored"], "cap": CAP,
                                "gapB0": v["gapB0"], "D0": v["D0"], "pxoff0": v["pxoff0"],
                                "pxoff_before_resolve": v["pxoff_before_resolve"], "b": v["b"],
                                "T_logit": v["T_logit"], "steps_run": v["steps_run"],
                                "TA": s1.get((a, s), {}).get("T_A"),
                                "A_retention": v["A_retention"] if a == "main" else None})
        if kB == 4:
            out["status"] = status
            # prereg outcome: seeds 5,6,7 each <= 0.5 x geomean T of SHUF-A and of COIN (all seeds), + S4 (subspace.py)
            def ok(s):
                return ("main", s) in T and math.exp(lnT(T[("main", s)]["T"])) <= 0.5 * min(geo["shuf"], geo["coin"])
            out["prereg_timing"] = {"seeds_5_7": {str(s): ok(s) for s in PREREG_SEEDS},
                                    "all_5_7": all(ok(s) for s in PREREG_SEEDS),
                                    "seeds_5_9": {str(s): ok(s) for s in range(5, 10)}}
    out.update(res)
    lr = [res[k]["law_read"]["outcome"] for k in ("k4", "k2") if k in res]
    out["law_read_overall"] = ("failed" if "failed" in lr else "held" if lr and all(x == "held" for x in lr)
                               else "unsettled")  # PREREG S1: held at every size run; failed if outside at any size
    return out


def boot_T(d, idx):
    """T_B per bootstrap resample of the 64 dev prompts (idx [n_boot, 64]); None-switch -> the last check (censored)."""
    cv = [c for c in d["curve"] if "D_B" in c]
    ts = np.array([c["t"] for c in cv])
    po, pb = np.array([c["p_off"] for c in cv]), np.array([c["p_B"] for c in cv])  # [checks, 64]
    g0 = np.asarray(d["p_base_off"])[idx].mean(1) - np.asarray(d["p_base_on"])[idx].mean(1)  # [n_boot]
    D = po[:, idx].mean(2) - pb[:, idx].mean(2) - g0[None]  # [checks, n_boot]
    hit = (D[:-1] >= 0.10) & (D[1:] >= 0.10)
    first = np.where(hit.any(0), hit.argmax(0), -1)
    T = np.where(first >= 0, ts[np.maximum(first, 0)], ts[-1]).astype(float)
    return np.log(np.maximum(T, 8)), int((first < 0).sum())


def prompt_level(runs, main_seeds, n_boot=2000):
    """TRUST RULES 1: 2,000 resamples of the 64 dev prompts (numpy seed 0); mean ln R per reference; 2.5/97.5 pct."""
    idx = np.random.default_rng(0).integers(0, 64, size=(n_boot, 64))
    L, cens = {}, 0
    for k, d in runs.items():
        L[k], c = boot_T(d, idx)
        cens += c
    out = {}
    for r in REFS:
        seeds = [s for s in main_seeds if ("main", s) in L and (r, s) in L]
        m = np.mean([L[("main", s)] - L[(r, s)] for s in seeds], 0)
        out[r] = {"mean_lnR": float(np.mean(m)), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
                  "R_lo": float(np.exp(np.percentile(m, 2.5))), "R_hi": float(np.exp(np.percentile(m, 97.5))),
                  "n_seeds": len(seeds)}
    out["n_boot"], out["run_resamples_without_switch"] = n_boot, cens
    return out


def fixture():
    """Synthetic check of the rule: true R 0.3 -> CONFIRMED, 1.0 -> REJECTED."""
    rng = np.random.default_rng(0)
    for trueR, want in ((0.3, "CONFIRMED"), (1.0, "REJECTED"), (0.5, None)):
        T = {}
        for s in range(5, 10):
            base = 296 * math.exp(rng.normal(0, 0.08))
            for arm in REFS:
                T[(arm, s)] = {"T": int(round(base * math.exp(rng.normal(0, 0.04)) / 8) * 8), "censored": False}
            T[("main", s)] = {"T": int(round(base * trueR * math.exp(rng.normal(0, 0.04)) / 8) * 8), "censored": False}
        gates = {"G": {"pass": True}}
        s, rr, *_ = status_of(T, list(range(5, 10)), gates)
        print(f"fixture true R {trueR}: {s}", {r: round(rr[r]["R"], 3) for r in REFS})
        if want:
            assert s == want, (trueR, s)
        s2, *_ = status_of(T, list(range(5, 10)), {"G": {"pass": False}})
        assert s2 == "UNCONFIRMED"
    T[("main", 5)] = {"T": None, "censored": True}
    T[("main", 6)] = {"T": None, "censored": True}
    s, *_ = status_of(T, list(range(5, 10)), {"G": {"pass": True}})
    assert s == "UNCONFIRMED", s
    print("fixture OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    if a.fixture:
        fixture()
        return
    out = analyse(load("stage1"), load("k4"), load("k2") or None)
    files = sorted(glob.glob(str(HERE / "data" / "*" / "*.json")))
    out["inputs_sha256"] = {str(Path(f).relative_to(HERE)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                            for f in files if not f.endswith("analysis.json")}
    (HERE / "data" / "analysis.json").write_text(json.dumps(out, indent=1))
    print("STATUS", out.get("status"))
    for k in ("k4", "k2"):
        if k in out:
            print(k, {r: (round(v["R"], 3), v["R_lo"] and round(v["R_lo"], 3), v["R_hi"] and round(v["R_hi"], 3))
                      for r, v in out[k]["ratios"].items()}, "geo", {a: round(v, 1) for a, v in out[k]["geo_T"].items()},
                  "gates", {g: v["pass"] for g, v in out[k]["gates"].items()}, "law", out[k]["law_read"]["outcome"])


if __name__ == "__main__":
    main()
