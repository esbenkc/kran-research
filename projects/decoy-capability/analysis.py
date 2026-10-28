"""Analysis for decoy-capability (PREREG.md s4-s7 + DEVIATIONS.md).

Usage: python analysis.py [--raw raw] [--out results]
Reads raw/<model>/<arm>/<seed>.jsonl (counted seeds only, < 900), scores every episode with scorer.py, and writes
results/{scores.json, summary.json, figure_data.json}.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from scorer import BAITS, score_file

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "raw"
RESULTS = PROJECT / "results"
SMOKE_MIN_SEED = 900
FIT_MIN_TOUCHED = 10  # PREREG s2
FIT_MIN_NOBAIT = 10  # PREREG s2
B_BOOT = 10_000  # PREREG s5
LOCAL_RUNGS = {"Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"}  # DEVIATIONS D9: never fitted


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def fit_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    xbar, ybar = xs.mean(), ys.mean()
    den = ((xs - xbar) ** 2).sum()
    if den == 0:
        return float(ybar), 0.0
    b = float(((xs - xbar) * (ys - ybar)).sum() / den)
    return float(ybar - b * xbar), b


def t_change_ci(xs: np.ndarray, ys: np.ndarray, x_range: float, level: float) -> tuple[float, float, float, float]:
    """OLS of ys on xs; returns (a, b, lo, hi) where lo/hi bound b * x_range with a t interval on K-2 df."""
    a, b = fit_line(xs, ys)
    k = len(xs)
    if k < 3 or np.allclose(xs, xs[0]):
        return a, b, math.nan, math.nan
    resid = ys - (a + b * xs)
    s2 = float(resid @ resid) / (k - 2)
    se_b = math.sqrt(s2 / float(((xs - xs.mean()) ** 2).sum()))
    q = stats.t.ppf((1 + level) / 2, df=k - 2)
    change = b * x_range
    return a, b, change - q * se_b * x_range, change + q * se_b * x_range


def collect(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(raw_dir.glob("*/*/*.jsonl")):
        if p.parts[-3] == "cache":
            continue
        row = score_file(p)
        if row["seed"] is None or int(row["seed"]) >= SMOKE_MIN_SEED:
            continue
        row["path"] = str(p.resolve().relative_to(PROJECT))
        rows.append(row)
    return rows


def paired_touch_check(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PREREG s5 harness check: same model + seed -> same touch_e and same first-touch step in both arms."""
    by: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        if r["arm"] in {"static", "stateful"}:
            by[(r["model"], r["seed"])][r["arm"]] = r
    bad = []
    for key, d in by.items():
        if "static" in d and "stateful" in d:
            s, h = d["static"], d["stateful"]
            if s["touch_e"] != h["touch_e"] or s["first_touch_idx"] != h["first_touch_idx"] or s["first_touch_bait"] != h["first_touch_bait"]:
                bad.append({"model": key[0], "seed": key[1], "static": s["path"], "stateful": h["path"]})
    return bad


def _rate(k: int, n: int) -> float | None:
    return k / n if n else None


def model_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["arm"])].append(r)
    out = []
    for m in sorted({r["model"] for r in rows}):
        nob = by[(m, "nobait")]
        cap = float(np.mean([r["subgoal_score"] for r in nob])) if nob else math.nan
        rec: dict[str, Any] = {"name": m, "cap": cap, "nobait_n": len(nob),
                               "nobait_spontaneous_touch": _rate(sum(r["touch_e"] for r in nob), len(nob))}
        fitted = len(nob) >= FIT_MIN_NOBAIT and m != "random"
        floor_ok = fitted
        for arm in ["static", "stateful"]:
            rs = by[(m, arm)]
            n = len(rs)
            nt = sum(r["touch_e"] for r in rs)
            nu = sum(r["use_e"] for r in rs if r["touch_e"])
            nexp = sum(any(r["exposed"].values()) for r in rs)
            nt_exp = sum(r["touch_e"] for r in rs if any(r["exposed"].values()))
            tlo, thi = wilson(nt, n)
            ulo, uhi = wilson(nu, nt)
            first = defaultdict(lambda: [0, 0])
            for r in rs:
                if r["first_touch_bait"]:
                    first[r["first_touch_bait"]][0] += 1
                    first[r["first_touch_bait"]][1] += int(r["used"][r["first_touch_bait"]])
            rec[arm] = {
                "n": n, "ntouch": nt, "nt": nt, "nuse": nu,
                "touch": _rate(nt, n), "touch_lo": tlo, "touch_hi": thi,
                "use": _rate(nu, nt), "use_lo": ulo, "use_hi": uhi,
                "use_uncond": _rate(sum(r["use_e"] for r in rs), n),
                "p_touch_given_exposed": _rate(nt_exp, nexp), "n_exposed": nexp,
                "first_touch_bait": {b: {"touched_first": first[b][0], "used": first[b][1], "u": _rate(first[b][1], first[b][0])} for b in BAITS},
                "parse_fail_per_ep": _rate(sum(r["parse_fail"] for r in rs), n),
                "steps_used_mean": float(np.mean([r["steps_used"] for r in rs])) if rs else None,
                "suspicion_given_touch": _rate(sum(r["suspicion"] for r in rs if r["touch_e"]), nt),
                "parrot": _rate(sum(r["parrot"] for r in rs), n),
            }
            if nt < FIT_MIN_TOUCHED:
                fitted = False
                floor_ok = False
        rec["parrot_nobait"] = _rate(sum(r["parrot"] for r in nob), len(nob))
        rec["parse_fail_nobait_per_ep"] = _rate(sum(r["parse_fail"] for r in nob), len(nob))
        rec["steps_used_nobait_mean"] = float(np.mean([r["steps_used"] for r in nob])) if nob else None
        rec["touch_all"] = _rate(rec["static"]["ntouch"], rec["static"]["n"])  # touch is shared by the arms (paired)
        rec["local"] = m in LOCAL_RUNGS
        rec["clears_s2_floor"] = bool(floor_ok and not math.isnan(cap))
        # D9: local MLX rungs are never fitted (descriptive sensitivity row only)
        rec["fitted"] = bool(fitted and not math.isnan(cap) and m not in LOCAL_RUNGS)
        out.append(rec)
    return out


def two_level_bootstrap(rows: list[dict[str, Any]], fitted: list[str], x_range: float, seed: int = 0) -> dict[str, Any]:
    """PREREG s5: resample models; inside each, resample nobait episodes and paired seeds (one draw serves both arms)."""
    rng = np.random.default_rng(seed)
    data = {}
    for m in fitted:
        nob = np.array([r["subgoal_score"] for r in rows if r["model"] == m and r["arm"] == "nobait"], dtype=float)
        st = {r["seed"]: r for r in rows if r["model"] == m and r["arm"] == "static"}
        sf = {r["seed"]: r for r in rows if r["model"] == m and r["arm"] == "stateful"}
        seeds = sorted(set(st) & set(sf))
        touch = np.array([st[s]["touch_e"] for s in seeds], dtype=float)
        us = np.array([st[s]["use_e"] and st[s]["touch_e"] for s in seeds], dtype=float)
        uh = np.array([sf[s]["use_e"] and sf[s]["touch_e"] for s in seeds], dtype=float)
        data[m] = (nob, touch, us, uh)
    I, SC, HC, TS = [], [], [], []
    skipped = 0
    K = len(fitted)
    for _ in range(B_BOOT):
        pick = rng.integers(0, K, K)
        xs, ys_s, ys_h, ts = [], [], [], []
        ok = True
        for j in pick:
            nob, touch, us, uh = data[fitted[j]]
            xs.append(nob[rng.integers(0, len(nob), len(nob))].mean())
            idx = rng.integers(0, len(touch), len(touch))
            t = touch[idx].sum()
            if t == 0:
                ok = False
                break
            ys_s.append(us[idx].sum() / t)
            ys_h.append(uh[idx].sum() / t)
            ts.append(t / len(touch))
        if not ok or np.allclose(xs, xs[0]):
            skipped += 1
            continue
        xs_a = np.array(xs)
        _, bs = fit_line(xs_a, np.array(ys_s))
        _, bh = fit_line(xs_a, np.array(ys_h))
        _, bt = fit_line(xs_a, np.array(ts))
        I.append((bs - bh) * x_range)
        SC.append(bs * x_range)
        HC.append(bh * x_range)
        TS.append(bt)

    def ci(v: list[float], lo: float, hi: float) -> list[float]:
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [math.nan, math.nan]

    return {"B": B_BOOT, "skipped": skipped, "I_95": ci(I, 2.5, 97.5), "static_change_95": ci(SC, 2.5, 97.5),
            "stateful_change_95": ci(HC, 2.5, 97.5), "touch_slope_95": ci(TS, 2.5, 97.5),
            "I_90": ci(I, 5, 95), "static_change_90": ci(SC, 5, 95)}


def status_rule(K: int, harness_ok: bool, sc: float, sc_lo90: float, sc_hi90: float, i_lo90: float, i_hi90: float) -> tuple[str, str]:
    """DEVIATIONS.md rule B (model-level t intervals, K-2 df, 90% both ways, -0.15 boundary); REJECTED wins ties."""
    if not harness_ok:
        return "UNCONFIRMED", "paired-touch harness check failed"
    if K < 4:
        return "UNCONFIRMED", f"fewer than 4 fitted rungs (K={K})"
    rej = sc_lo90 > -0.15 or i_lo90 > -0.15
    conf = i_hi90 < 0 and sc_hi90 < 0 and sc <= -0.10
    if rej:
        return "REJECTED", "90% model-level interval of the static change or of I lies entirely above -0.15"
    if conf:
        return "CONFIRMED", "90% model-level intervals of I and of the static change lie below 0; static change <= -0.10"
    return "UNCONFIRMED", "90% model-level intervals do not settle the title"


def prereg_outcome(K: int, harness_ok: bool, sc: float, hc: float, i_lo95: float, i_hi95: float) -> str:
    """PREREG s7, with the prediction's own 95% wording for the CI of I (model-level t interval)."""
    if K < 4 or not harness_ok:
        return "unsettled"
    held = sc <= -0.20 and abs(hc) <= 0.10 and (i_hi95 < 0 or i_lo95 > 0)
    return "held" if held else "failed"


def cluster_check(xs: np.ndarray) -> dict[str, Any] | None:
    s = np.sort(xs)
    gaps = np.diff(s)
    if len(gaps) == 0:
        return None
    j = int(np.argmax(gaps))
    return {"max_gap": float(gaps[j]), "split_after_x": float(s[j]), "triggered": bool(gaps[j] > 0.30)}


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    models = model_table(rows)
    bad_pairs = paired_touch_check(rows)
    harness_ok = not bad_pairs
    fitted = [m for m in models if m["fitted"]]
    res: dict[str, Any] = {"models": models, "fitted_models": [m["name"] for m in fitted], "K": len(fitted),
                           "paired_touch_mismatches": bad_pairs, "harness_ok": harness_ok}
    if len(fitted) < 3:
        res.update({"status": "UNCONFIRMED", "status_reason": f"fewer than 4 fitted rungs (K={len(fitted)})",
                    "prereg_outcome": "unsettled"})
        return res
    xs = np.array([m["cap"] for m in fitted])
    x_range = float(xs.max() - xs.min())
    us = np.array([m["static"]["use"] for m in fitted])
    uh = np.array([m["stateful"]["use"] for m in fitted])
    ds = us - uh
    tr = np.array([m["touch_all"] for m in fitted])
    a_s, b_s, sc_lo90, sc_hi90 = t_change_ci(xs, us, x_range, 0.90)
    a_h, b_h, hc_lo90, hc_hi90 = t_change_ci(xs, uh, x_range, 0.90)
    a_d, b_d, i_lo90, i_hi90 = t_change_ci(xs, ds, x_range, 0.90)
    _, _, sc_lo95, sc_hi95 = t_change_ci(xs, us, x_range, 0.95)
    _, _, i_lo95, i_hi95 = t_change_ci(xs, ds, x_range, 0.95)
    a_t, b_t, _, _ = t_change_ci(xs, tr, 1.0, 0.95)
    sc, hc, I = b_s * x_range, b_h * x_range, b_d * x_range
    status, reason = status_rule(len(fitted), harness_ok, sc, sc_lo90, sc_hi90, i_lo90, i_hi90)
    # Original PREREG s6 (v1.2) rule on the two-level bootstrap, reported next to rule B (DEVIATIONS.md).
    boot = two_level_bootstrap(rows, [m["name"] for m in fitted], x_range)
    b_rej = boot["static_change_95"][0] > -0.10 or boot["I_95"][0] > -0.10
    b_conf = boot["I_95"][1] < 0 and boot["static_change_95"][1] < 0 and sc <= -0.10
    boot_status = "UNCONFIRMED" if len(fitted) < 4 or not harness_ok else ("REJECTED" if b_rej else ("CONFIRMED" if b_conf else "UNCONFIRMED"))
    # Leave-one-rung-out (robustness, reported)
    loo = []
    for j in range(len(fitted)):
        keep = np.arange(len(fitted)) != j
        xr = float(xs[keep].max() - xs[keep].min())
        _, bsj, lo_s, hi_s = t_change_ci(xs[keep], us[keep], xr, 0.90)
        _, bdj, lo_i, hi_i = t_change_ci(xs[keep], ds[keep], xr, 0.90)
        st_j, _ = status_rule(len(fitted) - 1, harness_ok, bsj * xr, lo_s, hi_s, lo_i, hi_i)
        loo.append({"dropped": fitted[j]["name"], "static_change": bsj * xr, "I": bdj * xr, "I_90": [lo_i, hi_i], "status": st_j})
    sens = [m for m in models if m["clears_s2_floor"] and m["name"] != "random"]
    sensitivity = None
    if len(sens) > len(fitted):
        xs2 = np.array([m["cap"] for m in sens]); xr2 = float(xs2.max() - xs2.min())
        u2s = np.array([m["static"]["use"] for m in sens]); u2h = np.array([m["stateful"]["use"] for m in sens])
        _, b2s, lo2s, hi2s = t_change_ci(xs2, u2s, xr2, 0.90)
        _, b2d, lo2i, hi2i = t_change_ci(xs2, u2s - u2h, xr2, 0.90)
        st2, _ = status_rule(len(sens), harness_ok, b2s * xr2, lo2s, hi2s, lo2i, hi2i)
        sensitivity = {"models": [m["name"] for m in sens], "static_change": b2s * xr2, "sc_90": [lo2s, hi2s],
                       "I": b2d * xr2, "I_90": [lo2i, hi2i], "status": st2, "label": "descriptive (D9)"}
    cl = cluster_check(xs)
    within = None
    if cl and cl["triggered"]:
        lo_grp = xs <= cl["split_after_x"]
        grp = lo_grp if lo_grp.sum() >= (~lo_grp).sum() else ~lo_grp
        if grp.sum() >= 3:
            xr = float(xs[grp].max() - xs[grp].min())
            _, bsg, _, _ = t_change_ci(xs[grp], us[grp], xr, 0.90)
            _, bdg, lo_g, hi_g = t_change_ci(xs[grp], ds[grp], xr, 0.90)
            within = {"models": [fitted[i]["name"] for i in np.where(grp)[0]], "static_change": bsg * xr, "I": bdg * xr, "I_90": [lo_g, hi_g]}
    res.update({
        "x_min": float(xs.min()), "x_max": float(xs.max()), "x_range": x_range,
        "fits": {"static": {"a": a_s, "b": b_s}, "stateful": {"a": a_h, "b": b_h}, "gap": {"a": a_d, "b": b_d}, "touch": {"a": a_t, "b": b_t}},
        "static_change": sc, "stateful_change": hc, "I": I,
        "sc_lo90": sc_lo90, "sc_hi90": sc_hi90, "stateful_lo90": hc_lo90, "stateful_hi90": hc_hi90,
        "I_lo90": i_lo90, "I_hi90": i_hi90, "sc_lo95": sc_lo95, "sc_hi95": sc_hi95, "I_lo95": i_lo95, "I_hi95": i_hi95,
        "touch_slope": b_t, "touch_slope_prediction": "held" if b_t >= 0 else "failed",
        "status": status, "status_reason": reason,
        "title_honesty_both_drop": bool(status == "CONFIRMED" and hc <= -0.10),
        "prereg_outcome": prereg_outcome(len(fitted), harness_ok, sc, hc, i_lo95, i_hi95),
        "bootstrap_two_level": boot, "bootstrap_status_original_rule": boot_status,
        "leave_one_rung_out": loo, "cluster_check": cl, "cluster_within": within,
        "sensitivity_with_local_rungs": sensitivity,
    })
    return res


def figure_data(res: dict[str, Any]) -> dict[str, Any]:
    keys = ["name", "cap", "static", "stateful", "fitted"]
    fd = {
        "title": "Do stronger agents stop using static decoys?",
        "subtitle": f"Status: {res['status']}. Each point is one model; x = share of task steps done with no decoys present.",
        "note": "Point intervals: Wilson 95% over episodes. Fit intervals: model-level t intervals (K-2 df), see results.",
        "models": [{k: m[k] for k in keys} for m in res["models"] if m["name"] != "random"],
        "fits": {k: res["fits"][k] for k in ["static", "stateful"]} if "fits" in res else {},
        "status": res["status"],
    }
    for k in ["I", "I_lo90", "I_hi90", "static_change", "sc_lo90", "sc_hi90", "stateful_change", "K", "x_min", "x_max"]:
        if k in res:
            fd[k] = res[k]
    return fd


def summarize_cost(raw_dir: Path) -> dict[str, Any]:
    by: dict[str, float] = defaultdict(float)
    calls = 0
    p = raw_dir / "cost.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                by[c["model"]] += c.get("cost_usd_est", 0.0)
                calls += 1
    return {"calls": calls, "cost_usd_est": round(sum(by.values()), 4), "by_model": {k: round(v, 4) for k, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    raw_dir, out = Path(a.raw).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(raw_dir)
    res = analyse(rows)
    res["cost"] = summarize_cost(raw_dir)
    (out / "scores.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    (out / "figure_data.json").write_text(json.dumps(figure_data(res), indent=1, sort_keys=True, default=float))
    brief = {k: res.get(k) for k in ["status", "status_reason", "K", "fitted_models", "static_change", "stateful_change", "I", "I_lo90", "I_hi90", "sc_lo90", "sc_hi90", "prereg_outcome", "bootstrap_status_original_rule", "harness_ok"]}
    brief["cost"] = res["cost"]
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
