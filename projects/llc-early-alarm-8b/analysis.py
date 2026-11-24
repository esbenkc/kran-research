"""Frozen-rule analysis (PREREG.md FROZEN RULE (2)-(8), commit 1b81a5f3e). Written by Iris from the PREREG text and
the JSON files only; it imports nothing from rl.py / llc.py / sgld.py.

Inputs (all under the project dir):
  data/llc/<size>/<run>_u<step>.json   one SGLD estimate: keys run, step, nbeta, burn, draws, losses (per chain, every
                                        step incl. burn-in: minibatch loss at the point before the update).
                                        <size> in k2, k3, k4; <run> = m8_k?_s? (main) or m8_c1_s? (control, read with
                                        that size's cue rows).
  data/runs/<run>.json                  rl.py output of a main run: T_dense (scored T), dense_curve.
  data/race/<size>/<run>.json           race readouts at the grid points: {"steps": [...], "D": [...],
                                        "ba_norm": [...], "grad_norm": [...]} (C1's D at THAT size's cue rows).
Output: data/analysis.json (+ a short printout).

Rules implemented (PREREG section 2):
  LLC(t) = n beta * mean of the pooled post-burn-in minibatch losses (2 chains x 75 draws), L(w*) = 0.
  chain SE(t) = n beta * batch-means SE over the pooled 150 draws (chain 1 then chain 2), 5 batches of 30.
  converged(t): |mean L over draws 51-75 - mean L over draws 26-50| <= 2 x batch-means SE of the loss (pooled chains).
  mains:  z(t) = (LLC_run(t) - mean_C1(t)) / max(sigma, SE_run(t)), sigma^2 = mean over the size's grid points of the
          3-seed C1 sample variance (ddof 1).
  alarm:  first grid t >= 32 with z >= 3 at t and at the next grid point; t_alarm = t. lead = T - t_alarm, counts if >= 8.
  C1 c vs others a, b: z_c(t) = (c(t) - (a(t)+b(t))/2) / max(sigma_ab, SE_c(t)), sigma_ab^2 = mean over points of
          (a-b)^2 / 2; same pair rule; the clause holds if no C1 seed alarms at any size.
  status precedence: UNCONFIRMED (gate fail / main missing) > REJECTED (< 2/3 leads at >= 2 sizes) >
          CONFIRMED (>= 2/3 leads at all 3 sizes and control clause holds) > UNCONFIRMED.
  law:    slope of ln(lead) on ln(T) over runs with a counted lead; needs >= 2 leads per size; seed-level 95%
          t-interval over per-seed slopes; held = interval inside [0.8, 1.2]; failed = interval disjoint; else unsettled.
  race:   rule (4) with sigma from the 3 C1 seeds (no chain-SE floor) on D, ba_norm, grad_norm.

    .venv/bin/python analysis.py [--gate data/gate/frozen.json] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SIZES = ("k2", "k3", "k4")
SEEDS = (5, 6, 7)
C1_SEEDS = (5, 6, 7)
PARENT_T = {"k2": 72, "k3": 136, "k4": 296}
STEP = {"k2": 8, "k3": 16, "k4": 32}
Z_ALARM, MIN_LEAD, WINDOW0 = 3.0, 8, 32
LAW_BAND = (0.8, 1.2)
RACE = ("D", "ba_norm", "grad_norm")


def grid(size: str) -> list[int]:
    """PREREG (2): every STEP from 32 to 1.25 x parent T (k2 32..88, k3 32..160, k4 32..352)."""
    top = 1.25 * PARENT_T[size]
    return [t for t in range(WINDOW0, int(top) + 1, STEP[size])]


def t_crit(df: int) -> float:
    """Two-sided 95% Student t quantile (scipy-free; table for the small df used here)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
    return table.get(df, 1.96)


def estimate(d: dict) -> dict:
    """LLC, batch-means SE and convergence from the saved per-draw losses of one estimate."""
    burn, draws, nbeta = int(d["burn"]), int(d["draws"]), float(d["nbeta"])
    post = [np.asarray(ch[burn:burn + draws], dtype=float) for ch in d["losses"]]
    assert all(len(p) == draws for p in post), "short chain"
    pooled = np.concatenate(post)                         # 150 draws: chain 1 then chain 2
    bm = pooled.reshape(5, -1).mean(axis=1)               # 5 batches of 30
    se_loss = float(bm.std(ddof=1) / math.sqrt(len(bm)))
    late = np.concatenate([p[50:75] for p in post]).mean()   # draws 51-75 (1-based)
    early = np.concatenate([p[25:50] for p in post]).mean()  # draws 26-50
    return {"llc": nbeta * float(pooled.mean()), "se": nbeta * se_loss, "converged": bool(abs(late - early) <= 2 * se_loss),
            "chains": [nbeta * float(p.mean()) for p in post]}


def first_pair(z: list[float], ts: list[int], thr: float = Z_ALARM, sign: int = 1):
    """First t with sign*z >= thr at t and at the next grid point (both must exist)."""
    for i in range(len(ts) - 1):
        if sign * z[i] >= thr and sign * z[i + 1] >= thr:
            return ts[i]
    return None


def load_llc(root: Path) -> dict:
    """{size: {run: {step: estimate}}}"""
    out: dict = {}
    for size in SIZES:
        for f in sorted((root / "data" / "llc" / size).glob("*_u*.json")):
            d = json.loads(f.read_text())
            out.setdefault(size, {}).setdefault(d["run"], {})[int(d["step"])] = estimate(d)
    return out


def analyse(llc: dict, T: dict, race: dict, gate_pass: bool | None) -> dict:
    res: dict = {"sizes": {}, "missing": []}
    lead_counts, ctrl_alarms, conv = {}, [], {"converged": 0, "total": 0}
    law_pts = []
    for size in SIZES:
        ts = grid(size)
        S = llc.get(size, {})
        c1 = {s: S.get(f"m8_c1_s{s}", {}) for s in C1_SEEDS}
        miss_c1 = [(f"m8_c1_s{s}", t) for s in C1_SEEDS for t in ts if t not in c1[s]]
        if miss_c1:
            res["missing"] += [f"{size}:{r}@{t}" for r, t in miss_c1]
        # C1 band per grid point
        C = np.array([[c1[s][t]["llc"] if t in c1[s] else np.nan for t in ts] for s in C1_SEEDS])
        SEc = np.array([[c1[s][t]["se"] if t in c1[s] else np.nan for t in ts] for s in C1_SEEDS])
        mean_c1 = np.nanmean(C, axis=0)
        sigma = float(math.sqrt(np.nanmean(np.nanvar(C, axis=0, ddof=1))))
        out = {"grid": ts, "mean_c1": mean_c1.tolist(), "sigma": sigma, "runs": {}, "controls": {}}
        n_lead = 0
        for seed in SEEDS:
            run = f"m8_{size}_s{seed}"
            R = S.get(run, {})
            if any(t not in R for t in ts) or run not in T or T[run] is None:
                res["missing"].append(f"{size}:{run}")
                out["runs"][run] = {"missing": True}
                continue
            lam = np.array([R[t]["llc"] for t in ts])
            se = np.array([R[t]["se"] for t in ts])
            for t in ts:
                conv["total"] += 1
                conv["converged"] += int(R[t]["converged"])
            z = ((lam - mean_c1) / np.maximum(sigma, se)).tolist()
            ta = first_pair(z, ts)
            Tr = int(T[run])
            lead = (Tr - ta) if ta is not None else None
            counts = lead is not None and lead >= MIN_LEAD
            # second-of-pair lead (reported, not scored)
            lead2 = (Tr - ts[ts.index(ta) + 1]) if ta is not None else None
            neg = first_pair(z, ts, sign=-1)
            n_lead += int(counts)
            rr = {"T": Tr, "llc": lam.tolist(), "se": se.tolist(), "z": z, "t_alarm": ta, "lead": lead,
                  "lead_counts": counts, "lead_second_of_pair": lead2, "neg_pair_t": neg,
                  "z_at_last_pre": next((z[i] for i in range(len(ts) - 1, -1, -1) if ts[i] <= Tr - MIN_LEAD), None)}
            # race members, same rule, sigma from the 3 C1 seeds
            rr["race"] = {}
            for m in RACE:
                try:
                    xr = np.array([race[size][run][m][race[size][run]["steps"].index(t)] for t in ts], dtype=float)
                    xc = np.array([[race[size][f"m8_c1_s{s}"][m][race[size][f"m8_c1_s{s}"]["steps"].index(t)]
                                    for t in ts] for s in C1_SEEDS], dtype=float)
                except (KeyError, ValueError):
                    rr["race"][m] = None
                    continue
                sg = float(math.sqrt(np.mean(np.var(xc, axis=0, ddof=1))))
                zm = ((xr - xc.mean(0)) / sg).tolist() if sg > 0 else [math.inf * np.sign(v) if v else 0.0
                                                                     for v in (xr - xc.mean(0))]
                tm = first_pair(zm, ts)
                rr["race"][m] = {"z": zm, "t_alarm": tm, "llc_not_later": (ta is not None and (tm is None or ta <= tm))}
            if counts:
                law_pts.append((size, seed, Tr, lead))
            out["runs"][run] = rr
        lead_counts[size] = n_lead
        # control clause: each C1 seed vs the other two
        for s in C1_SEEDS:
            o = [x for x in C1_SEEDS if x != s]
            if any(t not in c1[x] for x in C1_SEEDS for t in ts):
                out["controls"][f"m8_c1_s{s}"] = {"missing": True}
                continue
            c = np.array([c1[s][t]["llc"] for t in ts])
            se_c = np.array([c1[s][t]["se"] for t in ts])
            a = np.array([c1[o[0]][t]["llc"] for t in ts])
            b = np.array([c1[o[1]][t]["llc"] for t in ts])
            sab = float(math.sqrt(np.mean((a - b) ** 2 / 2)))
            zc = ((c - (a + b) / 2) / np.maximum(sab, se_c)).tolist()
            tc = first_pair(zc, ts)
            out["controls"][f"m8_c1_s{s}"] = {"z": zc, "sigma_ab": sab, "t_alarm": tc,
                                              "neg_pair_t": first_pair(zc, ts, sign=-1)}
            if tc is not None:
                ctrl_alarms.append(f"{size}:m8_c1_s{s}@{tc}")
        resid = C - mean_c1
        r1 = [np.corrcoef(row[:-1], row[1:])[0, 1] for row in resid if np.all(np.isfinite(row)) and row.std() > 0]
        out["c1_resid_lag1_autocorr"] = float(np.mean(r1)) if r1 else None
        res["sizes"][size] = out
    res["lead_counts"] = lead_counts
    res["control_alarms"] = ctrl_alarms
    res["control_clause_holds"] = not ctrl_alarms
    res["convergence"] = conv
    main_missing = [m for m in res["missing"] if ":m8_k" in m]
    fails = [s for s in SIZES if lead_counts.get(s, 0) < 2]
    if gate_pass is False:
        status = "UNCONFIRMED"
        why = "gate failed"
    elif main_missing or gate_pass is None:
        status, why = "UNCONFIRMED", ("main run missing: " + ", ".join(main_missing)) if main_missing else "gate result missing"
    elif len(fails) >= 2:
        status, why = "REJECTED", f"lead in < 2/3 seeds at {fails}"
    elif not fails and res["control_clause_holds"]:
        status, why = "CONFIRMED", "lead in >= 2/3 seeds at all 3 sizes, control clause holds"
    else:
        status, why = "UNCONFIRMED", (f"lead fails at exactly 1 size {fails}" if fails else
                                      f"control clause fails: {ctrl_alarms}")
    res["status"], res["status_reason"] = status, why
    # law
    per_size = {s: sum(1 for p in law_pts if p[0] == s) for s in SIZES}
    law = {"points": law_pts, "per_size": per_size}
    if all(per_size[s] >= 2 for s in SIZES):
        x = np.log([p[2] for p in law_pts]); y = np.log([p[3] for p in law_pts])
        law["pooled_slope"] = float(np.polyfit(x, y, 1)[0])
        seed_slopes = []
        for seed in SEEDS:
            pts = [p for p in law_pts if p[1] == seed]
            if len({p[0] for p in pts}) >= 2:
                seed_slopes.append(float(np.polyfit(np.log([p[2] for p in pts]), np.log([p[3] for p in pts]), 1)[0]))
        law["seed_slopes"] = seed_slopes
        if len(seed_slopes) >= 2:
            m, sd = float(np.mean(seed_slopes)), float(np.std(seed_slopes, ddof=1))
            h = t_crit(len(seed_slopes) - 1) * sd / math.sqrt(len(seed_slopes))
            lo, hi = m - h, m + h
            law["seed_level"] = [m, lo, hi]
            if LAW_BAND[0] <= lo and hi <= LAW_BAND[1]:
                law["outcome"] = "held"
            elif hi < LAW_BAND[0] or lo > LAW_BAND[1]:
                law["outcome"] = "failed"
            else:
                law["outcome"] = "unsettled"
        else:
            law["outcome"] = "unsettled"
    else:
        law["outcome"] = "unsettled"
    res["law"] = law
    # seed-level headline: z at the last grid point <= T-8, per size, t-interval over seeds
    sl = {}
    for size in SIZES:
        zs = [r["z_at_last_pre"] for r in res["sizes"][size]["runs"].values() if not r.get("missing")
              and r["z_at_last_pre"] is not None]
        if len(zs) >= 2:
            m, sd = float(np.mean(zs)), float(np.std(zs, ddof=1))
            h = t_crit(len(zs) - 1) * sd / math.sqrt(len(zs))
            sl[size] = [m, m - h, m + h, len(zs)]
    res["seed_level_z_last_pre"] = sl
    # race wording rule
    llc_first = all(all(v is None or v["llc_not_later"] for v in r["race"].values())
                    for s in SIZES for r in res["sizes"][s]["runs"].values() if not r.get("missing") and r["t_alarm"])
    res["wording_alarm_ok"] = bool(llc_first and res["control_clause_holds"] and status == "CONFIRMED")
    return res


def selftest():
    """Synthetic checks of the rule: (a) pure noise -> no CONFIRMED; (b) a planted early step -> CONFIRMED."""
    rng = np.random.default_rng(0)

    def fake(step_at: dict | None):
        llc, race = {}, {}
        T = {}
        for size in SIZES:
            ts = grid(size)
            llc[size] = {}
            race[size] = {}
            for name in [f"m8_{size}_s{s}" for s in SEEDS] + [f"m8_c1_s{s}" for s in C1_SEEDS]:
                base = 50 + rng.normal(0, 1.0, size=len(ts))
                if step_at and "c1" not in name:
                    T[name] = PARENT_T[size]
                    base = base + np.array([6.0 if t >= step_at[size] else 0.0 for t in ts])
                elif "c1" not in name:
                    T[name] = PARENT_T[size]
                llc[size][name] = {t: {"llc": float(v), "se": 0.2, "converged": True} for t, v in zip(ts, base)}
                race[size][name] = {"steps": ts, "D": list(rng.normal(0, 1, len(ts))),
                                    "ba_norm": list(rng.normal(0, 1, len(ts))), "grad_norm": list(rng.normal(0, 1, len(ts)))}
        return llc, T, race

    llc, T, race = fake(None)
    a = analyse(llc, T, race, True)
    llc, T, race = fake({"k2": 48, "k3": 80, "k4": 160})
    b = analyse(llc, T, race, True)
    print("selftest null:", a["status"], a["status_reason"], "| planted:", b["status"], b["status_reason"],
          "law", b["law"]["outcome"], b["law"].get("seed_level"))
    assert a["status"] != "CONFIRMED" and b["status"] == "CONFIRMED"
    c = analyse(llc, T, race, False)
    assert c["status"] == "UNCONFIRMED"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="data/gate/frozen.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    llc = load_llc(HERE)
    T = {}
    for f in (HERE / "data" / "runs").glob("m8_k*_s*.json"):
        T[f.stem] = json.loads(f.read_text()).get("T_dense")
    race = {}
    for size in SIZES:
        for f in (HERE / "data" / "race" / size).glob("*.json"):
            race.setdefault(size, {})[f.stem] = json.loads(f.read_text())
    gp = HERE / a.gate
    gate_pass = json.loads(gp.read_text()).get("pass") if gp.exists() else None
    res = analyse(llc, T, race, gate_pass)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print("STATUS", res["status"], "-", res["status_reason"])
    print("leads per size", res["lead_counts"], "| control clause holds", res["control_clause_holds"],
          res["control_alarms"])
    print("law", res["law"]["outcome"], res["law"].get("seed_level"), "| convergence", res["convergence"])
    print("seed-level z at last point <= T-8:", res["seed_level_z_last_pre"])


if __name__ == "__main__":
    main()
