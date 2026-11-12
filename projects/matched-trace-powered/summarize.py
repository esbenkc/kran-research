"""matched-trace: status inputs from the raw files only (numpy; seconds). PREREG v1.1 @ 1b1edb23.

    /usr/bin/python3 summarize.py            -> data/summary.json + proof-matrix.md

Inputs: data/runs/joint_s0/curve.jsonl (dev checks every 50 updates + training windows), data/runs/joint_s0/run.json,
data/reads/raw/joint_s0_final.npz (R8 TEST joint read, read.py joint()), optional data/reads/raw/start_test.npz (the
same read of the start adapter). Points = x100 in the post; this file keeps fractions.
DEVIATIONS D1 (7463868a): y_P primary = mean dP x (c_notX - c_X) (key y_P_rb); void if the direct-minus-RB CI excludes 0.

Definitions (PREREG 'Reads', 'Status rule'):
  y_R = mean(c_nb - c_lb)   real trace, population switch (' Need' banned at token 0 in both builds)
  y_P = mean(c_n  - c_nc)   planted cue, population switch (none vs none + per-row cue, delta 0.10)
  D_R = mean max(0, 1 - exp(sum_t lp_lb_nb - lp_nb))   exact MC TV of the real trace on the banned pair
  D_P = mean(min(pX_off + delta, 0.99) - pX_off)      exact token-1 TV of the cue
  s   = y / (D x cheat_off), cheat_off = mean c_nb (R) or mean c_n (P)   ceiling share; ratio of means per resample
  Prompt bootstrap: 2,000 resamples of TEST prompts (seed 0), the same resample for every quantity (paired).
  P onset = first dev check (every 50) with dev_forced_switch >= 0.05 whose next check is also >= 0.05; onset
  exposure = cum_P_tv_mass at the first of the two.
  R TV mass = sum over 50-update windows of probed rows x mean(dev_D at the window's two ends) (trapezoid).
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUN = HERE / "data" / "runs" / "joint_s0"
FINAL = HERE / "data" / "reads" / "raw" / "joint_s0_final.npz"
START = HERE / "data" / "reads" / "raw" / "start_test.npz"
NB, SEED, ONSET = 2000, 0, 0.05


def boot_idx(groups):
    ug = np.unique(groups)
    rows = [np.where(groups == g)[0] for g in ug]
    rng = np.random.default_rng(SEED)
    return ug, rows, rng.integers(0, len(ug), (NB, len(ug)))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def read_cells(path, delta=0.10, seeds=None):
    z = np.load(path)
    m = z["split"] == "test"
    if seeds is not None:  # replicate cells: independent CRN draw halves of the same read
        m &= np.isin(z["seed"], seeds)
    g = z["prompt_idx"][m]
    c = {k: z[k][m].astype(np.float64) for k in ("c_n", "c_nb", "c_lb", "c_nlb", "c_nc", "c_X", "c_notX", "c_lc")}
    tvr = np.maximum(0.0, 1.0 - np.exp((z["lp_lb_nb"][m].astype(np.float64) - z["lp_nb"][m]).sum(1)))
    dP = np.minimum(z["pX_off"][m] + delta, 0.99) - z["pX_off"][m]
    ug, rows, B = boot_idx(g)
    # per-prompt sums and counts -> every bootstrap statistic is a ratio of resampled sums (paired across quantities)
    per = lambda x: np.array([x[r].sum() for r in rows])
    n = np.array([len(r) for r in rows], np.float64)
    S = {k: per(v) for k, v in c.items()}
    S["tvR"], S["dP"] = per(tvr), per(dP)
    # Felix's identity (proposed DEVIATIONS D1, #19735): the cue moves only p(X) at token 0, so per row
    # E[c_n - c_nc] = (p_on - p_off) x (E[c | not X] - E[c | X]) = dP x (c_notX - c_X): same estimand, lower variance
    S["yPrb"] = per(dP * (c["c_notX"] - c["c_X"]))

    def stat(Sx, nx):
        mean = lambda k: Sx[k].sum(-1) / nx.sum(-1)
        yR, yP = mean("c_nb") - mean("c_lb"), mean("c_n") - mean("c_nc")
        DR, DP = mean("tvR"), mean("dP")
        sR, sP = yR / (DR * mean("c_nb")), yP / (DP * mean("c_n"))
        sPrb = mean("yPrb") / (DP * mean("c_n"))
        return {"y_R": yR, "y_P": yP, "y_R_text": mean("c_nb") - mean("c_nlb"), "D_R": DR, "D_P": DP, "s_R": sR,
                "s_P": sP, "s_P_minus_s_R": sP - sR, "y_P_rb": mean("yPrb"), "s_P_rb": sPrb,
                "s_P_rb_minus_s_R": sPrb - sR, "y_P_minus_y_P_rb": yP - mean("yPrb"), "forced_switch": mean("c_notX") - mean("c_X"),
                "cheat_none": mean("c_n"), "cheat_none_banned": mean("c_nb"), "cheat_leaky_banned": mean("c_lb"),
                "cheat_cue": mean("c_nc"), "cheat_joint": mean("c_lc")}

    pt = stat(S, n)
    bs = stat({k: v[B] for k, v in S.items()}, n[B])
    out = {k: {"v": float(pt[k]), "ci": ci(bs[k])} for k in pt}
    out["n_rows"], out["n_prompts"], out["R"] = int(m.sum()), int(len(ug)), int(len(np.unique(z["seed"][m])))
    return out


def pair_swap_null(path):
    """Control: swap the probe-off and probe-on labels of each TEST prompt's real-trace pairs at random (a sign flip
    of the prompt's mean CRN gap); 2,000 draws. Under no switch the observed y_R sits inside this band."""
    z = np.load(path)
    m = z["split"] == "test"
    g, d = z["prompt_idx"][m], (z["c_nb"][m] - z["c_lb"][m]).astype(np.float64)
    ug = np.unique(g)
    per = np.array([d[g == u].mean() for u in ug])
    rng = np.random.default_rng(1)
    null = (per[None] * rng.choice([-1.0, 1.0], (NB, len(ug)))).mean(1)
    obs = float(per.mean())
    return {"observed": obs, "band": ci(null), "p": float((np.abs(null) >= abs(obs)).mean())}


def curve_cells():
    cv = [json.loads(l) for l in open(RUN / "curve.jsonl")]
    run = json.load(open(RUN / "run.json")) if (RUN / "run.json").exists() else {}
    checks = [r for r in cv if r["step"] % 50 == 0]
    rows_per_update = 32  # 4 probed prompts x G 8 (run.json args P 8, G 8)
    rmass, last = 0.0, None
    for r in checks:
        if last is not None and "dev_D" in r and "dev_D" in last:
            rmass += rows_per_update * (r["step"] - last["step"]) * 0.5 * (r["dev_D"] + last["dev_D"])
        r["cum_R_tv_mass"] = rmass
        last = r
    onset = None
    for i in range(len(checks) - 1):
        a, b = checks[i], checks[i + 1]
        if a.get("dev_forced_switch", -1) >= ONSET and b.get("dev_forced_switch", -1) >= ONSET:
            onset = {"step": a["step"], "confirmed_at": b["step"], "cue_tv_mass": a.get("cum_P_tv_mass", 0.0),
                     "forced": [a["dev_forced_switch"], b["dev_forced_switch"]]}
            break
    keep = ("step", "gap", "gap_P", "dev_forced_switch", "dev_D", "cheat_none", "cheat_leaky", "cheat_cue",
            "cum_P_tv_mass", "cum_R_tv_mass", "cum_probed_rows", "cue_p_off_mean", "cue_cap_rows", "x_rate_unprobed",
            "x_rate_probed", "train_cheat_unprobed", "train_cheat_probed", "train_cheat_probed_x",
            "train_cheat_probed_notx", "s")
    return {"checks": [{k: r.get(k) for k in keep} for r in checks], "onset": onset,
            "steps_done": run.get("steps_done"), "stop_clock": run.get("stop_clock"),
            "cum_P_tv_mass_end": (checks[-1].get("cum_P_tv_mass") if checks else None)}


def main():
    s = {"prereg": "PREREG.md v1.1 @ 1b1edb23", "curve": curve_cells()}
    on = s["curve"]["onset"]
    if FINAL.exists():
        f = read_cells(FINAL)
        s["final"] = f
        s["null_pair_swap"] = pair_swap_null(FINAL)
        s["final_half_r0_3"] = read_cells(FINAL, seeds=[0, 1, 2, 3])
        s["final_half_r4_7"] = read_cells(FINAL, seeds=[4, 5, 6, 7])
        steps = s["curve"]["steps_done"] or 0
        s["cue_tv_mass_final"] = 0.10 * 32 * steps  # exact up to capped rows (cue_cap_rows in the curve)
        # DEVIATIONS D1: the Rao-Blackwellised y_P is primary unless the direct-minus-RB CI excludes 0 (then void)
        lo, hi = f["y_P_minus_y_P_rb"]["ci"]
        d1 = not (lo > 0 or hi < 0)
        sp, dk = ("s_P_rb", "s_P_rb_minus_s_R") if d1 else ("s_P", "s_P_minus_s_R")
        s["primary_P"] = {"D1_valid": d1, "s_P_key": sp, "diff_key": dk, "y_P_key": "y_P_rb" if d1 else "y_P"}
        learned = on is not None and f[sp]["ci"][0] > 0
        yes = learned and f[dk]["ci"][0] > 0 and f["y_R"]["ci"][1] < 0.03
        no = f["y_R"]["ci"][0] >= 0.03 or (-f[dk]["ci"][1] > 0 and f["s_R"]["ci"][0] > 0)
        s["status"] = "CONFIRMED" if yes and not no else "REJECTED" if no else "UNCONFIRMED"
        s["status_inputs"] = {"P_onset_rule_met": on is not None, "s_P_ci_low_gt_0": f[sp]["ci"][0] > 0,
                              "paired_s_P_minus_s_R_ci_low_gt_0": f[dk]["ci"][0] > 0,
                              "y_R_ci_high_lt_3pts": f["y_R"]["ci"][1] < 0.03,
                              "y_R_ci_low_ge_3pts": f["y_R"]["ci"][0] >= 0.03,
                              "s_R_minus_s_P_ci_low_gt_0_and_s_R_ci_low_gt_0":
                                  bool(-f[dk]["ci"][1] > 0 and f["s_R"]["ci"][0] > 0)}
        s["exposure_ratio_final_over_onset"] = (s["cue_tv_mass_final"] / on["cue_tv_mass"]) if on else None
        # Kasper #19748: the real trace's OWN summed mass at the final adapter (trapezoid on the 50-grid, the last
        # check's dev_D carried flat to the final update) over P's onset mass; this is the ratio that backs a YES
        ck = s["curve"]["checks"]
        rm = ck[-1]["cum_R_tv_mass"] + 32 * (steps - ck[-1]["step"]) * (ck[-1]["dev_D"] or 0.0) if ck else None
        s["R_tv_mass_final"] = rm
        s["exposure_ratio_R_final_over_P_onset"] = (rm / on["cue_tv_mass"]) if (on and rm is not None) else None
        st = read_cells(START) if START.exists() else None
        if st:
            s["start"] = st
        s["scout_clauses"] = {
            "P_ge_5pts": f[s["primary_P"]["y_P_key"]]["v"] >= 0.05,
            "R_lt_1p5pts": f["y_R"]["v"] < 0.015,
            "two_x_onset": bool(on and s["exposure_ratio_final_over_onset"] >= 2),
            "probe_on_cheat_moves_ge_5pts": (abs(f["cheat_leaky_banned"]["v"] - st["cheat_leaky_banned"]["v"]) >= 0.05
                                             if st else None),
            "n_seeds": 1, "note": "n = 1 joint run; '2 of 2 seeds' cells were not run -> outcome at best unsettled"}
    (HERE / "data" / "summary.json").write_text(json.dumps(s, indent=1))
    matrix(s)
    print(json.dumps({k: s.get(k) for k in ("status", "status_inputs", "exposure_ratio_final_over_onset",
                                             "exposure_ratio_R_final_over_P_onset", "R_tv_mass_final")}, indent=1))
    print("onset", on)
    if "final" in s:
        for k in ("y_R", "y_P_rb", "y_P", "y_P_minus_y_P_rb", "s_R", "s_P_rb", "s_P", "s_P_rb_minus_s_R",
                  "s_P_minus_s_R", "D_R", "D_P", "forced_switch"):
            print(k, s["final"][k])


def matrix(s):
    L = ["# proof matrix — matched-trace", "", "Built by summarize.py from the raw files; points = x100. One joint run "
         "(seed 0), so seed never varies; the headline rests on the two cues of the same run, on independent CRN "
         "draw halves of the end read (rebuild cells), and on the start-policy control.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    on = s["curve"]["onset"]
    L.append(f"| planted onset (dev forced switch >= 5 at two checks) | 0 | planted cue | none | "
             f"{'update %d (cue TV mass %.0f), forced %.1f / %.1f pts' % (on['step'], on['cue_tv_mass'], 100*on['forced'][0], 100*on['forced'][1]) if on else 'not reached by update %s' % s['curve']['steps_done']} "
             f"| data/runs/joint_s0/curve.jsonl |")
    if "final" in s:
        f = s["final"]
        nl = s.get("null_pair_swap")
        if nl:
            L.append(f"| null | 0 | real trace, probe-on/off labels swapped per prompt (sign flip, 2,000) | pair-swap null | "
                     f"null 95% band [{100*nl['band'][0]:+.2f}, {100*nl['band'][1]:+.2f}] pts; observed {100*nl['observed']:+.2f}; "
                     f"two-sided p {nl['p']:.2f} | data/reads/raw/joint_s0_final.npz |")
        fmt = lambda k: "%+.2f [%+.2f, %+.2f] pts" % (100 * f[k]["v"], 100 * f[k]["ci"][0], 100 * f[k]["ci"][1])
        sfmt = lambda k: "%+.3f [%+.3f, %+.3f]" % (f[k]["v"], f[k]["ci"][0], f[k]["ci"][1])
        art = "data/reads/raw/joint_s0_final.npz"
        L.append(f"| headline | 0 | real trace (y_R) | none | {fmt('y_R')}, D_R {100*f['D_R']['v']:.1f} pts | {art} |")
        L.append(f"| headline | 0 | planted cue (y_P, D1 primary: D_P x forced) | none | {fmt('y_P_rb')}, D_P {100*f['D_P']['v']:.1f} pts | {art} |")
        L.append(f"| headline | 0 | planted cue (y_P, plain CRN) | none | {fmt('y_P')} | {art} |")
        L.append(f"| D1 check | 0 | direct minus RB y_P | none | {fmt('y_P_minus_y_P_rb')} (D1 void if CI excludes 0) | {art} |")
        L.append(f"| headline | 0 | ceiling share s_P - s_R (paired, D1) | none | {sfmt('s_P_rb_minus_s_R')} | {art} |")
        L.append(f"| headline | 0 | ceiling share s_R / s_P (D1) | none | {sfmt('s_R')} / {sfmt('s_P_rb')} | {art} |")
        L.append(f"| diagnostic | 0 | forced cue switch | none | {fmt('forced_switch')} | {art} |")
        for hk, lab in (("final_half_r0_3", "CRN draws 0-3"), ("final_half_r4_7", "CRN draws 4-7")):
            h = s[hk]
            hf = lambda k: "%+.2f [%+.2f, %+.2f]" % (100 * h[k]["v"], 100 * h[k]["ci"][0], 100 * h[k]["ci"][1])
            L.append(f"| replicate | 0 | {lab}: y_R / y_P_rb (pts) | none | {hf('y_R')} / {hf('y_P_rb')}; s_P-s_R "
                     f"{h['s_P_rb_minus_s_R']['v']:+.3f} [{h['s_P_rb_minus_s_R']['ci'][0]:+.3f}, {h['s_P_rb_minus_s_R']['ci'][1]:+.3f}] | {art} |")
        L.append(f"| status | 0 | frozen rule | none | {s['status']} | data/summary.json |")
        if "start" in s:
            st = s["start"]
            g = lambda k: "%+.2f [%+.2f, %+.2f] pts" % (100 * st[k]["v"], 100 * st[k]["ci"][0], 100 * st[k]["ci"][1])
            L.append(f"| control | - | start policy, real trace | untrained start | {g('y_R')} | data/reads/raw/start_test.npz |")
            L.append(f"| control | - | start policy, planted cue (D1) | untrained start | {g('y_P_rb')} | data/reads/raw/start_test.npz |")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
