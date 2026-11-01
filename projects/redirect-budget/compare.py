"""Measured shifts vs bounds (PREREG sec 5-7): scout clauses C1-C4 per row, the RANK status test, baselines, forward test.

Measured, per head and prompt, in the trained run (capture, test prompts): TV = 1/2 sum_j |a_on,j - a_off,j| at the
answer token; logit oscillation Omega_meas = max_j dz_j - min_j dz_j, dz = z_on - z_off (all keys j <= answer).
Comparison unit: U_h = mean over prompts of the measured value, B_h = mean over prompts of the bound; under iff U <= B.

    python compare.py   (after bounds.py)   -> data/summary.json, proof-matrix.md rows (proof_matrix.py)
"""
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HD = 128
RUNS = {"mlp8_none_s0": "17", "mlp8_none_s1": "17", "mlp8_none_s2": "17",
        "grader_s0": "06", "grader_s1": "06", "monitor_s0": "06", "monitor_s1": "06"}
FORWARD = ["mlp8_none_s3", "mlp8_none_s4"]
BLOCKS = {"17": list(range(20, 28)), "06": [26, 27]}
VROWS = ("S", "Q", "K", "C")


def measured(cap, blocks):
    tv, om = [], []
    for b in blocks:
        a = cap[f"b{b}_attn"].astype(np.float64)                      # [2,n,H,T]
        tv.append(0.5 * np.abs(a[1] - a[0]).sum(-1))
        qc = cap[f"b{b}_qhat"].astype(np.float64)                     # [2,n,H,128]
        kc = cap[f"b{b}_khat"].astype(np.float64)                     # [2,n,G,T,128]
        H, G = qc.shape[2], kc.shape[2]
        kq = np.repeat(kc, H // G, axis=2)
        z = np.einsum("cnhd,cnhtd->cnht", qc, kq) / np.sqrt(HD)
        dz = z[1] - z[0]
        om.append(dz.max(-1) - dz.min(-1))
    return np.concatenate(tv, 1), np.concatenate(om, 1)               # [n, heads]


def top3(names, score, tiebreak=None):
    """indices of the 3 largest scores; ties within 1e-6 (relative) broken by tiebreak (desc) then head index."""
    score = np.asarray(score, np.float64)
    tb = np.zeros_like(score) if tiebreak is None else np.asarray(tiebreak, np.float64)
    key = sorted(range(len(names)), key=lambda j: (-round(score[j] / max(abs(score).max(), 1e-300), 6), -tb[j], j))
    return [names[j] for j in key[:3]]


def status_test(rank, car):
    """PREREG sec 7 on the frozen RANK top-3 per model and the carriers given."""
    hits = {}
    for k in rank["17"]["top3"]:
        hits[k] = {r: bool(car[r]["defined"] and car[r]["carrier"] in rank[RUNS[r]]["top3"][k]) for r in RUNS}
    H = sum(hits["RANK"].values())
    # (i) last-layer random: one 3-set of the 16 last-layer heads per model, exact P(hits >= H)
    last = {"17": [f"27.{h}" for h in range(16)], "06": [f"27.{h}" for h in range(16)]}
    sets = list(itertools.combinations(range(16), 3))
    cnt = 0
    for s17 in sets:
        h17 = sum(car[r]["defined"] and car[r]["carrier"] in [last["17"][x] for x in s17] for r in RUNS if RUNS[r] == "17")
        for s06 in sets:
            h06 = sum(car[r]["defined"] and car[r]["carrier"] in [last["06"][x] for x in s06] for r in RUNS if RUNS[r] == "06")
            cnt += (h17 + h06) >= H
    p_last = cnt / len(sets) ** 2
    base_hits = {k: sum(hits[k].values()) for k in ("base_sink_mass", "opq", "opk", "sink_divergence")}
    ties = {k: rank["17"]["top3"][k] == rank["17"]["top3"]["RANK"] and rank["06"]["top3"][k] == rank["06"]["top3"]["RANK"]
            for k in base_hits}
    beaten = all(base_hits[k] < H and not ties[k] for k in base_hits) and p_last <= 0.05
    nund = sum(not car[r]["defined"] for r in RUNS)
    if nund >= 3:
        st = "UNCONFIRMED"
    elif H >= 5 and beaten:
        st = "CONFIRMED"
    else:
        st = "REJECTED"
    status = {"H": H, "hits_by_run": hits["RANK"], "p_last_layer_random_ge_H": p_last, "baseline_hits": base_hits,
              "baseline_top3_equal_to_RANK": ties, "reported_rows_hits": {k: sum(hits[k].values()) for k in ("RANK0", "RANK_Q", "RANK_C", "PROXY")},
              "undefined_runs": [r for r in RUNS if not car[r]["defined"]], "distinct_defined_carriers":
              sorted({(RUNS[r], car[r]["carrier"]) for r in RUNS if car[r]["defined"]}), "status": st}
    return status


def main() -> None:
    bj = json.load(open(HERE / "data" / "bounds.json"))
    car = json.load(open(HERE / "data" / "carriers.json"))
    out = {"bounds_py_sha256": bj["bounds_py_sha256"], "runs": {}}
    per_run_heads = {}
    for run, m in RUNS.items():
        cap = np.load(HERE / "raw" / "capture" / f"{m}_{run}.npz")
        tv, om = measured(cap, BLOCKS[m])
        bz = np.load(HERE / "data" / "bounds" / f"{run}.npz")
        names = [str(h) for h in bz["heads"]]
        per_run_heads[run] = names
        r = {"n_prompts": int(tv.shape[0]), "heads": {}}
        for j, h in enumerate(names):
            d = {"U_tv": float(tv[:, j].mean()), "U_logit": float(om[:, j].mean())}
            for row in VROWS:
                d[f"B_{row}_tv"] = float(bz[f"{row}_tv"][:, j].mean())
                d[f"B_{row}_logit"] = float(bz[f"{row}_logit"][:, j].mean())
            r["heads"][h] = d
        for row in VROWS:
            for sp, U in (("tv", tv), ("logit", om)):
                B = bz[f"{row}_{sp}"]
                r[f"cell_share_under_{row}_{sp}"] = float((U <= B).mean())
                r[f"max_cell_ratio_{row}_{sp}"] = float((U / np.maximum(B, 1e-300)).max())
        (HERE / "data" / "measured").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(HERE / "data" / "measured" / f"{run}.npz", heads=np.array(names), tv=tv, logit=om)
        out["runs"][run] = r

    # ---- scout clauses C1-C4 for every row and both spaces ----
    rng = np.random.default_rng(0)
    draws = {m: [rng.choice(len(per_run_heads[next(r for r in RUNS if RUNS[r] == m)]), 3, replace=False) for _ in range(50)]
             for m in ("17", "06")}
    clauses = {}
    for row in VROWS:
        for sp in ("tv", "logit"):
            under = [out["runs"][r]["heads"][h][f"U_{sp}"] <= out["runs"][r]["heads"][h][f"B_{row}_{sp}"]
                     for r in RUNS for h in out["runs"][r]["heads"]]
            c2 = {}
            hits = 0
            for r in RUNS:
                c = car[r]
                hd = out["runs"][r]["heads"]
                names = per_run_heads[r]
                t3 = top3(names, [hd[h][f"B_{row}_{sp}"] for h in names])
                hit = c["defined"] and c["carrier"] in t3
                hits += hit
                if c["defined"]:
                    U, B = hd[c["carrier"]][f"U_{sp}"], hd[c["carrier"]][f"B_{row}_{sp}"]
                    c2[r] = {"carrier": c["carrier"], "U": U, "B": B, "B_over_U": B / U if U > 0 else None,
                             "ok": bool(U <= B <= 10 * U), "top3": t3, "hit": bool(hit)}
                else:
                    c2[r] = {"carrier": None, "top3": t3, "hit": False}
            # C4: 50 random 3-sets per model, scored on all its runs
            rh = []
            for k in range(50):
                n = 0
                for r in RUNS:
                    c = car[r]
                    names = per_run_heads[r]
                    if c["defined"] and names.index(c["carrier"]) in draws[RUNS[r]][k]:
                        n += 1
                rh.append(n)
            clauses[f"{row}_{sp}"] = {"C1_share_heads_under": float(np.mean(under)), "C1": bool(np.mean(under) >= 0.95),
                                      "C2": bool(all(v.get("ok", False) for v in c2.values() if v["carrier"])),
                                      "C2_rows": c2, "C3_hits": int(hits), "C3": bool(hits >= 5),
                                      "C4_share_draws_ge": float(np.mean(np.array(rh) >= hits)),
                                      "C4": bool(np.mean(np.array(rh) >= hits) <= 0.05)}
            cl = clauses[f"{row}_{sp}"]
            cl["prereg_holds"] = bool(cl["C1"] and cl["C2"] and cl["C3"] and cl["C4"])
    out["scout_clauses"] = clauses
    nund = sum(not car[r]["defined"] for r in RUNS)
    s = clauses["S_tv"]
    out["prereg_outcome"] = "unsettled" if nund >= 3 else ("held" if s["prereg_holds"] else "failed")

    # ---- RANK status test (sec 7) and baselines ----
    status = {}
    rank = {}
    for m, tag in (("17", "rank_17"), ("06", "rank_06_monitor")):
        bz = np.load(HERE / "data" / "bounds" / f"{tag}.npz")
        names = [str(h) for h in bz["heads"]]
        hj = bj[tag]
        cap = np.load(HERE / "raw" / "capture" / (f"17_base_dev.npz" if m == "17" else "06_base_monitor.npz"))
        a = np.concatenate([cap[f"b{b}_attn"].astype(np.float64) for b in BLOCKS[m]], 2)   # [2,n,heads,T]
        scores = {"RANK": bz["RANK_logit"].mean(0), "RANK0": bz["RANK0_logit"].mean(0), "RANK_Q": bz["RANK_Q_logit"].mean(0),
                  "RANK_C": bz["RANK_C_logit"].mean(0), "PROXY": bz["PROXY_logit"].mean(0),
                  "base_sink_mass": a[0, :, :, 0].mean(0),
                  "opq": np.array([hj[h]["opq"] for h in names]), "opk": np.array([hj[h]["opk"] for h in names]),
                  "sink_divergence": np.abs((a[1, :, :, 0] - a[0, :, :, 0]).mean(0))}
        t = {k: top3(names, v, tiebreak=(scores["opq"] if k == "opk" else None)) for k, v in scores.items()}
        third = np.sort(scores["RANK"])[-3]
        rank[m] = {"names": names, "top3": t, "RANK_scores": dict(zip(names, map(float, scores["RANK"]))),
                   "heads_within_5pct_of_3rd": int((scores["RANK"] >= 0.95 * third).sum()),
                   "saturated_share": float(np.mean([hj[h]["c1_saturated_share_block"] for h in names]))}
    if (HERE / "data" / "bounds" / "rank_06_grader.npz").exists():
        g = np.load(HERE / "data" / "bounds" / "rank_06_grader.npz")["RANK_logit"].mean(0)
        rank["06"]["grader_vs_monitor_rank_maxdiff"] = float(np.abs(g - np.load(HERE / "data" / "bounds" / "rank_06_monitor.npz")["RANK_logit"].mean(0)).max())
    status = status_test(rank, car)
    # Kasper #11768, reported only: s1 counted with its argmax 27.12 (ratio rule at 1.15 instead of 1.20)
    car_alt = {r: dict(v) for r, v in car.items()}
    car_alt["mlp8_none_s1"]["defined"] = True
    status["sensitivity_s1_defined_ratio_1p15"] = status_test(rank, car_alt)
    st, H, p_last, base_hits, ties = status["status"], status["H"], status["p_last_layer_random_ge_H"], status["baseline_hits"], status["baseline_top3_equal_to_RANK"]
    out["rank"] = rank
    out["status"] = status
    # Felix #11773: place of each defined carrier under every frozen ranking (1 = top); Tove #11772: headroom used
    scores_by = {}
    for m, tag in (("17", "rank_17"), ("06", "rank_06_monitor")):
        bz = np.load(HERE / "data" / "bounds" / f"{tag}.npz")
        names = [str(h) for h in bz["heads"]]
        cap = np.load(HERE / "raw" / "capture" / (f"17_base_dev.npz" if m == "17" else "06_base_monitor.npz"))
        a = np.concatenate([cap[f"b{b}_attn"].astype(np.float64) for b in BLOCKS[m]], 2)
        scores_by[m] = {"names": names, "RANK": bz["RANK_logit"].mean(0), "RANK0": bz["RANK0_logit"].mean(0),
                        "RANK_Q": bz["RANK_Q_logit"].mean(0), "RANK_C": bz["RANK_C_logit"].mean(0), "PROXY": bz["PROXY_logit"].mean(0),
                        "base_sink_mass": a[0, :, :, 0].mean(0), "opq": np.array([bj[tag][h]["opq"] for h in names]),
                        "opk": np.array([bj[tag][h]["opk"] for h in names]),
                        "sink_divergence": np.abs((a[1, :, :, 0] - a[0, :, :, 0]).mean(0))}
    table, headroom = {}, {}
    for r in RUNS:
        c = car[r]
        if not c["defined"]:
            continue
        m = RUNS[r]
        sb = scores_by[m]
        names = sb["names"]
        row = {}
        for k, v in sb.items():
            if k == "names":
                continue
            tb = sb["opq"] if k == "opk" else np.zeros(len(names))   # PREREG sec 7 (iv): GQA ties by ||W_Q diag g||op
            order = sorted(range(len(names)), key=lambda j: (-v[j], -tb[j], j))
            row[k] = 1 + [names[j] for j in order].index(c["carrier"])
        hd = out["runs"][r]["heads"]
        row["measured_tv"] = 1 + sorted(names, key=lambda h: (-hd[h]["U_tv"], names.index(h))).index(c["carrier"])
        row["measured_logit"] = 1 + sorted(names, key=lambda h: (-hd[h]["U_logit"], names.index(h))).index(c["carrier"])
        row["n_heads"] = len(names)
        table[r] = {"carrier": c["carrier"], "place": row}
        rc = dict(zip(names, sb["RANK_C"]))
        rk = dict(zip(names, sb["RANK"]))
        top = rank[m]["top3"]["RANK"]
        headroom[r] = {"carrier": c["carrier"],
                       "carrier_used_of_RANK_C": hd[c["carrier"]]["U_logit"] / rc[c["carrier"]],
                       "carrier_used_of_RANK": hd[c["carrier"]]["U_logit"] / rk[c["carrier"]],
                       "carrier_used_of_row_C": hd[c["carrier"]]["U_logit"] / hd[c["carrier"]]["B_C_logit"],
                       "top3_used_of_RANK_C": {h: hd[h]["U_logit"] / rc[h] for h in top},
                       "top3_used_of_RANK": {h: hd[h]["U_logit"] / rk[h] for h in top},
                       "median_head_used_of_RANK_C": float(np.median([hd[h]["U_logit"] / rc[h] for h in names]))}
    out["carrier_places"] = table
    out["headroom"] = headroom
    out["spearman_RANK_vs_measured_logit"] = {}
    from scipy.stats import spearmanr
    for r in RUNS:
        m = RUNS[r]
        names = scores_by[m]["names"]
        hd = out["runs"][r]["heads"]
        out["spearman_RANK_vs_measured_logit"][r] = float(spearmanr(scores_by[m]["RANK"], [hd[h]["U_logit"] for h in names])[0])
    # forward test (sec 8)
    out["forward"] = {r: {"carrier": car[r]["carrier"], "defined": car[r]["defined"],
                          "hit": bool(car[r]["defined"] and car[r]["carrier"] in rank["17"]["top3"]["RANK"])}
                      for r in FORWARD if r in car}
    (HERE / "data" / "summary.json").write_text(json.dumps(out, indent=1))
    print("STATUS", st, "H", H, "p_last", round(p_last, 4), "baselines", base_hits, "ties", ties)
    print("RANK top3", {m: rank[m]["top3"]["RANK"] for m in rank}, "within5%", {m: rank[m]["heads_within_5pct_of_3rd"] for m in rank})
    for r, t in table.items():
        print("places", r, t)
    for r, h in headroom.items():
        print("headroom", r, {k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in h.items()})
    print("prereg", out["prereg_outcome"], {k: (v["C1_share_heads_under"], v["C2"], v["C3_hits"], v["C4_share_draws_ge"]) for k, v in clauses.items()})


if __name__ == "__main__":
    main()
