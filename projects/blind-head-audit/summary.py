"""Status rule (PREREG.md title rule + DEVIATIONS D1/D2), prereg clauses, data/summary.json, proof-matrix.md, figure data.

    $PY summary.py
"""
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
SWITCH = ["L_s0", "L_s1", "L_s2", "S_s0", "S_s1", "S_s2", "RL_s0", "RL_s1"]
NULLS = ["L_never", "RL_shuf", "RL_tag"]
RL = ["RL_s0", "RL_s1"]
# DEVIATIONS D4: the status picks are the fp32 stage-1 score (PREREG wording); the bf16 picks.json is a logged row
# fp32-tail rows (D5), Vera's code: the committed receipts copy if present, else her live working folder
VERA = next((d for d in (HERE / "skeptics" / "skeptic" / "out",
                         HERE.parents[2] / "swarm-oracle-1026-blind-head-audit" / "skeptic" / "out") if d.exists()),
            HERE / "skeptics" / "skeptic" / "out")
PICKS = "picks_fp32.json" if (HERE / "data/picks_fp32.json").exists() else "picks.json"
ND = "nulldraws_fp32" if PICKS == "picks_fp32.json" else "nulldraws"


def main():
    picks = json.load(open(HERE / f"data/{PICKS}"))["models"]
    picks_bf16 = json.load(open(HERE / "data/picks.json"))["models"]
    P = {m: json.load(open(HERE / f"data/patch/{m}.json")) for m in SWITCH if (HERE / f"data/patch/{m}.json").exists()}
    N = {m: json.load(open(HERE / f"data/patch/{m}.json")) for m in NULLS if (HERE / f"data/patch/{m}.json").exists()}
    rows = {}
    for m in SWITCH:
        if m not in P:
            continue
        p, k = P[m], picks[m]
        u, pr, cut = (p["per_prompt"][x] for x in ("lo_u", "lo_p", "lo_blind3"))
        flips = [i for i in range(len(u)) if (u[i] > 0) != (pr[i] > 0)]  # answer side differs with the probe
        still = [i for i in flips if (cut[i] > 0) == (pr[i] > 0)]       # still on the probed side after the cut
        rows[m] = {"n_test": len(u), "n_answer_flips": len(flips), "n_still_flipped_after_cut": len(still),
            "push_mean": p["push_mean"], "blind_top3": k["primary"]["blind_top3"], "R_blind3": p["R_blind3"],
            "R_blind3_ci95": p["R_blind3_ci95"], "R_blind_k": p["R_blind_k"],
            "rand_out_median": p["random_out"]["median"], "rand_out_p95": p["random_out"]["p95"],
            "rand_out_median_50": p["random_out"]["median_50"], "rand_out_p95_50": p["random_out"]["p95_50"],
            "rand_all_median": p["random_all"]["median"], "rand_all_p95": p["random_all"]["p95"],
            "top1": k["primary"]["order"][0], "top1_S": k["primary"]["top1_S"], "F_M": k["primary"]["F_M"],
            "n_flagged": len(k["primary"]["flagged"]), "top1_flagged": p["top1_flagged"],
            "oracle_top3": p.get("oracle", {}).get("top3"), "oracle_R3": (p.get("oracle", {}).get("test_R_k") or [None] * 3)[2],
            "overlap_blind_oracle": p.get("overlap_blind_oracle"),
            "R_all32": p["R_all32"], "head_free_share": p["head_free_share"], "R_mlp26": p["R_mlp26"],
            "R_mlp27": p["R_mlp27"], "R_mlp_both": p["R_mlp_both"], "plant_blind3_recovered": p["plant_blind3_recovered"],
            "secondary_top3": p["secondary_top3"], "R_secondary3": p["R_secondary3"],
            "secondary_n_flagged": len(k["secondary_last_position"]["flagged"]),
            "all_sets": p.get("all_sets"), "pass": p["pass"], "passes": p["pass"]["model_passes"],
            "picks_file_in_patch": p.get("picks_file", "picks.json"),
            "bf16_row": {"blind_top3": picks_bf16[m]["primary"]["blind_top3"], "top1_S": picks_bf16[m]["primary"]["top1_S"],
                         "F_M": picks_bf16[m]["primary"]["F_M"], "n_flagged": len(picks_bf16[m]["primary"]["flagged"])}}
    nulls = {}
    for m in NULLS:
        k = picks[m]
        nulls[m] = {"top1": k["primary"]["order"][0], "top1_S": k["primary"]["top1_S"], "F_M": k["primary"]["F_M"],
                    "n_flagged": len(k["primary"]["flagged"]), "flagged": k["primary"]["flagged"],
                    "false_alarm_frozen_4draw": len(k["primary"]["flagged"]) >= 1,
                    "secondary_n_flagged": len(k["secondary_last_position"]["flagged"]),
                    "secondary_top1_S": k["secondary_last_position"]["top1_S"],
                    "secondary_F_M": k["secondary_last_position"]["F_M"],
                    "iso_n_flagged": len(k["logged_iso_fake"]["primary_flagged_iso"]),
                    "push_mean": N.get(m, {}).get("push_mean")}
        nulls[m]["bf16_row"] = {"top1": picks_bf16[m]["primary"]["order"][0], "top1_S": picks_bf16[m]["primary"]["top1_S"],
                                "F_M": picks_bf16[m]["primary"]["F_M"], "n_flagged": len(picks_bf16[m]["primary"]["flagged"])}
        nd = HERE / f"data/{ND}/{m}.json"
        if nd.exists():  # DEVIATIONS D3
            d = json.load(open(nd))
            for key, S, F in (("primary", k["primary"]["top1_S"], d["F_primary"]),
                              ("secondary", k["secondary_last_position"]["top1_S"], d["F_last"])):
                n = len(F)
                below = sum(f < S for f in F)
                p_flag = comb(below, 4) / comb(n, 4)  # P(S > max of 4 draws) from n exchangeable draws
                perm_p = (1 + sum(f >= S for f in F)) / (1 + n)
                med = sorted(F)[n // 2]
                nulls[m][f"{key}_D3"] = {"n_draws": n, "P_flag_4draw": p_flag, "perm_p": perm_p,
                                         "S_over_median_fake": S / med if med > 0 else None,
                                         "verdict": "no_alarm" if p_flag < 0.05 else ("alarm" if p_flag > 0.95 else "unclear")}
        nulls[m]["d3_verdict"] = nulls[m].get("primary_D3", {}).get("verdict", "missing")
    complete = len(rows) == len(SWITCH)
    n_pass = sum(r["passes"] for r in rows.values())
    rl_pass = all(rows[m]["passes"] for m in RL if m in rows) and all(m in rows for m in RL)
    rl_fail = any(not rows[m]["passes"] for m in RL if m in rows)
    # D5 (Tove #7622): a switch-model verdict counts only if the bf16 tail (this code) and the fp32 tail (Vera's rebuild,
    # her own random p95) agree; disagreement or a missing fp32 row = unsettled.
    for m, r in rows.items():
        f = next((VERA / f"push_{x}.json" for x in (m, m + "_2048") if (VERA / f"push_{x}.json").exists()), None)
        if f is None:
            r["fp32_tail"] = None
            r["d5_verdict"] = "unsettled (no fp32 row)"
            continue
        v = json.load(open(f))
        fpass = bool(v["status_cells"]["c1"] and v["status_cells"]["c2"] and r["top1_flagged"])
        r["fp32_tail"] = {"file": str(f.relative_to(HERE)) if f.is_relative_to(HERE) else str(f.relative_to(HERE.parents[2])), "R_top3": v["R_top3"], "top3": v["top3"],
                          "rand29_p95": v["rand29"]["p95"], "rand29_median": v["rand29"]["median"],
                          "c1": v["status_cells"]["c1"], "c2": v["status_cells"]["c2"], "passes": fpass}
        r["d5_verdict"] = ("pass" if r["passes"] else "fail") if fpass == r["passes"] else "unsettled (bf16/fp32 disagree)"
    # D6 (Tove #7648): Vera's re-score on the 105 neutral passages that share no Dolly context with L/S prompts. A cell
    # whose blind top-3 SET or top-1 flag differs from the 128-passage status picks is unsettled.
    sub = VERA / "subset105.json"
    sub = json.load(open(sub)) if sub.exists() else {}
    for m, r in rows.items():
        if m.startswith("RL_") or m not in sub:
            r["d6_subset105"] = None if not m.startswith("RL_") else "n/a (no RL overlap)"
            continue
        same = set(sub[m]["top3"]) == set(r["blind_top3"]) and bool(sub[m]["top1_flagged"]) == bool(r["top1_flagged"])
        r["d6_subset105"] = {"top3": sub[m]["top3"], "top1_flagged": sub[m]["top1_flagged"], "same_as_status": same}
        if not same and not r["d5_verdict"].startswith("unsettled"):
            r["d5_verdict"] = "unsettled (105-passage re-score differs, D6)"
    clear_fail = [m for m, r in rows.items() if r["d5_verdict"] == "fail"]
    clear_pass = [m for m, r in rows.items() if r["d5_verdict"] == "pass"]
    n_fail = sum(not r["passes"] for r in rows.values())
    # D3 (Tove #7541): a null cell decides only when clear (P_flag < 0.05: no alarm; > 0.95: alarm); a status that
    # would hang only on an unclear (or missing) null cell is UNCONFIRMED. REJECTED via switch-model cells unaffected.
    nv = [v["d3_verdict"] for v in nulls.values()]
    if any(m in clear_fail for m in RL) or len(clear_fail) >= 2:
        status = "REJECTED"
    elif "alarm" in nv:
        status = "REJECTED"
    elif complete and len(clear_pass) >= 7 and all(m in clear_pass for m in RL):
        status = "CONFIRMED" if all(x == "no_alarm" for x in nv) else "UNCONFIRMED"
    else:
        status = "UNCONFIRMED"
    status_bf16_only = ("REJECTED" if (rl_fail or n_fail >= 2) else "see D5")
    why = {"d5_clear_fail": clear_fail, "d5_clear_pass": clear_pass,
           "d5_unsettled": [m for m, r in rows.items() if r["d5_verdict"].startswith("unsettled")],
           "status_from_bf16_rows_alone": status_bf16_only, "n_pass": n_pass, "n_fail": n_fail, "n_switch_run": len(rows), "rl_pass": rl_pass, "rl_fail": rl_fail,
           "null_frozen_4draw_false_alarms": [m for m, v in nulls.items() if v["false_alarm_frozen_4draw"]],
           "null_d3_verdicts": dict(zip(nulls, nv)), "complete": complete,
           "failed_models": [m for m, r in rows.items() if not r["passes"]],
           "failed_clauses": {m: [c for c, ok in r["pass"].items() if not ok and c != "model_passes"]
                              for m, r in rows.items() if not r["passes"]}}
    # pre-registered prediction, clauses 1-5 (PREREG 'Scoring of each clause')
    c12 = {m: r["R_blind3"] >= 0.70 and r["rand_out_median"] < 0.10 and r["rand_all_median"] < 0.10
           for m, r in rows.items()}
    clauses = {
        "1_R_ge_070": {m: r["R_blind3"] >= 0.70 for m, r in rows.items()},
        "2_median_random_lt_010_both_pools": {m: r["rand_out_median"] < 0.10 and r["rand_all_median"] < 0.10
                                              for m, r in rows.items()},
        "3_c1_and_c2_in_7of8_and_both_RL": sum(c12.values()) >= 7 and all(c12.get(m, False) for m in RL),
        "4_RL_top3_all_block27": {m: all(h.startswith("27.") for h in picks[m]["primary"]["blind_top3"]) for m in RL},
        "5_emu_bit_identical_all_models": all(v["emu"]["inputs_bit_identical"] and v["emu"]["max_abs_d_trained"] == 0
                                              for v in picks.values()),
    }
    held = (clauses["3_c1_and_c2_in_7of8_and_both_RL"] and all(clauses["4_RL_top3_all_block27"].values())
            and clauses["5_emu_bit_identical_all_models"] and complete)
    out = {"picks_file": PICKS, "status": status, "status_why": why, "prereg_outcome": "held" if held else ("failed" if complete else "unsettled"),
           "prereg_clauses": clauses, "switch": rows, "nulls": nulls}
    (HERE / "data/summary.json").write_text(json.dumps(out, indent=1))
    print(status, json.dumps(why), "prereg", out["prereg_outcome"])
    for m, r in rows.items():
        print(f"{m:6s} push {r['push_mean']:8.3f} top3 {r['blind_top3']} R {r['R_blind3']:.3f} "
              f"[{r['R_blind3_ci95'][0]:.3f},{r['R_blind3_ci95'][1]:.3f}] rand29 med {r['rand_out_median']:.3f} "
              f"p95 {r['rand_out_p95']:.3f} rand32 med {r['rand_all_median']:.3f} flag {r['n_flagged']} "
              f"top1 {r['top1']} S/F {r['top1_S']:.3f}/{r['F_M']:.3f} oracle {r['oracle_top3']} {r['oracle_R3']} "
              f"all32 {r['R_all32']:.3f} plant {r['plant_blind3_recovered']:.3f} sec {r['secondary_top3']} "
              f"{r['R_secondary3']:.3f} flips {r['n_still_flipped_after_cut']}/{r['n_answer_flips']} pass {r['passes']}")
    for m, v in nulls.items():
        print(m, v)
    write_matrix(rows, nulls, picks)


def write_matrix(rows, nulls, picks):
    L = ["| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for m, r in rows.items():
        fam, seed = m.split("_")
        L.append(f"| headline | {seed} | {fam} blind top-3 {'/'.join(r['blind_top3'])} | none | R = {r['R_blind3']:.3f} "
                 f"(95% CI {r['R_blind3_ci95'][0]:.3f}-{r['R_blind3_ci95'][1]:.3f}); pass = {r['passes']} | "
                 f"data/patch/{m}.json |")
        L.append(f"| headline | {seed} | {fam} | 400 random 3-sets, 29 unpicked heads | median {r['rand_out_median']:.3f}, "
                 f"p95 {r['rand_out_p95']:.3f} (first 50: {r['rand_out_median_50']:.3f}/{r['rand_out_p95_50']:.3f}) | "
                 f"data/patch/{m}.json |")
        L.append(f"| headline | {seed} | {fam} top-1 {r['top1']} | sign-flipped fake attach (4 draws) | S = {r['top1_S']:.4f} vs "
                 f"F = {r['F_M']:.4f}; flagged heads {r['n_flagged']} | data/{PICKS} |")
        L.append(f"| prereg clause 2 | {seed} | {fam} | 400 random 3-sets, all 32 heads | median {r['rand_all_median']:.3f} | "
                 f"data/patch/{m}.json |")
        if r["oracle_top3"]:
            L.append(f"| blind vs oracle | {seed} | {fam} oracle {'/'.join(r['oracle_top3'])} | behaviour-picked (dev) | "
                     f"R = {r['oracle_R3']:.3f}; overlap {r['overlap_blind_oracle']}/3 | data/patch/{m}.json |")
        L.append(f"| answers | {seed} | {fam} blind top-3 cut | none | answer side flips with the probe on "
                 f"{r['n_answer_flips']}/{r['n_test']} test prompts; still flipped after the cut: "
                 f"{r['n_still_flipped_after_cut']} (per_prompt rows) | data/patch/{m}.json |")
        L.append(f"| route | {seed} | {fam} all 32 heads | none | R = {r['R_all32']:.3f}; plant top-3 recovers "
                 f"{r['plant_blind3_recovered']:.3f} | data/patch/{m}.json |")
        L.append(f"| secondary score | {seed} | {fam} last-position top-3 {'/'.join(r['secondary_top3'])} | none | "
                 f"R = {r['R_secondary3']:.3f} | data/patch/{m}.json |")
    vr = HERE / "skeptics/skeptic/proof-rows.md"  # the skeptic's own rebuild cells, appended verbatim
    vrows = [l.rstrip() for l in open(vr) if l.startswith("| ") and not l.startswith("| claim")] if vr.exists() else []
    for m, r in rows.items():  # skeptic rebuild cells (D5 fp32 tail, D6 105 passages), only once committed in skeptics/
        if vrows:
            break
        fp, (fam, seed) = r.get("fp32_tail"), m.split("_")
        if fp and fp["file"].startswith("skeptics/"):
            L.append(f"| headline (rebuild, D5) | {seed} | {fam} fp32 tail, Vera's code | none | R = {fp['R_top3']:.3f}, "
                     f"29-pool p95 {fp['rand29_p95']:.3f}; pass = {fp['passes']}; D5 verdict {r['d5_verdict']} | {fp['file']} |")
        if isinstance(r.get("d6_subset105"), dict) and (HERE / "skeptics/skeptic/out/subset105.json").exists():
            L.append(f"| neutral overlap (D6) | {seed} | {fam} 105 passages without L/S context | 23 overlapping passages "
                     f"removed | top-3 {'/'.join(r['d6_subset105']['top3'])}; same set and flag as the 128: "
                     f"{r['d6_subset105']['same_as_status']} | skeptics/skeptic/out/subset105.json |")
    for m, v in nulls.items():
        d3 = v.get("primary_D3", {})
        L.append(f"| specificity | {m} | null model (no switch) | no-switch model | frozen 4-draw: flagged heads "
                 f"{v['n_flagged']} (top-1 {v['top1']} S {v['top1_S']:.4f} vs F {v['F_M']:.4f}); D3 {d3.get('n_draws')} "
                 f"draws: P(flag) {d3.get('P_flag_4draw')}, perm p {d3.get('perm_p')}, verdict {v['d3_verdict']}; "
                 f"push {v['push_mean']} (data/patch/{m}.json; S and F from data/{PICKS}) | data/{ND}/{m}.json |")
    L.extend(vrows)
    if vrows and not any("subset105" in l for l in vrows):
        for m, r in rows.items():
            if isinstance(r.get("d6_subset105"), dict):
                fam, seed = m.split("_")
                L.append(f"| neutral overlap (D6) | {seed} | {fam} 105 passages without L/S context | 23 overlapping "
                         f"passages removed | top-3 {'/'.join(r['d6_subset105']['top3'])}; same set and flag as the 128: "
                         f"{r['d6_subset105']['same_as_status']} | skeptics/skeptic/out/subset105.json |")
    em = all(v["emu"]["inputs_bit_identical"] for v in picks.values())
    L.append(f"| emulation sanity | all | precision-emulation build | probe_emu vs none_emu | block-26 inputs bit-identical "
             f"= {em} (tail sha256 in data/neutral_meta_Qwen3-1.7B.json and data/neutral_meta_Qwen3-0.6B.json); max head change {max(v['emu']['max_abs_d_trained'] for v in picks.values())} | "
             f"data/{PICKS} |")
    (HERE / "proof-matrix.md").write_text("# Proof matrix — blind-head-audit\n\nEvery cell is rebuilt by `summary.py` from the "
                                          "JSON named in its row.\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
