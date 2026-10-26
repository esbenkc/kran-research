"""Status rule (PREREG.md title rule + DEVIATIONS D1/D2), prereg clauses, data/summary.json, proof-matrix.md, figure data.

    $PY summary.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SWITCH = ["L_s0", "L_s1", "L_s2", "S_s0", "S_s1", "S_s2", "RL_s0", "RL_s1"]
NULLS = ["L_never", "RL_shuf", "RL_tag"]
RL = ["RL_s0", "RL_s1"]


def main():
    picks = json.load(open(HERE / "data/picks.json"))["models"]
    P = {m: json.load(open(HERE / f"data/patch/{m}.json")) for m in SWITCH if (HERE / f"data/patch/{m}.json").exists()}
    N = {m: json.load(open(HERE / f"data/patch/{m}.json")) for m in NULLS if (HERE / f"data/patch/{m}.json").exists()}
    rows = {}
    for m in SWITCH:
        if m not in P:
            continue
        p, k = P[m], picks[m]
        rows[m] = {
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
            "all_sets": p.get("all_sets"), "pass": p["pass"], "passes": p["pass"]["model_passes"]}
    nulls = {}
    for m in NULLS:
        k = picks[m]
        nulls[m] = {"top1": k["primary"]["order"][0], "top1_S": k["primary"]["top1_S"], "F_M": k["primary"]["F_M"],
                    "n_flagged": len(k["primary"]["flagged"]), "flagged": k["primary"]["flagged"],
                    "false_alarm": len(k["primary"]["flagged"]) >= 1,
                    "secondary_n_flagged": len(k["secondary_last_position"]["flagged"]),
                    "iso_n_flagged": len(k["logged_iso_fake"]["primary_flagged_iso"]),
                    "push_mean": N.get(m, {}).get("push_mean")}
    complete = len(rows) == len(SWITCH)
    n_pass = sum(r["passes"] for r in rows.values())
    rl_pass = all(rows[m]["passes"] for m in RL if m in rows) and all(m in rows for m in RL)
    rl_fail = any(not rows[m]["passes"] for m in RL if m in rows)
    any_fa = any(v["false_alarm"] for v in nulls.values())
    n_fail = sum(not r["passes"] for r in rows.values())
    if rl_fail or n_fail >= 2 or any_fa:
        status = "REJECTED"
    elif complete and n_pass >= 7 and rl_pass and not any_fa:
        status = "CONFIRMED"
    else:
        status = "UNCONFIRMED"
    why = {"n_pass": n_pass, "n_switch_run": len(rows), "rl_pass": rl_pass, "rl_fail": rl_fail,
           "null_false_alarms": [m for m, v in nulls.items() if v["false_alarm"]], "complete": complete}
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
    out = {"status": status, "status_why": why, "prereg_outcome": "held" if held else ("failed" if complete else "unsettled"),
           "prereg_clauses": clauses, "switch": rows, "nulls": nulls}
    (HERE / "data/summary.json").write_text(json.dumps(out, indent=1))
    print(status, json.dumps(why), "prereg", out["prereg_outcome"])
    for m, r in rows.items():
        print(f"{m:6s} push {r['push_mean']:8.3f} top3 {r['blind_top3']} R {r['R_blind3']:.3f} "
              f"[{r['R_blind3_ci95'][0]:.3f},{r['R_blind3_ci95'][1]:.3f}] rand29 med {r['rand_out_median']:.3f} "
              f"p95 {r['rand_out_p95']:.3f} rand32 med {r['rand_all_median']:.3f} flag {r['n_flagged']} "
              f"top1 {r['top1']} S/F {r['top1_S']:.3f}/{r['F_M']:.3f} oracle {r['oracle_top3']} {r['oracle_R3']} "
              f"all32 {r['R_all32']:.3f} plant {r['plant_blind3_recovered']:.3f} sec {r['secondary_top3']} "
              f"{r['R_secondary3']:.3f} pass {r['passes']}")
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
                 f"F = {r['F_M']:.4f}; flagged heads {r['n_flagged']} | data/picks.json |")
        L.append(f"| prereg clause 2 | {seed} | {fam} | 400 random 3-sets, all 32 heads | median {r['rand_all_median']:.3f} | "
                 f"data/patch/{m}.json |")
        if r["oracle_top3"]:
            L.append(f"| blind vs oracle | {seed} | {fam} oracle {'/'.join(r['oracle_top3'])} | behaviour-picked (dev) | "
                     f"R = {r['oracle_R3']:.3f}; overlap {r['overlap_blind_oracle']}/3 | data/patch/{m}.json |")
        L.append(f"| route | {seed} | {fam} all 32 heads | none | R = {r['R_all32']:.3f}; plant top-3 recovers "
                 f"{r['plant_blind3_recovered']:.3f} | data/patch/{m}.json |")
        L.append(f"| secondary score | {seed} | {fam} last-position top-3 {'/'.join(r['secondary_top3'])} | none | "
                 f"R = {r['R_secondary3']:.3f} | data/patch/{m}.json |")
    for m, v in nulls.items():
        L.append(f"| specificity | {m} | null model (no switch) | no-switch model | flagged heads {v['n_flagged']} "
                 f"(top-1 S {v['top1_S']:.4f} vs F {v['F_M']:.4f}); push {v['push_mean']} | data/picks.json |")
    em = all(v["emu"]["inputs_bit_identical"] for v in picks.values())
    L.append(f"| emulation sanity | all | precision-emulation build | probe_emu vs none_emu | block-26 inputs bit-identical "
             f"= {em}; max head change {max(v['emu']['max_abs_d_trained'] for v in picks.values())} | "
             f"raw/neutral/*/meta.json, data/picks.json |")
    (HERE / "proof-matrix.md").write_text("# Proof matrix — blind-head-audit\n\nEvery cell is rebuilt by `summary.py` from the "
                                          "JSON named in its row.\n\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
