"""Collect every scored cell into DATA/summary.json and proof-matrix.md (one row per cell, artifact paths relative to
this folder), and apply the PREREG FROZEN RULE mechanically to whatever cells exist. Port of the parent summary.py
(same keys: runs.<name>.whole_test.strict, T75, early_stop_step, audit.*, attn.*), plus: seed-level report (Wilson per
seed, t-interval over seeds clipped to [0, 1], min over seeds, k of n with Clopper-Pearson), status, prereg outcome,
the map predictions scored, and the 'same route at two sizes' read. Reads DATA/runs, whole, audit, after_map,
research_checks/{map,route,prediction}.json, research_checks/after_attn/. No GPU (numpy + scipy).

    $PY summary.py
"""
import json
import math

import numpy as np
from scipy import stats

from cfgm import DATA, FIRST, LAST, MODEL, NH, NL, OUT

ARM = {"all8": f"all-modules LoRA, blocks {FIRST}-{LAST}", "mlp8": f"MLP-only LoRA, blocks {FIRST}-{LAST}, heads frozen",
       "mlplast": f"last-MLP LoRA (block {LAST} MLP only)"}


def load(d):
    return {p.stem: json.load(open(p)) for p in sorted((DATA / d).glob("*.json"))}


def loss_slope(glog, last=512):
    g = np.array(glog[-last:])
    if len(g) < 16:
        return None
    return float(np.polyfit(g[:, 0], g[:, 1], 1)[0] * 1000)


def wilson(p, n=128):
    k = round(p * n); z = 1.959964
    c = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def seed_level(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None
    m = float(np.mean(xs))
    r = {"n": n, "mean": m, "min": float(min(xs)), "values": xs}
    if n >= 2:
        sd = float(np.std(xs, ddof=1))
        h = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        r.update({"t95": [max(0.0, m - h), min(1.0, m + h)], "method": f"95% t-interval over {n} seeds (df {n - 1}), "
                  "clipped to [0, 1]", "width_zero": sd == 0.0})
    k = sum(x >= 0.75 for x in xs)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    r["k_ge_0.75"] = k; r["k_of_n_cp95"] = [lo, hi]
    return r


def main() -> None:
    runs, whole, audit, after = load("runs"), load("whole"), load("audit"), load("after_map")
    t75_ref = runs.get("all8_none_s0", {}).get("T75")
    S = {"model": MODEL, "runs": {}, "t75_all8_s0": t75_ref}
    for name, r in runs.items():
        g = np.array(r["grad_log"])
        seed = r["args"]["seed"]
        same = runs.get(f"all8_none_s{seed}")
        t75 = (same or {}).get("T75") or t75_ref
        win = g[g[:, 0] <= t75] if t75 else g
        row = {"arm": r["args"]["arm"], "null": r["args"]["null"], "input": r["args"].get("input", "A"), "seed": seed,
               "steps_done": r["steps_done"], "early_stop_step": r["early_stop_step"], "T75": r["T75"],
               "train_s": r["train_s"], "resumed_at": r.get("resumed_at", []), "lora_params": r["lora_params"],
               "rms_grad_median_window": float(np.median(win[:, 3])), "window_T75": t75,
               "rms_grad_median_all": float(np.median(g[:, 3])),
               "loss_last128": float(g[-128:, 1].mean()), "loss_last128_median": float(np.median(g[-128:, 1])),
               "loss_slope_per_1k_last512": loss_slope(r["grad_log"]),
               "budget_2xT75": 2 * t75 if t75 else None,
               "reached_2xT75": bool(t75 and r["steps_done"] >= 2 * t75),
               "cached_test": {k: r["test_cached"][k] for k in ("strict", "switch_correct", "reverse_strict",
                                                                "share_unprobed_B", "share_probed_A", "shift_mean")},
               "cached_test_unmerged_strict": r["test_cached_unmerged"]["strict"],
               "dev_curve": [(c["step"], round(c["strict"], 4)) for c in r["dev_curve"]]}
        if name in whole:
            w = whole[name]
            row["whole_test"] = {k: w[k] for k in ("strict", "switch_correct", "reverse_strict", "share_unprobed_B",
                                                   "share_probed_A", "shift_mean", "bit_identical_prompts")}
            row["whole_test"]["wilson95"] = wilson(w["strict"], w["n_prompts"])
            row["whole_repeat_changed"] = [w["info"]["cond0_repeat_changed_prompts"], w["info"]["cond1_repeat_changed_prompts"]]
            # G1 (PREREG FROZEN RULE): eager-tail strict switch within 5 points of the whole-graph strict switch
            row["G1_gap"] = abs(r["test_cached"]["strict"] - w["strict"])
            row["G1_pass"] = row["G1_gap"] <= 0.05
            row["tail_vs_whole_argmax_gap"] = abs(r["test_cached"]["switch_correct"] - w["switch_correct"])
        if name in audit:
            a = audit[name]
            row["audit"] = {"clamp_all_removed": a["clamp_all_test"]["removed"],
                            "clamp_all_blind_removed": a["clamp_all_test"]["blind_removed"],
                            "R_blind3": a.get("R_blind3"), "null_p95": a.get("null_p95"),
                            "blind3_heads": a["blind_greedy_test"][-1]["heads"] if "blind_greedy_test" in a else None,
                            "clamp_by_block_removed": {b: v["removed"] for b, v in a["clamp_by_block_test"].items()},
                            "lora_split_strict": {k: v["strict"] for k, v in a.get("lora_split_test_unmerged", {}).items()}}
        if name in after:
            c = after[name]["cells"]
            row["after_map_79"] = {k.split("@")[0]: round(v["auroc"], 4) for k, v in c.items() if k.endswith("@79")}
        S["runs"][name] = row
    if "base" in whole:
        S["base_whole"] = {k: whole["base"][k] for k in ("strict", "switch_correct", "share_unprobed_B", "share_probed_A")}

    # mechanism: answer-token attention shift per head (research_checks/map_states.py attn)
    att = OUT / "research_checks" / "after_attn"
    same_route = {}
    for f in sorted(att.glob("attn_*_test.json")):
        tag = f.stem[len("attn_"):-len("_test")]
        h = json.load(open(f))["heads"]
        big = sorted(((k, v) for k, v in h.items() if v["tv_shift"] > 0.3), key=lambda kv: -kv[1]["tv_shift"])
        top = max(h.items(), key=lambda kv: kv[1]["tv_shift"])
        early = max(v["tv_shift"] for k, v in h.items() if int(k.split(".")[0]) <= LAST - 2)
        S.setdefault("attn", {})[tag] = {"heads_tv_gt_0.3": {k: v["tv_shift"] for k, v in big},
                                         f"max_tv_blocks_{FIRST}_{LAST - 2}": early,
                                         "top_head": [top[0], top[1]["tv_shift"], top[1]["on_argmax"]]}
        if tag.startswith("mlp8_none_s") and tag.endswith("_adapter"):
            same_route[tag] = bool(top[1]["tv_shift"] > 0.30 and int(top[0].split(".")[0]) >= FIRST + 1
                                   and 70 <= top[1]["on_argmax"] <= 78)
    S["same_route"] = {"per_model": same_route, "k": sum(same_route.values()), "n": len(same_route),
                       "holds": sum(same_route.values()) >= 2}

    # seed-level report and the FROZEN RULE status
    w = lambda n: S["runs"].get(n, {}).get("whole_test", {}).get("strict")
    mlp = [w(f"mlp8_none_s{s}") for s in range(3)]
    al = [w(f"all8_none_s{s}") for s in range(3)]
    last = [w(f"mlplast_none_s{s}") for s in range(2)]
    S["seed_level"] = {"mlp8": seed_level(mlp), "all8": seed_level(al), "mlplast": seed_level(last)}
    mlp_s = [x for x in mlp if x is not None]
    all_s = [x for x in al if x is not None]
    g1 = lambda n: S["runs"].get(n, {}).get("G1_pass", False)
    base_ok = S.get("base_whole", {}).get("strict", 1.0) < 0.25
    formed = [s for s in range(3) if mlp[s] is not None and mlp[s] >= 0.75 and g1(f"mlp8_none_s{s}")]
    all_formed = sum(x >= 0.75 for x in all_s)

    def null_settled(s):
        r = S["runs"].get(f"mlp8_none_s{s}")
        if r is None or mlp[s] is None or mlp[s] >= 0.25:
            return False
        need = max(2048, 2 * (t75_ref or 0), 2 * ((runs.get(f"all8_none_s{s}") or {}).get("T75") or 0))
        return r["steps_done"] >= need and f"all8_none_s{s}" in runs

    def grad_ok(s):
        r, c = S["runs"].get(f"mlp8_none_s{s}"), S["runs"].get(f"all8_none_s{s}") or S["runs"].get("all8_none_s0")
        return bool(r and c and r["rms_grad_median_window"] <= 3 * c["rms_grad_median_window"]
                    and c["rms_grad_median_window"] <= 3 * r["rms_grad_median_window"])
    mark = w("mlp8_none_s0_mark")
    if len(formed) >= 2 and all_formed >= 2 and base_ok:
        status = "CONFIRMED"
    elif (len(mlp_s) == 3 and all(x < 0.25 for x in mlp_s) and all(null_settled(s) for s in range(3))
          and all_formed >= 2 and all(grad_ok(s) for s in range(3)) and mark is not None and mark >= 0.75):
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"
    S["status"] = {"status": status, "mlp8_formed_seeds_G1": formed, "all8_formed": all_formed,
                   "n_mlp8_scored": len(mlp_s), "n_all8_scored": len(all_s), "base_below_0.25": base_ok,
                   "mlp8_null_settled": {s: null_settled(s) for s in range(3)}, "mark8": mark,
                   "rule": "PREREG.md FROZEN RULE (pushed e9e49b3ae)"}
    held = (len(mlp_s) == 3 or sum(x >= 0.75 for x in mlp_s) >= 2) and len(last) == 2 and None not in last
    S["prereg_outcome"] = (("held" if sum(x >= 0.75 for x in mlp_s) >= 2 and all(x < 0.25 for x in last) else "failed")
                           if held else "unsettled")
    pj = OUT / "research_checks" / "prediction.json"
    if pj.exists():
        p = json.load(open(pj))
        S["prediction"] = p
        if None not in last:
            S["prediction_scored"] = {"last_mlp_holds": all(x <= p["last_mlp"]["holds_if_every_seed_at_most"] for x in last)}
        if len(mlp_s) == 3:
            k = sum(x >= 0.75 for x in mlp_s)
            obs = "FORMS" if k >= 2 else ("DOES NOT FORM" if all(x < 0.25 for x in mlp_s) else "BETWEEN")
            S.setdefault("prediction_scored", {})["route"] = {"predicted": p["route"]["predicted"], "observed": obs,
                                                              "right": obs == p["route"]["predicted"]}
    (DATA / "summary.json").write_text(json.dumps(S, indent=1))

    # proof matrix
    L = ["# Proof matrix — probe-switch-frozen-heads-8b", "",
         f"One row per cell. Model {MODEL}. Switch = strict (PREREG: top-1 strictly above every other token, both "
         "conditions), 128 held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); "
         "'tail' = the eager training tail on cached inputs. Built by summary.py from the artifacts in the last column.",
         "", "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    for name, row in S["runs"].items():
        mark_run = row["input"] == "M"
        variant = ARM[row["arm"]] + (" (visible marker word as trigger)" if mark_run else "")
        claim = ("headline" if row["arm"] in ("mlp8", "all8") and row["null"] == "none" and not mark_run
                 else ("MARK-8" if mark_run else row["arm"]))
        ctrl = ("positive control" if row["arm"] == "all8" or mark_run
                else ("shuffled labels" if row["null"] == "shuffled" else "none"))
        if "whole_test" in row:
            wt = row["whole_test"]
            L.append(f"| {claim} | {row['seed']} | {variant} | {ctrl} | whole strict {wt['strict']:.3f} (Wilson 95% "
                     f"{wt['wilson95'][0]:.3f}-{wt['wilson95'][1]:.3f}), argmax {wt['switch_correct']:.3f}, unprobed-B "
                     f"{wt['share_unprobed_B']:.3f}, probed-A {wt['share_probed_A']:.3f} ({row['steps_done']} updates; "
                     f"G1 gap {row['G1_gap']:.3f}) | data/whole/{name}.json |")
        c = row["cached_test"]
        L.append(f"| {claim} | {row['seed']} | {variant} (tail) | {ctrl} | tail strict {c['strict']:.3f}, argmax "
                 f"{c['switch_correct']:.3f}; T75 {row['T75']}; steps {row['steps_done']} | data/runs/{name}.json |")
        L.append(f"| stall check | {row['seed']} | {variant} | {ctrl} | median RMS grad/param (upd 1..{row['window_T75']}) "
                 f"{row['rms_grad_median_window']:.2e}; loss last 128 median {row['loss_last128_median']:.4f} | "
                 f"data/runs/{name}.json |")
        if "audit" in row:
            a = row["audit"]
            L.append(f"| audit | {row['seed']} | {variant} | random 3-head null | clamp all {NL * NH} heads removes "
                     f"{a['clamp_all_removed']:.3f} of the push; blind top-3 {a['blind3_heads']} R "
                     f"{a['R_blind3'] if a['R_blind3'] is None else round(a['R_blind3'], 3)} (null p95 "
                     f"{a['null_p95'] if a['null_p95'] is None else round(a['null_p95'], 3)}) | data/audit/{name}.json |")
        if "after_map_79" in row:
            mx = max(row["after_map_79"].items(), key=lambda kv: kv[1])
            L.append(f"| after-map | {row['seed']} | {variant} | none | max answer-token AUROC {mx[1]:.3f} at {mx[0]} "
                     f"| data/after_map/{name}.json |")
    if "base_whole" in S:
        b = S["base_whole"]
        L.append(f"| headline | - | untrained model | untrained model | whole strict {b['strict']:.3f}, unprobed-B "
                 f"{b['share_unprobed_B']:.3f} | data/whole/base.json |")
    if "prediction" in S:
        p = S["prediction"]
        L.append(f"| map prediction | - | untrained model, block-{LAST} post-attention, answer token | 200-flip label "
                 f"null | predicted last-MLP strict {p['last_mlp']['predicted_strict_switch']:.3f} (acc "
                 f"{p['last_mlp']['acc']:.3f}) | research_checks/prediction.json |")
        L.append(f"| map prediction | - | untrained model, route rule | 200-flip label null | {p['route']['predicted']}: "
                 f"{len(p['route']['heads_ge_0.30_on_70_78'])} heads >= 0.30 on 70-78, {len(p['route']['readable_cells'])} "
                 f"readable source cells; answer-token max AUROC {p['answer_token_max_auroc_tail_inputs'][1]:.3f} | "
                 f"research_checks/prediction.json |")
    for tag, v in S.get("attn", {}).items():
        L.append(f"| same route | {tag.split('_s')[-1].split('_')[0] if '_s' in tag else '-'} | {tag} | "
                 f"{'untrained model' if tag == 'base' else 'none'} | top TV head {v['top_head'][0]} TV "
                 f"{v['top_head'][1]:.3f} -> pos {v['top_head'][2]}; {len(v['heads_tv_gt_0.3'])} heads > 0.3 | "
                 f"research_checks/after_attn/attn_{tag}_test.json |")
    for arm, sl in S["seed_level"].items():
        if sl and sl["n"] >= 2:
            L.append(f"seed-level: {sl['mean']:.3f} [{sl['t95'][0]:.3f}, {sl['t95'][1]:.3f}] ({sl['method']}; arm {arm}, "
                     f"whole-graph strict switch; min {sl['min']:.3f}; {sl['k_ge_0.75']} of {sl['n']} seeds >= 0.75)")
    for f in sorted([*(OUT / "skeptics").glob("*/proof_rows.md"), *(OUT / "skeptics").glob("*/proof-cells.md")]):
        L += [ln for ln in f.read_text().splitlines() if ln.startswith("| ") and not ln.startswith("| claim")
              and not ln.startswith("|---")]
    (OUT / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-6:]))
    print("STATUS", json.dumps(S["status"]), "prereg", S["prereg_outcome"])


if __name__ == "__main__":
    main()
