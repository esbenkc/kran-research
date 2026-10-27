"""Collect every scored cell into data/summary.json and proof-matrix.md (one row per cell, artifact paths relative to
this folder). Reads data/runs/*.json (train.py), data/whole/*.json (eval_whole.py), data/audit/*.json (audit.py),
data/after_map/*.json, research_checks/map.json (Iris, base map). No GPU.

    PY summary.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def load(d):
    return {p.stem: json.load(open(p)) for p in sorted((HERE / "data" / d).glob("*.json"))}


def loss_slope(glog, last=512):
    g = np.array(glog[-last:])
    if len(g) < 16:
        return None
    return float(np.polyfit(g[:, 0], g[:, 1], 1)[0] * 1000)  # loss change per 1000 updates


def main() -> None:
    runs, whole, audit, after = load("runs"), load("whole"), load("audit"), load("after_map")
    t75_ref = runs.get("all8_none_s0", {}).get("T75")
    S = {"runs": {}, "t75_all8_s0": t75_ref}
    for name, r in runs.items():
        g = np.array(r["grad_log"])
        seed = r["args"]["seed"]
        same = runs.get(f"all8_none_s{seed}")
        t75 = (same or {}).get("T75") or t75_ref
        win = g[g[:, 0] <= t75] if t75 else g
        row = {"arm": r["args"]["arm"], "null": r["args"]["null"], "seed": seed, "steps_done": r["steps_done"],
               "early_stop_step": r["early_stop_step"], "T75": r["T75"], "train_s": r["train_s"],
               "lora_params": r["lora_params"],
               "rms_grad_median_window": float(np.median(win[:, 3])), "window_T75": t75,
               "rms_grad_median_all": float(np.median(g[:, 3])),
               "loss_last128": float(g[-128:, 1].mean()), "loss_slope_per_1k_last512": loss_slope(r["grad_log"]),
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
            row["whole_repeat_changed"] = [w["info"]["cond0_repeat_changed_prompts"], w["info"]["cond1_repeat_changed_prompts"]]
            # G1: cached-tail pair both-right vs whole graph, within 5 points
            row["G1_gap"] = abs(r["test_cached"]["switch_correct"] - w["switch_correct"])
            row["G1_pass"] = row["G1_gap"] <= 0.05
            row["tail_vs_whole_strict_gap"] = abs(r["test_cached"]["strict"] - w["strict"])
        if name in audit:
            a = audit[name]
            row["audit"] = {"clamp_all_removed": a["clamp_all_test"]["removed"],
                            "clamp_all_blind_removed": a["clamp_all_test"]["blind_removed"],
                            "R_blind3": a.get("R_blind3"), "null_p95": a.get("null_p95"),
                            "blind3_heads": a["blind_greedy_test"][-1]["heads"] if "blind_greedy_test" in a else None}
        if name in after:
            c = after[name]["cells"]
            row["after_map_79"] = {k.split("@")[0]: round(v["auroc"], 4) for k, v in c.items() if k.endswith("@79")}
        S["runs"][name] = row
    if "base" in whole:
        S["base_whole"] = {k: whole["base"][k] for k in ("strict", "switch_correct", "share_unprobed_B", "share_probed_A")}
    mp = HERE / "research_checks" / "map.json"
    if mp.exists():
        m = json.load(open(mp))
        S["map_source"] = "research_checks/map.json"
    (HERE / "data" / "summary.json").write_text(json.dumps(S, indent=1))

    # proof matrix
    L = ["# Proof matrix — probe-switch-frozen-heads", "",
         "One row per cell. Switch = strict (PREREG: top-1 strictly above every other token, both conditions), 128 "
         "held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); 'tail' = the "
         "eager training tail on cached inputs. Built by summary.py from the artifacts named in the last column.", "",
         "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
    label = {"all8": "all-modules LoRA, blocks 20-27", "mlp8": "MLP-only LoRA, blocks 20-27, heads frozen",
             "mlp27": "last-MLP LoRA (block 27 MLP only)"}
    for name, row in S["runs"].items():
        claim = "headline" if row["arm"] in ("mlp8", "all8") and row["null"] == "none" else row["arm"]
        ctrl = "positive control" if row["arm"] == "all8" else ("shuffled labels" if row["null"] == "shuffled" else "none")
        if "whole_test" in row:
            w = row["whole_test"]
            L.append(f"| {claim} | {row['seed']} | {label[row['arm']]} | {ctrl} | whole strict {w['strict']:.3f}, argmax "
                     f"{w['switch_correct']:.3f}, unprobed-B {w['share_unprobed_B']:.3f}, probed-A {w['share_probed_A']:.3f} "
                     f"({row['steps_done']} updates) | data/whole/{name}.json |")
        c = row["cached_test"]
        L.append(f"| {claim} | {row['seed']} | {label[row['arm']]} (tail) | {ctrl} | tail strict {c['strict']:.3f}, argmax "
                 f"{c['switch_correct']:.3f}; T75 {row['T75']}; steps {row['steps_done']} | data/runs/{name}.json |")
        L.append(f"| stall check | {row['seed']} | {label[row['arm']]} | {ctrl} | median RMS grad/param (upd 1..{row['window_T75']}) "
                 f"{row['rms_grad_median_window']:.2e}; loss last 128 {row['loss_last128']:.4f}; slope/1k "
                 f"{row['loss_slope_per_1k_last512']:.4f} | data/runs/{name}.json (grad_log) |")
        if "audit" in row:
            a = row["audit"]
            L.append(f"| audit | {row['seed']} | {label[row['arm']]} | none | clamp all 128 heads removes {a['clamp_all_removed']:.3f}"
                     f" of the push; R_blind3 {a['R_blind3']} (null p95 {a['null_p95']}) | data/audit/{name}.json |")
        if "after_map_79" in row:
            mx = max(row["after_map_79"].items(), key=lambda kv: kv[1])
            L.append(f"| after-map | {row['seed']} | {label[row['arm']]} | none | max answer-token AUROC {mx[1]:.3f} at {mx[0]} "
                     f"| data/after_map/{name}.json |")
    if "base_whole" in S:
        b = S["base_whole"]
        L.append(f"| headline | - | untrained model | untrained model | whole strict {b['strict']:.3f}, unprobed-B "
                 f"{b['share_unprobed_B']:.3f} | data/whole/base.json |")
    if mp.exists():
        e = m["eager"]
        mx = max(((k, v["logistic"]["auroc"]) for k, v in e.items() if k.startswith("in") and k.endswith("@79")), key=lambda kv: kv[1])
        L.append(f"| map clause 1 | - | base model, eager tail | 200-flip label null | max answer-token AUROC over block inputs "
                 f"20-27 {mx[1]:.3f} ({mx[0]}) | research_checks/map.json |")
    (HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
