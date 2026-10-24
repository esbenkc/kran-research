"""Status numbers from data/ (run export_compact.py first): per model the switch gate, G1, the head test, the goal's
three prediction clauses, and the status rule of PREREG.md + DEVIATIONS.md (D1 steps, D2 restricted random control).

    $PY projects/probe-switch-heads/table.py   ->  data/summary.json (and a printed table)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
D = HERE / "data"
L_RUNS = ["lora_none_s0_2048", "lora_none_s1", "lora_none_s2"]
S_RUNS = ["compile_probe_none_s0", "compile_probe_none_s1", "compile_probe_none_s2"]


def load(p: Path):
    return json.load(open(p)) if p.exists() else None


def heads_row(h: dict) -> dict:
    cut, ind = h["test"]["cut"], h["test"]["induce"]
    out = {"test_orig_shift_mean": h["test_orig"]["shift_mean"],
           "greedy_all_dev": h["greedy_all_dev"], "greedy_27_dev": h["greedy_27_dev"],
           "best_nonpicked_dev_single": h.get("best_nonpicked_dev_single"),
           "rand_all_draws_containing_picked": h.get("rand_all_draws_containing_picked")}
    for k in (1, 2, 3):
        for pool in ("all", "27"):
            c = cut[f"greedy_{pool}_k{k}"]
            out[f"R_greedy_{pool}_k{k}"] = c["removed"]
            out[f"differs_greedy_{pool}_k{k}"] = c["differs"]
            out[f"probedA_greedy_{pool}_k{k}"] = c["share_probed_A"]
            out[f"recovered_greedy_{pool}_k{k}"] = ind[f"greedy_{pool}_k{k}"]["recovered"]
    out["R_all_32"] = cut["all_32"]["removed"]
    for pool in ("all", "27", "ex"):
        if f"random_{pool}_cut" in h:
            out[f"random_{pool}_cut_p95"] = h[f"random_{pool}_cut"]["p95"]
            out[f"random_{pool}_cut_median"] = float(np.median(h[f"random_{pool}_cut"]["values"]))
            out[f"random_{pool}_cut_values"] = h[f"random_{pool}_cut"]["values"]
    n = sum(1 for k in cut if k.startswith("rand_27_"))
    out["random_27_differs_median"] = float(np.median([cut[f"rand_27_{j:02d}"]["differs"] for j in range(n)]))
    for kind in ("zero", "mean"):
        z = h["test"][kind]["greedy_all_k3"]
        out[f"{kind}_k3"] = {"removed": z["removed"], "differs": z["differs"],
                             "unprobed_top_unchanged": z["unprobed_top_unchanged"],
                             "share_unprobed_B": z["share_unprobed_B"], "share_probed_A": z["share_probed_A"]}
    return out


def main() -> None:
    models = {}
    for pop, runs in (("L", L_RUNS), ("S", S_RUNS)):
        for r in runs:
            m = {"population": pop}
            if pop == "L":
                w0, w1 = load(D / "L" / "whole" / f"{r}__emu0.json"), load(D / "L" / "whole" / f"{r}__emu1.json")
                tr = load(D / "L" / "runs" / f"{r}.json")
                if w0:
                    m.update(whole_differs=w0["differs"], whole_pair=w0["switch_correct"], whole_shift=w0["shift_mean"],
                             whole_repeat_changed=[w0["info"]["cond0_repeat_changed_prompts"],
                                                   w0["info"]["cond1_repeat_changed_prompts"]])
                if w1:
                    m.update(emu_differs=w1["differs"], emu_bit_identical=w1["bit_identical_prompts"],
                             emu_share_A=float(np.mean([x["top"] == x["a_id"] for x in w1["rows"]])))
                if tr:
                    m["cached_differs"] = tr["test_cached"]["differs"]
            else:
                w0 = load(D / "S" / "whole" / f"{r}.json")
                if w0:
                    m.update(whole_pair=w0["both_correct"], whole_acc=w0["acc"],
                             whole_differs=float(np.mean([w0["rows"][k]["top"] != w0["rows"][k + 1]["top"]
                                                          for k in range(0, len(w0["rows"]), 2)])))
            h = load(D / pop / "heads" / f"{r}.json")
            if h:
                m.update(heads_row(h))
                m["cached_differs"] = h["test_orig"]["differs"]
            ha = load(D / pop / "heads" / f"{r}_ans.json")
            if ha:
                m["answer_only_R_greedy_all_k3"] = ha["test"]["cut"]["greedy_all_k3"]["removed"]
                m["answer_only_greedy_all_dev"] = ha["greedy_all_dev"]
            if "whole_pair" in m:
                m["gate_pass"] = m["whole_pair"] >= 0.75
            if "whole_differs" in m and "cached_differs" in m:
                m["G1_pass"] = abs(m["whole_differs"] - m["cached_differs"]) <= 0.05
            models[r] = m
    # status population
    L_pass = [r for r in L_RUNS if models[r].get("gate_pass")]
    S_pass = [r for r in S_RUNS if models[r].get("gate_pass")]
    pop = L_pass if len(L_pass) >= 2 else (S_pass if len(S_pass) >= 2 else [])
    status, why = "UNCONFIRMED", "no population passes the switch gate"
    if pop:
        ok = [r for r in pop if models[r].get("G1_pass") and "R_greedy_all_k3" in models[r]]
        best = {r: max(models[r][f"R_greedy_all_k{k}"] for k in (1, 2, 3)) for r in ok}
        yes = [r for r in ok if models[r]["R_greedy_all_k3"] >= 0.80 and models[r].get("random_ex_cut_p95", 1) < 0.30]
        no = [r for r in ok if best[r] < 0.80]
        if len(yes) >= 2:
            status, why = "CONFIRMED", f"top-3 R >= 0.80 and restricted random p95 < 0.30 in {yes}"
        elif len(no) >= 2:
            status, why = "REJECTED", f"best dev-picked <= 3 heads R < 0.80 in {no}"
        else:
            status, why = "UNCONFIRMED", "neither rule holds in >= 2 models"
    # the goal's prediction (L only, as written)
    L = [models[r] for r in L_RUNS]
    c1 = sum(m.get("whole_differs", 0) >= 0.75 for m in L) >= 2
    c2 = all(m.get("emu_differs", 1) < 0.60 for m in L)
    c3 = sum(m.get("differs_greedy_27_k3", 1) < 0.60 and m.get("random_27_differs_median", 0) >= 0.60 for m in L) >= 2
    nv = load(D / "L" / "whole" / "lora_never_s0__emu0.json")
    never = ({"whole_differs": nv["differs"], "whole_shift_mean": nv["shift_mean"],
              "bit_identical_prompts": nv["bit_identical_prompts"]} if nv else None)
    summary = {"status_population": pop, "never_attached_null_s0": never, "status_rule": status, "why": why,
               "goal_prediction": {"clause1_switch": c1, "clause2_emulation": c2, "clause3_block27_heads": c3,
                                   "holds": bool(c1 and c2 and c3)},
               "models": models}
    (D / "summary.json").write_text(json.dumps(summary, indent=1))
    keys = ["whole_pair", "whole_differs", "cached_differs", "G1_pass", "emu_bit_identical", "R_greedy_all_k1",
            "R_greedy_all_k2", "R_greedy_all_k3", "random_ex_cut_p95", "random_all_cut_p95", "R_greedy_27_k3",
            "differs_greedy_27_k3", "random_27_differs_median"]
    for r, m in models.items():
        print(r, {k: (round(m[k], 3) if isinstance(m.get(k), float) else m.get(k)) for k in keys},
              [g["set"] for g in m.get("greedy_all_dev", [])][-1:])
    print("STATUS", status, why, "| goal prediction", summary["goal_prediction"])


if __name__ == "__main__":
    main()
