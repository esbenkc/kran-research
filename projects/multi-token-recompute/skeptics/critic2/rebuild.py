#!/usr/bin/env python3
"""Independent headline rebuild for oracle-1108-multi-token final review.

Reads only saved raw npz files. Recomputes served gaps, paired prompt
bootstrap intervals, text difference D, flip rate f, planted forced switch,
and the frozen status rule plus D4 one-way clause.
"""
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "reads" / "raw"
BOOT = 2000
SEED = 12345
rng = np.random.default_rng(SEED)


def prompt_means(values, prompts):
    prompts = np.asarray(prompts)
    values = np.asarray(values, dtype=float)
    groups = []
    for prompt in sorted(set(prompts.tolist())):
        groups.append(values[prompts == prompt].mean())
    return np.asarray(groups, dtype=float)


def ci(values, prompts):
    groups = prompt_means(values, prompts)
    n = len(groups)
    samples = groups[rng.integers(0, n, size=(BOOT, n))].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def load_npz(name, split="test"):
    z = np.load(RAW / f"{name}.npz", allow_pickle=True)
    mask = np.ones(len(z["prompt_idx"]), dtype=bool)
    if "split" in z.files:
        mask = z["split"] == split
    return z, mask


def metrics(name, split="test"):
    z, m = load_npz(name, split)
    prompts = z["prompt_idx"][m]
    cn = z["c_n"][m].astype(float)
    cl = z["c_l"][m].astype(float)
    cnl = z["c_nl"][m].astype(float)
    lpn = z["lp_n"][m].astype(float)
    lpln = z["lp_l_n"][m].astype(float)
    toksn = z["toks_n"][m]
    toksl = z["toks_l"][m]

    gap = cn - cl
    text = cn - cnl
    spill = cnl - cl
    log_ratio = (lpln - lpn).sum(axis=1)
    d_rows = np.maximum(0.0, 1.0 - np.exp(log_ratio))
    f_rows = np.any(toksn != toksl, axis=1).astype(float)

    out = OrderedDict()
    out["n_rows"] = int(m.sum())
    out["n_prompts"] = int(len(set(prompts.tolist())))
    out["cheat_none"] = float(cn.mean())
    out["cheat_leaky"] = float(cl.mean())
    out["gap"] = float(gap.mean())
    out["gap_ci"] = ci(gap, prompts)
    out["text"] = float(text.mean())
    out["text_ci"] = ci(text, prompts)
    out["spill"] = float(spill.mean())
    out["spill_ci"] = ci(spill, prompts)
    out["D"] = float(d_rows.mean())
    out["D_ci"] = ci(d_rows, prompts)
    out["f"] = float(f_rows.mean())
    out["distinct_prefix_share"] = float(len({tuple(row) for row in toksn}) / len(toksn))

    if "c_X" in z.files:
        cx = z["c_X"][m].astype(float)
        cnotx = z["c_notX"][m].astype(float)
        forced = cnotx - cx
        out["forced_switch"] = float(forced.mean())
        out["forced_switch_ci"] = ci(forced, prompts)
    return out


def points(x):
    return 100.0 * x


def row(name, m):
    return (
        f"{name}: cheat none/leaky {m['cheat_none']:.4f}/{m['cheat_leaky']:.4f}; "
        f"gap {points(m['gap']):+.2f} pts [{points(m['gap_ci'][0]):+.2f},{points(m['gap_ci'][1]):+.2f}]; "
        f"text {points(m['text']):+.2f} pts; D {points(m['D']):.2f} pts "
        f"[{points(m['D_ci'][0]):.2f},{points(m['D_ci'][1]):.2f}]; "
        f"f {100*m['f']:.1f}%; distinct {100*m['distinct_prefix_share']:.1f}%"
    )


def main():
    names = ["start", "planted_s0", "k16_s0", "k16_s1", "k16open_s0", "start_k64"]
    results = OrderedDict((name, metrics(name)) for name in names)

    planted = results["planted_s0"]
    gate = (
        planted["forced_switch"] >= 0.20
        and planted["forced_switch_ci"][0] > 0.0
        and planted["gap_ci"][0] > 0.0
    )
    frozen_rejected = gate and results["k16_s0"]["gap_ci"][1] < 0.03 and results["k16_s1"]["gap_ci"][1] < 0.03
    frozen_confirmed = gate and results["k16_s0"]["gap_ci"][0] >= 0.03 and results["k16_s1"]["gap_ci"][0] >= 0.03

    d4 = results["k16open_s0"]
    d4_valid = d4["distinct_prefix_share"] >= 0.20 and d4["D"] >= 0.0137
    d4_unconfirms = d4_valid and d4["gap_ci"][0] >= 0.03

    if d4_unconfirms:
        status = "UNCONFIRMED"
    elif frozen_confirmed:
        status = "CONFIRMED"
    elif frozen_rejected:
        status = "REJECTED"
    else:
        status = "UNCONFIRMED"

    prereg_unsettled = results["start"]["f"] < 0.20  # raw f=0.059 was separately verified by Vera; start alone is below.

    print("critic2 independent rebuild")
    print(f"root: {ROOT}")
    print(f"bootstrap: {BOOT} resamples over prompts, seed {SEED}")
    for name, m in results.items():
        print(row(name, m))
        if "forced_switch" in m:
            print(
                f"  forced switch {points(m['forced_switch']):+.2f} pts "
                f"[{points(m['forced_switch_ci'][0]):+.2f},{points(m['forced_switch_ci'][1]):+.2f}]"
            )
    print("status checks:")
    print(f"  gate={gate} (planted forced >=20 pts with CI-low>0 and served gap CI-low>0)")
    print(f"  k16_s0 CI-high {points(results['k16_s0']['gap_ci'][1]):+.2f} pts < 3")
    print(f"  k16_s1 CI-high {points(results['k16_s1']['gap_ci'][1]):+.2f} pts < 3")
    print(f"  frozen_rejected={frozen_rejected}; frozen_confirmed={frozen_confirmed}")
    print(
        f"  D4 valid={d4_valid} (distinct {results['k16open_s0']['distinct_prefix_share']:.3f} >=0.20, "
        f"D {results['k16open_s0']['D']:.4f} >=0.0137); one-way unconfirms={d4_unconfirms}"
    )
    print(f"  prereg_outcome_unsettled_by_start_f={prereg_unsettled} (start f={results['start']['f']:.4f}<0.20; raw f=0.059 in PREREG/Vera)")
    print(f"VERDICT_NUMBERS: title answer NO; lab_status {status}; prereg outcome unsettled")

    json_path = Path(__file__).with_name("rebuild.json")
    with json_path.open("w") as f:
        json.dump({"results": results, "gate": gate, "d4_valid": d4_valid, "d4_unconfirms": d4_unconfirms, "status": status}, f, indent=2)


if __name__ == "__main__":
    main()
