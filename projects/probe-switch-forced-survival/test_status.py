"""No-model test of status.py's frozen rule: builds fake kaudit/whole/runs files in a temp dir and checks every branch.

    .venv/bin/python test_status.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N = 128


def lo_for(share, sign_pos=True, rng=None):
    """log-odds with exactly round(share * N) positive values (the cut switch still fires there)"""
    k = round(share * N)
    v = np.full(N, -2.0)
    v[:k] = 2.0
    return v.tolist()


def write_run(d, name, S0=1.0, S=None, r8=0.9, bf16_S8=None, G1_gap=0.0, tie_first=False, whole=True, runs=True):
    S = S or [0.9] * 7 + [0.1] * 5
    lu = [-3.0] * N
    lp = lo_for(S0)
    tc = {}
    for k in range(1, 13):
        lo32 = lo_for(S[k - 1])
        lo16 = lo_for(bf16_S8) if (k == 8 and bf16_S8 is not None) else lo32
        tc[f"greedy_TB_k{k}"] = {"lo": lo16, "lo32": lo32, "strict": float(np.mean(np.array(lo16) > 0)),
                                 "strict32": float(np.mean(np.array(lo32) > 0))}
    for j in range(50):
        lo = lo_for(r8)
        tc[f"rand8_{j:02d}"] = {"lo": lo, "lo32": lo, "strict": r8, "strict32": r8}
    for k in ("all_heads", "all_heads+mlpA+mlpB", "mlpA", "mlpB", "mlpA+mlpB"):
        tc[k] = {"strict32": 0.0}
    argmax = S0
    tb = {"lo_unprobed": lu, "lo_probed": lp, "lo_unprobed32": lu, "lo_probed32": lp, "strict": S0, "strict32": S0,
          "argmax_switch": argmax, "T_B_floor_share": 0.0, "probed_lead_A_over_B_mean": 5.0}
    ka = {"args": {"kmax": 12}, "test_base": tb, "test_cut": tc,
          "headline": {"strict32_after_greedy_k": {"TB": S}, "strict32_smallest_k_below_0.25": {"TB": None},
                       "strict32_random_min_median_max": {"8": [r8, r8, r8]}},
          "greedy_dev": {"TB": [{"set": ["26.0"], "tie": False}]}}
    for sub in ("kaudit", "whole", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "kaudit" / f"{name}.json").write_text(json.dumps(ka))
    if whole:
        (d / "whole" / f"{name}.json").write_text(json.dumps({"switch_correct": argmax + G1_gap}))
    if runs and not name.startswith("warm"):
        picks = [{"step": 1, "tie_steps": [tie_first] + [False] * 5, "heads": ["26.0"], "secs": 1.0},
                 {"step": 9, "tie_steps": [False] * 6, "heads": ["26.1"], "secs": 1.0}]
        (d / "runs" / f"{name}.json").write_text(json.dumps({
            "picks": picks, "train_log": [[1, 0.1, 0.0, 1.0], [2, 0.1, 0.2, 1.0]],
            "test_cached": {"switch_strict32": S0}, "curve": []}))


def run_status(d):
    env = dict(os.environ, AP_OUT=str(d))
    out = subprocess.run([sys.executable, str(HERE / "status.py")], env=env, capture_output=True, text=True, cwd=HERE)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads((d / "status.json").read_text())


def case(name, setup, want_status, want_outcome=None):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "fx"
        setup(d)
        st = run_status(d)
        ok = st["status"] == want_status and (want_outcome is None or st["prereg_outcome"] == want_outcome)
        print(f"{'PASS' if ok else 'FAIL'} {name}: status {st['status']} outcome {st['prereg_outcome']} "
              f"counted {st['counted_k6']} notes {st['notes_k6']}")
        return ok


def k6(d, kw=None):
    for s in (0, 1, 2):
        write_run(d, f"k6_s{s}", **(kw or {}).get(s, {}))


def law(d, kmins):
    for A, km in kmins.items():
        for s in (0, 1, 2):
            S = [0.9] * (km - 1) + [0.1] * (13 - km) if km <= 12 else [0.9] * 12
            write_run(d, f"k{A}_s{s}", S=S)


results = [
    case("all yes + law holds -> CONFIRMED / held", lambda d: (k6(d), law(d, {3: 5, 1: 3})), "CONFIRMED", "held"),
    case("yes, law k3 k_min 6 > 5 -> CONFIRMED / failed", lambda d: (k6(d), law(d, {3: 6, 1: 3})), "CONFIRMED",
         "failed"),
    case("yes, no law arms -> CONFIRMED / unsettled", lambda d: k6(d), "CONFIRMED", "unsettled"),
    case("one seed S(8) = 0.30 -> REJECTED / failed",
         lambda d: k6(d, {1: {"S": [0.9] * 8 + [0.1] * 4}}), "REJECTED", "failed"),
    case("S(8) = 0.25 exactly counts as not stopped -> REJECTED",
         lambda d: k6(d, {2: {"S": [0.9] * 7 + [0.25] + [0.1] * 4}}), "REJECTED"),
    case("random R8 0.70 in a stopped seed -> UNCONFIRMED", lambda d: k6(d, {0: {"r8": 0.70}}), "UNCONFIRMED"),
    case("bf16 rows disagree (S8 bf16 0.30) -> UNCONFIRMED", lambda d: k6(d, {0: {"bf16_S8": 0.30}}),
         "UNCONFIRMED"),
    case("seed 1 faded, spare live -> CONFIRMED",
         lambda d: (k6(d, {1: {"S0": 0.5}}), write_run(d, "k6_s3")), "CONFIRMED"),
    case("two faded, one spare -> UNCONFIRMED",
         lambda d: (k6(d, {1: {"S0": 0.5}, 2: {"S0": 0.625}}), write_run(d, "k6_s3")), "UNCONFIRMED"),
    case("monitor tie at first re-pick -> UNCONFIRMED (no spare for it)",
         lambda d: (k6(d, {0: {"tie_first": True}}), write_run(d, "k6_s3")), "UNCONFIRMED"),
    case("G1 miss 7/128 = 0.055 -> UNCONFIRMED", lambda d: k6(d, {2: {"G1_gap": -7 / 128}}), "UNCONFIRMED"),
    case("G1 gap 6/128 = 0.047 is ok -> CONFIRMED", lambda d: k6(d, {2: {"G1_gap": -6 / 128}}), "CONFIRMED"),
    case("no whole-graph eval -> UNCONFIRMED", lambda d: k6(d, {0: {"whole": False}}), "UNCONFIRMED"),
    case("seed 2 missing -> UNCONFIRMED", lambda d: (write_run(d, "k6_s0"), write_run(d, "k6_s1")), "UNCONFIRMED"),
]
print(f"{sum(results)}/{len(results)} cases pass")
sys.exit(0 if all(results) else 1)
