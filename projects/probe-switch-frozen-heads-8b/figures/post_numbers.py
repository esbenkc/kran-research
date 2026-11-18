"""Print every number the post quotes, from summary.json files (parent 1.7B + this run's 8B).

Usage: python3 /tmp/selma_nums.py <project dir> [--s17 data/parent_17b_summary.json]
Reads <project>/data/summary.json (8B, same schema as the parent's summary.py) and the parent copy.
"""
import json
import math
import sys
from pathlib import Path

P = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
N = 128
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def seed_t(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = T975[n - 1] * sd / math.sqrt(n)
    return m, max(0, m - h), min(1, m + h)


def report(path, size):
    if not path.exists():
        print(f"## {size}: {path} missing")
        return
    s = json.loads(path.read_text())
    print(f"## {size} ({path})")
    for k in ("t75_all8_s0", "base_whole", "map_source"):
        if k in s:
            print(f"  {k}: {s[k]}")
    arms = {}
    for key, r in sorted(s["runs"].items()):
        w = r.get("whole_test") or {}
        c = r.get("cached_test") or {}
        st = w.get("strict")
        k = None if st is None else round(st * N)
        lo, hi = wilson(k, N) if k is not None else (None, None)
        a = r.get("audit") or {}
        print(f"  {key}: whole strict {k}/{N}" + (f" Wilson [{lo:.3f},{hi:.3f}]" if k is not None else "")
              + f" | cached {c.get('strict')} | T75 {r.get('T75')} stop {r.get('early_stop_step')} steps {r.get('steps_done')}"
              f" budget2x {r.get('budget_2xT75')} reached {r.get('reached_2xT75')} | push {w.get('shift_mean')}"
              f" | A-with {w.get('share_probed_A')} B-without {w.get('share_unprobed_B')} | G1 {r.get('G1_pass')}"
              f" | rms {r.get('rms_grad_median_window')} | params {r.get('lora_params')}")
        if a:
            print(f"      audit: blind3 {a.get('blind3_heads')} removed {a.get('R_blind3')} null_p95 {a.get('null_p95')}"
                  f" all {a.get('clamp_all_removed')}")
        if r.get("null", "none") == "none" and not key.endswith(("_8k", "_mark")) and st is not None:
            arms.setdefault(r["arm"], []).append(st)
    for arm, xs in arms.items():
        m, lo, hi = seed_t(xs)
        print(f"  SEED-LEVEL {arm}: n={len(xs)} mean {m:.3f}" + (f" t95 [{lo:.3f},{hi:.3f}]" if lo is not None else "")
              + f" min {min(xs):.3f} at75 {sum(x >= 0.75 for x in xs)}/{len(xs)} under25 {sum(x < 0.25 for x in xs)}/{len(xs)}")
    rms = {(r["arm"], r["seed"]): r.get("rms_grad_median_window") for r in s["runs"].values() if r.get("null", "none") == "none"}
    for (arm, seed), v in sorted(rms.items()):
        if arm == "mlp8" and ("all8", seed) in rms and v and rms[("all8", seed)]:
            print(f"  rms ratio mlp8/all8 s{seed}: {v / rms[('all8', seed)]:.2f}")
    at = s.get("attn") or {}
    for k, v in at.items():
        print(f"  attn {k}: {json.dumps(v)[:300]}")


report(P / "data/parent_17b_summary.json", "1.7B")
report(P / "data/summary.json", "8B")
for k in ("seed_level", "status", "prereg_outcome", "prediction", "prediction_scored", "same_route"):
    q = P / "data/summary.json"
    if q.exists():
        print(f"## summary.{k}: {json.dumps(json.loads(q.read_text()).get(k))[:1500]}")
for f in ("research_checks/prediction.json", "PREDICTIONS.md"):
    q = P / f
    if q.exists():
        print(f"## {f}\n" + q.read_text()[:2500])
