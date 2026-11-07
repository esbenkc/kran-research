"""critic2 (Kasper, cross-family final review; Claude restart after the gpt-5.5 cap) headline rebuild.

Reads ONLY per-prompt rows in data/runs/*.json (row = [prompt, probe, top1, logpA, logpB]); no model load.
Recomputes: strict switch per eval, the frozen RETURNED rule (dev strict >= 0.75 at 2 consecutive post-update evals),
final test strict on the merged clamped model, reversed share n, mean push, CONVERGED rule, LM-loss rise, then the
PREREG title status and outcome. Checks every recomputed value against the stored fields (mismatch count).
Null: sign-flip null on the final rows (P(strict count >= observed) if probe labels are swapped per prompt at random).
Run: python3 skeptics/critic2/rebuild.py  (from projects/head-clamp-retrain)
"""
import glob
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "data", "runs")
A, B = 32, 33


def pairs(rows):
    d = defaultdict(dict)
    for pr, probe, top, la, lb in rows:
        d[pr][int(probe)] = (top, la - lb)
    return [(d[p][0], d[p][1]) for p in sorted(d)]


def stats(rows):
    ps = pairs(rows)
    strict = sum(u[0] == B and p[0] == A for u, p in ps)
    rev = sum(u[0] == A and p[0] != A for u, p in ps)
    push = sum(p[1] - u[1] for u, p in ps) / len(ps)
    return len(ps), strict, rev, push


def null_p(rows, reps=20000, seed=0):
    """swap probe on/off per prompt with p 1/2: count of prompts that read as 'switch' (B off, A on)"""
    ps = pairs(rows)
    obs = sum(u[0] == B and p[0] == A for u, p in ps)
    rng = random.Random(seed)
    ge = 0
    for _ in range(reps):
        c = 0
        for u, p in ps:
            if rng.random() < 0.5:
                u, p = p, u
            c += u[0] == B and p[0] == A
        ge += c >= obs
    return (ge + 1) / (reps + 1)


out, mism = {}, []
for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
    r = json.load(open(f))
    name, rows = r["name"], r["rows"]
    steps = sorted(int(k[4:].split("/")[0]) for k in rows if k.startswith("step") and k.endswith("/dev_clamp"))
    curve = [(0, stats(rows["t0/dev_clamp"]))] + [(s, stats(rows[f"step{s}/dev_clamp"])) for s in steps]
    hits, ret_at = 0, None
    for s, (n, k, _, _) in curve[1:]:
        hits = hits + 1 if k / n >= 0.75 else 0
        if hits >= 2:
            ret_at = s
            break
    n, k, rev, push = stats(rows["final/test_merged_clamp"])
    L = r["loss"]
    m = max(1, len(L) // 5)
    dl = sum(L[-m:]) / m - sum(L[-2 * m:-m]) / m
    last3 = [c[1][1] / c[1][0] for c in curve[-3:]]
    conv = abs(dl) <= 0.02 and max(last3) - min(last3) <= 0.10
    lm = r["lm_loss"]
    rise = lm["final_clamp"] / lm["base"] - 1
    t0n, t0k, _, _ = stats(rows["t0/test_clamp"])
    row = {"arm": r["args"]["arm"], "seed": r["args"]["seed"], "heads": len(r["heads"]), "t0_test_strict": t0k / t0n,
           "returned_at": ret_at, "steps_run": curve[-1][0], "final_test_strict_k": k, "n": n,
           "final_test_strict": k / n, "reversed_k": rev, "push": round(push, 3), "dL_last20": round(dl, 4),
           "converged": conv, "lm_rise_final_clamp": round(rise, 5), "lm_rise_stored": round(r["lm_rise_vs_base"], 5),
           "null_p": null_p(rows["final/test_merged_clamp"], reps=4000)}
    out[name] = row
    # cross-check vs stored fields
    st = r["test_merged_clamp"]
    if abs(st["switch_correct"] - k / n) > 1e-9:
        mism.append((name, "test strict", st["switch_correct"], k / n))
    if abs(st["shift_mean"] - push) > 1e-2:
        mism.append((name, "push", st["shift_mean"], push))
    for c, (s, (cn, ck, _, _)) in zip(r["dev_curve"], curve):
        if c["step"] != s or abs(c["switch_correct"] - ck / cn) > 1e-9:
            mism.append((name, f"dev@{s}", c["switch_correct"], ck / cn))
    want = "returned" if ret_at else "budget"
    if r["stop_reason"] != want or (ret_at and r["steps_run"] != ret_at):
        mism.append((name, "stop", r["stop_reason"], r["steps_run"], want, ret_at))

for name, row in out.items():
    print(f"{name:16s} heads {row['heads']:2d} t0 {row['t0_test_strict']:.3f} ret@{row['returned_at']} "
          f"stop {row['steps_run']:4d} final {row['final_test_strict_k']:3d}/{row['n']} rev {row['reversed_k']} "
          f"push {row['push']:7.3f} dL {row['dL_last20']:+.4f} conv {row['converged']} "
          f"lm {100 * row['lm_rise_final_clamp']:+.3f}% null_p {row['null_p']:.4f}")

arm = lambda a: [out[k] for k in sorted(out) if out[k]["arm"] == a]
set_back = sum(r["returned_at"] is not None or r["final_test_strict"] >= 0.25 for r in arm("set"))
set_hold = sum(r["returned_at"] is None and r["final_test_strict"] < 0.25 and r["converged"] for r in arm("set"))
rand_on = sum(r["final_test_strict"] >= 0.75 for r in arm("random"))
lm_max = max(abs(r["lm_rise_final_clamp"]) for r in out.values())
status = "CONFIRMED" if set_hold >= 2 else "REJECTED" if set_back >= 2 else "UNCONFIRMED"
outcome = "held" if (set_hold >= 2 and rand_on >= 2 and lm_max < 0.02) else \
    "failed" if (set_back >= 2 or rand_on < 2 or lm_max >= 0.02) else "unsettled"
print(f"SET back {set_back}/3, SET held+converged {set_hold}/3, RANDOM >=0.75 {rand_on}/3, max |LM rise| "
      f"{100 * lm_max:.3f}% -> title {status}, prereg {outcome}")
print(f"ALL-32 held {sum(r['returned_at'] is None and r['final_test_strict'] < 0.25 for r in arm('all32'))}/3, "
      f"converged {sum(r['converged'] for r in arm('all32'))}/3")
print("mismatches vs stored fields:", len(mism), mism[:5])
json.dump({"runs": out, "status": status, "outcome": outcome, "mismatches": mism},
          open(os.path.join(HERE, "rebuild.json"), "w"), indent=1)
