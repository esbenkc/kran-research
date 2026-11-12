"""PREREG s8.1 (+ D1, D2 v2) learning-rate rule, applied mechanically to pilot runs at x_top, seed 0.

--arm sgd   : data/runs/sgdpilot_lr*.json, ladder 1 .. 0.01 (s8.1)       -> data/sgd_pick.json
--arm sgdfa : data/runs/sgdfapilot_lr*.json, ladder 10 .. 0.01 (D2 v2.2) -> data/sgdfa_pick.json
Descend the ladder. Stable = common.stable(). The descent may stop once a stable lr has T > 141. Among stable lrs that
switch, pick T closest to 141 on a log scale; ties -> smaller lr. No stable lr with T in [48, 4096] -> GATE_FAIL.
Prints the table and NEXT <lr> (run this pilot next), PICK <lr> or GATE_FAIL.

    /usr/bin/python3 pick_lr.py [--arm sgd|sgdfa]
"""
import argparse
import json
import math

from common import HERE, LADDER, T_of, load, stable

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="sgd", choices=["sgd", "sgdfa"])
arm = ap.parse_args().arm
ladder = LADDER if arm == "sgd" else [10, 5, 2] + LADDER
pname = lambda lr: f"{arm}pilot_lr{lr:g}"  # noqa: E731

rows, next_lr = [], None
for lr in ladder:
    r = load(pname(lr))
    if r is None:
        next_lr = lr
        break
    ok, why = stable(r)
    T = T_of(r)
    rows.append({"lr": lr, "run": pname(lr), "stable": ok, "why": why, "T": T, "steps_run": r["steps_run"]})
    if ok and T is not None and T > 141:
        next_lr = None
        break  # smaller lrs only move T further from 141

for x in rows:
    print(f"lr {x['lr']:<5g} stable {x['stable']!s:5} T {x['T']!s:>5} steps {x['steps_run']:>5}  {'; '.join(x['why'])}")
cand = [x for x in rows if x["stable"] and x["T"] is not None and 48 <= x["T"] <= 4096]
done = next_lr is None
out = {"rule": "PREREG s8.1 + D1" + (" + D2 v2" if arm == "sgdfa" else ""), "arm": arm, "pilots": rows}
if not done:
    out["decision"] = f"NEXT {next_lr:g}"
elif cand:
    best = min(cand, key=lambda x: (abs(math.log(x["T"] / 141)), x["lr"]))
    out.update({"decision": f"PICK {best['lr']:g}", "lr": best["lr"], "run": best["run"], "T": best["T"]})
else:
    out["decision"] = "GATE_FAIL"
print(out["decision"])
(HERE / "data" / f"{arm}_pick.json").write_text(json.dumps(out, indent=1))
