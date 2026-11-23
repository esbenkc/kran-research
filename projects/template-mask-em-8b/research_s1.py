"""Score the pre-data secondary prediction S1 (DEVIATIONS.md, 'Secondary prediction, pre-data').

Iris (research), room theory-1123-template-mask-em. Reads analysis.json only; reported, never a status input.

S1 (revised 2026-11-19 14:58Z, 2395efd62, before any answer existed), means over seeds 0, 1, 2:
  (i)  mean habit(C) >= mean habit(B) + 0.10
  (ii) mean cut(C)   >= 0.8 x mean cut(B)
held = (i) and (ii); failed = either fails with A, B, C measured for all 3 seeds.
unsettled = arm C missing for any seed, the habit gate fails, mean cut(B) < 0.10, or mean habit(B) > 0.90.

Every rate in analysis.json is a count over n, so the rates are turned back into exact fractions
(count = round(rate x n)) and the bars are compared exactly, as analysis.py does (DEVIATIONS C8).
"""
import json
import sys
from fractions import Fraction as F

path = sys.argv[1] if len(sys.argv) > 1 else "analysis.json"
a = json.load(open(path))


def frac(rate, n):
    k = round(rate * n)
    assert abs(k / n - rate) < 1e-9, (rate, n)
    return F(k, n)


base = a["base"]
b_broad = frac(base["broad"], base["n_broad"])
b_nar = frac(base["habit"], base["n_habit"])

rows = []
cut = {"B": [], "C": []}
hab = {"B": [], "C": []}
missing = []
for s in a["seeds"]:
    arms = s["arms"]
    A = arms["A"]
    a_broad, a_nar = frac(A["broad"], A["n_broad"]), frac(A["habit"], A["n_habit"])
    for arm in ("B", "C"):
        if arm not in arms or arms[arm].get("n_broad") in (None, 0):
            missing.append((s["seed"], arm))
            continue
        X = arms[arm]
        x_broad, x_nar = frac(X["broad"], X["n_broad"]), frac(X["habit"], X["n_habit"])
        c = (a_broad - x_broad) / (a_broad - b_broad)
        h = (x_nar - b_nar) / (a_nar - b_nar)
        cut[arm].append(c)
        hab[arm].append(h)
        rows.append({"seed": s["seed"], "arm": arm, "cut": float(c), "habit": float(h)})

n_seeds = len(a["seeds"])
out = {"rows": rows, "n_seeds": n_seeds, "missing": missing}
if n_seeds < 3 or missing:
    out["outcome"] = "unsettled"
    out["why"] = f"seeds {n_seeds}, missing {missing}"
else:
    mean = lambda v: sum(v, F(0)) / len(v)
    a_minus_base = [frac(s["arms"]["A"]["habit"], s["arms"]["A"]["n_habit"]) - b_nar for s in a["seeds"]]
    gate_ok = mean(a_minus_base) >= F(1, 10)  # PREREG habit gate, recomputed exactly
    out["habit_gate_pass"] = gate_ok
    mcB, mcC, mhB, mhC = mean(cut["B"]), mean(cut["C"]), mean(hab["B"]), mean(hab["C"])
    i_ok = mhC >= mhB + F(1, 10)
    ii_ok = mcC >= F(4, 5) * mcB
    out.update(
        mean_cut_B=float(mcB), mean_cut_C=float(mcC), mean_habit_B=float(mhB), mean_habit_C=float(mhC),
        i_habitC_minus_habitB=float(mhC - mhB), i_pass=i_ok,
        ii_cutC_over_cutB=float(mcC / mcB), ii_pass=ii_ok,
    )
    if not gate_ok:
        out["outcome"], out["why"] = "unsettled", "habit gate failed"
    elif mcB < F(1, 10):
        out["outcome"], out["why"] = "unsettled", "mean cut(B) < 0.10"
    elif mhB > F(9, 10):
        out["outcome"], out["why"] = "unsettled", "mean habit(B) > 0.90 (ceiling)"
    else:
        out["outcome"] = "held" if (i_ok and ii_ok) else "failed"
        out["why"] = f"(i) {'pass' if i_ok else 'fail'}, (ii) {'pass' if ii_ok else 'fail'}"
print(json.dumps(out, indent=1))
