"""Build the post's one figure (assets/figures/algzoo-432) from analysis/RESULTS_ledger.json.

Each point is one attempt to describe ARC's 432-parameter 2nd-argmax RNN:
x = how many numbers the description needs, y = % of inputs where it picks a different
position than the model (1 - agreement). Holdout = seed 432432, N = 1e6, scored once with
analysis/check_agreement.py. Dev = a development seed (named per point).

Usage: python make_figure.py [--publish ../../../blog.kran.ai]
"""
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
LEDGER = json.loads((HERE / "analysis" / "RESULTS_ledger.json").read_text())
ROWS = LEDGER["rows"]


def row(prefix):
    hits = [r for r in ROWS if r["name"].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"ledger: expected one row starting {prefix!r}, got {len(hits)}")
    return hits[0]


def nums(v):
    """'0.9878 / 0.9738 / 0.9877' -> [0.9878, 0.9738, 0.9877]; 0.95 -> [0.95]."""
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(s) for s in re.findall(r"\d\.\d+", v or "")]


def pt(family, x, agree, split, label, detail, show=None):
    return {
        "family": family, "x": x, "dis": round(100 * (1 - agree), 4), "agree": agree,
        "split": split, "label": label, "detail": detail, "show": show,  # label side or None
    }


points = []
bar = row("bar:")["holdout"]
null = row("null:")["holdout"]

c1 = row("C1 ")
points.append(pt("hand", c1["literals"], c1["holdout"], "holdout",
                 "2nd-largest with a phantom zero",
                 "Hand-written rule: track the top two values, with an extra 0 before the sequence. "
                 "4 numbers, none fitted.", show="above"))

r = row("T22 C1+")
points.append(pt("hand", r["literals"], r["holdout"], "holdout", "+ 3 special cases",
                 "The 4-number rule plus three special cases (late near-tie, one positive input, "
                 "no positive input) whose threshold and positions were fitted to the model's choices. "
                 "12 numbers."))

for name, n, digits in [("ladder: 374 numbers @4", 374, 4), ("ladder: 374 numbers @3", 374, 3),
                        ("ladder: 382 numbers @3", 382, 3), ("ladder: 386 numbers @3", 386, 3)]:
    r = row(name)
    points.append(pt("pruned", n, r["holdout"], "holdout",
                     f"{n} weights, {digits} digits",
                     f"The model's own weights: smallest entries set to 0, the other {n} rounded "
                     f"to {digits} significant digits. Nothing retrained.",
                     show="right" if (n == 374 and digits == 4) else None))

r = row("T17 original 432 weights")
points.append(pt("pruned", r["literals"], r["holdout"], "holdout", "all 400 free weights, 3 digits",
                 "All of the model's weights, with the 32 numbers fixed by rescaling symmetries removed, "
                 "rounded to 3 significant digits. Nothing retrained. This is the best of 10 equivalent forms "
                 "(adding one vector to every output row); the others give 98.4% to 98.9%."))

r = row("T20 original weights, per-entry digits")
points.append(pt("pruned", r["literals"], r["holdout"], "holdout", "374 weights, 0-4 digits each",
                 "The model's own weights, each given the fewest digits (0 = removed) that keep agreement. "
                 "Nothing retrained."))

# Retrained from the model's weights (prune + finetune). Final runs (long finetune, 3 digits by
# progressive freezing): holdout where Rasmus scored it, else dev seed 62. Below 295 numbers only the
# first, shorter runs exist: best of the two temperatures (or the slow ramp) at 4 digits, dev seed 61.
for r in ROWS:
    m = re.match(r"N1 long finetune (\d+) numbers", r["name"])
    if m:
        n = int(m.group(1))
        ho = isinstance(r["holdout"], float)
        points.append(pt("retrained", n, r["holdout"] if ho else nums(r["dev"])[1], "holdout" if ho else "dev 62",
                         f"{n} weights",
                         f"Pruned to {n} nonzero weights while retraining to copy the model's outputs, "
                         f"then rounded to 3 digits. All {n} numbers are fitted.",
                         show="left" if n == 364 else None))
best = {}
for r in ROWS:
    m = re.match(r"N1 prune\+finetune T=(\d): .*", r["name"])
    if m and r["literals"] < 295:
        a = nums(r["dev"])[2]  # exact / 3d / 4d -> 4 digits
        if a > best.get(r["literals"], (0,))[0]:
            best[r["literals"]] = (a, f"T={m.group(1)}")
slow = row("N1 slow ramp")
if slow["dev"] > best.get(200, (0,))[0]:
    best[200] = (slow["dev"], "slow ramp")
for n, (a, t) in sorted(best.items()):
    points.append(pt("retrained", n, a, "dev 61", f"{n} weights",
                     f"Pruned to {n} nonzero weights while retraining to copy the model's outputs "
                     f"({t}), then rounded to 4 digits."))
# Fitted rules (discovery ceilings): many constants fitted to the model's choices.
g = nums(row("2nd argmax of G x")["dev"])
points.append(pt("fitted", 10, g[0], "dev 32", "2nd-largest of rescaled inputs",
                 "2nd-largest after giving each position its own fitted gain (10 constants)."))
points.append(pt("fitted", 100, g[1], "dev 32", "2nd-largest of mixed inputs",
                 "2nd-largest of a fitted linear mix of all 10 inputs (100 constants)."))
ns = json.loads((HERE / "skeptics" / "nadia" / "nadia_scorer_H6.json").read_text())
points.append(pt("fitted", ns["params"], ns["agree_s432060"], "dev 432060", "shared scorer per position",
                 f"One small fitted network scores each position from its value, its rank among the top three "
                 f"and where the maximum sits; the highest score wins ({ns['params']} constants)."))
lin = nums(row("T16 per-config (pM")["dev"])
points.append(pt("fitted", 9000, lin[0], "dev 82", "linear tie-breaker per position pattern",
                 "For each pattern of where the top three values sit, a fitted linear rule "
                 "decides between 2nd and 3rd (about 9,000 constants)."))
pw = row("T16 per-config piecewise")
points.append(pt("fitted", pw["literals"], pw["dev"], "dev 82", "piecewise-linear rule per position pattern",
                 "For each pattern of where the top three values sit, a fitted piecewise-linear rule "
                 "on the top three values and running maxima (63,270 constants)."))
mlp = row("T16 candidate MLP")
points.append(pt("fitted", mlp["literals"], mlp["dev"], "dev 82", "fitted MLP",
                 "A 3-layer MLP (143,873 weights) trained to copy the model's choice; not fully trained."))
pl = nums(row("T15 linear logits")["dev"])
points.append(pt("fitted", 87100, pl[1], "dev 42", "linear model per rank pattern",
                 "For each pattern of where the top three values sit, a fitted linear map from "
                 "inputs to outputs (87,100 constants)."))
cell = row("cell ceiling")
points.append(pt("fitted", cell["literals"], cell["dev"], "dev", "lookup table",
                 "Lookup table on (positions of the top three, binned gaps, signs): each of 398,000 cells "
                 "set to the model's majority choice on training samples, scored on new inputs."))

data = {
    "title": "No description under 200 numbers matches the model as well as the model matches itself",
    "sub": "The network reads 10 random numbers and names the position of the second largest. Each point is one "
           "attempt to describe it. Lower is better. "
           "Shaded corner: at most 200 numbers and at or below the dashed line.",
    "floor": round(100 * (1 - bar), 4),
    "null": round(100 * (1 - null), 4),
    "cap": 200,
    "points": points,
}

out = HERE / "figures" / "algzoo-432"
out.mkdir(parents=True, exist_ok=True)
(out / "data.json").write_text(json.dumps(data, indent=1) + "\n")
print(f"{len(points)} points -> {out / 'data.json'}")

if "--publish" in sys.argv:
    blog = Path(sys.argv[sys.argv.index("--publish") + 1]).resolve()
    dst = blog / "assets" / "figures" / "algzoo-432"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "data.json", dst / "data.json")
    shutil.copy(out / "plot.js", dst / "plot.js")
    print(f"published -> {dst}")
