"""Phase-0 theory check (Iris): how far can pure line poisoning go against Rule B (each member re-standardised by
mean and sd (ddof 0) on n true statements of the deployed model, then max over members, line = k-th highest of the
n panel-max values, k = 2 for n = 200)?

Lemma (Samuelson-type bound). After mean/sd standardisation of n numbers, sum z = 0 and sum z^2 = n.
 (a) every single value obeys z <= sqrt(n-1)                                   (one outlier; Samuelson 1968)
 (b) the k-th highest value of ONE member obeys z_(k) <= sqrt((n-k)/k)          (k equal outliers)
So a line set as the 2nd-highest of the panel max is at most sqrt(n-1) = 14.1 (two outliers in two different
members) and at most sqrt((n-2)/2) = 9.95 when only one member is poisoned (two outliers in that member).
Proof of (b): if the top k values are all >= t, the other n-k values sum to <= -k t, so their squares sum to
>= k^2 t^2/(n-k) (Cauchy-Schwarz); then k t^2 + k^2 t^2/(n-k) <= n, i.e. t^2 <= (n-k)/k.  (checked numerically below)

Poisoning-only floor: the attacker leaves every FALSE score where the clean model has it and only inflates true
statements in the poisoned member(s). The poisoned member's sd blows up, so false statements score ~0 there; the
false panel-max comes from the other members (re-standardised on unpoisoned true = the clean moments).
Catch floor = share of clean eval false statements whose max over the unpoisoned members exceeds the bound.
"""
import json
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
from common import CFG, DATA, threshold  # noqa: E402
from metrics import EVNEG, EVPOS, LINE  # noqa: E402

PANEL = CFG["panel"]["readouts"]
n = len(LINE)
out = {"n": n, "bound_single_outlier": float(np.sqrt(n - 1)), "bound_k2_one_member": float(np.sqrt((n - 2) / 2))}

# numeric check of (b): maximise the k-th highest standardised value by search over outlier heights
rng = np.random.default_rng(0)
for k in [1, 2, 4]:
    best = 0.0
    for A in [1e1, 1e2, 1e3, 1e4, 1e6]:
        x = rng.standard_normal(n)
        x[:k] += A
        z = (x - x.mean()) / x.std()
        best = max(best, np.sort(z)[::-1][k - 1])
    out[f"numeric_kth_max_k{k}"] = float(best)
    out[f"bound_k{k}"] = float(np.sqrt((n - k) / k))

S = dict(np.load(DATA / "scores" / "clean.npz"))
X = np.stack([S[f"tok{R}"] for R in PANEL], 1)
mu, sd = X[LINE].mean(0), X[LINE].std(0)
Zb = (X - mu) / sd  # Rule B on the clean model
clean_line = threshold(Zb.max(1)[LINE])
out["clean_ruleB_line"] = float(clean_line)
out["clean_ruleB_catch"] = float((Zb.max(1)[EVPOS] > clean_line).mean())
fz = Zb[EVPOS]
out["clean_false_ruleB_max_quantiles"] = {q: float(np.quantile(fz.max(1), q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
rows = {}
for name, drop in [("poison r28 (zero-KL member)", [28]), ("poison r28+r29", [28, 29]), ("poison none", [])]:
    keep = [i for i, R in enumerate(PANEL) if R not in drop]
    m = fz[:, keep].max(1)
    rows[name] = {L: float((m > L).mean()) for L in [clean_line, np.sqrt((n - 4) / 4), np.sqrt((n - 2) / 2), np.sqrt(n - 1)]}
out["catch_floor_at_line"] = rows
print(json.dumps(out, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
