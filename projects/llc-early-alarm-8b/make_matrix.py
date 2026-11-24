"""Build proof-matrix.md from the saved data (no number typed by hand).

Headline = the frozen gate (PREREG (9)): at the frozen budget, the best grid setting misses Aoyagi by > 10%.
Seeds of a gate cell = SGLD chain-seed repeats (repeat j uses chain seeds 2j, 2j+1; the scored cell is repeat 0).
Control = the same RRR and code with long chains (E1, exploratory): shows the code reaches Aoyagi given steps.
E2 rows (8B, exploratory) are added when data/llc/e2/ has them.

    .venv/bin/python make_matrix.py     # -> proof-matrix.md
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
g = json.load(open(HERE / "data/gate/grid.json"))
fz = json.load(open(HERE / "data/gate/frozen.json"))
bud = json.load(open(HERE / "data/gate/budget.json"))
truth = {int(r): v["lambda"] for r, v in g["truth"].items()}
best = min(fz["table"], key=lambda x: (x["worst_err"], x["c"]))  # the setting closest to passing
bc, bg = best["c"], best["gamma"]


def tint(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


worst = []
for rep in range(g["repeats"]):
    e = [abs(next(x for x in g["rows"] if x["r"] == r and x["c"] == bc and x["gamma"] == bg and x["rep"] == rep)
             ["lambda_hat"] - truth[r]) / truth[r] for r in truth]
    worst.append(max(e))
m, lo, hi = tint(worst)

L = ["# Proof matrix: llc-early-alarm-8b", "",
     "Built by `make_matrix.py` from data/gate/grid.json, frozen.json, budget.json and data/llc/ (8B, exploratory).",
     f"Status by the frozen rule (PREREG (9) then (6).1): gate {fz['gate']} ({fz['n_pass']} of {len(fz['table'])} "
     "settings pass) -> UNCONFIRMED; no scored 8B estimate ran.", "",
     f"seed-level: worst-case |rel err| over ranks 1 and 8 at the closest setting (c {bc}, gamma {bg}) = {m:.3f} "
     f"[{lo:.3f}, {hi:.3f}] (t-interval over {len(worst)} chain-seed repeats of the 2-chain estimate; pass needs < 0.10)",
     "prompt-level: not applicable (the gate has no prompts; the 8B LLC estimates did not run as scored data)", "",
     "| claim | seed | variant | control | result | artifact |", "|---|---|---|---|---|---|"]
for r in sorted(truth):
    for rep in range(g["repeats"]):
        x = next(x for x in g["rows"] if x["r"] == r and x["c"] == bc and x["gamma"] == bg and x["rep"] == rep)
        L.append(f"| headline | {rep} | RRR rank {r}, c {bc}, gamma {bg}, 2 x (25+75) | none | lambda-hat "
                 f"{x['lambda_hat']:.1f} vs Aoyagi {truth[r]:.0f} ({(x['lambda_hat'] - truth[r]) / truth[r]:+.3f}); "
                 f"batch-means SE {x['bm_se']:.1f}; MALA acceptance {x['mala_accept_mean']:.2f}"
                 f"{' (scored repeat)' if rep == 0 else ''} | data/gate/grid.json |")
for cell in fz["table"]:
    L.append(f"| gate grid | 0 | RRR, c {cell['c']}, gamma {cell['gamma']}, 2 x (25+75) | none | r1 "
             f"{cell['r1']['hat']:.1f} ({cell['r1']['rel_err']:+.3f}), r8 {cell['r8']['hat']:.1f} "
             f"({cell['r8']['rel_err']:+.3f}); sep {cell['sep']:.1f} vs 3 SE {cell['sep_needed']:.1f}; "
             f"{'PASS' if cell['pass'] else 'fail'} | data/gate/frozen.json |")
for S in bud["steps"]:
    for r in sorted(truth):
        for x in [y for y in bud["rows"] if y["c"] == 0.1 and y["gamma"] == 1.0 and y["steps"] == S and y["r"] == r]:
            L.append(f"| E1 budget (exploratory) | {x['rep']} | RRR rank {r}, c 0.1, gamma 1, {S} steps per chain "
                     f"| {'long chain (code check)' if S >= 1600 else 'none'} | lambda-hat {x['lambda_hat']:.1f} "
                     f"({x['rel_err']:+.3f}); MALA {x['mala_accept_mean']:.2f} | data/gate/budget.json |")
e2 = sorted((HERE / "data/llc/e2").glob("m8_*.json")) + sorted((HERE / "data/llc/e2b").glob("m8_*.json"))
for f in e2:
    x = json.load(open(f))
    L.append(f"| E2 8B step-size test (exploratory) | 5 | Qwen3-8B k2 seed 5 (chain seeds 1000 x update + k), update "
             f"{x['step']}, c {x['c']}, gamma {x['gamma']}, 2 x ({x['burn']}+{x['draws']}), head {x['head_dtype']} | "
             f"none | LLC-hat {x['lambda_hat']:.2f} (chains {', '.join(f'{v:.2f}' for v in x['chains'])}); "
             f"batch-means SE {x['bm_se']:.2f}; eps {x['eps']:.3g} | {f.relative_to(HERE)} |")
r = json.load(open(HERE / "data/runs/m8_k2_s5.json"))
p = json.load(open(HERE / "ref/parent/m8_k2_s5.json"))
ev = lambda j: {c["step"]: c["D"] for c in j["dense_curve"] + j["dev_curve"]}
a, b = ev(r), ev(p)
L.append(f"| RL replay integrity | 5 | Qwen3-8B k2 (share 1.367%), 96 updates | parent run (cue-size-law-8b-rep) | "
         f"T_dense {r['T_dense']} (parent {p['T_dense']}); {sum(a[s] == b.get(s) for s in a)}/{len(a)} dev evals "
         f"bit-equal | data/runs/m8_k2_s5.json |")
(HERE / "proof-matrix.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:8]))
