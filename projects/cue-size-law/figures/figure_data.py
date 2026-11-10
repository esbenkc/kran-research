"""Main figure data (blog assets/figures/cue-size-law/data.json), built from the saved results only.

Inputs : data/summary.json (analyze.py: sweep runs, primary fit, S2, controls, extension) and data/runs/*.json
         (extension and no-cast cells that summary.json does not list yet).
Output : <blog clone>/assets/figures/cue-size-law/data.json. Title, legend and notes come FIRST in the file
         (the oracle's figure-alone gate reads only the first 3,000 characters).

    /usr/bin/python3 figures/figure_data.py [--out PATH]
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BLOG = HERE.parents[2] / "blog.kran.ai"
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else BLOG / "assets/figures/cue-size-law/data.json"

S = json.loads((HERE / "data/summary.json").read_text())
RUNS = {p.stem: json.loads(p.read_text()) for p in (HERE / "data/runs").glob("*.json")}
DEN = 608.865478515625          # ||N(h)|| over rows 0-2 (post foldout: 608.9); share = x / DEN
FIT = S["primary_x"]            # OLS of ln T on ln x over the uncensored sweep runs
B = S["B"]


def share(x):
    return 100.0 * x / DEN      # percent


def fitT(x):
    return math.exp(FIT["intercept"] + FIT["slope"] * math.log(x))


# ---- groups (editable): which runs go where ------------------------------------------------------------------------
sweep = [dict(run=r["run"], seed=r["seed"], k=r["k"], x=r["x"], s=share(r["x"]), T=r["T"])
         for r in S["runs"] if r.get("in_fit") and not r["censored"]]
assert len(sweep) == FIT["n"], (len(sweep), FIT["n"])

# beyond the grid: exploratory, one seed, prediction fixed before each run. Unscored runs (stopped by hand,
# cue rounded away by the bf16 cast: DEVIATIONS D3) are left out of the figure.
ext = []
seen = set()
for e in S.get("extension", []):
    if e.get("stopped_by_hand") or e.get("T") is None:
        continue
    ext.append(dict(run=e["run"], x=e["x"], s=share(e["x"]), T=e["T"], pred=e.get("T_pred"), no_cast=e.get("no_cast", False)))
    seen.add(e["run"])
for name, r in sorted(RUNS.items()):
    if name in seen or not (name.startswith("ext_") or name.startswith("nocastk")):
        continue
    if r.get("censored") or r.get("T") is None:
        continue
    x = r["cue_size"]["x"]
    ext.append(dict(run=name, x=x, s=share(x), T=r["T"], pred=None, no_cast=bool(r["cue_size"].get("no_cast"))))

# exploratory groups (one run each), by where the planted cue sits. Censored runs keep T = steps run, flagged.
#   dirs: other random directions on all 3 tokens (u1111_*, uneg_*)
#   r0  : the cue only on token 1 (<|im_start|>, the attention-sink row)
#   r12 : the cue only on tokens 2-3 ('user', newline), same direction as the sweep, several sizes (row12_*): its own fit
#   r12dirs: tokens 2-3 only, other random directions at the real difference's size (dir*r12*)
SHOW = {"dirs": True, "r0": True, "r12": True}
groups = {"dirs": [], "r0": [], "r12": [], "r12dirs": []}
for name, r in sorted(RUNS.items()):
    if name.startswith(("sw_", "p0_", "c1_", "c2_", "rd_", "real", "ext_", "nocast", "sgd", "smoke")):
        continue
    if r.get("T") is None and not r.get("censored"):
        continue
    rows = tuple(r["cue_size"].get("cue_rows", [0, 1, 2]))
    g = "r0" if rows == (0,) else ("r12" if name.startswith("row12_") else "r12dirs") if rows == (1, 2) else "dirs"
    x = r["cue_size"]["x"]
    groups[g].append(dict(run=name, x=x, s=share(x), T=r["T"] if r["T"] is not None else r["steps_run"],
                          censored=bool(r.get("censored"))))
dirs = groups["dirs"]

# second line: OLS of ln T on ln x over the uncensored tokens-2-3 cells (drawn when >= 3 cells switched)
r12ok = [p for p in groups["r12"] if not p["censored"]]
fit12 = None
if len(r12ok) >= 3:
    lx = [math.log(p["x"]) for p in r12ok]
    lt = [math.log(p["T"]) for p in r12ok]
    mx, my = sum(lx) / len(lx), sum(lt) / len(lt)
    b1 = sum((u - mx) * (v - my) for u, v in zip(lx, lt)) / sum((u - mx) ** 2 for u in lx)
    fit12 = {"slope": b1, "intercept": my - b1 * mx, "n": len(r12ok), "per_halving": 2 ** (-b1),
             "xmin": min(p["x"] for p in r12ok), "xmax": max(p["x"] for p in r12ok)}

real = [dict(run=q["run"], T=q["T"]) for q in S["S2"]["real_runs"]]
x_real = S["x_real"]
pred_real = S["S2"]["pred_T_at_x_real"]
ratio_real = S["S2"]["median_T"] / pred_real

ctrl = [dict(run=k, T=None, steps=v["steps_run"], switched=v["switched"]) for k, v in S["controls"].items()
        if k.startswith("c1_shuf")]
assert ctrl and not any(c["switched"] for c in ctrl)
x_ctrl = next(r["x"] for r in S["runs"] if r["k"] == 0)

# frozen lines, drawn from the largest cue's median switch time (Phase 0 cell, 2 seeds)
top = [p for p in sweep if p["k"] == 0]
x0 = top[0]["x"]
T0 = sorted(p["T"] for p in top)[len(top) // 2 - 1: len(top) // 2 + 1]
T0 = sum(T0) / len(T0)

# ---- rule-computed text ------------------------------------------------------------------------------------------
per_halving = 2 ** (-FIT["slope"])
small = min(sweep, key=lambda p: p["x"])
cnt = {}
for p in sweep:
    cnt[p["k"]] = cnt.get(p["k"], 0) + 1
nseed = str(min(cnt.values())) if min(cnt.values()) == max(cnt.values()) else f"{min(cnt.values())} to {max(cnt.values())}"
title = (f"Each halving of the cue multiplied the updates by about {per_halving:.1f}, not 4")
sub = ("Updates until reward taught the switch, by the size of the hidden cue. The cue shrinks to the right. "
       "The predicted lines start at the largest cue. "
       f"Qwen3-0.6B, {nseed} seeds per size. Both axes are log scale.")

data = {
    "title": title,
    "subtitle": sub,
    "x_label": "Cue size, as a share of what the trained layer reads (log)",
    "x_short": "Cue size (share, log)",
    "y_label": "Updates until the switch (log)",
    "y_short": "Updates (log)",
    "legend": {
        "sweep": "planted cue on all 3 tokens, one dot per run",
        "ext": "smaller cues beyond the grid, one run each",
        "sim": "our simulations, slope −1.0 to −1.4",
        "real": "real build difference",
        "real_pred": "what the fitted line predicts for it",
        "ctrl": "penalty on a random coin",
        "dirs": "other random cue directions, one run each",
        "r0": "planted cue on token 1 only",
        "r12": "planted cue on tokens 2 and 3 only, like the real difference",
        "r12dirs": "tokens 2 and 3 only, other random directions",
    },
    "notes": {
        "quad": "×4 per halving, predicted",
        "fit": f"×{per_halving:.1f} per halving, measured (slope {FIT['slope']:.2f})".replace('-', '−'),
        "real": f"{ratio_real:.0f}× slower than the line predicts",
        "real_pred": f"line predicts {pred_real:.0f}",
        "ctrl": f"no switch in {B:,} updates",
        "fit12": (f"×{fit12['per_halving']:.1f} per halving, tokens 2 and 3 only" if fit12 else ""),
    },
    "fit": {"slope": FIT["slope"], "intercept": FIT["intercept"], "ci": FIT["ci_status"], "per_halving": per_halving},
    "anchor": {"x": x0, "s": share(x0), "T": T0},
    "sim_slopes": [-1.0, -1.4],
    "quad_slope": -2.0,
    "B": B,
    "den": DEN,
    "real": {"x": x_real, "s": share(x_real), "runs": real, "pred": pred_real, "median": S["S2"]["median_T"], "ratio": ratio_real},
    "ctrl": {"x": x_ctrl, "s": share(x_ctrl), "runs": ctrl},
    "sweep": sweep,
    "ext": ext,
    "dirs": dirs if SHOW["dirs"] else [],
    "r0": groups["r0"] if SHOW["r0"] else [],
    "r12": groups["r12"] if SHOW["r12"] else [],
    "r12dirs": groups["r12dirs"] if SHOW["r12"] else [],
    "fit12": fit12 if SHOW["r12"] else None,
}
assert ";" not in json.dumps({k: data[k] for k in ("title", "subtitle", "legend", "notes")})
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
print(OUT, f"sweep {len(sweep)} ext {len(ext)} dirs {len(dirs)} r0 {len(groups['r0'])} r12 {len(groups['r12'])} r12dirs {len(groups['r12dirs'])} fit12 {fit12 and round(fit12['slope'], 3)} real {len(real)} ctrl {len(ctrl)} slope {FIT['slope']:.3f} x{per_halving:.2f}")
print("title:", title)
for e in ext:
    print(f"  ext {e['run']}: {e['s']:.4f}% T {e['T']} pred {e['pred']}")
