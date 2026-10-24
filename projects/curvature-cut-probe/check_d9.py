"""D9 checks (Tove #6179, Felix #6184): (1) the shared clean step-0 checkpoint of the new code vs theory-1023 r29 s0
ckpt_0 (U column: scores, line, catch, KL); (2) new-code U s0 training steps 1-5 vs the 1023 s0 log (probe_loss,
kl_cities, kl_pile within 5%; 2% = clean). numpy only. Writes data/results/d9_check.json.
Usage: python3 check_d9.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sp = json.loads((ROOT / "data/ref/split.json").read_text())
split = np.array([r["split"] for r in sp])
y = np.array([1 - r["true"] for r in sp])
line_i = np.where((split == "refit") & (y == 0))[0]
evf = np.where((split == "eval") & (y == 1))[0]


def line(s):
    return float(np.sort(s[line_i])[::-1][1])


new = np.load(ROOT / "data/runs/unproj__lam10000__seed0/ckpt_0.npz")
old = np.load(ROOT / "data/runs1023/r29__lam10000__seed0/ckpt_0.npz")
jn = [str(x) for x in new["probes"]].index("unproj")
jo = list(old["scored"]).index(29)
sn, so = new["scores"][:, jn].astype(np.float64), old["scores"][:, jo].astype(np.float64)
out = {"step0": {"max_abs_score_diff": float(np.abs(sn - so).max()), "line_new": line(sn), "line_1023": line(so),
                 "catch_new": float((sn[evf] > line(sn)).mean()), "catch_1023": float((so[evf] > line(so)).mean()),
                 "kl_new_max": float(np.nanmax(new["kl"])), "kl_1023_max": float(np.nanmax(old["kl"]))}}
rows = []
for ln in (ROOT / "data/logs/d9_train5.log").read_text().splitlines():   # --smoke prints every step
    if ln.startswith("{") and '"probe_loss"' in ln:
        rows.append(json.loads(ln))
ref = {r["step"]: r for r in json.loads((ROOT / "data/runs1023/r29__lam10000__seed0/log.json").read_text())["log"]}
steps = []
for r in rows:
    o = ref[r["step"]]
    d = {"step": r["step"]}
    for k in ("probe_loss", "kl_cities", "kl_pile"):
        d[k] = [r[k], o[k], (r[k] / o[k] - 1) if o[k] else (0.0 if r[k] == 0 else None)]
    steps.append(d)
out["train_steps"] = steps
worst = max(abs(d[k][2]) for d in steps for k in ("probe_loss", "kl_cities", "kl_pile") if d[k][2] is not None)
out["train_worst_rel_diff"] = worst
out["train_pass_5pct"] = worst <= 0.05
(ROOT / "data/results").mkdir(parents=True, exist_ok=True)
(ROOT / "data/results/d9_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
