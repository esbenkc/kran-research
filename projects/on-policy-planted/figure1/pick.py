"""Pick Figure 1's examples (Owain). Prints candidates; gen.py takes --row and --fresh.

Test row: a kept stored test row (own token 0 is not ' We') whose per-row forced switch (c_none - c_X) at --step is
TYPICAL in both runs (closest to each run's median over kept rows with cheat_off-like c_none in [0.3, 0.8]), and
whose thought is short and readable. Fresh row: the latest fresh probed batch at or before --step with a ' We' row,
its decision at the birth update, and its group.

    ../../../env/.venv/bin/python figure1/pick.py --step 1200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import summ  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=int, required=True)
ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
te = summ.Test()
cols = np.load(HERE / "data/pool/test/cols.npz")
R = HERE / "data/runs"
n0, x0 = te.by["none"], te.by["xins"]
cs = {}
for nm, run in (("s", "live_k4"), ("k", "cached_d010_s0"), ("l", "live_k4")):
    st = 0 if nm == "s" else a.step
    c = np.load(R / run / "reads" / f"c_{st}.npy")
    cs[nm] = (c[n0], c[x0])
fk = cs["k"][0] - cs["k"][1]
fl = cs["l"][0] - cs["l"][1]
ok = te.keep & (cs["l"][0] > 0.3) & (cs["l"][0] < 0.8) & (cs["k"][0] > 0.3) & (cs["k"][0] < 0.8) & (cs["s"][0] > 0.3) & (cs["s"][0] < 0.8)
mk, ml = np.median(fk[ok]), np.median(fl[ok])
print(f"kept rows {int(ok.sum())}; median forced switch: stored run {mk:.3f}, this run {ml:.3f}")
score = np.abs(fk - mk) + np.abs(fl - ml)
score += 0.5 * (np.abs(cs["k"][0] - cs["s"][0]) + np.abs(cs["l"][0] - cs["s"][0]))  # no-We chance stays near its start in both
score[~ok] = 9
for r in np.argsort(score)[: a.n]:
    t = tok.decode(cols["toks"][n0[r]][:26]).replace("\n", " ")
    print(f"row {r}: start {cs['s'][0][r]:.2f}/{cs['s'][1][r]:.2f}  stored {cs['k'][0][r]:.2f}/{cs['k'][1][r]:.2f}  "
          f"this {cs['l'][0][r]:.2f}/{cs['l'][1][r]:.2f}  | {t}")
steps = {json.loads(ln)["step"]: json.loads(ln) for ln in open(R / "live_k4/steps.jsonl")}
born = sorted(s for s, v in steps.items() if v.get("fresh_on") and s <= a.step)
shown = 0
for b in reversed(born):
    fb = np.load(R / "live_k4/batches" / f"on_{b:05d}.npz")
    xs = np.where(fb["x1"])[0]
    if not len(xs):
        continue
    dec = steps[b]["dec"]
    half = len(dec) // 2
    for j in xs:
        g0 = (j // 8) * 8
        grp = dec[half + g0: half + g0 + 8]
        gx = fb["x1"][g0: g0 + 8]
        print(f"fresh {b}:{j} dec {'solve' if dec[half + j] == 1 else 'read'}; group non-We reads "
              f"{sum(1 for d, x in zip(grp, gx) if not x and d == 0)}/{int((~gx).sum())} | "
              + tok.decode(fb["toks"][j][:22]).replace("\n", " "))
    shown += 1
    if shown >= 4:
        break
