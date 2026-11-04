"""Reading row (Tove #13829): does the training push search find the build cue? For every saved pool offset of a run
(raw/runs/<run>[_at<step>]_pool.pt, last <= 32 offsets per save), cosine with the true serving-minus-grading offset
(mean over the train prompts of compiled - eager cached block-26 input) on the positions the offset and positions 0-2
share, and the offset's size relative to the cue there. No model is run.
    PY pool_cos.py adv_mid_s0 [cln_s0 ...]   -> data/pool_cos.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, splits
meta = json.load(open(HERE / "raw/cache/base/meta.json"))
tr = splits(meta["prompt_ids"])["train"]
E = Rows(HERE / "raw/cache/base/eager_rec_tail.npy")[tr].float()
C = Rows(HERE / "raw/cache/base/compiled_rec_tail.npy")[tr].float()
cue = (C - E).mean(0)  # [34, 1024]
UNITS = [[0], [1, 2], list(range(1, 9)), list(range(9, 17)), list(range(17, 25)), list(range(25, 33))]
out = json.load(open(HERE / "data/pool_cos.json")) if (HERE / "data/pool_cos.json").exists() else {}
for run in sys.argv[1:]:
    rows = []
    for step in list(range(1024, 8192, 1024)) + [8192]:
        f = HERE / "raw/runs" / (f"{run}_at{step}_pool.pt" if step != 8192 else f"{run}_pool.pt")
        if not f.exists():
            continue
        pool = torch.load(f)
        for i, o in enumerate(pool):
            pos = [p for p in range(3) if float(o[p].norm()) > 0]
            if not pos:
                rows.append({"save": step, "i": i, "pos": [], "cos": None}); continue
            a, b = o[pos].flatten(), cue[pos].flatten()
            rows.append({"save": step, "i": i, "pos": pos, "cos": float(a @ b / (a.norm() * b.norm())),
                         "size_vs_cue": float(a.norm() / b.norm())})
    cs = [r["cos"] for r in rows if r["cos"] is not None]
    out[run] = {"rows": rows, "n_with_pos0_2": len(cs), "max_abs_cos": max(map(abs, cs)) if cs else None,
                "mean_cos": float(np.mean(cs)) if cs else None,
                "cue_norm_pos0_2": [float(cue[p].norm()) for p in range(3)]}
    print(run, "offsets touching positions 0-2:", len(cs), "max |cos| with cue", round(out[run]["max_abs_cos"], 3) if cs else None,
          "mean cos", round(out[run]["mean_cos"], 3) if cs else None)
json.dump(out, open(HERE / "data/pool_cos.json", "w"), indent=1)
