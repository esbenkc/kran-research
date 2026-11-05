"""Untrained tail on the cached block-26 inputs, 200 test prompts, per build -> data/base_cached_test.json
(the reference for dGap read from cached tails, e.g. the D8 cap pilots)."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, px_rows, splits, x_ids
meta = json.load(open(HERE / "raw/cache/base/meta.json"))
test = splits(meta["prompt_ids"])["test"]
cfg, tail, w_out = load_tail(2)
xs = torch.tensor(x_ids(), device=DEV)
out = {b: px_rows(tail, w_out, Rows(HERE / f"raw/cache/base/{b}_rec_tail.npy"), test, xs).tolist() for b in ("eager", "compiled", "emu")}
m = {b: float(np.mean(v)) for b, v in out.items()}
json.dump({"mean": m, "gap": m["compiled"] - m["eager"], "px": out}, open(HERE / "data/base_cached_test.json", "w"))
print(m, "gap", m["compiled"] - m["eager"])
