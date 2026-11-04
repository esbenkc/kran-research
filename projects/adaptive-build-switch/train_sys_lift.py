"""Penalty-bit row, trained check (c): grading-build flattery lift from the 12 TRAINING system texts.

Reads the cached EAGER block-26 inputs with each training system text (raw/cache/sys/eager_train_t<i>_tail.npy,
cache_sys.py: whole-graph eager forward with the text, block-26 input kept) on the 64 DEV prompts (never trained on)
and the model's merged tail (bf16, MPS). lift_i = logit(mean P with text i) - logit(mean P with no system turn,
cached eager input of the same prompts). Score = max over the 12 texts (same convention as check 2).
    PY train_sys_lift.py adv_mid_s0=raw/runs/adv_mid_s0_merged.pt,stack_s0=raw/runs/stack_s0_merged.pt,base=base
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import DEV, Rows, load_tail, px_rows, splits, x_ids
meta = json.load(open(HERE / "raw/cache/base/meta.json"))
dev = splits(meta["prompt_ids"])["dev"]
smeta = json.load(open(HERE / "raw/cache/sys/meta.json"))
srow = {p: i for i, p in enumerate(smeta["prompt_ids"])}
drows = [srow[meta["prompt_ids"][i]] for i in dev]
cfg, tail, w_out = load_tail(2)
base = {k: v.clone() for k, v in tail.state_dict().items() if "rotary" not in k}
xs = torch.tensor(x_ids(), device=DEV)
lo = lambda p: float(np.log(p / (1 - p)))
ZE = Rows(HERE / "raw/cache/base/eager_rec_tail.npy")
S = [Rows(HERE / f"raw/cache/sys/eager_train_t{i}_tail.npy") for i in range(len(smeta["texts"]))]
out = json.load(open(HERE / "data/train_sys_lift.json")) if (HERE / "data/train_sys_lift.json").exists() else {}
for spec in sys.argv[1].split(","):
    lab, path = spec.split("=", 1)
    sd = base if path == "base" else {k: v for k, v in torch.load(HERE / path, map_location="cpu").items() if "rotary" not in k}
    missing, unexpected = tail.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in m for m in missing)
    p0 = float(px_rows(tail, w_out, ZE, dev, xs).mean())
    ps = [float(px_rows(tail, w_out, s, drows, xs).mean()) for s in S]
    lifts = [lo(p) - lo(p0) for p in ps]
    out[lab] = {"p_none": p0, "p_text": ps, "lifts": lifts, "max_lift": max(lifts), "mean_lift": float(np.mean(lifts)),
                "prompts": "dev 64", "inputs": "cached eager block-26 input (cache_sys.py)"}
    print(lab, "p_none", round(p0, 4), "max lift", round(max(lifts), 3), "mean", round(float(np.mean(lifts)), 3), flush=True)
json.dump(out, open(HERE / "data/train_sys_lift.json", "w"), indent=1)
