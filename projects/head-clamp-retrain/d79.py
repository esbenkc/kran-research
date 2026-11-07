"""Assumption-free bound for the ALL-32 swap arm (PREREG v2): with every block-26/27 head of the probed run given the
unprobed slice, the answer logits depend on the probe only through the block-26 input at the answer position (79).
So strict <= d79 = share of TEST prompts whose cached block-26 input at position 79 differs in any bit between probe
off and on. Also reported for all splits and for every position (where the leak sits).

    $PY d79.py   -> data/d79.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
cdir = HERE / "raw" / "step2" / "Qwen3-1.7B" / "compile_C"
pid = json.load(open(cdir / "meta.json"))["prompt_ids"]
split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
z0 = np.load(cdir / "none_rec_tail.npy", mmap_mode="r")
z1 = np.load(cdir / "probe_rec_tail.npy", mmap_mode="r")
out = {"by_split": {}}
diff_any = np.zeros((len(pid), z0.shape[1]), dtype=bool)
nbits = np.zeros((len(pid), z0.shape[1]), dtype=np.int64)
for i in range(len(pid)):
    d = z0[i] != z1[i]
    diff_any[i] = d.any(-1)
    nbits[i] = d.sum(-1)
for s in ("train", "dev", "test"):
    ii = [i for i, p in enumerate(pid) if split_of[p] == s]
    out["by_split"][s] = {"n": len(ii), "d79": float(diff_any[ii, 79].mean()),
                          "mean_changed_dims_pos79": float(nbits[ii, 79].mean()),
                          "share_prompts_any_pos_differs": float(diff_any[ii].any(1).mean())}
out["share_differs_by_pos_test"] = diff_any[[i for i, p in enumerate(pid) if split_of[p] == "test"]].mean(0).round(4).tolist()
out["d79"] = out["by_split"]["test"]["d79"]
(HERE / "data" / "d79.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["by_split"]), "d79(test) =", out["d79"])
