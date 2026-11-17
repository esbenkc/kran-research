"""G0 (forward-only, from the cache; no training): is there a probe trace at the first tail block, and how readable is
it at the answer position?

1. d_ans = share of TEST prompts whose cached input to block FIRST at the answer position (the last token) differs in
   any bit between probe off and on (the parent's d79 at 1.7B = 1.000). Also per split and per position.
2. The cached inputs to block FIRST at the answer position and at position 71, probe off and on (fp32, exact copies of
   the bf16 cache values), go to OUT/cache34_pos.npz: the ALL-heads readout forecast (PREREG: Vera's
   skeptic/vera_readout.py) runs on this file, without the ~1 GB cache.

    HC_MODEL=Qwen/Qwen3-8B $PY g0.py      -> OUT/g0.json, OUT/cache34_pos.npz; exit 2 when d_ans < 0.5 (G0 null)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cfgm import CDIR, FIRST, MODEL, OUT  # noqa: E402



def f32(u16):
    return (np.ascontiguousarray(u16).astype(np.uint32) << 16).view(np.float32)


OUT.mkdir(parents=True, exist_ok=True)
pid = json.load(open(CDIR / "meta.json"))["prompt_ids"]
split_of = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(HERE / "raw" / "prompts.jsonl")}
z0 = np.load(CDIR / "none_rec_tail.npy", mmap_mode="r")
z1 = np.load(CDIR / "probe_rec_tail.npy", mmap_mode="r")
T = z0.shape[1]
idx = {s: [i for i, p in enumerate(pid) if split_of[p] == s] for s in ("train", "dev", "test")}
diff_any = np.zeros((len(pid), T), dtype=bool)
ndims = np.zeros((len(pid), T), dtype=np.int64)
for i in range(len(pid)):
    d = z0[i] != z1[i]
    diff_any[i] = d.any(-1)
    ndims[i] = d.sum(-1)
out = {"model": MODEL, "block": FIRST, "T": T, "answer_pos": T - 1, "by_split": {}}
for s, ii in idx.items():
    out["by_split"][s] = {"n": len(ii), "d_ans": float(diff_any[ii, T - 1].mean()),
                          "mean_changed_dims_ans": float(ndims[ii, T - 1].mean()),
                          "share_prompts_any_pos_differs": float(diff_any[ii].any(1).mean())}
out["share_differs_by_pos_test"] = diff_any[idx["test"]].mean(0).round(4).tolist()
out["d_ans"] = out["by_split"]["test"]["d_ans"]

# answer-position and position-71 rows (fp32, [P, H] each) for Vera's readout script and any rebuild without the cache
pos = {"ans": T - 1, "p71": 71}
arr = {f"{c}_{k}": f32(np.asarray(z[:, t])) for k, t in pos.items() for c, z in (("none", z0), ("probe", z1))}
np.savez_compressed(OUT / "cache34_pos.npz", **arr, prompt_ids=np.array(pid),
                    split=np.array([split_of[p] for p in pid]), positions=np.array([T - 1, 71]))
(OUT / "g0.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"d_ans": out["d_ans"], "by_split": out["by_split"]}))
sys.exit(2 if out["d_ans"] < 0.5 else 0)
