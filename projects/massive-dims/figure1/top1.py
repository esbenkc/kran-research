"""Figure 1 example check (Owain): first answer token (top-1) of each headline copy on chosen test prompts,
for the condition-on input, the condition-off input, and the (a) massive / (b) non-massive / (d) all patches at
tokens 0-2 (donor off -> recipient on), exactly as massive.py builds them. Light: one copy's tail at a time, CPU.

    NC_DEV=cpu env/.venv/bin/python swarm-oracle-1031-massive-dims/figure1/top1.py --prec bf16 --idx 74,76,124
Writes swarm-oracle-1031-massive-dims/figure1/top1_<prec>.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
HERE = Path(__file__).resolve().parent
PROJ = next(p for p in [HERE.parents[1] / "kran-research/projects/massive-dims", HERE.parent] if (p / "common.py").exists())
sys.path.insert(0, str(PROJ))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prec", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--idx", required=True)
a = ap.parse_args()
torch.set_num_threads(2)
T = C.T
FROZEN = json.load(open(PROJ / "data/frozen.json"))
M = FROZEN["massive_set"]
assert M == [1, 13, 35]
NONM = [d for d in range(1024) if d not in M]
IDX = [int(i) for i in a.idx.split(",")]


def copy_dims(rec, don, dims, toks=(0, 1, 2)):
    out = rec.copy()
    for t in toks:
        out[:, t, dims] = don[:, t, dims]
    return out


cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
xs = T.x_ids()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if a.prec == "fp32":
    w_out = w_out.float()

PROMPTS = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}
out = {"prec": a.prec, "idx": IDX, "copies": {}}
for name in C.HEADLINE:
    sw, path = C.COPIES[name]
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float() if a.prec == "fp32" else tail.to(torch.bfloat16)
    X = C.load_inputs(sw)
    on, off = X["on"][IDX], X["off"][IDX]
    meta = json.load(open(C.SWITCH[sw]["meta"]))
    te = C.test_idx(C.SWITCH[sw]["meta"])
    pids = [meta["prompt_ids"][te[i]] for i in IDX]
    rows = {"on": on, "off": off, "a_massive": copy_dims(on, off, M), "b_nonmassive": copy_dims(on, off, NONM),
            "d_all": copy_dims(on, off, list(range(1024)))}
    res = {"prompt_ids": pids}
    with torch.no_grad():
        for k, u in rows.items():
            h = T.bf16(u).to(T.DEV)
            if a.prec == "fp32":
                h = h.float()
            p = torch.softmax(tail(h, w_out).float(), -1)
            top = p.topk(3, -1)
            res[k] = [{"px": round(float(p[j, xs].sum()), 6),
                       "top": [(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values[j], top.indices[j])]}
                      for j in range(len(IDX))]
    out["copies"][name] = res
    print(name, pids, {k: [r["top"][0] for r in res[k]] for k in rows}, flush=True)
out["questions"] = {pid: PROMPTS[pid]["question"] for pid in out["copies"]["grader_s0"]["prompt_ids"]}
(HERE / f"top1_{a.prec}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", HERE / f"top1_{a.prec}.json")
