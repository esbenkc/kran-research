"""Figure 1 example check (Owain): the most likely first answer token (top-3) of each headline copy on chosen test
prompts, for the cue-on input, the cue-off input and the status arms of run_patch.py (V-only, K-only, K+V at tokens
0-2, all 32 query heads, query positions >= 3), built with the project's own patch engine (patch.py). Light: the
2-block tail on a few prompts, CPU.

    NC_DEV=cpu env/.venv/bin/python swarm-oracle-1101-sink-value/figure1/top1.py --prec bf16 --idx 74,76
Writes swarm-oracle-1101-sink-value/figure1/top1_<prec>.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
HERE = Path(__file__).resolve().parent
PROJ = next(p for p in [HERE.parents[1] / "kran-research/projects/sink-value", HERE.parent]
            if (p / "patch.py").exists())
sys.path.insert(0, str(PROJ))
import torch  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prec", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--idx", required=True)
a = ap.parse_args()
torch.set_num_threads(2)
T = C.T
IDX = [int(i) for i in a.idx.split(",")]
ALL16, F3 = list(range(16)), [0, 1, 2]
ARMS = {"G0_KV": P.arm(ALL16, ALL16, kpos=F3, vpos=F3), "V": P.arm(ALL16, ALL16, vpos=F3),
        "K": P.arm(ALL16, ALL16, kpos=F3)}

cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
if a.prec == "fp32":
    w_out = w_out.float()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(T.MODEL)
xs = T.x_ids()


def h_of(u16):
    h = T.bf16(u16).to(T.DEV)
    return h.float() if a.prec == "fp32" else h


@torch.no_grad()
def run(h, spec=None, donor=None):
    if spec is None:
        P.CTRL.update(mode="none", spec=None, donor=None)
    else:
        P.CTRL.update(mode="patch", spec=spec, donor=donor)
    p = torch.softmax(tail(h, w_out).float(), -1)
    P.CTRL.update(mode="none", spec=None, donor=None)
    top = p.topk(3, -1)
    return [{"px": round(float(p[j, xs].sum()), 6),
             "top": [(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values[j], top.indices[j])]}
            for j in range(h.shape[0])]


PROMPTS = {r["id"]: r for r in map(json.loads, open(PROJ / "src1025/raw/prompts.jsonl"))}
out = {"prec": a.prec, "idx": IDX, "copies": {}}
for name in C.HEADLINE:
    sw, path = C.COPIES[name]
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float() if a.prec == "fp32" else tail.to(torch.bfloat16)
    X = C.load_inputs(sw)
    on, off = h_of(X["on"][IDX]), h_of(X["off"][IDX])
    with torch.no_grad():
        P.CTRL.update(mode="capture")
        tail(off, w_out)
        donor = {i: P.CAP[i] for i in range(2)}
        P.CTRL.update(mode="none")
    meta = json.load(open(C.SWITCH[sw]["meta"]))
    te = C.test_idx(C.SWITCH[sw]["meta"])
    res = {"prompt_ids": [meta["prompt_ids"][te[i]] for i in IDX], "on": run(on), "off": run(off)}
    for k, spec in ARMS.items():
        res[k] = run(on, spec, donor)
    out["copies"][name] = res
    print(name, res["prompt_ids"], {k: [r["top"][0] for r in res[k]] for k in ("on", "off", *ARMS)}, flush=True)
out["questions"] = {pid: PROMPTS[pid]["question"] for pid in out["copies"]["grader_s0"]["prompt_ids"]}
(HERE / f"top1_{a.prec}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", HERE / f"top1_{a.prec}.json")
