"""Figure 1 example (Owain): the UNTRAINED tail (Qwen3-0.6B as released, blocks 26-27) on chosen test prompts,
cue-on input, PREREG s2 arm Z with the project's engine (patch.py mode "zero": pairs removed from q.k in both
blocks in one pass). Records head 27.8's answer-position attention on tokens 0, 1, 2 with no pairs removed, with
pairs 61-63 removed, and with each extra 3-set given on the command line. Light: a few prompts, CPU, fp32.

    NC_DEV=cpu ../../env/.venv/bin/python base_sink.py --idx 74 --sets n1=41,46,51
Writes base_sink.json next to this file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1] / "kran-research/projects/switch-slow-band"
sys.path.insert(0, str(PROJ))
import torch  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--idx", required=True)
ap.add_argument("--sets", nargs="*", default=[])
a = ap.parse_args()
torch.set_num_threads(2)
T = C.T
IDX = [int(i) for i in a.idx.split(",")]
SETS = {"none": None, "band": [61, 62, 63]}
for s in a.sets:
    k, v = s.split("=")
    SETS[k] = [int(x) for x in v.split(",")]

cfg, tail, w_out = T.load_tail()          # released weights, no copy loaded
P.install(tail)
tail.float()
w_out = w_out.float()
X = C.load_inputs("monitor")
on = T.bf16(X["on"][IDX]).to(T.DEV).float()
meta = json.load(open(C.SWITCH["monitor"]["meta"]))
te = C.test_idx(C.SWITCH["monitor"]["meta"])
out = {"idx": IDX, "prompt_ids": [meta["prompt_ids"][te[i]] for i in IDX], "head": "27.8", "sets": SETS, "att": {}}
for k, S in SETS.items():
    P.CTRL.update(mode="zero", zset=[S, S], zrec=[[], []])
    with torch.no_grad():
        tail(on, w_out)
    P.CTRL.update(mode="none")
    ans = P.CTRL["zrec"][1][0]["ans"]            # [B, 16, 3], block 27, answer position
    out["att"][k] = [[round(float(x), 6) for x in ans[j, 8]] for j in range(len(IDX))]
    print(k, out["att"][k], flush=True)
(HERE / "base_sink.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / "base_sink.json")
