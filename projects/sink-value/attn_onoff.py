"""Reading row (Tove #11235): per trained copy, per query head of blocks 26-27, the attention weight on tokens 0, 1, 2
(answer position, and mean over query positions 3-33), on the condition-on vs condition-off input. Shows how far a
head 'leaves the sink' when the cue changes. CPU fp32, own manual softmax (patch.py mode 'attn').

    env/.venv/bin/python kran-research/projects/sink-value/attn_onoff.py
"""
import json
import os

os.environ["NC_DEV"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

T = C.T
torch.set_num_threads(6)
cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
w_out = w_out.float()


def attn(u16):
    P.CTRL.update(mode="attn", attn=[[], []])
    with torch.no_grad():
        for s in range(0, len(u16), 25):
            tail(T.bf16(u16[s:s + 25]).float(), w_out)
    P.CTRL["mode"] = "none"
    return [torch.cat(P.CTRL["attn"][i]).double().numpy() for i in range(2)]


out = {}
for name, (sw, path) in C.COPIES.items():
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float()
    X = C.load_inputs(sw)
    res = {}
    for cond in ("on", "off"):
        A = attn(X[cond])
        for i, blk in enumerate((26, 27)):
            a = A[i]
            res.setdefault(str(blk), {})[cond] = {
                "answer_t0": a[:, :, -1, 0].mean(0).round(5).tolist(),
                "answer_t1": a[:, :, -1, 1].mean(0).round(5).tolist(),
                "answer_t2": a[:, :, -1, 2].mean(0).round(5).tolist(),
                "later_t0": a[:, :, 3:, 0].mean((0, 2)).round(5).tolist(),
                "later_t12": a[:, :, 3:, 1:3].sum(-1).mean((0, 2)).round(5).tolist(),
            }
    out[name] = res
    for blk in ("26", "27"):
        on, off = res[blk]["on"], res[blk]["off"]
        d0 = np.array(off["answer_t0"]) - np.array(on["answer_t0"])
        d2 = np.array(off["answer_t2"]) - np.array(on["answer_t2"])
        top = np.argsort(-np.abs(d0) - np.abs(d2))[:3]
        print(name, blk, "largest answer-token shifts (off - on):",
              [(int(h), round(float(d0[h]), 4), round(float(d2[h]), 4), round(on["answer_t0"][h], 3), round(on["answer_t2"][h], 4)) for h in top], flush=True)
json.dump(out, open(C.HERE / "data/attn_onoff.json", "w"), indent=1)
print("wrote data/attn_onoff.json")
