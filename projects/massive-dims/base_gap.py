"""Control: the UNTRAINED tail (base Qwen3-0.6B blocks 26-27, no LoRA) on the same condition-on / condition-off
inputs. If the untrained model already split P(X) by condition, the 'switch' would not be learned. Also runs the
(a) and (d) patches there for completeness. Writes data/base_gap.json.

    env/.venv/bin/python kran-research/projects/massive-dims/base_gap.py
"""
import json
import os

os.environ.setdefault("NC_DEV", "cpu")
import torch  # noqa: E402

import common as C  # noqa: E402

T = C.T
torch.set_num_threads(8)
M = json.load(open(C.HERE / "data/frozen.json"))["massive_set"]
cfg, tail, w_out = T.load_tail()
xs = torch.tensor(T.x_ids())
out = {}
for prec in ("bf16", "fp32"):
    tl, wo = (tail.float(), w_out.float()) if prec == "fp32" else (tail, w_out)

    @torch.no_grad()
    def px(u16):
        o = []
        for s in range(0, len(u16), 50):
            h = T.bf16(u16[s:s + 50])
            o.append(torch.softmax(tl(h.float() if prec == "fp32" else h, wo).float(), -1)[:, xs].sum(-1))
        return float(torch.cat(o).double().mean())

    for sw in C.SWITCH:
        X = C.load_inputs(sw)
        pa = X["on"].copy(); pa[:, :3, M] = X["off"][:, :3, M]
        pd = X["on"].copy(); pd[:, :3] = X["off"][:, :3]
        r = {"P_on": px(X["on"]), "P_off": px(X["off"]), "P_a_massive": px(pa), "P_d_all": px(pd)}
        r["gap_off_minus_on"] = r["P_off"] - r["P_on"]
        out[f"{sw}_{prec}"] = {k: round(v, 5) for k, v in r.items()}
        print(sw, prec, out[f"{sw}_{prec}"], flush=True)
json.dump(out, open(C.HERE / "data/base_gap.json", "w"), indent=1)
