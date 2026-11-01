"""R9 sink diagnostics (PREREG s2, descriptive; Iris #11190, Fesser et al. 2606.08105; Guo 2410.13835).
Per query head of blocks 26-27, on the untrained base and on each trained copy, condition-on input, CPU fp32:
  - value-norm ratio: |v_0| / mean_{j=3..33} |v_j| (the head's KV head),
  - output-norm ratio: |W_O,h v_0| / mean_{j=3..33} |W_O,h v_j| (what the head would write if it copied only token j),
  - effective rank of the head's outputs over query positions 3-33 (exp of the entropy of the normalised singular
    values), averaged over the 200 prompts.

    env/.venv/bin/python kran-research/projects/sink-value/r9_diag.py
"""
import json
import os

os.environ["NC_DEV"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers.integrations.sdpa_attention import repeat_kv  # noqa: E402

import common as C  # noqa: E402
import patch as P  # noqa: E402

T = C.T
torch.set_num_threads(4)
cfg, tail, w_out = T.load_tail()
base_sd = {k: v.clone() for k, v in tail.state_dict().items()}
P.install(tail)
w_out = w_out.float()
OUTS = {}


def ohook(i):
    def pre(mod, args):
        OUTS[i] = args[0].detach()
    return pre


hooks = [tail.layers[i].self_attn.o_proj.register_forward_pre_hook(ohook(i)) for i in range(2)]


def diag(u16):
    acc = {k: np.zeros((2, 16)) for k in ("vratio", "oratio", "erank")}
    n = 0
    with torch.no_grad():
        for s in range(0, len(u16), 25):
            P.CTRL["mode"] = "capture"
            tail(T.bf16(u16[s:s + 25]).float(), w_out)
            P.CTRL["mode"] = "none"
            b = min(25, len(u16) - s)
            for i in range(2):
                at = tail.layers[i].self_attn
                v = repeat_kv(P.CAP[i][1], at.num_key_value_groups).float()         # [B, 16, T, 128]
                vn = v.norm(dim=-1)
                acc["vratio"][i] += (vn[:, :, 0] / vn[:, :, 3:].mean(-1)).sum(0).numpy()
                Wo = at.o_proj.weight.float().view(-1, 16, 128)                     # [1024, 16, 128]
                ov = torch.einsum("bhtd,ehd->bhte", v, Wo).norm(dim=-1)
                acc["oratio"][i] += (ov[:, :, 0] / ov[:, :, 3:].mean(-1)).sum(0).numpy()
                o = OUTS[i].float().view(b, -1, 16, 128)[:, 3:].permute(0, 2, 1, 3)   # [B, 16, 31, 128]
                sv = torch.linalg.svdvals(o)
                p = sv / sv.sum(-1, keepdim=True)
                acc["erank"][i] += torch.exp(-(p * torch.log(p + 1e-12)).sum(-1)).sum(0).numpy()
            n += b
    return {k: {str(26 + i): np.round(v[i] / n, 4).tolist() for i in range(2)} for k, v in acc.items()}


out = {}
for sw in ("grader", "monitor"):
    X = C.load_inputs(sw)
    tail.load_state_dict(base_sd)
    tail.float()
    out[f"base_{sw}"] = diag(X["on"])
for name, (sw, path) in C.COPIES.items():
    X = C.load_inputs(sw)
    tail.load_state_dict(base_sd)
    tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    tail.float()
    out[name] = diag(X["on"])
for k, v in out.items():
    print(k, "27.8/27.9 vratio", v["vratio"]["27"][8], v["vratio"]["27"][9], "oratio", v["oratio"]["27"][8],
          v["oratio"]["27"][9], "erank", v["erank"]["27"][8], v["erank"]["27"][9],
          "| median vratio all", round(float(np.median(v["vratio"]["26"] + v["vratio"]["27"])), 3), flush=True)
json.dump(out, open(C.HERE / "data/r9_diag.json", "w"), indent=1)
print("wrote data/r9_diag.json")
