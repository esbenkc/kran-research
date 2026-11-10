"""x (PREREG s2) for the variant directions (u seed 1111, -u), light: CPU, only block 26's input-norm weight is loaded.
Same formula as scan.py: x = L2 over rows 0-2 of N(h_on) - N(h), h_on as fed (fp32 add, bf16 cast once).
Also recomputes x for u seed 1110 so it can be checked against data/scan.json.

    env/.venv/bin/python xvar.py   -> data/scan_u1111.json, data/scan_neg.json, data/scan_u1110_check.json
"""
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = Path(__file__).parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from rl import CACHE, cue_rows  # noqa: E402

EPS = 1e-6


def norm(h, w):
    h = h.float()
    return w.float() * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS))


def main():
    path = Path(snapshot_download("Qwen/Qwen3-0.6B"))
    w = None
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as sf:
            if "model.layers.26.input_layernorm.weight" in sf.keys():
                w = sf.get_tensor("model.layers.26.input_layernorm.weight")
    h0_u16 = np.ascontiguousarray(np.load(CACHE / "eager_rec_tail.npy", mmap_mode="r")[0, :3])
    h0 = torch.from_numpy(h0_u16.view(np.int16)).view(torch.bfloat16)
    n0 = norm(h0, w)
    alphas = [0.03125 / 2 ** k for k in range(6)]
    for tag, uf in (("u1111", "data/cue_u_1111.npy"), ("neg", "data/cue_u_neg.npy"), ("u1110_check", "data/cue_u.npy"),
                    ("ext", "data/cue_u.npy")):
        if tag == "ext":  # exploratory k = 6..9 (Tove #19413, Kasper #19415)
            alphas = [0.03125 / 2 ** k for k in range(6, 10)]
        out = {"u_file": uf, "cues": {}}
        for al in alphas:
            on = torch.from_numpy(cue_rows("rand", al, uf, h0_u16).view(np.int16)).view(torch.bfloat16)
            dn = norm(on, w) - n0
            d = on.float() - h0.float()
            out["cues"][str(al)] = {"alpha": al, "x": float(dn.norm()), "raw_l2": float(d.norm()),
                                    "frac_elems_unchanged_rows": [float((d[p] == 0).float().mean()) for p in range(3)],
                                    "x_if_no_rounding": None}
        json.dump(out, open(HERE / "data" / f"scan_{tag}.json", "w"), indent=1)
        print(tag, {k: round(v["x"], 4) for k, v in out["cues"].items()})


if __name__ == "__main__":
    main()
