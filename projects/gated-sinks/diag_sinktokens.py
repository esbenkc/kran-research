"""Where do idle heads put their attention? Share of attention by key token type, per block and head,
at idle query positions (q >= 8, next token not evidence) and at scored queries.

uv run python diag_sinktokens.py RUN CKPT  -> results/diag_sinktokens_<RUN>_<CKPT>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from gen import GenCfg, World
from model import ModelCfg, TinyLM

HERE = Path(__file__).parent


def main(run, ck):
    log = json.loads((HERE / "runs" / run / "log.json").read_text())
    g = GenCfg(**log["gen"])
    m = TinyLM(ModelCfg(**log["model"]))
    m.load_weights(str(HERE / "runs" / run / f"ckpt_{ck}.safetensors"))
    t, z, f, s = World(g).sample(128, 778000)
    _, _, info = m.trace(mx.array(t))
    types = {"bos": t == g.bos,
             "trigger": (t >= g.trigger) & (t < g.trigger + g.n_trig),
             "copy_trigger": (t >= g.trigger + g.n_trig) & (t < g.bos),
             "filler": (t >= g.filler0) & (t < g.filler0 + g.n_filler),
             "content": t < g.vocab}
    idle = ~s
    idle[:, :8] = False
    out = []
    for L, d in enumerate(info):
        a = np.array(d["attn"])
        for h in range(a.shape[1]):
            A = a[:, h]
            row = {"block": L + 1, "head": h}
            for nm, msk in types.items():
                share = (A * msk[:, None, :]).sum(-1)
                row[f"idle_{nm}"] = float(share[idle].mean())
                row[f"query_{nm}"] = float(share[s].mean())
            out.append(row)
            print(row["block"], h, {k: round(v, 2) for k, v in row.items() if k.startswith("idle")})
    p = HERE / "results" / f"diag_sinktokens_{run}_{ck}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
