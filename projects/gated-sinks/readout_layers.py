"""Where does each model keep the latents? Ridge readout R^2 of z at h_0..h_4 on rows P
(Tove #17309, Felix #17311). Ridge only, no SAE.

uv run python readout_layers.py RUN:CKPT [RUN:CKPT ...]  -> results/readout_layers.json (merged)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from gen import GenCfg, World
from measure import TEST_SEED, TRAIN_SEED, r2, ridge_fit
from model import ModelCfg, TinyLM

HERE = Path(__file__).parent


def acts(model, world, n, seed):
    toks, z, fresh, score = world.sample(n, seed)
    P = score.copy()
    P[:, :32] = False
    out = [[] for _ in range(model.c.n_layers + 1)]
    for i in range(0, n, 128):
        _, resid, _ = model.trace(mx.array(toks[i:i + 128]))
        mx.eval(*resid)
        for L, h in enumerate(resid):
            out[L].append(np.array(h)[P[i:i + 128]])
    return [np.concatenate(o) for o in out], np.repeat(z, P.sum(1), 0)


def main(specs):
    p = HERE / "results" / "readout_layers.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    for spec in specs:
        run, ck = spec.split(":")
        log = json.loads((HERE / "runs" / run / "log.json").read_text())
        w = World(GenCfg(**log["gen"]))
        m = TinyLM(ModelCfg(**log["model"]))
        m.load_weights(str(HERE / "runs" / run / f"ckpt_{ck}.safetensors"))
        Htr, Ztr = acts(m, w, 2048, TRAIN_SEED)
        Hte, Zte = acts(m, w, 1000, TEST_SEED)
        row = []
        for L in range(len(Htr)):
            mu, sd = Htr[L].mean(0), Htr[L].std(0) + 1e-6
            row.append(float(r2(ridge_fit((Htr[L] - mu) / sd, Ztr)((Hte[L] - mu) / sd), Zte).mean()))
        res[spec] = {"readout_r2_by_layer": row}
        print(spec, [round(x, 3) for x in row], flush=True)
    p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
