"""Head-ablation cell (Kasper #17276): for each head of the sink block, its first-token mass and the
drop in readout R^2 of the latents at the SAE layer (rows P) when that head's output is zeroed.

uv run python ablate.py RUN CKPT [--layer 2]  -> results/ablate_<RUN>_<CKPT>.json
Zeroing is done on the head's slice of the attention output before W_o (and before nothing else),
by wrapping Attention.__call__ in this script only; model.py is untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

import model as M
from gen import GenCfg, World
from measure import TEST_SEED, TRAIN_SEED, collect, r2, ridge_fit

HERE = Path(__file__).parent
_orig = M.Attention.__call__
ZERO = {"block": None, "head": None}


def patched(self, x, mask, want_attn=False):
    if ZERO["head"] is None or self is not ZERO["block"]:
        return _orig(self, x, mask, want_attn)
    o = self.o
    dh = self.dh
    h = ZERO["head"]
    keep = mx.array([0.0 if h * dh <= i < (h + 1) * dh else 1.0 for i in range(self.c.d)])

    class _Mask:
        def __call__(_, y):
            return o(y * keep)
    self.o = _Mask()
    try:
        return _orig(self, x, mask, want_attn)
    finally:
        self.o = o


M.Attention.__call__ = patched


def readout(model, world, layer, n_train=1024, n_test=1000):
    tr = collect(model, world, n_train, TRAIN_SEED, layer)
    te = collect(model, world, n_test, TEST_SEED, layer)
    Htr, Hte = tr["H"][tr["P"]], te["H"][te["P"]]
    Ztr = np.repeat(tr["z"], tr["P"].sum(1), 0)
    Zte = np.repeat(te["z"], te["P"].sum(1), 0)
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    return float(r2(ridge_fit((Htr - mu) / sd, Ztr)((Hte - mu) / sd), Zte).mean()), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ckpt", type=int)
    ap.add_argument("--layer", type=int, default=2)
    a = ap.parse_args()
    log = json.loads((HERE / "runs" / a.run / "log.json").read_text())
    g = GenCfg(**log["gen"])
    w = World(g)
    model = M.TinyLM(M.ModelCfg(**log["model"]))
    model.load_weights(str(HERE / "runs" / a.run / f"ckpt_{a.ckpt}.safetensors"))
    blk = model.blocks[a.layer - 1].attn
    base, te = readout(model, w, a.layer)
    sink = te["sink_heads"][a.layer - 1]
    rows = []
    for h in range(model.c.n_heads):
        ZERO.update(block=blk, head=h)
        r, _ = readout(model, w, a.layer)
        ZERO.update(block=None, head=None)
        rows.append({"head": h, "first_token_mass": float(sink[h]), "readout_r2_zeroed": r,
                     "readout_drop": base - r})
        print(rows[-1], flush=True)
    out = {"run": a.run, "ckpt": a.ckpt, "layer": a.layer, "readout_r2": base, "heads": rows}
    p = HERE / "results" / f"ablate_{a.run}_{a.ckpt}.json"
    p.write_text(json.dumps(out, indent=1))
    print("base readout", base)


if __name__ == "__main__":
    main()
