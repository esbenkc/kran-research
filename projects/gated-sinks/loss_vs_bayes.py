"""Held-out loss on fresh evidence tokens (positions >= 32) vs the exact Bayes loss and a no-latent
baseline (base logits only). Shows how much of the latent-driven loss gain a model has learned.

uv run python loss_vs_bayes.py RUN:CKPT [RUN:CKPT ...]  -> results/loss_vs_bayes.json (appends)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from bayes import Bayes
from gen import GenCfg, World
from model import ModelCfg, TinyLM

HERE = Path(__file__).parent


def main(specs):
    p = HERE / "results" / "loss_vs_bayes.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for spec in specs:
        run, ck = spec.split(":")
        log = json.loads((HERE / "runs" / run / "log.json").read_text())
        g = GenCfg(**log["gen"])
        w = World(g)
        m = TinyLM(ModelCfg(**log["model"]))
        m.load_weights(str(HERE / "runs" / run / f"ckpt_{ck}.safetensors"))
        t, z, f, s = w.sample(512, 778000)
        lp = []
        for i in range(0, 512, 128):
            lg = m(mx.array(t[i:i + 128, :-1]))
            l = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            lp.append(np.array(mx.take_along_axis(l, mx.array(t[i:i + 128, 1:])[..., None],
                                                  axis=-1)[..., 0]))
        lp = -np.concatenate(lp)
        fr = f[:, 1:].copy()
        fr[:, :31] = False                      # targets x_32..x_127
        _, bl = Bayes(w).run(t, f, bs=16)
        base = w.base - np.log(np.exp(w.base).sum())
        ub = -base[np.where(f[:, 1:], t[:, 1:], 0)]
        r = {"model": float(lp[fr].mean()), "bayes": float(np.nanmean(bl[fr])),
             "no_latent": float(ub[fr].mean())}
        r["share_of_gain"] = (r["no_latent"] - r["model"]) / (r["no_latent"] - r["bayes"])
        out[spec] = r
        print(spec, {k: round(v, 3) for k, v in r.items()})
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
