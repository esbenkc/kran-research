"""One real example for Figure 1 (Owain): the same held-out sequence through a standard-attention and a
gated-attention model.

uv run python example.py RUN_SOFTMAX:CKPT RUN_GATED:CKPT [--tag L2_m256_k8] [--seq 0]

Writes results/example.json with, per model: block-2 attention row of one trigger query (head mean and
per head), the sink mass of that row, one planted latent that is on in the sequence, and the activation
of that latent's best SAE feature (SAE seed 0) at every position, plus that latent's recovery score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from gen import GenCfg, World
from measure import TEST_SEED
from model import ModelCfg, TinyLM

HERE = Path(__file__).parent


def load(spec):
    run, ck = spec.split(":")
    rd = HERE / "runs" / run
    log = json.loads((rd / "log.json").read_text())
    model = TinyLM(ModelCfg(**log["model"]))
    model.load_weights(str(rd / f"ckpt_{ck}.safetensors"))
    return run, int(ck), log, model, rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("soft")
    ap.add_argument("gated")
    ap.add_argument("--tag", default="L2_m256_k8")
    ap.add_argument("--seq", type=int, default=0)
    a = ap.parse_args()
    out = {"models": []}
    for spec in (a.soft, a.gated):
        run, ck, log, model, rd = load(spec)
        g = GenCfg(**log["gen"])
        toks, z, fresh, score = World(g).sample(a.seq + 1, TEST_SEED)
        toks, z, score = toks[a.seq:a.seq + 1], z[a.seq], score[a.seq]
        q = int(np.nonzero(score[90:])[0][0] + 90) if score[90:].any() else 100
        _, resid, info = model.trace(mx.array(toks))
        att = np.array(info[1]["attn"][0, :, q, :q + 1])            # block 2, [H, q+1]
        h = np.array(resid[2][0])                                   # [T, d]
        tag = a.tag + f"_c{ck}"
        meas = json.loads((rd / f"measure_{tag}.json").read_text())
        acts = np.load(rd / f"acts_{tag}.npz")
        x = (h - acts["mu"]) / float(acts["scale"])
        pre = np.maximum((x - acts["b_dec_0"]) @ acts["W_enc_0"] + acts["b_enc_0"], 0)
        thr = np.sort(pre, 1)[:, -meas["k"]][:, None]
        f = np.where(pre >= thr, pre, 0)
        j = int(np.nonzero(z)[0][0])
        feat = meas["saes"][0]["best_feature"][j]
        out["models"].append({
            "run": run, "ckpt": ck, "arm": log["attn"], "query_pos": q,
            "tokens": toks[0].tolist(), "score_positions": np.nonzero(score)[0].tolist(),
            "attn_row_mean": att.mean(0).tolist(), "attn_row_heads": att.tolist(),
            "row_sink_mass": float(att.mean(0)[0]), "model_sink_mass": meas["sink_mass"],
            "latents_on": np.nonzero(z)[0].tolist(), "latent": j, "best_feature": feat,
            "feature_act": f[:, feat].tolist(),
            "latent_recovery": meas["saes"][0]["mcc_per_latent"][j],
            "model_recovery": meas["recovery"]})
    p = HERE / "results" / "example.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
