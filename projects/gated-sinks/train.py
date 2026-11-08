"""Train one tiny LM on the planted-latent data.

uv run python projects/gated-sinks/train.py --attn gated --seed 0 [--target 3.95] [--max-steps 8000]

Stops at the first held-out eval (every --eval-every steps) whose loss is <= --target, so models of
different attention variants end at matched held-out loss. Without --target it runs --max-steps.
Writes runs/<name>/{weights.safetensors, log.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from gen import GenCfg, World
from model import ModelCfg, TinyLM

HERE = Path(__file__).parent
EVAL_SEED = 999_000
EVAL_N = 512


def loss_fn(model, toks):
    logits = model(toks[:, :-1])
    return nn.losses.cross_entropy(logits, toks[:, 1:], reduction="mean")


def eval_loss(model, toks_eval, bs=128):
    tot = 0.0
    for i in range(0, len(toks_eval), bs):
        t = mx.array(toks_eval[i:i + bs])
        tot += loss_fn(model, t).item() * len(t)
    return tot / len(toks_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="softmax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-every-eval", action="store_true",
                    help="keep ckpt_<step>.safetensors at every eval (for matched-loss picking)")
    a = ap.parse_args()

    g = GenCfg()
    world = World(g)
    mc = ModelCfg(n_tokens=g.n_tokens, d=a.d, n_layers=a.layers, seq_len=g.seq_len, d_mlp=4 * a.d,
                  n_heads=a.heads,
                  attn=a.attn, gate_bias_init=a.gate_bias)
    name = a.name or f"{a.attn}_s{a.seed}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    mx.random.seed(a.seed)
    model = TinyLM(mc)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))

    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, a.lr, a.warmup), (lambda _s: a.lr)], [a.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=a.wd, betas=[0.9, 0.95])
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    def _step(toks):
        loss, grads = lg(model, toks)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    step = mx.compile(_step, inputs=state, outputs=state)

    toks_eval, _, _ = world.sample(EVAL_N, EVAL_SEED)
    hist, t0, stop_step, reached = [], time.time(), None, False
    for it in range(1, a.max_steps + 1):
        toks, _, _ = world.sample(a.batch, a.seed * 1_000_003 + it)
        loss = step(mx.array(toks))
        mx.eval(state)
        if it % a.eval_every == 0 or it == a.max_steps:
            el = eval_loss(model, toks_eval)
            hist.append({"step": it, "train": float(loss.item()), "eval": el,
                         "sec": round(time.time() - t0, 1)})
            print(json.dumps(hist[-1]), flush=True)
            if a.save_every_eval:
                model.save_weights(str(out / f"ckpt_{it}.safetensors"))
            if a.target is not None and el <= a.target:
                stop_step, reached = it, True
                break
    stop_step = stop_step or it
    model.save_weights(str(out / "weights.safetensors"))
    log = {"name": name, "attn": a.attn, "seed": a.seed, "args": vars(a), "gen": g.dict(),
           "model": mc.dict(), "n_params": int(n_params), "stop_step": stop_step,
           "reached_target": reached, "final_eval": hist[-1]["eval"], "history": hist,
           "eval_seed": EVAL_SEED, "eval_n": EVAL_N, "sec": round(time.time() - t0, 1)}
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print("done", name, stop_step, hist[-1]["eval"], f"{n_params/1e6:.2f}M params")


if __name__ == "__main__":
    main()
