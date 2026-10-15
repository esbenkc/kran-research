"""Appendix experiments for the 'If you're curious' section (post-hoc, run after the post went live).

Same pipeline, checkpoints and attack code as exp.py; raw logs go to results/appendix/raw/ so the
published figure and data/runs.jsonl stay untouched.

  bigc    Q1: push the gauge far past c = 16. Seed 0 lr search for SGD and AdamW at each new log2c
          on the same half-decade grid as exp.py: start at the attacker's c = 1 best lr and one
          half-decade either side, then keep stepping toward the best value until it is interior;
          if nothing recovers, step down while the lowest lr diverges or up while the highest is
          too slow. Then other seeds / data orders at the seed-0 best lr; rebalance at its c = 1 lr
          (one run per c, identical by construction).
  orders  Q2: more data orders / seeds at c = 1 and c = 16 for SGD and AdamW at the seed-0 best lr
          of the main grid (exp.best_lrs on results/raw). New seeds build their own unlearned
          checkpoint with exp.prepare.
"""
from __future__ import annotations

import argparse
import json
import math

import exp

MAIN_RAW = exp.RAW
APP_RAW = exp.HERE / "results" / "appendix" / "raw"
exp.RAW = APP_RAW  # prepare(), run_one(), grid_outcomes() and best_lrs() now use the appendix dir

LR_C1 = {"sgd": 3e-4, "adamw": 3e-5}  # seed-0 best lr at c = 1 in the main grid
MAX_TRIES = 10


def last_status(seed, attacker, k, lr):
    p = APP_RAW / f"seed{seed}" / f"{attacker}_log2c{k}_lr{lr:g}.jsonl"
    row = json.loads(p.read_text().splitlines()[-1])
    if not math.isfinite(row["loss"]):
        return "diverged"
    return "recovered" if row["acc_A"] >= exp.THRESH else "censored"


def grid(seed, attacker, k, model, state, batches):
    out = APP_RAW / f"seed{seed}"
    run = lambda lr: exp.run_one(model, state, seed, attacker, k, lr, batches, out)  # noqa: E731
    lr0 = LR_C1[attacker]
    for lr in (lr0, exp.half_decade(lr0, up=False), exp.half_decade(lr0, up=True)):
        run(lr)
    for _ in range(MAX_TRIES):
        g = exp.grid_outcomes(seed, attacker, k)
        ok = {lr: s for lr, s in g.items() if s is not None}
        lo, hi = min(g), max(g)
        if ok:
            b = min(ok, key=lambda lr: (ok[lr], lr))
            if b == lo:
                new = exp.half_decade(lo, up=False)
            elif b == hi:
                new = exp.half_decade(hi, up=True)
            else:
                return
        elif last_status(seed, attacker, k, lo) == "diverged":
            new = exp.half_decade(lo, up=False)
        elif last_status(seed, attacker, k, hi) == "censored":
            new = exp.half_decade(hi, up=True)
        else:
            return  # lower lrs too slow, higher lrs diverge: nothing recovers within the cap
        if new < 1e-8 or new > 1:
            return
        run(new)


def bigc(log2cs, phase, seeds, orders, attackers=("sgd", "adamw")):
    if phase in ("grid", "all"):
        state, batches = exp.prepare(0)
        model = exp.load_model()
        for k in log2cs:
            for attacker in attackers:
                grid(0, attacker, k, model, state, batches)
            exp.run_one(model, state, 0, "renorm", k, 3e-5, batches, APP_RAW / "seed0")
    if phase in ("reps", "all"):
        best = exp.best_lrs(0)
        for seed in seeds:
            state, batches = exp.prepare(seed)
            model = exp.load_model()
            for o in orders:
                for attacker in ("sgd", "adamw"):
                    for k in log2cs:
                        if (attacker, k) in best:
                            exp.run_one(model, state, seed, attacker, k, best[(attacker, k)], batches,
                                        APP_RAW / f"seed{seed}", tag=f"order{o}_" if o else "", order=o)


def orders(seeds, order_list):
    exp.RAW = MAIN_RAW
    best = exp.best_lrs(0)
    exp.RAW = APP_RAW
    for seed in seeds:
        state, batches = exp.prepare(seed)
        model = exp.load_model()
        for o in order_list:
            for attacker in ("sgd", "adamw"):
                for k in (0, 4):
                    exp.run_one(model, state, seed, attacker, k, best[(attacker, k)], batches,
                                APP_RAW / f"seed{seed}", tag=f"order{o}_" if o else "", order=o)


def stopcheck(seeds, log2cs):
    """Rerun main-order attacks at the seed-0 best lr with held-out and kept accuracy measured at the
    stop step (files stop_*.jsonl). Main c values use the main grid's best lr, new c the appendix's."""
    exp.RAW = MAIN_RAW
    best = exp.best_lrs(0)
    exp.RAW = APP_RAW
    best.update({key: lr for key, lr in exp.best_lrs(0).items() if abs(key[1]) > 4})
    for seed in seeds:
        state, batches = exp.prepare(seed)
        model = exp.load_model()
        for k in log2cs:
            for attacker in ("sgd", "adamw"):
                if (attacker, k) in best:
                    exp.run_one(model, state, seed, attacker, k, best[(attacker, k)], batches,
                                APP_RAW / f"seed{seed}", tag="stop_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bigc", "orders", "stopcheck"])
    ap.add_argument("--log2c", type=int, nargs="+", default=[6, 8, 10, 12, -8, -12])
    ap.add_argument("--phase", default="all", choices=["grid", "reps", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--orders", type=int, nargs="+", default=None)
    ap.add_argument("--attackers", nargs="+", default=["sgd", "adamw"])
    a = ap.parse_args()
    if a.mode == "bigc":
        bigc(a.log2c, a.phase, a.seeds, a.orders or [0], a.attackers)
    elif a.mode == "stopcheck":
        stopcheck(a.seeds, a.log2c)
    else:
        orders(a.seeds, a.orders or list(range(3, 13)))
