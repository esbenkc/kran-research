"""Scouting sweep (seed 0 only): which (p, frac, K) show a memorising phase AND then grok?

Output: data/scout/scout.json. Used once to pick the train fraction per p. Not part of the figure.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import train

HERE = Path(__file__).parent
OUT = HERE / "data" / "scout"


def job(cfg):
    p, frac, K = cfg
    r = train.adamw_train(p, K, frac, 0, lr=3e-3, wd=1.0, init_scale=1.0, max_steps=40000)
    return dict(p=p, frac=frac, K=K, n_train=r["n_train"], mem=r["mem"] and r["mem"]["step"],
                grok_first=r["grok_first"], grok=r["grok"] and r["grok"]["step"],
                last=r["hist"][-1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [(p, f, m * p) for p in [7, 11, 17, 23, 31] for f in [0.3, 0.5, 0.7, 0.8] for m in [2, 4, 8]]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for row in ex.map(job, cfgs):
            rows.append(row)
            print(row["p"], row["frac"], row["K"], row["mem"], row["grok_first"], row["grok"],
                  round(row["last"]["train_acc"], 3), round(row["last"]["test_acc"], 3), flush=True)
    json.dump(rows, open(OUT / "scout.json", "w"), indent=1, default=float)
