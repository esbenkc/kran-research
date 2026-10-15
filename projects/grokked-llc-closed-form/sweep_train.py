"""Main training sweep: every (p, K, seed) with a per-p train fraction. Saves mem + grok checkpoints
to data/ckpt/ and one training curve per run (json). Recipe: full-batch AdamW, lr 3e-3, decoupled
weight decay 1.0, init N(0,1/d) for W and N(0,1/K) for V, at most 80k steps.

Train fraction per p (from the seed-0 scout in data/scout/scout.json): small p needs more data to grok,
and a larger train set N keeps K(3p-2) under the identifiability cap p(N-1) for more cells.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import train

HERE = Path(__file__).parent
OUT = HERE / "data" / "ckpt"

FRAC = {7: 0.8, 11: 0.8, 17: 0.7, 23: 0.7, 31: 0.5}
KMULT = [3, 4, 8]
SEEDS = [0, 1, 2]
LR, WD, INIT, MAX_STEPS = 3e-3, 1.0, 1.0, 80000


def job(cfg, out=None):
    p, K, seed, frac = cfg
    r = train.adamw_train(p, K, frac, seed, lr=LR, wd=WD, init_scale=INIT, max_steps=MAX_STEPS)
    tag = train.save_run(r, Path(out) if out else OUT)
    return tag, r["mem"] and r["mem"]["step"], r["grok_first"], r["grok"] and r["grok"]["step"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="*", help="only these p")
    ap.add_argument("--K", type=int, nargs="*", help="only these K")
    ap.add_argument("--frac", type=float, help="override the train fraction for the chosen p")
    ap.add_argument("--out", type=str, help="checkpoint dir (default data/ckpt)")
    a = ap.parse_args()
    if a.frac is not None:
        for p in a.p:
            FRAC[p] = a.frac
    if a.out:
        OUT = Path(a.out)
    cfgs = [(p, m * p, s) for p in FRAC for m in KMULT for s in SEEDS]
    if a.p:
        cfgs = [c for c in cfgs if c[0] in a.p]
    if a.K:
        cfgs = [(p, K, s) for p in sorted({c[0] for c in cfgs}) for K in a.K for s in SEEDS]
    cfgs = [(p, K, s, FRAC[p]) for p, K, s in cfgs]  # pass frac explicitly (workers re-import FRAC)
    cfgs.sort(key=lambda c: -c[0] * c[1])  # big jobs first
    with ProcessPoolExecutor(max_workers=8) as ex:
        for out in ex.map(job, cfgs, [str(OUT)] * len(cfgs)):
            print(*out, flush=True)
