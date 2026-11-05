"""Phase-0 design measurements on SAVED models only (no training).

(a) --time: wall time of one training refill (search.fit_offsets, UNITS_TRAIN, 8 starts x 60 steps, 32 train prompts,
    eager cache, MPS bf16) at several --chunk values, on one merged tail. Footprint: run under watchdog.py.
(b) --rand: unoptimised lift (log-odds of mean P(X) on the 64 dev prompts, eager cache) under 32 fixed random pushes
    per unit, each position on the edge of the search ball (radius_of: 2% of the mean norm, 0.5% at position 0).
    Iris #14553: if twins (stack_s*) react to random pushes, arm R penalises the switch itself and is no null.
    PY measure0.py --time --models cln_s0=raw/runs/cln_s0_merged.pt
    PY measure0.py --rand --models base=base,stack_s0=...,cln_s0=...      -> data/measure0_rand.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from search import UNITS_TRAIN, fit_offsets, pos_norms, radius_of  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--time", action="store_true")
ap.add_argument("--rand", action="store_true")
ap.add_argument("--chunks", default="8,48")
ap.add_argument("--n-rand", type=int, default=32)
ap.add_argument("--nfit", type=int, default=32)
ap.add_argument("--steps", type=int, default=60)
a = ap.parse_args()

meta = json.load(open(HERE / "raw/cache/base/meta.json"))
idx = splits(meta["prompt_ids"])
ZE = Rows(HERE / "raw/cache/base/eager_rec_tail.npy")
norms = pos_norms(ZE, idx["train"])
xs = torch.tensor(x_ids(), device=DEV)
cfg, tail, w_out = load_tail(2)
base_sd = {k: v.detach().clone() for k, v in tail.state_dict().items() if "rotary" not in k}
for p in tail.parameters():
    p.requires_grad_(False)
lo = lambda q: float(np.log(max(q, 1e-9) / max(1 - q, 1e-9)))
out = {}
for spec in a.models.split(","):
    lab, path = spec.split("=", 1)
    sd = base_sd if path == "base" else {k: v for k, v in torch.load(HERE / path, map_location="cpu").items() if "rotary" not in k}
    tail.load_state_dict({k: v.to(torch.bfloat16).to(DEV) for k, v in sd.items()}, strict=False)
    tail.eval()
    if a.time:
        fit_idx = list(np.random.default_rng(30_000).choice(idx["train"], a.nfit, replace=False))
        E = ZE[fit_idx].float().to(DEV)
        for ch in [int(c) for c in a.chunks.split(",")]:
            torch.mps.empty_cache()
            t0 = time.time()
            found = fit_offsets(tail, w_out, E, xs, norms, UNITS_TRAIN, steps=a.steps, seed=123, starts=8, chunk=ch)
            torch.mps.synchronize()
            dt = time.time() - t0
            out[f"{lab}_chunk{ch}_nfit{a.nfit}_steps{a.steps}"] = {"s": round(dt, 1), "steps": a.steps, "lift": [round(r["lift"], 3) for r in found]}
            print("TIME", lab, "chunk", ch, "nfit", a.nfit, f"{dt:.1f}s", "lift/unit", [round(r["lift"], 3) for r in found], flush=True)
    if a.rand:
        E = ZE[idx["dev"]].float().to(DEV)
        g = torch.Generator(device="cpu").manual_seed(55_555)
        T, H = E.shape[1], E.shape[2]

        @torch.no_grad()
        def pmean(off):
            return float(torch.softmax(tail((E + off).to(torch.bfloat16), w_out).float(), -1)[:, xs].sum(-1).mean())
        p0 = pmean(torch.zeros(T, H, device=DEV))
        res = {"p0": p0, "units": []}
        for unit in UNITS_TRAIN:
            rad = radius_of(norms.cpu(), unit)
            lifts = []
            for _ in range(a.n_rand):
                d = torch.randn(T, H, generator=g)
                lifts.append(lo(pmean((rad[:, None] * d / d.norm(dim=-1, keepdim=True)).to(DEV))) - lo(p0))
            res["units"].append({"unit": unit, "median": float(np.median(lifts)), "max": float(np.max(lifts)),
                                 "min": float(np.min(lifts)), "mean_abs": float(np.mean(np.abs(lifts))), "lifts": lifts})
        out[lab] = res
        print("RAND", lab, "p0", round(p0, 4), "median/max per unit",
              [(round(u["median"], 3), round(u["max"], 3)) for u in res["units"]], flush=True)
(HERE / "data").mkdir(exist_ok=True)
fn = "measure0_time.json" if a.time else "measure0_rand.json"
prev = json.loads((HERE / "data" / fn).read_text()) if (HERE / "data" / fn).exists() else {}
prev.update(out)
(HERE / "data" / fn).write_text(json.dumps(prev, indent=1))
