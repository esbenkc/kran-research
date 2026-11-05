"""Arm C token set Y: print the base model's mean first-token distribution on the 382 train prompts (eager cache) and
the step-0 push-search lift (8 starts x 60 steps, the arm-P search) for candidate sets, so Y can be matched to X in
base mass and in step-0 search lift (Vera #14565). Saved: data/pick_y.json.
    PY pick_y.py --top 40                       (list)
    PY pick_y.py --cands 'name:tok1|tok2,...;name2:...'   (search lift per candidate, base model)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from search import UNITS_TRAIN, fit_offsets, pos_norms  # noqa: E402
from tail import DEV, Rows, load_tail, px_rows, splits, x_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--cands", default="")
ap.add_argument("--chunk", type=int, default=2)
a = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
meta = json.load(open(HERE / "raw/cache/base/meta.json"))
idx = splits(meta["prompt_ids"])
ZE = Rows(HERE / "raw/cache/base/eager_rec_tail.npy")
cfg, tail, w_out = load_tail(2)
for p in tail.parameters():
    p.requires_grad_(False)
tail.eval()
xs = x_ids()
fn = HERE / "data/pick_y.json"
out = json.loads(fn.read_text()) if fn.exists() else {}
with torch.no_grad():
    pr = torch.cat([torch.softmax(tail(ZE[idx["train"][i:i + 32]].to(DEV), w_out), -1).float().cpu()
                    for i in range(0, len(idx["train"]), 32)]).mean(0)
out["px_base_train"] = float(pr[xs].sum())
top = torch.topk(pr, a.top)
out["top"] = [[int(i), tok.convert_ids_to_tokens(int(i)), float(v)] for v, i in zip(top.values, top.indices)]
for i, t, v in out["top"]:
    print(f"{i:>7} {t!r:>16} {v:.4f}{'  X' if i in xs else ''}")
print("P(X) base train", round(out["px_base_train"], 4))
if a.cands:
    norms = pos_norms(ZE, idx["train"])
    fit_idx = list(np.random.default_rng(30_000).choice(idx["train"], 32, replace=False))
    E = ZE[fit_idx].float().to(DEV)
    out.setdefault("cands", {})
    for c in a.cands.split(";"):
        name, toks = c.split(":", 1)
        ids = xs if name == "X" else [tok.convert_tokens_to_ids(t) for t in toks.split("|")]
        assert name == "X" or all(isinstance(i, int) and i >= 0 and i not in xs for i in ids), (name, ids)
        ys = torch.tensor(ids, device=DEV)
        found = fit_offsets(tail, w_out, E, ys, norms, UNITS_TRAIN, steps=60, seed=1_000_000, starts=8, chunk=a.chunk)
        out["cands"][name] = {"ids": ids, "toks": toks.split("|"), "p_base_train": float(pr[ids].sum()),
                              "lift_step0": [r["lift"] for r in found], "p0_fit": found[0]["p0"]}
        print("CAND", name, "P base", round(float(pr[ids].sum()), 4), "lift/unit", [round(r["lift"], 3) for r in found], flush=True)
fn.write_text(json.dumps(out, indent=1))
