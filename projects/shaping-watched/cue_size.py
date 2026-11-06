"""Size of the probe's footprint where the fed update sees it: block-26 inputs, leaky (ref) vs none, recorded in the
real compiled graph. Ours (hook at q_norm 24) vs the parent numeric-chameleon-rl (hook at q_norm 14, same recorder).
Reported at the decision position (last prompt token) over all prompts: share of elements that differ, mean |diff|
relative to mean |h|.   python cue_size.py -> data/cue_size.json"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
PARENT = Path.home() / "work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/cache/smoke"  # parent base cache deleted; smoke = same recipe, 8 dev prompts

def bf(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def stats(n, r, pos):
    a, b = bf(np.ascontiguousarray(n[:, pos])), bf(np.ascontiguousarray(r[:, pos]))
    d = np.abs(a - b)
    return {"share_elements_differ": float((d > 0).mean()), "share_prompts_any_diff": float((d.max(1) > 0).mean()),
            "mean_absdiff_over_mean_abs": float(d.mean() / np.abs(a).mean()), "N": int(a.shape[0])}

out = {}
n = np.load(HERE / "raw/cache/base26/none_rec_tail.npy", mmap_mode="r")[:, 0]
r = np.load(HERE / "raw/cache/base26/ref_rec_tail.npy", mmap_mode="r")[:, 0]
out["ours_q_norm24_at_block26_decision_pos"] = stats(n, r, n.shape[1] - 2)
if (PARENT / "none_rec_tail.npy").exists():
    pn = np.load(PARENT / "none_rec_tail.npy", mmap_mode="r"); pr = np.load(PARENT / "ref_rec_tail.npy", mmap_mode="r")
    out["parent_q_norm14_at_block26_decision_pos"] = stats(pn, pr, pn.shape[1] - 1)
json.dump(out, open(HERE / "data/cue_size.json", "w"), indent=1)
print(json.dumps(out, indent=1))
