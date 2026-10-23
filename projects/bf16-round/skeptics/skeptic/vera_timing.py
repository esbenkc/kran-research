"""Vera: recompute speed cost from build's raw per-round timings (raw/timing/" + (__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "timing.json") + ")."""
import json, numpy as np
t = json.load(open("/Users/esben/work/oracle-1023-bf16-round/kran-research/projects/bf16-round/raw/timing/" + (__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "timing.json") + ""))
ms = {k: np.array(v) for k, v in t["ms"].items()}
rng = np.random.default_rng(0)
print({k: len(v) for k, v in ms.items()})
for k in ms:
    if k == "none": continue
    cost = np.median(ms[k]) / np.median(ms["none"]) - 1
    # unpaired bootstrap of median ratio, and paired per-round ratio
    b = [np.median(rng.choice(ms[k], len(ms[k]))) / np.median(rng.choice(ms["none"], len(ms["none"]))) - 1 for _ in range(10000)]
    r = ms[k] / ms["none"] - 1
    bp = [np.median(rng.choice(r, len(r))) for _ in range(10000)]
    print(f"{k}: median {np.median(ms[k]):.2f} vs none {np.median(ms['none']):.2f} ms; cost {100*cost:+.2f}% "
          f"[unpaired boot {100*np.percentile(b,2.5):+.2f}, {100*np.percentile(b,97.5):+.2f}]; "
          f"paired per-round median {100*np.median(r):+.2f}% [{100*np.percentile(bp,2.5):+.2f}, {100*np.percentile(bp,97.5):+.2f}]; max {np.max(ms[k]):.1f}")
