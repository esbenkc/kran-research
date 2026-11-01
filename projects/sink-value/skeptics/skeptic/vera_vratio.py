"""First-token value norm / mean value norm of tokens 3-33, per KV head, blocks 26-27, my engine, condition-on input,
mean over 200 prompts (own check of the post's 'about 8% of a typical token value'). CPU fp32. Usage: python vera_vratio.py"""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vera_eng import Eng, S, BS
for fam, seed in [("grader", "base"), ("grader", "s1"), ("monitor", "s0")]:
    E = Eng(fam, seed, "fp32", "cpu"); acc = {0: [], 1: []}
    for i in range(0, len(E.CON), BS):
        S["mode"] = "capture"; S["donor"] = {}; E.run(E.CON[i:i + BS])
        for L in (0, 1):
            v = S["donor"][L][1].float().norm(dim=-1)  # N, 8, T
            acc[L].append((v[:, :, 0] / v[:, :, 3:].mean(-1)).numpy())
    S["mode"] = "none"
    r = {L: np.concatenate(acc[L]).mean(0) for L in (0, 1)}
    print(fam, seed, "block26", r[0].round(3).tolist(), "block27", r[1].round(3).tolist(), "mean all", round(float(np.mean(np.concatenate([r[0], r[1]]))), 3), flush=True)
