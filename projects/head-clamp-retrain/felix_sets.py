"""Frozen clamp sets for PREREG v2, read off the saved first audit (no model run).

Source: probe-switch-heads data/L/heads/<run>.json, field dev_single_cut_removed (64 dev prompts),
branch swarm/oracle-1024-probed-only-switch. Writes data/sets.json.
"""
import json
import os
import sys

import random

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/L/heads")
RUNS = ["lora_none_s0_2048", "lora_none_s1", "lora_none_s2"]
HEADS = [f"{l}.{h}" for l in (26, 27) for h in range(16)]

out = {}
for i, run in enumerate(RUNS):
    d = json.load(open(os.path.join(SRC, f"{run}.json")))["dev_single_cut_removed"]
    top = max(d.items(), key=lambda kv: kv[1])
    scout = [h for h, v in d.items() if v >= 0.95 * top[1]]          # literal scout rule (relative)
    scout_abs = [h for h, v in d.items() if v >= top[1] - 0.05]      # literal scout rule (absolute)
    st = sorted([h for h, v in d.items() if v >= 0.05], key=lambda h: -d[h])
    pool = [h for h in HEADS if h not in st]
    rng = random.Random(1107 + i)                                     # Iris prereg_v2, Vera vera_draws.py
    key = lambda x: (int(x.split(".")[0]), int(x.split(".")[1]))
    draws = [sorted(rng.sample(pool, len(st)), key=key) for _ in range(3)]
    free = min(HEADS, key=lambda h: (abs(d[h]),) + key(h))          # smallest |share|; ties: lower block, index
    out[f"s{i}"] = {"run": run, "top1": [top[0]], "scout_rel5": scout, "scout_abs5": scout_abs,
                    "set": st, "set_scores": {h: d[h] for h in st}, "random": draws,
                    "all32": HEADS, "free": free, "leave_one_free": [h for h in HEADS if h != free]}
    print(f"s{i}", "top1", top[0], "scout", scout, "set", st, "random", draws, "free", free)

os.makedirs("data", exist_ok=True)
json.dump(out, open("data/sets.json", "w"), indent=1)
