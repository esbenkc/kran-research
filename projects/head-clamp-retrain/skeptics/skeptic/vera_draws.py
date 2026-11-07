"""Vera: re-derive the frozen SET / TOP-1 / RANDOM / LEAVE-ONE-FREE names in prereg_v2 from the saved audit."""
import json, random
R = "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/L/heads/"
runs = {0: "lora_none_s0_2048", 1: "lora_none_s1", 2: "lora_none_s2"}
heads = [f"{b}.{h}" for b in (26, 27) for h in range(16)]
key = lambda h: (int(h.split(".")[0]), int(h.split(".")[1]))
for s, r in runs.items():
    d = json.load(open(R + r + ".json"))["dev_single_cut_removed"]
    S = sorted([h for h in heads if d[h] >= 0.05], key=lambda h: -d[h])
    top1 = max(heads, key=lambda h: d[h])
    free = min(heads, key=lambda h: (abs(d[h]),) + key(h))
    pool = [h for h in heads if h not in S]
    rng = random.Random(1107 + s)
    draws = [sorted(rng.sample(pool, len(S)), key=key) for _ in range(3)]
    print(f"s{s} SET {S} TOP1 {top1} FREE {free} ({d[free]:.6f})")
    for i, dr in enumerate(draws): print(f"   draw{i+1} {dr}")
    print("   near line:", {h: round(d[h], 4) for h in heads if 0.03 < d[h] < 0.07})
