"""Main-arm per-seed slopes refit on the sizes each S9 seed used (descriptive). Reads my check_final.json + check_s9.json."""
import json, math, numpy as np
from vera_check import ols, tint
c = json.load(open("check_final.json")); s9 = json.load(open("check_s9.json"))
runs = [r for r in c["runs"].values() if r["arm"] == "cue" and r["k"] is not None and r["alpha"] > 0]
for s, rs in s9["runs"].items():
    ks = sorted(r["k"] for r in rs if r["in_fit"] and r["T"] is not None)
    sl = []
    for ms in (5, 6, 7):
        u = sorted([r for r in runs if r["seed"] == ms and r["k"] in ks], key=lambda r: r["k"])
        sl.append(ols([math.log(r["share_logged"]) for r in u], [math.log(r["T"]) for r in u])[0])
    m, sd, ci = tint(sl)
    print(f"S9 seed {s} sizes {ks}: S9 slope {s9['slopes'].get(s, float('nan')):.4f}; main seeds same sizes {[round(x,4) for x in sl]} -> {m:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
