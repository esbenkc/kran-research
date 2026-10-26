"""Vera: blind picks + flags from my own score runs (out/score{06,17}_*.json). Usage: vera_picks.py SIZE_TAG BASE m1 m2 ..."""
import sys, json
import numpy as np
tag, base, models = sys.argv[1], sys.argv[2], sys.argv[3:]
B = json.load(open(f"out/score{tag}_{base}.json"))
heads = B["heads"]
out = {}
for m in models:
    M = json.load(open(f"out/score{tag}_{m}.json"))
    row = {}
    for suf in ("", "_last"):
        S = np.array(M["d"]["probe" + suf]) - np.array(B["d"]["probe" + suf])
        F_flip = max(float(np.max(np.array(M["d"][f"flip{s}" + suf]) - np.array(B["d"][f"flip{s}" + suf]))) for s in range(4))
        F_iso = max(float(np.max(np.array(M["d"][f"fake{s}" + suf]) - np.array(B["d"][f"fake{s}" + suf]))) for s in range(4))
        order = sorted(range(32), key=lambda i: (-S[i], i))  # ties by head index
        emu = float(np.max(np.abs(np.array(M["d"]["emu" + suf]))))
        base_rank = sorted(range(32), key=lambda i: (-B["d"]["probe" + suf][i], i))
        row["primary" if suf == "" else "secondary_last"] = dict(
            top8=[heads[i] for i in order[:8]], S_top8=[round(float(S[i]), 4) for i in order[:8]],
            F_flip=round(F_flip, 4), F_iso=round(F_iso, 4),
            flagged_flip=[heads[i] for i in order if S[i] > F_flip], flagged_iso=[heads[i] for i in order if S[i] > F_iso],
            emu_max_d=emu, top1_base_rank=base_rank.index(order[0]) + 1)
    out[m] = row
    p, s = row["primary"], row["secondary_last"]
    print(f"{m}: top3 {p['top8'][:3]} S {p['S_top8'][:3]} F_flip {p['F_flip']} F_iso {p['F_iso']} flagged {p['flagged_flip']} "
          f"| last: top3 {s['top8'][:3]} S {s['S_top8'][:3]} F_flip {s['F_flip']} flagged {s['flagged_flip']} | emu {p['emu_max_d']} base-rank(top1) {p['top1_base_rank']}")
json.dump(out, open(f"out/picks{tag}.json", "w"), indent=1)
