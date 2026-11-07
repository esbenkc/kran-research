"""Vera: near-tie set per saved seed, from the saved single-head cut scores (no model).
Two readings of 'within 5% of the top': relative (s >= 0.95*top) and absolute (s >= top-0.05)."""
import json, glob, os
ROOTS = {
 "L (probe-switch-heads, blocks 26-27 trained)": "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/L/heads/*.json",
 "S (probe-sensing-mps detectors)": "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/S/heads/*.json",
}
for lab, pat in ROOTS.items():
    print("==", lab)
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        s = d.get("dev_single_cut_removed")
        if not s: print(os.path.basename(f), "no single scores"); continue
        last2 = {h: v for h, v in s.items() if h.split(".")[0] in ("26", "27")}
        top = max(last2.values())
        rel = sorted([h for h, v in last2.items() if v >= 0.95 * top], key=lambda h: -last2[h])
        ab = sorted([h for h, v in last2.items() if v >= top - 0.05], key=lambda h: -last2[h])
        top6 = sorted(last2.items(), key=lambda kv: -kv[1])[:6]
        print(os.path.basename(f), "top", round(top, 3), "| rel5%", rel, "| abs0.05", ab)
        print("   top6", [(h, round(v, 3)) for h, v in top6])
        print("   greedy", [(g["set"], round(g["dev_removed"], 3)) for g in d.get("greedy_all_dev", [])])
