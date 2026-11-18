"""Host-confound and luck checks: seed 0 ran on the mini, seeds 1-2 on Esben's laptop.
Compares pre-onset per-update training stats (updates 1-500, before any seed's onset) across seeds,
and counts the carrying ' We' thoughts each seed had seen by each read.
Usage: python3 vera_host.py <project dir>
"""
import json, sys
import numpy as np

P = sys.argv[1]
KEYS = ["cheat_probed", "cheat_unprobed", "cheat_probed_x", "cheat_probed_notx", "x_rate_probed", "x_rate_unprobed",
        "kl_dec", "g_dec_preclip", "live_probed_groups", "M_read_mean", "M_solve_mean", "corr_x_s", "t"]


def load(run):
    rows = {}
    for l in open(f"{P}/data/runs/{run}/steps.jsonl"):
        d = json.loads(l); rows[d["step"]] = d
    return rows


runs = {0: "kl0_s0", 1: "kl0_s1", 2: "kl0_s2"}
R = {s: load(r) for s, r in runs.items()}
print("pre-onset window: updates 1-500 (every seed's s_P <= 0.012 there)")
print(f"{'stat':20s} " + " ".join(f"{'s'+str(s)+' mean':>12s}" for s in R) + "   s0 inside s1-s2 range (+-2 se)?")
for k in KEYS:
    ms, ses = [], []
    for s in R:
        a = np.array([R[s][i][k] for i in range(1, 501) if R[s][i].get(k) is not None and not (isinstance(R[s][i][k], float) and np.isnan(R[s][i][k]))], float)
        ms.append(a.mean()); ses.append(a.std(ddof=1) / np.sqrt(len(a)))
    lo = min(ms[1] - 2 * ses[1], ms[2] - 2 * ses[2]); hi = max(ms[1] + 2 * ses[1], ms[2] + 2 * ses[2])
    print(f"{k:20s} " + " ".join(f"{m:12.4f}" for m in ms) + f"   {'yes' if lo <= ms[0] <= hi else 'NO'}  (lap range {lo:.4f}..{hi:.4f}, s0 se {ses[0]:.4f})")

print("\ncumulative probed rows carrying ' We' (sum n_probed_x over updates; fresh-thought updates only in brackets)")
for s in R:
    cum, cumf, line = 0, 0, []
    for i in range(1, max(R[s]) + 1):
        d = R[s][i]; cum += d["n_probed_x"]
        if d["fresh_on"]:
            cumf += d["n_probed_x"]
        if i % 100 == 0:
            line.append(f"{i}:{cum}[{cumf}]")
    print(f"  s{s}: " + " ".join(line))

print("\nreward edge on carrying rows: mean(cheat_probed_x - cheat_probed_notx) per 100-update block (the signal the planted word could ride)")
for s in R:
    out = []
    for a in range(0, max(R[s]), 100):
        v = [R[s][i]["cheat_probed_x"] - R[s][i]["cheat_probed_notx"] for i in range(a + 1, a + 101)
             if i in R[s] and R[s][i]["n_probed_x"] > 0 and R[s][i]["cheat_probed_x"] is not None]
        out.append(f"{a+100}:{np.mean(v):+.3f}")
    print(f"  s{s}: " + " ".join(out))
