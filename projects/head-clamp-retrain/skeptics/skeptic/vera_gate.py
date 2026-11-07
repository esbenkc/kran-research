"""Vera: rebuild the t=0 gate from per-prompt rows only (data/t0/s*.json rows: [prompt, probe, top1, logpA, logpB]).
strict s = share of prompts with probe-on top1 == A(32) AND probe-off top1 == B(33); n = reversed (on != A, off == A).
Also strict from logp (A>B on, B>A off) as a cross-check of the top-1 field, and the rule PRIMARY = UMEAN only if
SET and TOP-1 umean t=0 test strict < 0.25 on all 3 seeds."""
import json, collections
P = "/Users/esben/work/theory-1107-near-tie-clamp/kran-research/projects/head-clamp-retrain/data/t0/"
A, B = 32, 33
def score(rows):
    by = collections.defaultdict(dict)
    for pid, probe, top1, la, lb in rows: by[pid][probe] = (top1, la, lb)
    s = n = s_lp = 0; unB = 0
    for pid, d in by.items():
        on, off = d[1], d[0]
        s += on[0] == A and off[0] == B; n += on[0] != A and off[0] == A
        s_lp += on[1] > on[2] and off[2] > off[1]; unB += off[0] == B
    N = len(by); return dict(N=N, s=s / N, n=n / N, s_logp=s_lp / N, unprobedB=unB / N)
res = {}
for seed in (0, 1, 2):
    d = json.load(open(P + f"s{seed}.json"))
    for key, rows in d["rows"].items():
        if key.endswith("/test"):
            r = score(rows); res[f"s{seed}/{key}"] = r
            stored = d["parent"]["test"] if key.startswith("parent") else d["arms"][key.split("/")[0]]["test"]
            flag = "" if abs(r["s"] - stored["switch_correct"]) < 1e-9 else f"  MISMATCH stored {stored['switch_correct']}"
            print(f"s{seed} {key:<22} strict {r['s']:.3f} (logp {r['s_logp']:.3f}) n {r['n']:.3f} unprobedB {r['unprobedB']:.3f}{flag}")
um = all(res[f"s{s}/{a}_umean/test"]["s"] < 0.25 for s in (0, 1, 2) for a in ("set", "top1"))
print("PRIMARY by rule:", "UMEAN" if um else "SWAP")
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
