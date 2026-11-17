"""Vera null tests. (1) Permutation of the warning label within (template, trap level) strata:
   U - W (present) and D under both fit readings, 10,000 shuffles.
(2) URL-slip rate: share of runs with >=1 GET that lacks the run's /<token>/ prefix, per cell.
Usage: python3 vera_null.py OUTDIR TAG
"""
import sys, os, json, collections as C
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vera_rebuild import load

outdir, tag = sys.argv[1], sys.argv[2]
rows = load(outdir, tag)
man = [json.loads(l) for l in open(os.path.join(outdir, tag, "manifest.jsonl"))]
# attribute each prefix-less GET to a run by time order is not possible from the log alone; use raw actions instead
slip = C.Counter(); n = C.Counter()
for m in man:
    p = os.path.join(outdir, tag, m["raw_file"])
    if not os.path.exists(p):
        continue
    acts = json.load(open(p)).get("actions", [])
    bad = any(a.get("tool") == "fetch" and a.get("path") and not a["path"].startswith("/" + m["token"] + "/") for a in acts)
    slip[m["cell"]] += bad; n[m["cell"]] += 1
print("URL slip (>=1 fetch outside /<token>/):", {c: f"{slip[c]}/{n[c]}" for c in sorted(n)})

rng = np.random.default_rng(7)
def stat(rs, metric, lab):
    def rate(cell): 
        v = [r[metric] for r, l in zip(rs, lab) if l == cell]
        return sum(v) / len(v)
    uw = rate("neutral-cued") - rate("warning-cued")
    d = uw - (rate("neutral-none") - rate("warning-none"))
    return uw, d
sub = [r for r in rows if r["cell"] in ("neutral-cued", "warning-cued", "neutral-none", "warning-none")]
lab = [r["cell"] for r in sub]
for metric in ("fit_exact", "fit_tok"):
    obs = stat(sub, metric, lab)
    strata = C.defaultdict(list)
    for k, r in enumerate(sub):
        strata[(r["template"], r["cell"].split("-")[1])].append(k)
    ge = np.zeros(2); B = 10000
    for _ in range(B):
        lab2 = list(lab)
        for idx in strata.values():
            perm = rng.permutation([lab[k] for k in idx])
            for k, l in zip(idx, perm):
                lab2[k] = l
        s = stat(sub, metric, lab2)
        ge += [abs(s[0]) >= abs(obs[0]) - 1e-12, abs(s[1]) >= abs(obs[1]) - 1e-12]
    print(f"{metric}: U-W {obs[0]:+.3f} perm p={ge[0]/B:.4f};  D {obs[1]:+.3f} perm p={ge[1]/B:.4f}")
