"""Manipulation check on landed data: what did the model actually SEE around the canary link?

For every run in the 5 status arms, find the first observation that contains the run's canary id (hid) and
classify the link markup: html <a> with rel nofollow / html <a> without it / markdown [..](..) / other.
Also: harness_followed vs access-log follow, and whether arms share site template + seed at each index i.
"""
import json, re, sys
from collections import Counter, defaultdict

OUT = sys.argv[1]
rows = {}
for line in open(f"{OUT}/manifest.jsonl"):
    r = json.loads(line); rows[(r["cell"], r["i"])] = r
log = defaultdict(set)
for line in open(f"{OUT}/access.log"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 7 and p[2] == "GET":
        log[p[6]].add(p[3])

def classify(obs, hid):
    k = obs.find(hid)
    # look back to the start of the enclosing tag / markdown link
    a = obs.rfind("<a", 0, k); md = obs.rfind("](", 0, k)
    if a != -1 and obs.find(">", a) > k:  # hid inside an <a ...> tag
        tag = obs[a:obs.find(">", a) + 1]
        return "a+nofollow" if re.search(r'rel\s*=\s*"?[^">]*nofollow', tag) else "a-plain"
    if md != -1 and k - md < 80:
        return "markdown"
    return "other"

tab = defaultdict(Counter); mism = Counter(); nf_text = Counter(); sites = defaultdict(dict)
for (cell, i), r in sorted(rows.items()):
    arm = cell.replace("-free", "")
    d = json.load(open(f'{OUT}/{r["raw_file"]}'))
    obs = [t["text"] for t in d["transcript"] if t["role"] == "observation" and r["hid"] in t["text"]]
    tab[arm][classify(obs[0], r["hid"]) if obs else "never-seen"] += 1
    nf_text[arm] += sum("nofollow" in t["text"] for t in d["transcript"] if t["role"] == "observation") > 0
    fol = any(r["hid"] in p for p in log[f"{cell}/{i:03d}"])
    mism[arm] += fol != bool(d["harness_followed"])
    sites[i][arm] = (r["template"], r["seed"], r["run_seed"], r["paraphrase"])

for arm in ["raw", "rawnf", "md", "mdhtml", "mdnf", "mdpad8", "gatemd"]:
    print(f"{arm:7s} canary markup seen: {dict(tab[arm])}  runs whose observations contain 'nofollow': {nf_text[arm]}  log!=harness_followed: {mism[arm]}")
five = ["raw", "rawnf", "md", "mdhtml", "mdnf"]
same = sum(len({v[a][:3] for a in five if a in v}) == 1 for v in sites.values() if all(a in v for a in five))
print(f"site indices with all 5 arms: {sum(all(a in v for a in five) for v in sites.values())}; same (template, seed, run_seed) across the 5 arms: {same}")
