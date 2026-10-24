"""Vera: rebuild the whole-graph switch numbers from build's raw rows (own code).
Per seed and build: answer differs, pair both-right (probed=A and unprobed=B), mean shift, SHA identity, A/B shares.
Also: split disjointness (ids and bodies) and Wilson intervals.
"""
import json, glob, os, collections, math
import numpy as np

P = "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/raw"
prompts = [json.loads(l) for l in open(f"{P}/prompts.jsonl")]
split = {r["id"]: r["split"] for r in prompts}
body = lambda t: t.split("\n\nWhich would you rather")[0]
bodies = collections.defaultdict(set)
for r in prompts: bodies[r["split"]].add(body(r["text"]))
print("split sizes", collections.Counter(split.values()))
print("body overlap train&test", len(bodies["train"] & bodies["test"]), "dev&test", len(bodies["dev"] & bodies["test"]),
      "train&dev", len(bodies["train"] & bodies["dev"]), "unique test bodies", len(bodies["test"]))
qs = collections.Counter(r["text"][r["text"].find("\n\nWhich"):] for r in prompts)
print("distinct question tails", len(qs))

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

for f in sorted(glob.glob(f"{P}/step2/Qwen3-1.7B/whole/*.json")):
    d = json.load(open(f)); rows = d["rows"]
    by = collections.defaultdict(dict)
    for r in rows: by[r["prompt"]][r["probe"]] = r
    ids = sorted(by)
    assert all(split[i] == "test" for i in ids) and len(ids) == 128 and all(set(by[i]) == {0, 1} for i in ids)
    a, b = rows[0]["a_id"], rows[0]["b_id"]
    t0 = np.array([by[i][0]["top"] for i in ids]); t1 = np.array([by[i][1]["top"] for i in ids])
    lo0 = np.array([by[i][0]["logp_a"] - by[i][0]["logp_b"] for i in ids]); lo1 = np.array([by[i][1]["logp_a"] - by[i][1]["logp_b"] for i in ids])
    same_sha = sum(by[i][0]["logits_sha256"] == by[i][1]["logits_sha256"] for i in ids)
    pair = int(((t1 == a) & (t0 == b)).sum()); diff = int((t1 != t0).sum())
    lo_ci = wilson(pair, 128)
    print(f"{os.path.basename(f)}: differs {diff}/128, pair both-right {pair}/128 (Wilson {lo_ci[0]:.3f}-{lo_ci[1]:.3f}), "
          f"mean shift {np.mean(lo1-lo0):.2f} nats (min {np.min(lo1-lo0):.2f}), SHA identical {same_sha}/128, "
          f"unprobed A/B/other {np.mean(t0==a):.3f}/{np.mean(t0==b):.3f}/{np.mean(~np.isin(t0,[a,b])):.3f}, "
          f"probed A/B/other {np.mean(t1==a):.3f}/{np.mean(t1==b):.3f}/{np.mean(~np.isin(t1,[a,b])):.3f} | "
          f"build says differs {d['differs']}, pair {d['switch_correct']}, shift {d['shift_mean']:.2f}, bitid {d['bit_identical_prompts']}")
