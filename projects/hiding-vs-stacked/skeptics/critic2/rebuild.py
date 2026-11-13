# critic2 (Kasper) rebuild: follows per cell from the server log, own code.
# Run from projects/hiding-vs-stacked:  python3 skeptics/critic2/rebuild.py
import json, collections, random
P = "web/"
M = [json.loads(l) for l in open(P + "raw/manifest.jsonl")]
M = [m for m in M if m["phase"] == "counted"]
hits = collections.defaultdict(set)
for l in open(P + "logs/access.log"):
    f = l.rstrip("\n").split("\t")
    if len(f) > 4 and f[2] == "GET" and f[4] == "200":
        hits[f[3].split("/")[1]].add(f[3])
print("counted runs", len(M), "unique site tokens", len({m["token"] for m in M}))
fol, n = collections.Counter(), collections.Counter()
ag, agn = collections.Counter(), collections.Counter()
css = off = 0
for m in M:
    h = hits[m["token"]]; k = m["cell"]; y = m["fit_path"] in h
    n[k] += 1; fol[k] += y; ag[(k, m["model"])] += y; agn[(k, m["model"])] += 1
    css += m["css_path"] in h; off += m["offtask_path"] in h
for k in sorted(n):
    print(f"{k:12s} {fol[k]:4d}/{n[k]} = {100*fol[k]/n[k]:.1f}%")
print("site.css fetched", css, "| off-task link followed", off)
models = sorted({m["model"] for m in M})
for c in "A A2 C E B B2".split():
    print(c, "neutral->stacked per agent:", [(mo, ag[("neutral-"+c, mo)], ag[("stacked-"+c, mo)], agn[("stacked-"+c, mo)]) for mo in models])
rng = random.Random(7)
for c in ("B", "B2"):
    rows = [m["fit_path"] in hits[m["token"]] for m in M if m["cell"] == "stacked-" + c]
    bs = sorted(sum(rng.choice(rows) for _ in rows) / len(rows) for _ in range(4000))
    print(c, "stacked pooled 95% (unstratified run bootstrap, seed 7):", round(100*bs[100], 1), round(100*bs[3899], 1))
sc = [json.loads(l) for l in open(P + "results/scores_counted.jsonl")]
cue, cn = collections.Counter(), collections.Counter()
for x in sc:
    cn[x["hiding"]] += 1; cue[x["hiding"]] += bool(x["cue"])
print("cue_grep cue=1 per row:", {h: (cue[h], cn[h]) for h in sorted(cn)})
c = [json.loads(l) for l in open(P + "raw/cost.jsonl")]
print("experiment API cost (cost.jsonl sum cost_usd_est):", round(sum(x["cost_usd_est"] for x in c), 4), "over", len(c), "calls")
