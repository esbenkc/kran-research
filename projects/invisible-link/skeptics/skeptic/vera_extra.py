"""Vera extra checks on the counted phase (3 status agents), own code.
1. Off-task (nonfit) hidden link: raw HTML (hiding cue in text) vs markdown (cue erased), per level and pooled.
   Permutation null: shuffle the view label within agent x template x level strata (5000).
2. Seed-style robustness: split-half (paraphrase 0-2 vs 3-5) title gap.
Run: python3 vera_extra.py <project_dir>
"""
import json, glob, os, sys, collections
import numpy as np

proj = sys.argv[1]
rng = np.random.default_rng(7)
EXC = {"claude-haiku-4-5-20251001"}
man = {}
for l in open(os.path.join(proj, "raw/manifest.jsonl")):
    m = json.loads(l)
    if m.get("phase") == "counted":
        man[m["token"]] = m
log = collections.defaultdict(set)
for line in open(os.path.join(proj, "logs/access.log")):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 5 and f[2] == "GET" and f[4] == "200":
        log[f[3]].add(1)
runs = []
for fp in glob.glob(os.path.join(proj, "raw/counted/**/*.json"), recursive=True):
    d = json.load(open(fp)); s = d["site"]
    if d["model"] in EXC:
        continue
    m = man[s["token"]]
    obs = [t["text"] for t in d["transcript"] if t["role"] != "assistant"]
    runs.append(dict(model=d["model"], level=m["cell"].split("-")[0], view=m["view"], arm=m["arm"],
                     template=m["template"], para=m["paraphrase"], exposed=any(s["hid"] in o for o in obs),
                     follow=s["hidden_path"] in log))


def rate_by(sel, key):
    agents = sorted({r["model"] for r in sel})
    out = {}
    for v in sorted({r[key] for r in sel}):
        out[v] = np.mean([np.mean([r["follow"] for r in sel if r["model"] == a and r[key] == v]) for a in agents])
    return out


non = [r for r in runs if r["arm"] == "hidden_nonfit" and r["view"] in ("raw", "md") and r["exposed"]
       and r["level"] in ("stuck", "rstuck")]
print("1. off-task hidden link, follow rate by view (agents equal weight):")
for lvl in ("stuck", "rstuck", "both"):
    sel = non if lvl == "both" else [r for r in non if r["level"] == lvl]
    rb = rate_by(sel, "view")
    obs = rb["raw"] - rb["md"]
    strata = collections.defaultdict(list)
    for r in sel:
        strata[(r["model"], r["template"], r["level"])].append(r)
    null = []
    for _ in range(5000):
        sh = []
        for rs in strata.values():
            vs = [r["view"] for r in rs]; rng.shuffle(vs)
            sh += [dict(r, view=v) for r, v in zip(rs, vs)]
        q = rate_by(sh, "view"); null.append(q["raw"] - q["md"])
    null = np.array(null)
    per = {a: (sum(r["follow"] for r in sel if r["model"] == a and r["view"] == "raw"),
               sum(1 for r in sel if r["model"] == a and r["view"] == "raw"),
               sum(r["follow"] for r in sel if r["model"] == a and r["view"] == "md"),
               sum(1 for r in sel if r["model"] == a and r["view"] == "md")) for a in sorted({r["model"] for r in sel})}
    print(f"  {lvl:6s} raw {rb['raw']:.3f} md {rb['md']:.3f} diff {obs:+.3f}; two-sided perm P = "
          f"{(np.abs(null) >= abs(obs) - 1e-9).mean():.4f}; per agent raw f/n, md f/n: {per}")

print("2. split-half title gap (STUCK, hidden arms, raw+md, exposed):")
tit = [r for r in runs if r["level"] == "stuck" and r["arm"] in ("hidden_fit", "hidden_nonfit")
       and r["view"] in ("raw", "md") and r["exposed"]]
for name, ps in (("paraphrase 0-2", {0, 1, 2}), ("paraphrase 3-5", {3, 4, 5})):
    sel = [r for r in tit if r["para"] in ps]
    rb = rate_by(sel, "arm")
    print(f"  {name}: fit {rb['hidden_fit']:.3f} nonfit {rb['hidden_nonfit']:.3f} gap {rb['hidden_fit'] - rb['hidden_nonfit']:+.3f} (n={len(sel)})")
