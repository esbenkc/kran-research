"""Extra checks on counted raw: exposure (key shown to the agent), per-arm D_add with its own
shuffle null, and per-agent key rate among exposed runs."""
import json, os, sys, collections as C
import numpy as np
sys.argv = ["x", "counted/vera_tags.jsonl", "key"]
import vera_rebuild as vr
RAW = os.environ.get("RAW", "../../raw")  # projects/residue-as-trait/raw from skeptics/skeptic
exp = {}
for dp, _, fs in os.walk(RAW):
    for f in fs:
        if f.endswith(".jsonl"):
            for l in open(os.path.join(dp, f)):
                ep = json.loads(l)
                if "episode" not in ep: continue
                key = ep["episode"]["key"]
                obs = [a.get("observation") or "" for a in ep["actions"]]
                first_seen = next((i for i, o in enumerate(obs) if key in o), None)
                route = any("report" in json.dumps(a.get("args", {})) and a["tool"].startswith("api") for a in ep["actions"])
                exp[(ep["model"], ep["episode"]["task"], ep["cell"], ep["seed"])] = dict(
                    exposed=first_seen is not None, steps=ep["steps_used"], cap=ep["steps_used"] >= ep["max_steps"],
                    report=ep["flags"].get("report_accepted"), route=route)
rows = vr.rows
for r in rows:
    r.update(exp[(r["model"], r["scenario"], r["cell"], r["seed"])])
print("per agent: key/n | exposed/n | key|exposed | cap hits | report accepted | route tried")
for a in sorted({r["model"] for r in rows}):
    s = [r for r in rows if r["model"] == a]; e = [r for r in s if r["exposed"]]
    print(f"  {a:28s} {sum(r['y'] for r in s):2d}/{len(s)} | {len(e):2d}/{len(s)} | {sum(r['y'] for r in e)}/{len(e)} | "
          f"{sum(r['cap'] for r in s)} | {sum(bool(r['report']) for r in s)} | {sum(r['route'] for r in s)}")
print("key in unexposed runs:", sum(r["y"] for r in rows if not r["exposed"]))
# per-arm D_add with its own within-scenario agent-label shuffle null
for arm in ("named", "runbook"):
    sub = [r for r in rows if r["cell"] == arm]
    Y, ag, sc, arms, seeds = vr.table(sub)
    d = vr.fast_dadd(Y); rng = np.random.default_rng(20261104); null = []
    for i in range(2000):
        Z = Y.copy()
        for s in range(len(sc)):
            Z[:, s, :] = rng.permutation(Y[:, s, :].reshape(-1)).reshape(Y[:, s, :].shape)
        null.append(vr.fast_dadd(Z))
    null = np.array(null)
    print(f"arm {arm}: D_add {d:+.4f}, null mean {null.mean():+.4f} q95 {np.quantile(null,.95):+.4f}, p {(1+(null>=d).sum())/2001:.4f}; per agent key {[int(Y[i].sum()) for i in range(len(ag))]}")
# cross-arm Spearman of per-agent rates
from scipy.stats import spearmanr
ra = {arm: [np.mean([r['y'] for r in rows if r['cell'] == arm and r['model'] == a]) for a in sorted({r['model'] for r in rows})] for arm in ("named", "runbook")}
print("per-agent rate named", np.round(ra['named'], 3), "runbook", np.round(ra['runbook'], 3), "Spearman", round(spearmanr(ra['named'], ra['runbook']).correlation, 3))
