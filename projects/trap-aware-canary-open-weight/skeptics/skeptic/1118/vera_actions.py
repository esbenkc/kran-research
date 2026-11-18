# Vera: does every counted fit GET in access.log come from a model action? (harness-artefact null)
# For each run: model fetch URLs from raw actions (tool == fetch) vs GET paths in access.log that carry the run's token or hid.
import json, sys, collections
from urllib.parse import urlparse

OUT = sys.argv[1]
for tag in ["gpt-oss-20b-d4", "qwen3-8b"]:
    d = f"{OUT}/{tag}"
    man = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
    log = collections.Counter()
    for l in open(f"{d}/access.log"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 5 and p[2] == "GET":
            log[p[3]] += 1
    mism = collections.Counter()
    ex = []
    for m in man:
        raw = json.load(open(f"{d}/raw/{m['cell']}/{m['i']:03d}.json"))
        acts = [urlparse(a.get("url") or a.get("args", {}).get("url", "")).path for a in raw["actions"] if a.get("tool") == "fetch"]
        model_fit = sum(1 for a in acts if a == m["fit_path"])
        log_fit = log.get(m["fit_path"], 0)
        if model_fit != log_fit:
            mism[m["cell"]] += 1
            ex.append((m["cell"], m["i"], model_fit, log_fit))
        mine_log = sum(c for p, c in log.items() if p.startswith(f"/{m['token']}/"))
        mine_act = sum(1 for a in acts if a.startswith(f"/{m['token']}/"))
        if mine_log != mine_act:
            mism["tokdir:" + m["cell"]] += 1
            ex.append(("tokdir", m["cell"], m["i"], mine_act, mine_log))
    print(tag, "runs", len(man), "fit-GET count mismatches (model actions vs server log):", dict(mism), ex[:6])
