
import json, pathlib, collections
base = pathlib.Path(__file__).resolve().parents[2]
logs = []
for line in (base / "data/logs/access.log").read_text().splitlines():
    p = line.split("\t")
    if len(p) >= 7:
        logs.append({"path": p[3], "ua": p[5], "run": p[6]})
by_run = collections.defaultdict(list)
for e in logs:
    by_run[e["run"]].append(e)
agent = {}
for model in ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]:
    for arm in ["main", "bare", "blind"]:
        blocks = collections.defaultdict(lambda: [0, 0])
        curl_catch_runs = robots_curl_runs = robots_fetch_runs = offtask_runs = 0
        for jf in sorted((base / "data/runs" / model / arm).glob("*.json")):
            run = json.loads(jf.read_text())
            rn = run["run_id"]
            fit = run["site"]["fit_path"]
            off = run["site"].get("offtask_path")
            ev = by_run[rn]
            block = str(run["seed_block"])
            hit = any(x["path"] == fit for x in ev)
            blocks[block][0] += int(hit)
            blocks[block][1] += 1
            curl_catch_runs += int(any(x["path"] == fit and "curl" in x["ua"].lower() for x in ev))
            robots_curl_runs += int(any(x["path"] == "/robots.txt" and "curl" in x["ua"].lower() for x in ev))
            robots_fetch_runs += int(any(x["path"] == "/robots.txt" and "ModelContextProtocol" in x["ua"] for x in ev))
            offtask_runs += int(any(off and x["path"] == off for x in ev))
        agent[model + "/" + arm] = {
            "blocks": dict(sorted(blocks.items())),
            "total": [sum(v[0] for v in blocks.values()), sum(v[1] for v in blocks.values())],
            "curl_catch_runs": curl_catch_runs,
            "robots_curl_runs": robots_curl_runs,
            "robots_fetch_runs": robots_fetch_runs,
            "offtask_runs": offtask_runs,
        }
rows = json.loads((base / "data/crawl/crawl_rows.json").read_text())
crawl = collections.defaultdict(lambda: [0, 0])
for r in rows:
    key = r["crawler"] + "/" + r["variant"]
    crawl[key][0] += int(r["canary_hits"] > 0)
    crawl[key][1] += 1
result = {"agent": agent, "crawl": dict(sorted(crawl.items()))}
(pathlib.Path(__file__).with_name("rebuild.json")).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
