"""Smoke COST only (PREREG s7): per-agent key cost per run, web cost per run, projected spend for each N.
Never reads smoke rates. python price_smoke.py -> results/smoke_price.json"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent
key_cost: dict[str, float] = defaultdict(float)
for line in (P / "key" / "raw" / "cost.jsonl").read_text().splitlines():
    r = json.loads(line)
    key_cost[r["model"]] += r["cost_usd_est"]
n_key = {m: len(list((P / "key" / "raw_smoke" / m).glob("*/*.jsonl"))) for m in key_cost}
per_run = {m: key_cost[m] / n_key[m] for m in key_cost}
key_mean = sum(per_run.values()) / len(per_run)
web_total = sum(json.loads(x)["cost_usd_est"] for x in (P / "web" / "raw" / "cost.jsonl").read_text().splitlines())
n_web = len(list((P / "web" / "raw" / "smoke").glob("*/*/*.json")))
web_run = web_total / n_web
smoke = sum(key_cost.values()) + web_total
proj = {f"N{n}_web{w}": round(smoke + 6 * 5 * n * key_mean + 3 * 2 * w * web_run, 3)
        for n in (40, 36, 32, 30, 24, 20) for w in (40, 30)}
out = {"key_runs": n_key, "key_cost_per_run": {m: round(v, 5) for m, v in per_run.items()},
       "key_mean_per_run": round(key_mean, 5), "web_runs": n_web, "web_cost_per_run": round(web_run, 5),
       "smoke_spend": round(smoke, 4), "projected": proj}
(P / "results").mkdir(exist_ok=True)
(P / "results" / "smoke_price.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps(out, indent=1, sort_keys=True))
