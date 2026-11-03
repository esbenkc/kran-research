"""EXPLORATORY drift check (DEVIATIONS D4): the 1102 open-route cell open_c4, UNCHANGED, on fresh seeds.

env.py, agent.py, scorer.py in this folder are byte-identical copies of kran-research projects/open-route at f1b23ec8
(branch swarm/defense-1102-open-route; sha1 in drift/SHA1SUMS). No pointer line, no storage.md, 1102 denials.
Models: claude-haiku-4-5-20251001, gpt-5.4-mini; seeds 40..69; 60 runs. Replies cache in drift/raw/cache, cost in
drift/raw/cost.jsonl; episodes in ../raw_drift/<model>/open_c4/<seed>.jsonl. Feeds no status, law or prediction.

with-model-keys uv run --with openai python drift/run_drift.py [--score]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent import COST_PATH, run_episode  # noqa: E402  (1102 agent.py: cache + cost under drift/raw/)
from scorer import score  # noqa: E402

OUT = HERE.parent / "raw_drift"
MODELS = ["claude-haiku-4-5-20251001", "gpt-5.4-mini"]
SEEDS = range(40, 70)
MAIN_COST = HERE.parent / "raw" / "cost.jsonl"
BUDGET = 4.50


def spent() -> float:
    tot = 0.0
    for p in (MAIN_COST, COST_PATH):
        if p.exists():
            tot += sum(json.loads(x).get("cost_usd_est", 0.0) for x in p.read_text().splitlines() if x.strip())
    return tot


def one(model: str, seed: int) -> str:
    p = OUT / model / "open_c4" / f"{seed}.jsonl"
    if p.exists():
        return "skip"
    if spent() > BUDGET:
        return "budget"
    run_episode(model, "open_c4", seed, out_path=p)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if not a.score:
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(one, m, s): (m, s) for s in SEEDS for m in MODELS}
            res = [f.result() for f in cf.as_completed(futs)]
        print(json.dumps({"runs": len(res), "status": {k: res.count(k) for k in set(res)}, "spent_usd_total": round(spent(), 4)}))
    out = {}
    for m in MODELS:
        rows = [score(json.loads(p.read_text())) for p in sorted((OUT / m / "open_c4").glob("*.jsonl"))]
        out[m] = {"n": len(rows), "key": sum(r["key"] for r in rows), "key_outside_route": sum(r["key_outside_route"] for r in rows),
                  "key_in_route_call": sum(r["key_in_route_call"] for r in rows),
                  "first_storage_reply": {k: sum(r["first_storage_reply"] == k for r in rows) for k in ("token_missing", "scope", "size", "other", "none")},
                  "key_by_first_reply": {k: sum(r["key"] for r in rows if r["first_storage_reply"] == k) for k in ("token_missing", "scope")}}
    (HERE.parent / "results").mkdir(exist_ok=True)
    (HERE.parent / "results" / "drift_d4.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
