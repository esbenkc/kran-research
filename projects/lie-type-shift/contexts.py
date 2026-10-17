"""Rebuild the DolusChat context for every eval prompt used in probe-goodhart.

probe-goodhart (branch swarm/oracle-1014-probe-goodhart, projects/probe-goodhart/common.py) picked its
100 held-out eval prompts with `load_splits()`: keep DolusChat rows whose reference lie_type is omission,
fabrication or exaggeration, shuffle with seed 0, bucket by setting into 3 groups, eval = group 2 rows
300..399. Each rollout row stores prompt_id = sha256(system_message + user query)[:12]. This script
repeats that split, checks every prompt_id in data/rollouts/ is found, and writes data/contexts.jsonl.

    uv run python projects/lie-type-shift/contexts.py
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"


def _bucket(setting: str) -> int:
    return int(hashlib.sha256(setting.encode()).hexdigest(), 16) % 3


def pid(r) -> str:
    return hashlib.sha256((r["system_message"] + r["user_query"]["content"]).encode()).hexdigest()[:12]


def eval_rows(seed=0, n_eval=100):
    from datasets import load_dataset
    d = load_dataset("AlignmentResearch/DolusChat")["train"]
    rows = [r for r in d if r["lie_type"].rstrip(".") in ("omission", "fabrication", "exaggeration")]
    random.Random(seed).shuffle(rows)
    g = {0: [], 1: [], 2: []}
    for r in rows:
        g[_bucket(r["context"]["setting"])].append(r)
    return g[2][300:300 + n_eval]


def main() -> None:
    rows = eval_rows()
    out = {}
    for r in rows:
        c = r["context"]
        out[pid(r)] = {
            "prompt_id": pid(r),
            "system_message": r["system_message"],
            "sensitive_info": c["sensitive_info"],
            "non_sensitive_info": list(c["non_sensitive_info"]),
            "setting": c["setting"],
            "query": r["user_query"]["content"],
            "ref_lie_type": r["lie_type"].rstrip("."),
            "ref_truthful": r["responses"]["truthful"],
            "ref_deceptive": r["responses"]["deceptive"],
        }
    needed = set()
    for f in sorted((DATA / "rollouts").glob("eval_*.jsonl")):
        for line in f.open():
            needed.add(json.loads(line)["prompt_id"])
    missing = needed - set(out)
    assert not missing, f"{len(missing)} rollout prompt_ids not rebuilt, e.g. {sorted(missing)[:3]}"
    with (DATA / "contexts.jsonl").open("w") as f:
        for k in sorted(needed):
            f.write(json.dumps(out[k]) + "\n")
    from collections import Counter
    print(len(needed), "contexts;", Counter(out[k]["ref_lie_type"] for k in needed))


if __name__ == "__main__":
    main()
