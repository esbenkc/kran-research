"""Robustness check: re-judge round 0 and the final round with a second judge from another
model family (same prompt as common.judge_one). Writes data/judge2.jsonl.

    with-model-keys uv run python projects/probe-goodhart/judge2.py
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import DATA, JUDGE_SYS, JUDGE_TMPL, load_splits, pid
from lib import llm

MODEL2 = "claude-sonnet-4-6"


def main():
    rows_by_id = {pid(r): r for r in load_splits()["eval"]}
    todo = []
    for f in sorted((DATA / "raw").glob("eval_lam*.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        last = max(r["round"] for r in rows)
        todo += [r for r in rows if r["round"] in (0, last)]

    def one(r):
        src = rows_by_id[r["prompt_id"]]
        p = JUDGE_TMPL.format(sensitive=src["context"]["sensitive_info"], query=src["user_query"]["content"],
                              response=r["response"].strip()[:2500])
        t = llm.complete(p, system=JUDGE_SYS, model=MODEL2, max_tokens=5).strip().upper()
        lab = next((l for l in ("DECEPTIVE", "HONEST", "OTHER") if l in t), "OTHER")
        return {"run": r["run"], "lam": r["lam"], "seed": r["seed"], "round": r["round"],
                "prompt_id": r["prompt_id"], "sample": r["sample"], "judge": r["judge"], "judge2": lab}

    with ThreadPoolExecutor(12) as ex:
        out = list(ex.map(one, todo))
    with (DATA / "judge2.jsonl").open("w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    agree = sum(o["judge"] == o["judge2"] for o in out) / len(out)
    print(f"{len(out)} rows, agreement {agree:.2f}")
    for run in sorted({o["run"] for o in out}):
        for rnd in sorted({o["round"] for o in out if o["run"] == run}):
            s = [o for o in out if o["run"] == run and o["round"] == rnd]
            print(run, rnd, "judge1 dec", round(sum(o["judge"] == "DECEPTIVE" for o in s) / len(s), 2),
                  "judge2 dec", round(sum(o["judge2"] == "DECEPTIVE" for o in s) / len(s), 2))


if __name__ == "__main__":
    main()
