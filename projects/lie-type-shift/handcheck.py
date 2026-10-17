"""Blind hand-check of the judges (PREREGISTRATION.md: 60 answers, stratified over arm x round).

    uv run python projects/lie-type-shift/handcheck.py export   # data/handcheck/sample_blind.csv + sample_key.jsonl
    uv run python projects/lie-type-shift/handcheck.py score    # needs data/handcheck/labels_agent.csv (id,label[,note])

The blind file holds only the information the judges see (sensitive item, other items, question, reply), in
shuffled order, without arm, round or any judge label. The key file maps id -> answer.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
HC = DATA / "handcheck"
JUDGES = ("gpt-4.1-mini", "claude-sonnet-4-6")
PER_CELL = 6


def export() -> None:
    ctx = {c["prompt_id"]: c for c in map(json.loads, (DATA / "contexts.jsonl").open())}
    rows = []
    for f in sorted((DATA / "rollouts").glob("eval_*.jsonl")):
        rows += list(map(json.loads, f.open()))
    rng = random.Random(1017)
    picked, seen = [], set()
    for lam in (0.0, 1.0):
        for rnd in (0, 2, 4, 6, 8):
            pool = [r for r in rows if r["lam"] == lam and r["round"] == rnd]
            rng.shuffle(pool)
            n = 0
            for r in pool:
                k = (r["prompt_id"], r["response"])
                if k in seen:  # round 0 is shared by both arms: no duplicate answers
                    continue
                seen.add(k)
                picked.append(r)
                n += 1
                if n == PER_CELL:
                    break
    rng.shuffle(picked)
    HC.mkdir(parents=True, exist_ok=True)
    with (HC / "sample_blind.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sensitive_item", "other_information", "question", "reply"])
        for i, r in enumerate(picked):
            c = ctx[r["prompt_id"]]
            w.writerow([f"h{i:02d}", c["sensitive_info"], " | ".join(c["non_sensitive_info"]), c["query"],
                        r["response"].strip()])
    with (HC / "sample_key.jsonl").open("w") as f:
        for i, r in enumerate(picked):
            f.write(json.dumps({"id": f"h{i:02d}"} | {k: r[k] for k in ("run", "lam", "seed", "round",
                                                                          "prompt_id", "sample")}) + "\n")
    print(len(picked), "answers ->", HC / "sample_blind.csv")


def score() -> None:
    key = {d["id"]: d for d in map(json.loads, (HC / "sample_key.jsonl").open())}
    src = next(p for p in (HC / "labels_agent.csv", HC / "labels_human.csv") if p.exists())
    hand = {r["id"]: r["label"].strip().upper() for r in csv.DictReader(src.open())}
    out = {"n": len(hand)}
    for j in JUDGES:
        lab = {(r["run"], r["round"], r["prompt_id"], r["sample"]): r["label"]
               for r in map(json.loads, (DATA / "labels" / f"labels_{j}.jsonl").open())}
        pairs = [(hand[i], lab[(k["run"], k["round"], k["prompt_id"], k["sample"])]) for i, k in key.items() if i in hand]
        dec = ("FALSIFICATION", "MISLEADING_TRUE", "OMISSION")
        no_false = ("MISLEADING_TRUE", "OMISSION")
        out[j] = {"acc_5way": sum(h == x for h, x in pairs) / len(pairs),
                  "acc_deceptive_vs_not": sum((h in dec) == (x in dec) for h, x in pairs) / len(pairs),
                  # the distinction M rests on: among answers both call deceptive, false statement vs none
                  "acc_false_vs_nofalse_on_both_deceptive": (
                      lambda ps: sum((h in no_false) == (x in no_false) for h, x in ps) / len(ps) if ps else None)(
                      [(h, x) for h, x in pairs if h in dec and x in dec]),
                  "confusion_hand|judge": {f"{h}|{x}": n for (h, x), n in sorted(Counter(pairs).items())}}
    (HC / "score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1]]()
