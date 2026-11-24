"""Vera (skeptic): independent rebuild of the J1 headline from raw rows. Written BEFORE J1 landed (2026-11-23),
from PREREG.md and the field names Bruno posted in the room (#29381, #29420); key names are checked at run time
and the script stops with the missing key rather than guessing.

Run from projects/cot-necessity:  python3 skeptics/skeptic/vera_j1.py
Imports no project code. Reads:
  data/runs/main_k8_s{0,1,2}/evals/step_NNNN.jsonl   policy completions (task accuracy, thought length)
  data/runs/main_k8_s*/steps.jsonl, data/lambda.json  the reward ramp
  data/reader/gate.json, data/reader/scores/main_k8_s*/step_NNNN.jsonl   reader scores
"""
import glob
import json
import math
import random
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vera_rule import status, wilson  # noqa: E402

P = Path(os.environ.get("COTN_DIR") or Path(__file__).resolve().parents[2])
ANS = re.compile(r"<answer>\s*(heads|tails)\s*</answer>\s*$", re.I)


def load(p):
    return [json.loads(l) for l in open(p)]


def need(row, *keys):
    for k in keys:
        if k in row:
            return row[k]
    sys.exit(f"missing key {keys} in row with keys {sorted(row)}")


def parse(completion):
    """PREREG :61-67: thought = whole completion minus the final strict answer span and the <reasoning> tags."""
    m = ANS.search(completion)
    pred = m.group(1).lower() if m else None
    body = completion[: m.start()] if m else completion
    thought = body.replace("<reasoning>", "").replace("</reasoning>", "")
    return pred, thought


def run_rows(run):
    out = {}
    for f in sorted(glob.glob(str(P / f"data/runs/{run}/evals/step_*.jsonl"))):
        step = int(re.search(r"step_(\d+)", f).group(1))
        out[step] = load(f)
    return out


def main():
    report = {"runs": {}}
    seeds = []
    gate = json.load(open(P / "data/reader/gate.json"))
    report["gate"] = gate
    thr = need(gate, "threshold", "logit_threshold")
    for s in range(3):
        run = f"main_k8_s{s}"
        evals = run_rows(run)
        cps, mism = [], 0
        for step, rows in sorted(evals.items()):
            sc_file = P / f"data/reader/scores/{run}/step_{step:04d}.jsonl"
            scores = {r["id"]: r for r in load(sc_file)} if sc_file.exists() else {}
            task_k = reader_k = 0
            chars = []
            cells = {}
            for r in rows:
                comp = need(r, "completion", "text")
                pred, thought = parse(comp)
                capped = bool(r.get("capped", False))
                ok = (pred == r["answer"]) and not capped
                task_k += ok
                chars.append(len(thought))
                if "thought_chars" in r and r["thought_chars"] != len(thought):
                    mism += 1
                cells[(r["start"], r["answer"])] = cells.get((r["start"], r["answer"]), 0) + 1
                if scores:
                    sc = scores[r["id"]]
                    # primary logit reader: predicts heads iff logit(heads) - logit(tails) > calibrated threshold
                    reader_k += (need(sc, "logit") > thr) == (r["answer"] == "heads")
            cps.append(dict(step=step, n=len(rows), task_k=task_k, reader_k=reader_k,
                            mean_chars=statistics.mean(chars), cells=sorted(cells.values())))
        report["runs"][run] = dict(checkpoints=cps, thought_chars_mismatch=mism)
        seeds.append(cps)
    st, v = status(seeds)
    report["status"] = st
    report["per_seed"] = [dict(seed_yes=x["seed_yes"], seed_no=x["seed_no"], rows=x["rows"]) for x in v]
    print(json.dumps(report, indent=1, default=str))
    (Path(__file__).parent / "vera_j1.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
