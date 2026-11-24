"""Vera (skeptic): independent rebuild of Step 0 (necessity) from the raw NDIF rows, plus a data audit of inputs/.

Run from projects/cot-necessity:  python3 skeptics/skeptic/vera_step0.py
Reads only data/step0/necessity_k*.jsonl and inputs/*.jsonl (raw rows); no project code is imported.
"""
import collections
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # projects/cot-necessity


def auroc(pairs):
    pos = [x for x, y in pairs if y]
    neg = [x for x, y in pairs if not y]
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return s / len(pos) / len(neg)


def load(p):
    return [json.loads(l) for l in open(p)]


def step0():
    out = {}
    for k in [1, 2, 3, 4, 8]:
        f = HERE / f"data/step0/necessity_k{k}.jsonl"
        if not f.exists():
            continue
        rows = load(f)
        d = [(r["logit_heads"] - r["logit_tails"], r["answer"] == "heads") for r in rows]
        med = statistics.median(x for x, _ in d)
        res = dict(
            n=len(d),
            cells=sorted(collections.Counter((r["start"], r["answer"]) for r in rows).values()),
            argmax_acc=sum((x > 0) == y for x, y in d) / len(d),
            share_heads=sum(x > 0 for x, _ in d) / len(d),
            auroc=round(auroc(d), 4),
            median_threshold_acc=round(sum((x > med) == y for x, y in d) / len(d), 3),
            distinct_scores=len({x for x, _ in d}),
        )
        for st in ["heads", "tails"]:
            res[f"auroc_start_{st}"] = round(auroc([(r["logit_heads"] - r["logit_tails"], r["answer"] == "heads")
                                                    for r in rows if r["start"] == st]), 4)
        if k == 8:
            rng = random.Random(0)
            ys = [y for _, y in d]
            null = []
            for _ in range(500):
                rng.shuffle(ys)
                null.append(auroc([(x, y) for (x, _), y in zip(d, ys)]))
            null.sort()
            res["label_shuffle_auroc_95"] = [round(null[12], 4), round(null[487], 4)]
        out[k] = res
    return out


def audit():
    key = lambda r: (r["start"], tuple(r["acts"]))
    out = {}
    for k in [1, 2, 3, 4, 8]:
        sp = {s: load(HERE / f"inputs/{s}_k{k}.jsonl") for s in ["train", "calib", "eval", "necessity"]}
        bad = sum(((r["start"] == "heads") ^ (sum(r["acts"]) % 2 == 1)) != (r["answer"] == "heads")
                  for s in sp.values() for r in s)
        tr = set(map(key, sp["train"]))
        o = dict(label_errors=bad, possible_prompts=2 * 2 ** k, train_distinct=len(tr))
        for s in ["calib", "eval", "necessity"]:
            ks = list(map(key, sp[s]))
            o[f"{s}_in_train"] = sum(x in tr for x in ks)
            o[f"{s}_distinct"] = len(set(ks))
            o[f"{s}_2x2"] = sorted(collections.Counter((r["start"], r["answer"]) for r in sp[s]).values())
        o["eval_in_calib"] = len(set(map(key, sp["eval"])) & set(map(key, sp["calib"])))
        out[k] = o
    return out


if __name__ == "__main__":
    res = {"step0": step0(), "audit": audit()}
    print(json.dumps(res, indent=1))
    (Path(__file__).parent / "vera_step0.json").write_text(json.dumps(res, indent=1))
