"""Validate the local generating judge (judge_think.py) against the blind hand labels (hand_labels_cammarata.json).

    python3 projects/neuron-sweep/validate_think.py data/think_val.jsonl

Splits: dev = pools a + dev_x10 (rubric examples were written with these in view), held-out = pools b + c,
pool_e = new blind 10x pool labelled on 2026-10-17 before any judge saw it. Prints confusion (hand rows x judge
columns), accuracy, Cohen's kappa, effect-vs-none accuracy, and unrelated precision/recall.
"""
import json
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
K = ["none", "one", "unrelated", "broken"]
lab = json.load(open(D / "hand_labels_cammarata.json"))


def summary(pairs):
    c = Counter(pairs)
    n = sum(c.values())
    po = sum(c[(k, k)] for k in K) / n
    pe = sum(sum(c[(k, x)] for x in K) * sum(c[(x, k)] for x in K) for k in K) / n / n
    eff = sum((h != "none") == (j != "none") for h, j in pairs) / n
    return {"n": n, "acc": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3), "effect_acc": round(eff, 3),
            "unrelated_hand": sum(c[("unrelated", x)] for x in K), "unrelated_judge": sum(c[(x, "unrelated")] for x in K),
            "unrelated_both": c[("unrelated", "unrelated")],
            "unparsed": sum(1 for _, j in pairs if j not in K),
            "confusion_hand_rows_judge_cols": {k: [c[(k, x)] for x in K] for k in K}}


def main(files):
    rows = [json.loads(l) for f in files for l in open(Path(f) if Path(f).exists() else D.parent / f)]
    out = {}
    for split, pre in (("dev", ("a", "d")), ("heldout", ("b", "c")), ("pool_e", ("e",)),
                       ("heldout+e", ("b", "c", "e")), ("all", ("a", "b", "c", "d", "e"))):
        pairs = [(lab[r["id"]], r["label"]) for r in rows if r.get("id") and r["id"][0] in pre and r["id"] in lab]
        if pairs:
            out[split] = summary(pairs)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(sys.argv[1:])
    (D / "validation" / "cammarata_local_14b_generating.json").write_text(json.dumps(res, indent=1))
