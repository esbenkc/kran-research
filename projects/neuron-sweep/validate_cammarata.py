"""Validate the local Cammarata judge (judge_cammarata.py) against the blind hand labels (hand_labels_cammarata.json).

    python3 projects/neuron-sweep/validate_cammarata.py data/cam_val.jsonl data/cam_e.jsonl

Threshold for "effect": P(A) below the 5th percentile of P(A) on the no-push arm (rule fixed before scoring).
Prints 4x4 confusion (hand rows x judge columns), accuracy, Cohen's kappa, per split (dev = pools a + dev_x10,
held-out = pools b + c, e = new blind 10x pool), and the same for plain argmax.
"""
import json
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
K = ["none", "one", "unrelated", "broken"]
lab = json.load(open(D / "hand_labels_cammarata.json"))


def threshold(rows):
    pa = sorted(r["p_A"] for r in rows if r["kind"] == "none" and not r.get("id"))
    return pa[int(0.05 * len(pa))]


def label(r, t):
    if t is None:
        return K[max(range(4), key=lambda i: r["p_" + "ABCD"[i]])]
    if r["p_A"] >= t:
        return "none"
    return K[1 + max(range(3), key=lambda i: r["p_" + "BCD"[i]])]


def summary(pairs):
    c = Counter(pairs)
    n = sum(c.values())
    po = sum(c[(k, k)] for k in K) / n
    pe = sum(sum(c[(k, x)] for x in K) * sum(c[(x, k)] for x in K) for k in K) / n / n
    eff = Counter(((h != "none"), (j != "none")) for h, j in pairs)
    return {"n": n, "acc": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3),
            "effect_acc": round((eff[(True, True)] + eff[(False, False)]) / n, 3),
            "confusion_hand_rows_judge_cols": {k: [c[(k, x)] for x in K] for k in K}}


def main(files):
    rows = [json.loads(l) for f in files for l in open(Path(f) if Path(f).exists() else D.parent / f)]
    t = threshold(rows)
    out = {"threshold_pA": t}
    for rule, tt in (("calibrated", t), ("argmax", None)):
        for split, pre in (("dev", ("a", "d")), ("heldout", ("b", "c")), ("pool_e", ("e",)), ("all", ("a", "b", "c", "d", "e"))):
            pairs = [(lab[r["id"]], label(r, tt)) for r in rows if r.get("id") and r["id"][0] in pre and r["id"] in lab]
            if pairs:
                out[f"{rule}/{split}"] = summary(pairs)
        npush = [label(r, tt) for r in rows if r["kind"] == "none" and not r.get("id")]
        out[f"{rule}/nopush_rates"] = {k: round(v / len(npush), 3) for k, v in Counter(npush).items()}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(sys.argv[1:])
    (D / "validation" / "cammarata_local_14b.json").write_text(json.dumps(res, indent=1))
