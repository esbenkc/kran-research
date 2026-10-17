"""Validate a local judge (judge_big.py output) against the blind Cammarata-schema labels.

    cd kran-research && uv run python projects/neuron-sweep/validate_big.py data/big_val.jsonl

Pools e+f (200 runs at 10x, two labellers) are the main test; pools a/b/c have labeller 1 only.
Pass rule (posted in the room before the first judge row): on e+f, kappa >= 0.55 against EACH labeller, and the judge
calls 'unrelated' on <= 5% of the runs that both labellers call 'none' or 'one'.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P / "data"
CLS = ["none", "one", "unrelated", "broken"]


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in CLS) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def confusion(a: list[str], b: list[str]) -> dict:
    return {x: [sum(1 for p, q in zip(a, b) if p == x and q == y) for y in CLS] for x in CLS}


def main():
    judged = {}
    for l in open(P / sys.argv[1]):
        r = json.loads(l)
        judged[r["id"]] = r["label"]
    l1 = json.loads((DATA / "hand_labels_cammarata.json").read_text())
    l2 = json.loads((DATA / "hand_labels_cammarata_l2.json").read_text())
    ef = [i for i in l2 if i in judged]
    abc = [i for i in l1 if i[0] in "abc" and i in judged]
    out = {"n_ef": len(ef), "n_abc": len(abc), "unparsed": sum(v == "unparsed" for v in judged.values())}
    j = lambda ids: [judged[i] if judged[i] in CLS else "none" for i in ids]
    out["ef_vs_l1"] = {"kappa": round(kappa(j(ef), [l1[i] for i in ef]), 3),
                       "raw": round(sum(x == l1[i] for x, i in zip(j(ef), ef)) / len(ef), 3),
                       "confusion_judge_rows_l1_cols": confusion(j(ef), [l1[i] for i in ef])}
    out["ef_vs_l2"] = {"kappa": round(kappa(j(ef), [l2[i] for i in ef]), 3),
                       "raw": round(sum(x == l2[i] for x, i in zip(j(ef), ef)) / len(ef), 3),
                       "confusion_judge_rows_l2_cols": confusion(j(ef), [l2[i] for i in ef])}
    out["l1_vs_l2_ef"] = round(kappa([l1[i] for i in ef], [l2[i] for i in ef]), 3)
    plain = [i for i in ef if l1[i] in ("none", "one") and l2[i] in ("none", "one")]
    out["false_unrelated_on_plain"] = {"n": len(plain), "called_unrelated": sum(judged[i] == "unrelated" for i in plain)}
    out["false_unrelated_rate"] = round(out["false_unrelated_on_plain"]["called_unrelated"] / max(1, len(plain)), 3)
    unrel = [i for i in ef if "unrelated" in (l1[i], l2[i])]
    out["unrelated_recall_either"] = {"n": len(unrel), "judge_unrelated": sum(judged[i] == "unrelated" for i in unrel)}
    if abc:
        out["abc_vs_l1"] = {"kappa": round(kappa(j(abc), [l1[i] for i in abc]), 3),
                            "confusion_judge_rows_l1_cols": confusion(j(abc), [l1[i] for i in abc])}
    out["pass"] = bool(out["ef_vs_l1"]["kappa"] >= 0.55 and out["ef_vs_l2"]["kappa"] >= 0.55
                       and out["false_unrelated_rate"] <= 0.05)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
