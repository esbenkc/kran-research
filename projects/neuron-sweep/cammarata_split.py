"""Cammarata split of the steering runs at 10 times max: none / one job (incl. related offshoots) / unrelated / broken.

    python3 projects/neuron-sweep/cammarata_split.py

Inputs: blind hand labels of pools e + f (200 runs at 10x: 100 neuron pushes, 100 random-direction pushes, arm
hidden from both labellers). Labeller 1 = data/hand_labels_cammarata.json (lead), labeller 2 =
data/hand_labels_cammarata_l2.json (fresh agent that saw only the rubric and the texts).
"Agreed" = the label both gave; rows where they differ count as disputed. Writes data/cammarata_split.json.
"""
import json
import math
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
K = ["none", "one", "unrelated", "broken"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]; sum of probabilities of tables as or less likely than the observed one."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    obs = p(a)
    return sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if p(x) <= obs * (1 + 1e-9))


def kappa(pairs):
    c = Counter(pairs)
    n = sum(c.values())
    po = sum(c[(k, k)] for k in K) / n
    pe = sum(sum(c[(k, x)] for x in K) * sum(c[(x, k)] for x in K) for k in K) / n / n
    return round((po - pe) / (1 - pe), 3), round(po, 3)


def main():
    rows = [json.loads(l) for f in ("hand_pool_e.jsonl", "hand_pool_f.jsonl") for l in open(D / f)]
    l1 = json.load(open(D / "hand_labels_cammarata.json"))
    l2 = json.load(open(D / "hand_labels_cammarata_l2.json"))
    out = {"n_runs": len(rows)}
    k, po = kappa([(l1[r["id"]], l2[r["id"]]) for r in rows])
    out["agreement"] = {"kappa": k, "raw": po,
                        "confusion_l1_rows_l2_cols": {a: [sum(1 for r in rows if l1[r["id"]] == a and l2[r["id"]] == b)
                                                          for b in K] for a in K}}
    for name, lab in (("labeller1", lambda r: l1[r["id"]]), ("labeller2", lambda r: l2[r["id"]]),
                      ("agreed", lambda r: l1[r["id"]] if l1[r["id"]] == l2[r["id"]] else "disputed"),
                      ("either_unrelated", lambda r: "unrelated" if "unrelated" in (l1[r["id"]], l2[r["id"]]) else "other")):
        res = {}
        for arm in ("neuron", "direction"):
            A = [lab(r) for r in rows if r["kind"] == arm]
            c = Counter(A)
            eff = sum(c[x] for x in ("one", "unrelated", "broken"))
            res[arm] = {"n": len(A), "counts": dict(c),
                        "unrelated_share": round(c["unrelated"] / len(A), 3), "unrelated_ci": wilson(c["unrelated"], len(A)),
                        "unrelated_of_effects": round(c["unrelated"] / eff, 3) if eff else None,
                        "one_of_effects": round(c["one"] / eff, 3) if eff else None}
        a, b = res["neuron"]["counts"].get("unrelated", 0), res["neuron"]["n"] - res["neuron"]["counts"].get("unrelated", 0)
        cc, d = res["direction"]["counts"].get("unrelated", 0), res["direction"]["n"] - res["direction"]["counts"].get("unrelated", 0)
        res["fisher_p_unrelated_neuron_vs_direction"] = round(fisher_two_sided(a, b, cc, d), 3)
        out[name] = res
    print(json.dumps(out, indent=1))
    (D / "cammarata_split.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
