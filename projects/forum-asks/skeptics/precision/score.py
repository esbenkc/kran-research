"""Score judgements.jsonl (my hand labels). Strict: borderline/partial/unclear count as no. Lenient: as yes."""
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = [json.loads(l) for l in open(HERE / "judgements.jsonl")]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.0f}% (95% CI {100 * (c - h):.0f}-{100 * (c + h):.0f}%)"


def ok(v, strict):
    return v == "yes" if strict else v != "no"


def norm(name):
    return "".join(c for c in unicodedata.normalize("NFKC", name or "") if unicodedata.category(c)[0] in "LN").lower()


rand = [j for j in J if j["set"] == "random"]
top = [j for j in J if j["set"] == "top10"]
for strict in (True, False):
    tag = "strict" if strict else "lenient"
    print(f"--- {tag}")
    print("random: real ask           ", wilson(sum(ok(j["ask"], strict) for j in rand), len(rand)))
    print("random: real ask AND AI    ", wilson(sum(ok(j["ask"], strict) and ok(j["ai"], strict) for j in rand), len(rand)))
    print("random: statement faithful ", wilson(sum(ok(j["faithful"], strict) for j in rand), len(rand)))
    print("top10: ask for group's exp ", wilson(sum(ok(j["ask"], strict) and ok(j["same_experiment"], strict) for j in top), len(top)))
    per = defaultdict(lambda: [set(), set()])
    for j in top:
        per[j["rank"]][0].add(norm(j["author"]))
        if ok(j["ask"], strict) and ok(j["same_experiment"], strict):
            per[j["rank"]][1].add(norm(j["author"]))
    print("top10: people with a true ask for the group's experiment, by rank (listed n_askers in top50.json):")
    claimed = {g["rank"]: g["n_askers"] for g in json.load(open(HERE.parents[1] / "data" / "top50.json"))[:10]}
    print("   ", ", ".join(f"r{r}: {len(v[1])} (listed {claimed[r]})" for r, v in sorted(per.items())))
