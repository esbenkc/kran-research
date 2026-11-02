"""Re-check after D6 (askers counted per person). For the new top 10 in data/top50.json, count the people
whose snippet I judged a true ask for the group's experiment (judgements.jsonl, strict and lenient), next to
the listed n_askers. Also flag same-forum name collisions (two user_ids, one normalised name, one forum)."""
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
top = json.load(open(HERE.parents[1] / "data" / "top50.json"))


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


# Same-experiment calls for new top-10 members that I had only judged in the random sample.
SAME_FROM_RANDOM = {"ea-vcbHaqotqvM3c9vTH-h0": "yes"}
J = {}
for line in open(HERE / "judgements.jsonl"):
    j = json.loads(line)
    if j["set"] == "random" and j["sid"] in SAME_FROM_RANDOM:
        J.setdefault(j["sid"], {**j, "same_experiment": SAME_FROM_RANDOM[j["sid"]]})
    elif j["set"] != "random":
        J[j["sid"]] = j
missing = [a["sid"] for g in top[:10] for a in g["asks"] if a["sid"] not in J]
print("top-10 members without my judgement:", missing)
members = [J[a["sid"]] for g in top[:10] for a in g["asks"]]
for strict in (True, False):
    tag = "strict" if strict else "lenient"
    print(f"--- {tag}: true ask for group's experiment",
          wilson(sum(ok(j["ask"], strict) and ok(j["same_experiment"], strict) for j in members), len(members)))
    cells = []
    for g in top[:10]:
        ppl = {norm(a["author"]) for a in g["asks"]
               if ok(J[a["sid"]]["ask"], strict) and ok(J[a["sid"]]["same_experiment"], strict)}
        cells.append(f"r{g['rank']}: {len(ppl)} (listed {g['n_askers']})")
    print("   ", ", ".join(cells))

clash = []
for g in top:
    seen = defaultdict(set)
    for a in g["asks"]:
        seen[(a["forum"], norm(a["author"]))].add(a["user_id"])
    clash += [(g["rank"], k) for k, v in seen.items() if len(v) > 1]
print("same-forum name collisions in top 50:", clash)
