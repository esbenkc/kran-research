"""T7 agreement: Iris hand labels (labels.jsonl) vs gpt-4.1-mini judge and string match (key.json).
Hand REFUSE/COMPLY/INCOHERENT; judge refusal/compliance/incoherent; str_ref bool.
Binary comparison: refused = REFUSE (hand) / 'refusal' (judge) / str_ref True. Incoherent counted separately.
"""
import json
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent
key = json.loads((D / "key.json").read_text())["key"]
lab = {r["uid"]: r for r in map(json.loads, (D / "labels.jsonl").open())}
truthy = lambda v: str(v).lower() == "true"


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


rows = []
for u, k in key.items():
    h = lab[u]["label"]
    if h == "SKIP":
        continue
    rows.append({"uid": u, "arm": k["arm"], "hand": h, "judge": k["judge"], "str": truthy(k["str_ref"]),
                 "hard": lab[u]["hard"]})
hand = [r["hand"] == "REFUSE" for r in rows]
judge = [r["judge"] == "refusal" for r in rows]
strm = [r["str"] for r in rows]
out = {"n": len(rows),
       "hand_vs_judge": dict(zip(("agree", "kappa"), kappa(hand, judge))),
       "hand_vs_string": dict(zip(("agree", "kappa"), kappa(hand, strm))),
       "judge_incoherent": sum(r["judge"] == "incoherent" for r in rows),
       "hand_incoherent": sum(r["hand"] == "INCOHERENT" for r in rows),
       "per_arm": {}, "disagreements_judge": []}
by = defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)
for arm, rs in by.items():
    out["per_arm"][arm] = {"n": len(rs), "hand_refused": sum(r["hand"] == "REFUSE" for r in rs),
                           "judge_refused": sum(r["judge"] == "refusal" for r in rs),
                           "judge_incoherent": sum(r["judge"] == "incoherent" for r in rs),
                           "string_refused": sum(r["str"] for r in rs),
                           "agree_hand_judge": sum((r["hand"] == "REFUSE") == (r["judge"] == "refusal") for r in rs)}
for r in rows:
    if (r["hand"] == "REFUSE") != (r["judge"] == "refusal"):
        out["disagreements_judge"].append({k: r[k] for k in ("uid", "arm", "hand", "judge", "hard")})
(D / "agreement.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
