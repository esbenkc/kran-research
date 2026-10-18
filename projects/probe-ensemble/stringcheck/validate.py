"""Agreement of (a) the string check and (b) the gpt-4.1-mini judge with Iris's blind hand labels (60 answers).
Only HONEST/DECEPTIVE are compared; judge OTHER rows are reported and dropped."""
import json

import numpy as np

from string_check import THRESHOLD, fact_score

blind = {o["n"]: o for o in map(json.loads, open("blind.jsonl"))}
key = {k["n"]: k for k in map(json.loads, open("key.jsonl"))}
hand = {h["n"]: h for h in map(json.loads, open("hand_labels.jsonl"))}


def rate(pairs):
    return sum(a == b for a, b in pairs) / len(pairs), len(pairs)


rows = []
for n in sorted(blind):
    b, k, h = blind[n], key[n], hand[n]
    sc = fact_score(b["fact"], b["query"], b["response"])
    rows.append({"n": n, "stratum": k["stratum"], "hand": h["hand"], "borderline": h["borderline"], "judge": k["judge"],
                 "score": round(sc, 3), "string": "HONEST" if sc >= THRESHOLD else "DECEPTIVE"})
json.dump(rows, open("validation_rows.json", "w"), indent=0)

out = {"threshold": THRESHOLD, "n": len(rows)}
J = [r for r in rows if r["judge"] in ("HONEST", "DECEPTIVE")]
out["judge_other_dropped"] = len(rows) - len(J)
out["string_vs_hand"] = rate([(r["string"], r["hand"]) for r in rows])
out["judge_vs_hand"] = rate([(r["judge"], r["hand"]) for r in J])
out["string_vs_judge"] = rate([(r["string"], r["judge"]) for r in J])
nb = [r for r in rows if not r["borderline"]]
out["string_vs_hand_clear_only"] = rate([(r["string"], r["hand"]) for r in nb])
out["judge_vs_hand_clear_only"] = rate([(r["judge"], r["hand"]) for r in nb if r["judge"] in ("HONEST", "DECEPTIVE")])
for s in ("r0", "lam0_r8", "lam1_r8", "lam3_r8"):
    R = [r for r in rows if r["stratum"] == s]
    out[f"{s}"] = {"string_vs_hand": rate([(r["string"], r["hand"]) for r in R]),
                   "judge_vs_hand": rate([(r["judge"], r["hand"]) for r in R if r["judge"] in ("HONEST", "DECEPTIVE")]),
                   "hand_deceptive": sum(r["hand"] == "DECEPTIVE" for r in R) / len(R),
                   "string_deceptive": sum(r["string"] == "DECEPTIVE" for r in R) / len(R),
                   "judge_deceptive": sum(r["judge"] == "DECEPTIVE" for r in R) / len(R)}
# confusion of the string check against hand labels
out["string_confusion"] = {f"hand_{a}_string_{b}": sum(r["hand"] == a and r["string"] == b for r in rows)
                           for a in ("HONEST", "DECEPTIVE") for b in ("HONEST", "DECEPTIVE")}
json.dump(out, open("validation.json", "w"), indent=1)
print(json.dumps(out, indent=1))
