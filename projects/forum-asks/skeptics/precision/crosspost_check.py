"""Side check found while judging the top 10: n_askers counts forum accounts (user_id), and LW and EA
accounts differ, so one person who crossposts to both forums counts as two askers.
Recount n_askers in top50.json by normalised author name and show how the ranks would move.
Writes crosspost_check.json next to this file."""
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
top = json.load(open(ROOT / "data" / "top50.json"))


def norm(name):
    if not name:
        return None
    s = "".join(c for c in unicodedata.normalize("NFKC", name) if unicodedata.category(c)[0] in "LN")
    return s.lower()


rows = []
for g in top:
    people = {norm(a["author"]) or f"id:{a['user_id']}" for a in g["asks"]}
    dup = sorted({a["author"] for a in g["asks"]
                  if len({b["user_id"] for b in g["asks"] if norm(b["author"]) == norm(a["author"])}) > 1})
    same_text = len({re.sub(r"\W+", "", a["snippet"]) for a in g["asks"]}) < len(g["asks"])
    rows.append({"rank": g["rank"], "n_askers": g["n_askers"], "n_people": len(people), "double_counted": dup,
                 "identical_snippet_in_group": same_text, "span": g["span"], "karma": g["karma"],
                 "statement": g["statement"]})

hit = [r for r in rows if r["n_people"] < r["n_askers"]]
print(f"{len(hit)} of {len(rows)} top-50 groups count one person twice:")
for r in hit:
    print(f"  rank {r['rank']:2d}: n_askers {r['n_askers']} -> {r['n_people']} people  ({', '.join(r['double_counted'])})")
rerank = sorted(rows, key=lambda r: (-r["n_people"], -r["span"], -r["karma"]))
print("top 10 re-ranked by people:", [r["rank"] for r in rerank[:10]])
print("groups with only 1 person:", [r["rank"] for r in rows if r["n_people"] < 2])
json.dump({"groups": rows, "rerank_by_people": [r["rank"] for r in rerank]}, open(HERE / "crosspost_check.json", "w"),
          indent=1, ensure_ascii=False)
