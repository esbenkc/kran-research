"""Summarize N1 (random-context floor) from skeptics/skeptic2/out/n1_judgments.jsonl with the frozen one-job rule."""
import json
from collections import Counter
from math import comb
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
R = [json.loads(l) for l in open(OUT / "n1_judgments.jsonl")]
res = {}
for j in sorted({r["judge"] for r in R}):
    ok = [r for r in R if r["judge"] == j and r["label_ok"]]
    fit = [r for r in ok if r["parsed"]["main_type"] != "NONE" and r["parsed"]["main_count"] >= 15]
    one = [r for r in fit if (r["hits"] or 0) >= 4]
    hits = [r["hits"] for r in ok if r["hits"] is not None]
    res[j] = {"n": len(ok), "types": dict(Counter(r["parsed"]["main_type"] for r in ok)),
              "fit>=15": len(fit), "hits>=4": sum(h >= 4 for h in hits), "n_detect": len(hits), "one_job": len(one),
              "mean_fit": sum(r["parsed"]["main_count"] for r in ok) / len(ok)}
# chance that 5 random picks out of 10 hit >= 4 of the 5 true ones (hypergeometric)
res["chance_hits>=4"] = (comb(5, 4) * comb(5, 1) + 1) / comb(10, 5)
res["one_job_floor_both_judges"] = sum(v["one_job"] for k, v in res.items() if isinstance(v, dict)) / sum(
    v["n"] for k, v in res.items() if isinstance(v, dict))
json.dump(res, open(OUT / "n1_summary.json", "w"), indent=1)
print(json.dumps(res, indent=1))
