"""Smoke table: python smoke_table.py [raw_smoke] -> per model x cell rates of every say/trip variant, and a
dump of window thoughts for the hand audit (results_smoke/window_thoughts.jsonl, blind to trip: no trip field)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from scorer import score

ROOT = Path(__file__).resolve().parent
src = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "raw_smoke")
out_dir = ROOT / "results_smoke"
out_dir.mkdir(exist_ok=True)
rows = [score(json.loads(p.read_text())) for p in sorted(src.glob("*/*/*.jsonl"))]
(out_dir / "scores.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
blind = [{"model": r["model"], "cell": r["cell"], "seed": r["seed"], "thought": r.get("window_thought"),
          **{k: r.get(k) for k in r if k.startswith("says")}} for r in rows if r.get("window_reply") is not None]
(out_dir / "window_thoughts.jsonl").write_text("".join(json.dumps(b, sort_keys=True) + "\n" for b in blind))

cols = ["says_prereg_prereg", "says_clause_prereg", "says_prereg_strict", "says_clause_strict", "says_raw", "says_aware",
        "trip_any", "trip_presented"]
g = collections.defaultdict(list)
for r in rows:
    g[(r["model"], r["cell"])].append(r)
print(f"{'model':28s} {'cell':12s} n exp " + " ".join(c.replace('says_', 's_')[:13].rjust(13) for c in cols) + "  words")
for (m, c), rs in sorted(g.items()):
    ex = [r for r in rs if r.get("window_reply") is not None]
    vals = []
    for col in cols:
        base = ex if col.startswith("says") else rs
        vals.append(f"{sum(bool(r.get(col)) for r in base)}/{len(base)}".rjust(13))
    words = sum(r["thought_words_mean"] for r in rs) / len(rs)
    print(f"{m:28s} {c:12s} {len(rs)} {len(ex):3d} " + " ".join(vals) + f"  {words:.1f}")
# say x trip 2x2 per prompt arm (primary variants as currently set in scorer.PRIMARY)
for arm in ("off", "on"):
    t = collections.Counter((bool(r["says"]), bool(r["trip"])) for r in rows if r["prompt"] == arm and r.get("window_reply") is not None)
    print(f"prompt {arm}: says&trip {t[(True, True)]}  says&no-trip {t[(True, False)]}  no-say&trip {t[(False, True)]}  no-say&no-trip {t[(False, False)]}")
spent = sum(json.loads(l)["cost_usd_est"] for l in (ROOT / "raw" / "cost.jsonl").read_text().splitlines() if l.strip())
print(f"runs {len(rows)}  logged spend ${spent:.4f}  per run ${spent / max(1, len(rows)):.5f}")
