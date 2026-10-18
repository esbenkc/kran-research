"""Convert judge2_*.jsonl (my Claude-haiku labels) to eval-like rows for vera_check.py (label = judge_deceptive)."""
import json, sys
for f in sys.argv[1:]:
    out = f.replace(".jsonl", "_rows.jsonl")
    with open(out, "w") as fo:
        for l in open(f):
            r = json.loads(l)
            fo.write(json.dumps(dict(run=r["run"], round=r["round"], prompt_id=r["prompt_id"], sample=r["sample"],
                                     judge=r["j2"], judge_deceptive=int(r["j2"] == "DECEPTIVE"))) + "\n")
    print(out)
