"""D4 check: the API rejects `temperature` for claude-haiku-4-5 (400 'Extra inputs are not permitted'), so the judge
samples at its default. Test-retest: label the headline cells a SECOND time (no cache), same model/prompt, and report
agreement and the split recomputed on the second labels.
Usage: with-model-keys uv run --no-project --with anthropic python retest.py  -> data/judge_retest/*.json"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import judge as J

P = Path(__file__).parent
OUT = P / "data/judge_retest"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ["base", "G@30", "G_on@30", "G_off@30", "G_offK@30"]


def fresh(prompt):
    r = J.CLIENT.messages.create(model=J.MODEL, max_tokens=5, system=J.SYSTEM, messages=[{"role": "user", "content": prompt}])
    J.USAGE["in"] += r.usage.input_tokens
    J.USAGE["out"] += r.usage.output_tokens
    out = "".join(b.text for b in r.content if getattr(b, "type", None) == "text").strip().upper()
    for lab in ("REFUSE", "COMPLY", "INCOHERENT"):
        if lab in out:
            return lab
    return "UNK:" + out[:20]


res = {}
for c in CELLS:
    f = OUT / f"{c}.json"
    if not f.exists():
        rows = [json.loads(l) for l in open(P / "data/runs" / f"{c}.jsonl")]
        with ThreadPoolExecutor(8) as ex:
            labs = list(ex.map(lambda r: fresh(J.TMPL.format(p=r["prompt"], r=r["reply"])), rows))
        f.write_text(json.dumps({"cell": c, "labels": labs}))
    a = json.loads((P / "data/judge" / f"{c}.json").read_text())["labels"]
    b = json.loads(f.read_text())["labels"]
    res[c] = {"agree": sum(x == y for x, y in zip(a, b)) / len(a), "comply_1": a.count("COMPLY") / len(a),
              "comply_2": b.count("COMPLY") / len(b)}
    print(c, res[c], flush=True)
b0 = res["base"]["comply_2"]
res["split_retest"] = {u: 100 * (res[f"{u}@30"]["comply_2"] - b0) for u in ("G", "G_on", "G_off", "G_offK")}
res["off_ratio_retest"] = res["split_retest"]["G_off"] / res["split_retest"]["G"]
usd = J.USAGE["in"] / 1e6 + J.USAGE["out"] / 1e6 * 5
res["usd"] = usd
(OUT / "summary.json").write_text(json.dumps(res, indent=1))
with open(P / "data/judge/spend.log", "a") as fh:
    fh.write(f"retest n={len(CELLS) * 200} in={J.USAGE['in']} out={J.USAGE['out']} usd={usd:.4f}\n")
print(json.dumps(res, indent=1))
