"""Vera: rebuild base_steer at (15,1.5) and the XSTest arm (D5), build labels + my judge. Own code."""
import json
from pathlib import Path
from vera_judge import judge_many
R = Path(__file__).resolve().parents[2] / "kran-research/projects/refusal-addback/runs"
M = {"REFUSE": "refusal", "COMPLY": "compliance", "INCOHERENT": "incoherent"}
bs = [r for r in map(json.loads, open(R / "test_arms.jsonl")) if r["arm"] == "base_steer"]
for r, l in zip(bs, judge_many([(r["prompt"], r["response"]) for r in bs])): r["vj"] = M.get(l, l)
for sp in ("harmful", "harmless"):
    x = [r for r in bs if r["split"] == sp]
    print(f"base model + add-back (15,1.5) {sp}: build {sum(r['judge_label']=='refusal' for r in x)}/{len(x)} refusal, "
          f"{sum(r['judge_label']=='incoherent' for r in x)} incoh; mine {sum(r['vj']=='refusal' for r in x)}/{len(x)} refusal, "
          f"{sum(r['vj']=='incoherent' for r in x)} incoh")
b = [json.loads(l) for l in open(R / "base/xstest.judged.jsonl")]
s = [json.loads(l) for l in open(R / "strip_s0/xstest.judged.jsonl")]
ok = {r["i"] for r in b if r["judge"] == "compliance"}
for name, sel in (("stripped", lambda r: r["layer"] is None), ("stripped+add-back(15,1.5)", lambda r: r["layer"] == 15)):
    x = [r for r in s if sel(r)]
    bad = [r for r in x if r["judge"] in ("refusal", "incoherent")]
    print(f"XSTest {name}: refused/garbled {len(bad)}/{len(x)} = {len(bad)/len(x):.3f}; new vs base-answered {sum(r['i'] in ok for r in bad)}/{len(ok)}")
print(f"XSTest base: refused {sum(r['judge']=='refusal' for r in b)}/{len(b)}")
