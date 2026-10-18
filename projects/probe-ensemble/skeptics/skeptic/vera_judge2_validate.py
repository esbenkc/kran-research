"""Validate vera_judge2 rubric on DolusChat reference replies (known labels) for the 100 eval prompts."""
import json
from concurrent.futures import ThreadPoolExecutor
import vera_judge2 as J
rows = []
for pid, p in J.PROMPTS.items():
    rows.append(dict(run="ref", round=-1, prompt_id=pid, sample=0, response=p["dolus_truthful"], judge_deceptive=0))
    rows.append(dict(run="ref", round=-1, prompt_id=pid, sample=1, response=p["dolus_deceptive"], judge_deceptive=1))
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(J.judge, rows))
ok = sum((r["j2"] == "DECEPTIVE") == bool(r["j1"]) for r in res if r["j2"] != "OTHER")
n = sum(r["j2"] != "OTHER" for r in res)
print(f"reference accuracy {ok}/{n} = {ok/n:.3f}; OTHER {len(res)-n}")
for lab in (0, 1):
    sub = [r for r in res if r["j1"] == lab]
    print(f" true label {lab}: called DECEPTIVE {sum(r['j2']=='DECEPTIVE' for r in sub)}/{len(sub)}")
json.dump(res, open("judge2_reference.json", "w"))
