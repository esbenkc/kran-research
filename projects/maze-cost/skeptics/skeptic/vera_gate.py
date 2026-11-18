"""Vera: rebuild the gate count from transcripts: the answer string must appear in an observation of a /r/ page."""
import json,glob,sys,re
for m in sys.argv[1:]:
    ok=0;n=0;rows=[]
    for f in sorted(glob.glob(f"{m}/raw/visible/*.json")):
        d=json.load(open(f));n+=1
        ans=(d.get("answer") or "").strip()
        obs=[t["text"] for t in d["transcript"] if t["role"]=="observation"]
        nums=re.findall(r"\$?\d[\d,]*(?:\.\d+)?",ans)
        hit=bool(nums) and any(all(x in o for x in nums) for o in obs if "/r/" in o or "2025" in o)
        ok+=hit; rows.append((f.split('/')[-1],ans[:40],hit,d.get("correct")))
    print(m.split('/')[-1],f"{ok}/{n}",[r for r in rows if r[2]!=bool(r[3])])
