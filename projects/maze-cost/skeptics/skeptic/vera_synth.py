"""Vera: synthetic raw data in Bruno's contract format to unit-check vera_check.py (h known)."""
import json, os, sys, random
out=sys.argv[1]; h=float(sys.argv[2]); n=int(sys.argv[3]); random.seed(5)
for arm in ("maze",):
    os.makedirs(f"{out}/raw/{arm}",exist_ok=True); os.makedirs(f"{out}/access/{arm}",exist_ok=True)
    for i in range(n):
        calls=[];acc=[];k=random.choice([1,2,3])
        for s in range(1,k):  # pre-entry fetches of site pages
            calls.append(dict(step=s,prompt_tokens=500,candidates_tokens=50,thoughts_tokens=0,action={"tool":"fetch","url":f"/p{s}.html"}))
            acc.append(dict(path=f"/p{s}.html",status=200,bytes=900,cpu_ns=300000))
        calls.append(dict(step=k,prompt_tokens=600,candidates_tokens=50,thoughts_tokens=0,action={"tool":"fetch","url":"/m/a0.html"}))
        acc.append(dict(path="/m/a0.html",status=200,bytes=1028,cpu_ns=300000))
        d=1
        for s in range(k+1,13):
            if random.random()<h:
                calls.append(dict(step=s,prompt_tokens=600+300*d,candidates_tokens=50,thoughts_tokens=0,action={"tool":"answer","text":"not found"}));break
            calls.append(dict(step=s,prompt_tokens=600+300*d,candidates_tokens=50,thoughts_tokens=0,action={"tool":"fetch","url":f"/m/a{d}.html"}))
            acc.append(dict(path=f"/m/a{d}.html",status=200,bytes=1028,cpu_ns=300000)); d+=1
        json.dump(dict(run_index=i,seed_block=i%3,calls=calls),open(f"{out}/raw/{arm}/{i}.json","w"))
        open(f"{out}/access/{arm}/{i}.jsonl","w").write("\n".join(json.dumps(a) for a in acc)+"\n")
