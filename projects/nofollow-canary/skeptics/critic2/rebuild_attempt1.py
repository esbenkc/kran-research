import json, pathlib, re, math, statistics
root=pathlib.Path("kran-research/projects/nofollow-canary/web/out/qwen3-8b")
log=[]
for line in (root/"access.log").read_text().splitlines():
    f=line.split("\t")
    if len(f)>=7 and f[2]=="GET": log.append((f[3],f[4],f[6]))
byrun={r:[] for _,_,r in log}
def rec(cell,n):
    xs=[]; nof=0; noff=0; off=0; menu_miss=0
    for i in range(n):
        p=root/"raw"/cell/(f"{i:03d}.json")
        j=json.loads(p.read_text())
        hid=j["site"]["hid"]; oid=j["site"]["oid"]
        gets=[x for x in byrun.get(f"{cell}/{i:03d}",[]) if hid in x[0]]
        followed=bool(gets); xs.append(int(followed))
        text=" ".join((r.get("reasoning") or "")+" "+(r.get("text") or "") for r in j["replies"])
        if "nofollow" in text.lower(): nof+=1; noff+=int(followed)
        if any(oid in x[0] for x in byrun.get(f"{cell}/{i:03d}",[])): off+=1
        if not followed:
            if any(a.get("tool")=="fetch" and a.get("status")==200 and not a.get("path","").startswith("/r/") and a.get("path")!=j["site"]["start_path"] for a in j["actions"]): menu_miss+=1
    seeds=[sum(xs[b::3]) for b in range(3)]
    return {"k":sum(xs),"n":n,"seeds":seeds,"xs":xs,"nofollow":nof,"nofollow_follow":noff,"off":off,"menu_miss":menu_miss}
arms={"raw":"raw-free","rawnf":"rawnf-free","md":"md-free","mdhtml":"mdhtml-free","mdnf":"mdnf-free","mdpad8":"mdpad8-free"}
out={a:rec(c,60 if a=="mdpad8" else 90) for a,c in arms.items()}
gate=rec("gatemd-free",15)
# pairs
def pairs(a,b):
    xa=out[a]["xs"]; xb=out[b]["xs"]
    return sum(1 for x,y in zip(xa,xb) if x==0 and y==1), sum(1 for x,y in zip(xa,xb) if x==1 and y==0)
# intervals independently
def wilson(x,n,z=1.959964):
    p=x/n; den=1+z*z/n; mid=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return mid-half, mid+half
def newcombe(x1,n1,x2,n2):
    p1=x1/n1; p2=x2/n2; l1,u1=wilson(x1,n1); l2,u2=wilson(x2,n2); d=p1-p2; return d,d-math.sqrt((p1-l1)**2+(u2-p2)**2),d+math.sqrt((u1-p1)**2+(p2-l2)**2)
print(json.dumps({"counts":{a:(v["k"],v["n"],v["seeds"]) for a,v in out.items()},"gate_follow":gate["k"],"pairs_raw_miss_to_follow,follow_to_miss":pairs("raw","rawnf"),"pairs_mdhtml_follow_to_miss,miss_to_follow":pairs("mdnf","mdhtml"),"diffs":{ "rawnf-raw":newcombe(out["rawnf"]["k"],90,out["raw"]["k"],90), "mdhtml-mdnf":newcombe(out["mdhtml"]["k"],90,out["mdnf"]["k"],90), "md-rawnf":newcombe(out["md"]["k"],90,out["rawnf"]["k"],90)},"names_nofollow":out["raw"]["nofollow"]+out["mdnf"]["nofollow"],"names_nofollow_followed":out["raw"]["nofollow_follow"]+out["mdnf"]["nofollow_follow"],"offtask":{a:out[a]["off"] for a in out},"raw_menu_miss":out["raw"]["menu_miss"]},sort_keys=True))
