# critic2 (openai/gpt-5.5) rebuild, tool call 25 of its session, verbatim (was: python3 -c '...').
# Run from /Users/esben/work/oracle-1123-gated-refusal-cont (paths are relative to the work dir).
import json, pathlib, statistics; P=pathlib.Path("kran-research/projects/gated-refusal-8b/data"); base={};
for line in (P/"base/gen_test.judged.jsonl").open():
 r=json.loads(line); base[(r["kind"],r["i"])] = r["judge"]
B=sum(base[("harmful",i)]=="refusal" for i in range(200)); ok=[i for i in range(200) if base[("harmless",i)]=="compliance"]
out=[]; rgs=[]
for s in range(3):
 d=P/f"strip_s{s}"; gate=json.loads((d/"gate.json").read_text())["test_flags"]; rows={}
 for line in (d/"gen_test.judged.jsonl").open():
  r=json.loads(line); rows[(r["set"],r["kind"],r["i"])] = r["judge"]
 def lab(arm,k,i):
  f=gate[k][i]
  st={"S":"none","G":"push" if f else "none","RG0":"rand0" if f else "none","RG1":"rand1" if f else "none","U":"push","C":"CANNED" if f else "none"}[arm]
  return "refusal" if st=="CANNED" else rows[(st,k,i)]
 row={"seed":s,"flags_h":sum(gate["harmful"]),"flags_x":sum(gate["harmless"])}
 for arm in ["S","G","RG0","RG1","U","C"]:
  hr=sum(lab(arm,"harmful",i)=="refusal" for i in range(200)); nh=sum(lab(arm,"harmless",i) in ("refusal","incoherent") for i in ok); row[arm]=(hr,round(hr/B,4),nh)
 for arm in ["RG0","RG1"]: rgs.append(row[arm][1])
 out.append(row)
print("B",B,"base_harmless_answered",len(ok)); print("rows",out); print("G_mean_restore",round(statistics.mean([x["G"][1] for x in out]),4),"G_new_sum",sum(x["G"][2] for x in out),"G_new_mean",round(statistics.mean([x["G"][2] for x in out]),3),"RG_mean",round(statistics.mean(rgs),4)); res=json.load(open("kran-research/projects/gated-refusal-8b/research_checks/judge_resample.json")); print("resample",[r["gated_only"]["sum"] for r in res["replicates"]],"passing",res["passing_gated_only"],"i3",[res["key_items_i3_i115"][f"{s}_i3"]["labels"].count("refusal") for s in range(3)])
