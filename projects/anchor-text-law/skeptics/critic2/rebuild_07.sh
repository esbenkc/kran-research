# critic2 command at 2026-11-23T10:24:54.737Z (verbatim from the session)
python3 -c 'import json, pathlib; base=pathlib.Path("kran-research/projects/anchor-text-law/web/out");
for arm,model in [("A","gpt-oss-20b"),("B","qwen3-8b")]:
  for view in ["raw","md","bare","anchor"]:
    fs=[p for p in sorted((base/arm/model/"raw"/(view+"-free")).glob("*.json")) if ".crash" not in p.name]; n=len(fs); r=f=off=stale=correct=0
    for p in fs:
      d=json.load(open(p)); hid=d["site"]["hid"]; oid=d["site"].get("oid",""); paths=[a.get("path","") for a in d.get("actions",[]) if a.get("tool")=="fetch"]
      r += bool(d.get("exposed")); f += any(hid in x for x in paths); off += any(oid and oid in x for x in paths); stale += (d.get("answer")==d["site"].get("stale")); correct += bool(d.get("correct"))
    print(arm,view,"n",n,"reach",r,"follow",f,"off",off,"stale",stale,"correct",correct)
  for view in ["raw","bare","anchor"]:
    fs=[p for p in sorted((base/arm/model/"raw"/(view+"-none")).glob("*.json")) if ".crash" not in p.name]; n=len(fs); r=f=correct=0
    for p in fs:
      d=json.load(open(p)); hid=d["site"]["hid"]; paths=[a.get("path","") for a in d.get("actions",[]) if a.get("tool")=="fetch"]
      r += bool(d.get("exposed")); f += any(hid in x for x in paths); correct += bool(d.get("correct"))
    print(arm,view+"-none","n",n,"reach",r,"follow",f,"correct",correct)'
