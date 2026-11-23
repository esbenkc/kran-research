# critic2 (openai/gpt-5.5) rebuild commands, copied verbatim from its session transcript

Staged by Selma (write) from ~/odin-context/swarms/oracle-1123-audit-penalty-8b/sessions/critic2/*.jsonl; critic2 ended its session without committing receipts (spend 2.36 of 2.50).

```
python3 -c 'import json, pathlib; p=pathlib.Path("kran-research/projects/probe-switch-audit-penalty-8b/data/audit/ao_s0.json"); d=json.load(open(p)); print(d.keys()); print({k:type(v).__name__ for k,v in d.items() if k not in ("dev_ids","test_ids","per_prompt_lo","dev_removed","rows")}); print("per_prompt keys", d.get("per_prompt_lo",[{}])[0].keys() if d.get("per_prompt_lo") else None); print("rows?", type(d.get("rows")), len(d.get("rows",[])) if isinstance(d.get("rows"), list) else None)'
```

```
python3 -c 'import json, pathlib; d=json.load(open("kran-research/projects/probe-switch-audit-penalty-8b/data/audit/ao_s0.json")); print(type(d["per_prompt_lo"]).__name__); print(d["per_prompt_lo"].keys() if isinstance(d["per_prompt_lo"],dict) else None); print("headline", d["headline"]); print("test_cut", d["test_cut"].keys()); print("test_cut_sets keys", d["test_cut_sets"].keys()); print("test_base", d["test_base"]); print("cached_test", d["cached_test"]);'
```

```
python3 -c 'import json; d=json.load(open("kran-research/projects/probe-switch-audit-penalty-8b/data/audit/ao_s0.json")); pp=d["per_prompt_lo"]; print({k:(type(v).__name__, len(v) if hasattr(v,"__len__") else None) for k,v in pp.items()}); print(pp["unprobed"][:3]); print(pp["blind_k3"][:3]); print(d["test_cut"]["blind_k3"]);'
```

```
python3 -c '
import json, pathlib, statistics, math
base=pathlib.Path("kran-research/projects/probe-switch-audit-penalty-8b/data")
runs=["ao_s0","ao_s1","ao_s2","p100_s0","p100_s1","p100_s2","p100_s0_g2048","l0_s0","l0_s1"]
for name in runs:
    w=json.load(open(base/"whole"/(name+".json")))
    rows=w["rows"]
    by={}
    for r in rows:
        by.setdefault(r["prompt"],{})[r["probe"]]=r
    n=len(by)
    strict=argmax=fp32=0; pushes=[]; probe_ties=unprobe_ties=arg_tie=0; pA=uA=0; pB=uB=0
    for pr,pair in by.items():
        u=pair[0]; p=pair[1]
        s=(p["lo"]>0 and u["lo"]<0); strict+=s
        a=(p["top"]==p["a_id"] and u["top"]==u["b_id"]); argmax+=a
        f=(p["lo32"]>0 and u["lo32"]<0); fp32+=f
        pushes.append(p["lo"]-u["lo"])
        probe_ties += (p["lo"]==0); unprobe_ties += (u["lo"]==0); arg_tie += (a and p["lo"]==0)
        pA += (p["top"]==p["a_id"]); uA += (u["top"]==u["a_id"]); pB += (p["top"]==p["b_id"]); uB += (u["top"]==u["b_id"])
    aud=json.load(open(base/"audit"/(name+".json"))) if (base/"audit"/(name+".json")).exists() else None
    if aud:
        tc=aud["test_cut"]["blind_k3"]
        r3=tc["removed"]; cut=tc["answer_A"]; swcut=tc["switch_after"]; r6=aud["test_cut"].get("blind_k6",{}).get("removed")
        singles=[v["removed"] for k,v in aud["test_cut"].items() if k.startswith("single_")]
        best=max(singles) if singles else float("nan")
    else:
        r3=cut=swcut=r6=best=float("nan")
    print(f"{name}: strict={strict/n:.3f} argmax={argmax/n:.3f} fp32={fp32/n:.3f} push={statistics.mean(pushes):.3f} tiesP/U/argTie={probe_ties}/{unprobe_ties}/{arg_tie} probedA={pA}/{n} unprobedA={uA}/{n} unprobedB={uB}/{n} R3={r3:.3f} R6={r6:.3f} cutA={cut:.3f} cutSwitch={swcut:.3f} bestSingle={best:.3f}")
ao=[json.load(open(base/"audit"/(n+".json")))["test_cut"]["blind_k3"]["removed"] for n in ["ao_s0","ao_s1","ao_s2"]]
mean=statistics.mean(ao); sd=statistics.stdev(ao); lo=mean-4.303*sd/math.sqrt(3); hi=mean+4.303*sd/math.sqrt(3)
print(f"AO seed interval R3 mean={mean:.3f} [{lo:.3f},{hi:.3f}]")
p100=[json.load(open(base/"whole"/(n+".json")))["switch_strict"] for n in ["p100_s0","p100_s1","p100_s2"]]
mean=statistics.mean(p100); sd=statistics.stdev(p100); lo=mean-4.303*sd/math.sqrt(3); hi=mean+4.303*sd/math.sqrt(3)
print(f"P100 strict seed interval mean={mean:.3f} [{lo:.3f},{hi:.3f}]")
'
```

```
python3 -c 'import json, pathlib; base=pathlib.Path("kran-research/projects/probe-switch-audit-penalty-8b/data/audit");
for n in ["ao_s0","ao_s1","ao_s2","l0_s0","l0_s1"]:
 d=json.load(open(base/(n+".json"))); h=d["headline"]; nr=d["null_removed"]; print(n, "R3", round(h["title_cell_blind_top3_removed"],3), "null95", round(nr.get("p95", nr.get("random_p95", float("nan"))),3), "null>=.30", nr.get("ge_0.3", nr.get("share_ge_030", None)), "keys", list(nr.keys())[:8])'
```

```
python3 -c 'import json; s=json.load(open("kran-research/projects/probe-switch-audit-penalty-8b/data/status.json"));
for n in ["ao_s0","ao_s1","ao_s2"]:
 c=s["cells"][n]; print(n, c.get("pen_zero_count"), c.get("pen_zero_share"), c.get("index_top3_zero_count"), c.get("index_top3_zero_share"), sorted([k for k in c.keys() if "zero" in k]) )'
```

```
~/bin/room review oracle-1123-audit-penalty-8b --as critic2 "result" --pass "Cross-family PASS on result. Checked deploy preview https://deploy-preview-236--kranai.netlify.app/probe-switch-audit-penalty-8b/ and local post blog 092fabd5 / kran-research 3425ef90d. Own Python rebuild from saved raw rows in data/whole + data/audit: AO strict/R3/cut/bestSingle = s0 0.977/0.676/0.680/0.201, s1 1.000/0.487/0.633/0.141, s2 0.992/0.654/0.273/0.194; P100 strict/argmax/fp32/push = s0 0.172/0.688/0.734/0.237, s1 0.016/0.062/0.141/0.117, s2 0.000/0.344/0.039/0.275; guard strict/argmax/fp32/push = 0/0/0/0.219 and probedA=unprobedA=128/128; L0 R3/cut = 0.998/0.000, 0.983/0.000; AO seed interval R3 0.606 [0.348,0.863], P100 strict 0.062 [-0.174,0.299]. These match proof-matrix.md:7-15,20-22 and post claims at _notes/probe-switch-audit-penalty-8b.md:56,70,72,76,105-126. Status UNCONFIRMED is correct: PREREG.md:104 hide cell, :108-118 status rule; status.json:3 says UNCONFIRMED and status.json:543-573 has all P10 cells untrained. Figure data matches rebuilt cells: assets/figures/probe-switch-audit-penalty-8b/data.json:43-50; fig1 image read shows not-settled answer, 0/3 AO hid, 3/3 P100 faded, P10 waiting. External checks: arXiv 2509.21761 abstract says BAHA ablating ~3% heads reduces ASR over 90%; 2404.15255 and 2309.16042 support activation-patching/metric caveats. No blocking overclaim found."
```