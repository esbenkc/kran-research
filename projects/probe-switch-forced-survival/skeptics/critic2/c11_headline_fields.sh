# critic2 (openai/gpt-5.5) tool call 11, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 -c 'import json, pathlib; base=pathlib.Path("kran-research/projects/probe-switch-forced-survival/data/kaudit");
for name in ["k6_s0","k6_s1","k6_s2","l0_s0","l0_s1"]:
 d=json.load(open(base/(name+".json"))); print("--",name); 
 for k in ["headline","cached_test","greedy_dev"]: print(k, type(d.get(k)).__name__, d.get(k) if k=="headline" else (list(d.get(k,{}).keys())[:20] if isinstance(d.get(k),dict) else d.get(k)))
'
