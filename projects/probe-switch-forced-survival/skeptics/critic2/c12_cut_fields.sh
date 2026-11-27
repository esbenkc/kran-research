# critic2 (openai/gpt-5.5) tool call 12, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 -c 'import json; d=json.load(open("kran-research/projects/probe-switch-forced-survival/data/kaudit/k6_s0.json"));
for k,v in d["test_cut"].items(): print(k, type(v).__name__, (list(v.keys())[:10] if isinstance(v,dict) else (len(v) if hasattr(v,"__len__") else v)))
'
