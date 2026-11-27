# critic2 (openai/gpt-5.5) tool call 42, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 - <<'PY'
import json, subprocess, pathlib, statistics
branch='origin/swarm/oracle-1127-forced-switch-head'
base='projects/probe-switch-forced-survival/data/kaudit/'
for s in range(3):
    d=json.loads(subprocess.check_output(['git','-C','kran-research','show',f'{branch}:{base}k6_s{s}.json']))
    h=d['headline']; sk=h['strict32_after_greedy_k']['TB']; rr=h['strict32_random_min_median_max']['8']; print(s, 'S6', round(100*sk[5],1), 'S8', round(100*sk[7],1), 'S12', round(100*sk[11],1), 'R8med%', round(100*rr[1]), 'kmin', h['strict32_smallest_k_below_0.25']['TB'])
d=json.loads(subprocess.check_output(['git','-C','kran-research','show',f'{branch}:{base}k6_s1.json']))
print('s1 T1/T16 S1-5', d['headline']['strict32_after_greedy_k']['T1'][:5], d['headline']['strict32_after_greedy_k']['T16'][:5])
print('s1 rand8 min%', 100*d['headline']['strict32_random_min_median_max']['8'][0])
print('median kmin counted', statistics.median([7,13,2]))
PY
