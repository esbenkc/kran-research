# critic2 (openai/gpt-5.5) tool call 22, verbatim from its session; cwd = /Users/esben/work/oracle-1127-forced-switch-head
python3 - <<'PY'
import json, subprocess, statistics
branch='origin/swarm/oracle-1127-forced-switch-head'
base='projects/probe-switch-forced-survival/data/kaudit/'
vals=[]; rows=[]
for s in range(3):
    raw=subprocess.check_output(['git','-C','kran-research','show',f'{branch}:{base}k6_s{s}.json'])
    d=json.loads(raw); h=d['headline']; sk=h['strict32_after_greedy_k']['TB']; rr=h['strict32_random_min_median_max']['8']; km=h['strict32_smallest_k_below_0.25']['TB']
    vals.append(sk[7])
    rows.append((s,h['base_strict32'],sk[0],sk[1],sk[5],sk[7],sk[11],km,rr[0],rr[1],rr[2],d['cached_test'].get('switch_strict32')))
for r in rows:
    print('seed %d base %.6f S1 %.6f S2 %.6f S6 %.6f S8 %.6f S12 %.6f kmin %s R8 min/med/max %.6f %.6f %.6f cached %.6f'%r)
print('status', 'REJECTED' if any(v>=0.25 for v in vals) else 'CONFIRMED', 'S8mean %.6f'%statistics.mean(vals))
PY
