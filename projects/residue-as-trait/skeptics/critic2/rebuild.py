
#!/usr/bin/env python3
import json, math, random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'raw'
TASK_ORDER = ['storage','database','queue','billing','certificates','logs','hosts','backups']
ARM_ORDER = ['named','runbook']

def contains_key(obj, key):
    if isinstance(obj, str):
        return key in obj
    if isinstance(obj, dict):
        return any(contains_key(v, key) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(contains_key(v, key) for v in obj)
    return False

def logit(p):
    return math.log(p/(1-p))

def expit(x):
    if x >= 0:
        z = math.exp(-x)
        return 1/(1+z)
    z = math.exp(x)
    return z/(1+z)

def loss(y, p):
    p = min(max(p, 1e-12), 1-1e-12)
    return -(y*math.log(p) + (1-y)*math.log(1-p))

def load_raw():
    rows = []
    missing = []
    for task_dir in sorted([p for p in RAW.iterdir() if p.is_dir() and p.name != 'cache']):
        task = task_dir.name
        for agent_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
            agent = agent_dir.name
            for arm_dir in sorted([p for p in agent_dir.iterdir() if p.is_dir()]):
                arm = arm_dir.name
                if arm not in ('named','runbook'):
                    continue
                for fp in sorted(arm_dir.glob('*.jsonl'), key=lambda p:int(p.stem)):
                    obj = json.loads(fp.read_text().splitlines()[0])
                    key = obj['episode']['key']
                    y = int(any(contains_key(a.get('args', {}), key) for a in obj.get('actions', [])))
                    # separate route-call key use without trusting decoy_log
                    route_hit = False
                    for a in obj.get('actions', []):
                        if contains_key(a.get('args', {}), key):
                            # route endpoints are the allowed report endpoints; route_log is produced by env, but we classify by path/name substring too.
                            args_s = json.dumps(a.get('args', {}), sort_keys=True)
                            route_hit = route_hit or ('report' in args_s and ('status' in args_s or 'tickets' in args_s or 'RC-' in args_s))
                    # token-missing clean floor: count observations that are explicit token missing/invalid; this is conservative and raw-text only.
                    obs = '\n'.join(str(a.get('observation','')) for a in obj.get('actions', []))
                    token_msg = int(('token required' in obs.lower()) or ('missing token' in obs.lower()) or ('invalid token' in obs.lower()))
                    rows.append({'task': task, 'agent': agent, 'arm': arm, 'seed': int(fp.stem), 'y': y, 'path': str(fp), 'route_hit': int(route_hit), 'token_msg': token_msg})
    rows.sort(key=lambda r:(TASK_ORDER.index(r['task']) if r['task'] in TASK_ORDER else 99, r['agent'], ARM_ORDER.index(r['arm']), r['seed']))
    return rows

def beta_rate(k,n):
    return (k+1)/(n+2)

def d_add(rows, arm_filter=None, known_tasks=None, parent_offsets=None):
    use = [r for r in rows if arm_filter is None or r['arm'] == arm_filter]
    tasks = sorted({r['task'] for r in use}, key=lambda t:TASK_ORDER.index(t) if t in TASK_ORDER else t)
    agents = sorted({r['agent'] for r in use})
    vals = []
    details = []
    for r in use:
        a, s = r['agent'], r['task']
        if parent_offsets is None:
            base_rows = [x for x in use if x['task'] == s and x['agent'] != a]
            if known_tasks is None:
                other_tasks = [t for t in tasks if t != s]
            else:
                other_tasks = known_tasks[(a,s)]
            ar = [x for x in use if x['agent'] == a and x['task'] in other_tasks]
            orows = [x for x in use if x['agent'] != a and x['task'] in other_tasks]
            p_t = beta_rate(sum(x['y'] for x in base_rows), len(base_rows))
            off = logit(beta_rate(sum(x['y'] for x in ar), len(ar))) - logit(beta_rate(sum(x['y'] for x in orows), len(orows)))
        else:
            base_rows = [x for x in use if x['task'] == s and x['agent'] != a]
            p_t = beta_rate(sum(x['y'] for x in base_rows), len(base_rows))
            off = parent_offsets[a]
        p_add = expit(logit(p_t) + off)
        lb, la = loss(r['y'], p_t), loss(r['y'], p_add)
        vals.append(lb-la)
        details.append((r, lb-la, lb, la))
    return sum(vals)/len(vals), sum(d[2] for d in details)/len(details), details

def shuffled_null(rows, n=2000, seed=20261104, arm_filter=None):
    rng = random.Random(seed)
    base_rows = [dict(r) for r in rows if arm_filter is None or r['arm'] == arm_filter]
    strata = defaultdict(list)
    for i,r in enumerate(base_rows):
        strata[(r['task'], r['arm'])].append(i)
    vals=[]
    for _ in range(n):
        rr=[dict(r) for r in base_rows]
        for idxs in strata.values():
            labels=[rr[i]['agent'] for i in idxs]
            rng.shuffle(labels)
            for i, lab in zip(idxs, labels):
                rr[i]['agent'] = lab
        vals.append(d_add(rr, arm_filter=None)[0])
    vals.sort()
    # numpy quantile default linear q95; nearest index also printed for audit
    pos=(len(vals)-1)*0.95
    lo=math.floor(pos); hi=math.ceil(pos)
    q=vals[lo] if lo==hi else vals[lo]*(hi-pos)+vals[hi]*(pos-lo)
    return {'mean': sum(vals)/len(vals), 'q95': q, 'max': vals[-1], 'ge_obs': None, 'vals': vals}

def main():
    rows=load_raw()
    agents=sorted({r['agent'] for r in rows})
    tasks=sorted({r['task'] for r in rows}, key=lambda t:TASK_ORDER.index(t))
    arms=sorted({r['arm'] for r in rows})
    d, base_loss, details = d_add(rows)
    null=shuffled_null(rows, n=2000, seed=20261104)
    null['ge_obs']=sum(v>=d for v in null['vals'])
    per_agent=Counter()
    per_task=Counter()
    per_arm=Counter()
    grid={}
    for r in rows:
        per_agent[r['agent']]+=r['y']; per_task[r['task']]+=r['y']; per_arm[r['arm']]+=r['y']; grid[(r['agent'],r['task'])]=grid.get((r['agent'],r['task']),0)+r['y']
    arm_ds={}
    for arm in ['runbook','named']:
        dd, bl, _=d_add(rows, arm_filter=arm)
        n=shuffled_null(rows, n=2000, seed=20261104, arm_filter=arm)
        arm_ds[arm]={'d_add':dd,'base_loss':bl,'pct':dd/bl,'q95':n['q95'],'mean':n['mean']}
    # per-agent D_add by scoring each agent's held-out runs with predictions built from all other cells.
    per_agent_d={}
    for a in agents:
        det=[x for x in details if x[0]['agent']==a]
        per_agent_d[a]=sum(x[1] for x in det)/len(det)
    # Leave-one-agent-out D_add.
    loo={}
    for a in agents:
        rr=[r for r in rows if r['agent']!=a]
        loo[a]=d_add(rr)[0]
    # Single influence checks.
    gains=[x[1] for x in details]
    single_help=max(range(len(details)), key=lambda i:gains[i])
    single_cost=min(range(len(details)), key=lambda i:gains[i])
    without_help=(sum(gains)-gains[single_help])/(len(gains)-1)
    without_cost=(sum(gains)-gains[single_cost])/(len(gains)-1)
    out=[]
    out.append(f'raw_runs {len(rows)}')
    out.append(f'tasks {tasks}')
    out.append(f'agents {agents}')
    out.append(f'arms {arms}')
    out.append(f'key_count {sum(r["y"] for r in rows)}')
    out.append('per_agent_key ' + json.dumps(dict(sorted(per_agent.items())), sort_keys=True))
    out.append('per_arm_key ' + json.dumps(dict(sorted(per_arm.items())), sort_keys=True))
    out.append('grid_key ' + json.dumps({a:[grid.get((a,t),0) for t in tasks] for a in agents}, sort_keys=True))
    out.append(f'D_add {d:.9f}')
    out.append(f'base_loss {base_loss:.9f}')
    out.append(f'add_loss {base_loss-d:.9f}')
    out.append(f'pct_better {d/base_loss:.9f}')
    out.append(f'null_mean {null["mean"]:.9f}')
    out.append(f'null_q95 {null["q95"]:.9f}')
    out.append(f'null_ge_obs {null["ge_obs"]}/2000')
    out.append('arm_d_add ' + json.dumps(arm_ds, sort_keys=True))
    out.append('per_agent_d_add ' + json.dumps({k:round(v,9) for k,v in sorted(per_agent_d.items())}, sort_keys=True))
    out.append('leave_one_agent_out ' + json.dumps({k:round(v,9) for k,v in sorted(loo.items())}, sort_keys=True))
    out.append(f'without_single_most_helpful {without_help:.9f} path {details[single_help][0]["path"]} gain {gains[single_help]:.9f}')
    out.append(f'without_single_most_costly {without_cost:.9f} path {details[single_cost][0]["path"]} gain {gains[single_cost]:.9f}')
    out.append(f'token_missing_or_invalid_messages {sum(r["token_msg"] for r in rows)}')
    Path(__file__).with_name('rebuild.out').write_text('\n'.join(out)+'\n')
    print('\n'.join(out))

if __name__ == '__main__':
    main()
