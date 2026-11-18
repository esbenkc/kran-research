# cwd: /Users/esben/work/defense-1117-trap-aware-canary
python3 - <<'PY'
import json, collections, math, statistics
from pathlib import Path
p=Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/scores.jsonl')
rows=[json.loads(l) for l in p.read_text().splitlines()]
want=['warning-cued','neutral-cued','warning-none','neutral-none','warning-safe','neutral-safe']
print('rows',len(rows), collections.Counter(r['cell'] for r in rows))
for cell in want:
    rs=[r for r in rows if r['cell']==cell]
    if not rs: continue
    def c(k): return sum(1 for r in rs if r.get(k))
    print(cell, 'n',len(rs), 'fit',c('fit'), 'fit_correct',sum(1 for r in rs if r.get('fit') and r.get('correct')), 'correct',c('correct'), 'fit_tok',c('fit_tok'), 'fit_tok_404',c('fit_tok_404'), 'cap_hit',c('cap_hit'), 'format_fail',c('format_fail'))
    print(' seeds fit', {s:sum(1 for r in rs if r['seed']==s and r.get('fit')) for s in [0,1,2]}, 'seed n',{s:sum(1 for r in rs if r['seed']==s) for s in [0,1,2]})
# derived claims
W=[r for r in rows if r['cell']=='warning-cued']; U=[r for r in rows if r['cell']=='neutral-cued']
WA=[r for r in rows if r['cell']=='warning-none']; UA=[r for r in rows if r['cell']=='neutral-none']
print('D fit', (sum(r['fit'] for r in U)/60-sum(r['fit'] for r in W)/60)-(sum(r['fit'] for r in UA)/60-sum(r['fit'] for r in WA)/60))
print('D fit_tok', (sum(r['fit_tok'] for r in U)/60-sum(r['fit_tok'] for r in W)/60)-(sum(r['fit_tok'] for r in UA)/60-sum(r['fit_tok'] for r in WA)/60))
print('hidden_rt trap pages', sum(r.get('hidden_rt',False) for r in W+U), 'of', len(W+U))
print('knowing among follows', sum(r.get('knowing_rt',False) for r in W+U if r.get('fit')), 'of', sum(r.get('fit',False) for r in W+U), 'split', sum(r.get('knowing_rt',False) for r in W if r.get('fit')), '/', sum(r.get('fit',False) for r in W), sum(r.get('knowing_rt',False) for r in U if r.get('fit')), '/', sum(r.get('fit',False) for r in U))
# drop-only: fit_tok true but fit false; exactly one? use access log + manifest for occurrences containing hid
manifest=[json.loads(l) for l in Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/manifest.jsonl').read_text().splitlines()]
by_token={m['token']:m for m in manifest}
access=[]
for line in Path('kran-research/projects/trap-aware-canary-open-weight/web/out/qwen3-8b/access.log').read_text().splitlines():
    parts=line.split('\t')
    if len(parts)>=6 and parts[2]=='GET': access.append(parts[3])
for cell in ['neutral-cued','warning-cued','neutral-none','warning-none']:
    ms=[m for m in manifest if m['cell']==cell]
    drop=[]; outside=[]
    for m in ms:
        code_paths=[path for path in access if m['hid'] in path]
        fit=any(path==m['fit_path'] for path in access)
        if code_paths and not fit:
            drop.append((m['i'], len(code_paths), code_paths[:3]))
        if any((path.startswith('/'+m['token']+'/') is False) and (m['hid'] in path) for path in access):
            outside.append(m['i'])
    print('drop_only', cell, len(drop), 'ones', sum(1 for _,n,_ in drop if n==1), 'outside_code', len(outside), 'sample', drop[:2])
PY
