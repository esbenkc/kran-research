#!/bin/bash
# progress.sh <run>: one line per logged step (loss, KL, Rule B / plain catch, d/v range)
grep '^{' data/logs/attack_$1.log | python3 -c "
import sys,json
for l in sys.stdin:
    r=json.loads(l)
    if 'lam' in r: print(r); continue
    dv=r.get('d_over_v'); s=f\"{r['step']:>3} \"
    s+=' '.join(f'{k}={r[k]:.4g}' for k in ('probe_loss','kl_cities','kl_pile','ruleB_catch_eval','ruleB_line','max_catch_eval','max_line','sec') if k in r)
    if dv: s+=f' d/v={min(dv):.2f}..{max(dv):.2f} maxz={max(r[\"maxabsz_mom\"]):.2f}'
    print(s)
"
