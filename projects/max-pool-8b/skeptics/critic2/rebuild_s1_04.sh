python3 - <<'PY'
import json, numpy as np, math
from pathlib import Path
base=Path('kran-research/projects/max-pool-8b/data')
rows=json.loads((base/'ref/split.json').read_text())
ref_true=np.array([r['split']=='refit' and r['true']==1 for r in rows])
eval_false=np.array([r['split']=='eval' and r['true']==0 for r in rows])
eval_true=np.array([r['split']=='eval' and r['true']==1 for r in rows])
print('n ref_true eval_false eval_true', ref_true.sum(), eval_false.sum(), eval_true.sum())
def catch(state, key='max'):
    a=np.load(base/'states'/state/'scores.npz')[key]
    ref=a[ref_true]
    # second-highest line, strict above line
    line=np.sort(ref)[-2]
    return float((a[eval_false] > line).mean()), float((a[eval_true] > line).mean()), line
for state in ['clean','main__seed0__step150','main__seed1__step150','main__seed2__step150','random__seed0__step150','random__seed1__step150']:
    print(state)
    for key,name in [('mean','avg18'),('max','max18'),('last','last18')]:
        c,f,l=catch(state,key)
        print(' ',name, round(100*c,3), round(100*f,3), 'line', round(l,4))
# seed interval t with df=2 from per seed max catches, hard-code t_0.975,2
per=np.array([catch(s)[0] for s in ['main__seed0__step150','main__seed1__step150','main__seed2__step150']])
mean=per.mean(); sd=per.std(ddof=1); t=4.302652729911275; half=t*sd/math.sqrt(3)
print('main max per', [round(x*100,3) for x in per], 'mean', mean*100, 'tCI', (mean-half)*100, (mean+half)*100)
PY
