# Mean whole-graph P(X) over the checkpoints after takeoff (4,096..8,192) per build, main seeds.
# Tests whether the serving 'overshoot' at 8,192 (s0 0.564 vs untrained 0.393) is a level shift or a checkpoint swing.
import json, glob, os, numpy as np
E='/Users/esben/work/oracle-1030-stack-conditional/kran-research/projects/stack-conditional/data/eval/'
base={b: json.load(open(E+f'base/{b}_0.json'))['px_mean'] for b in ['eager','compiled','emu']}
out={'untrained':{b:round(v,4) for b,v in base.items()}}
for s in ['stack_s0_lr0.0001_kl1','stack_s1_lr0.0001_kl1']:
    r={}
    for b in ['eager','compiled','emu']:
        v=[json.load(open(E+f'{s}/{b}_{k}.json'))['px_mean'] for k in (4096,5120,6144,7168,8192) if os.path.exists(E+f'{s}/{b}_{k}.json')]
        r[b]={'n_ckpt':len(v),'mean':round(float(np.mean(v)),4),'sd':round(float(np.std(v,ddof=1)),4) if len(v)>1 else None,'vals':[round(x,3) for x in v]}
    out[s]=r
print(json.dumps(out,indent=1))
