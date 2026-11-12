"""critic2 (Kasper) final cross-family review rebuild for optimizer-cue-size.
Part 1 and part 2 are the code critic2 (openai/gpt-5.5) ran inline on 2026-11-12 ~08:10 PST,
saved verbatim by the Claude fallback after critic2 hit its API cap. Only change: root is
resolved from this file's location (ROOT), and part 1's broken D4 tail (it loaded adamwtm_a0.03125_s0,
which is stored under the adamwtm_pick.json name) is dropped; part 2 does D4 with the pick.
Run: /usr/bin/python3 rebuild.py  (numpy, scipy)
"""
import pathlib, os
ROOT=pathlib.Path(__file__).resolve().parents[2]  # projects/optimizer-cue-size
print("=== part 1: headline 24 runs, pooled OLS, contrasts, bootstrap ===")
def _part1():
    import json, math, pathlib, itertools, statistics
    import numpy as np
    from scipy import stats
    root=ROOT
    runs=root/'data'/'runs'
    OPTS=['adamw','sgd','sign']; SIZES=[1/32,1/64,1/128,1/256]; SEEDS=[0,1]
    # selected pilot for SGD top seed0
    sgd_pick=json.load(open(root/'data'/'sgd_pick.json'))
    
    def load(name):
        return json.load(open(runs/f'{name}.json'))
    
    def run_name(opt, alpha, seed): return f'{opt}_a{alpha:g}_s{seed}'
    
    def all_evals(r): return sorted(r.get('dev_curve',[])+r.get('dense_curve',[]), key=lambda c:c['step'])
    
    def own_T(r):
        gap0=r.get('gap0')
        ev=all_evals(r)
        for i,c in enumerate(ev[:-1]):
            d0=c['gap']-gap0
            d1=ev[i+1]['gap']-gap0
            if d0>=0.10 and d1>=0.10:
                return c['step']
        return None
    rows=[]; mism=[]
    for opt in OPTS:
      for a in SIZES:
        for s in SEEDS:
          name=run_name(opt,a,s)
          if opt=='sgd' and a==1/32 and s==0:
            name=sgd_pick['run']
          r=load(name)
          T=own_T(r)
          if T!=r.get('T_dense'):
            mism.append((name,T,r.get('T_dense')))
          rows.append({'opt':opt,'alpha':a,'seed':s,'run':name,'x':float(r['cue_size']['x']),'T':T if T is not None else int(r['steps_run']),'censored':T is None})
    print('rows',len(rows),'mismatches',mism,'censored',sum(r['censored'] for r in rows))
    print('T table')
    for a in SIZES:
      line=[f'a={a:g} x={np.mean([r["x"] for r in rows if r["alpha"]==a]):.5g}']
      for opt in OPTS:
        line.append(opt+':'+'/'.join(str(r['T']) for r in rows if r['opt']==opt and r['alpha']==a))
      print(' '.join(line))
    # pooled opt-specific intercept+slope OLS
    arms=OPTS
    X=np.zeros((len(rows),2*len(arms))); y=np.log([r['T'] for r in rows])
    for i,r in enumerate(rows):
        k=arms.index(r['opt']); X[i,2*k]=1; X[i,2*k+1]=math.log(r['x'])
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    res=y-X@beta; df=len(y)-X.shape[1]; sd=math.sqrt(float(res@res/df)); cov=sd*sd*np.linalg.inv(X.T@X); q=stats.t.ppf(0.975,df)
    print('df',df,'sd',sd)
    for opt in arms:
        k=2*arms.index(opt)+1; se=math.sqrt(cov[k,k]); b=beta[k]
        print('slope',opt,b,b-q*se,b+q*se,'per_halving',2**(-b))
    def contrast(a,b):
        v=np.zeros(len(beta)); v[2*arms.index(a)+1]=1; v[2*arms.index(b)+1]=-1
        d=float(v@beta); se=math.sqrt(float(v@cov@v)); return d,d-q*se,d+q*se
    for c in [('sgd','adamw'),('sign','adamw')]: print('contrast',c,contrast(*c))
    # bootstrap seed 0, resample within optimizer x alpha
    rng=np.random.default_rng(0); cells={}
    for r in rows: cells.setdefault((r['opt'],r['alpha']),[]).append(r)
    outs={('sgd','adamw'):[],('sign','adamw'):[]}
    keys=list(cells)
    for _ in range(10000):
        s=[]
        for k in keys:
            g=cells[k]
            idx=rng.integers(0,len(g),len(g))
            s.extend([g[i] for i in idx])
        Xb=np.zeros((len(s),2*len(arms))); yb=np.log([r['T'] for r in s])
        for i,r in enumerate(s):
            j=arms.index(r['opt']); Xb[i,2*j]=1; Xb[i,2*j+1]=math.log(r['x'])
        bb,*_=np.linalg.lstsq(Xb,yb,rcond=None)
        for a,b in outs:
            outs[(a,b)].append(bb[2*arms.index(a)+1]-bb[2*arms.index(b)+1])
    for k,v in outs.items(): print('boot',k,np.percentile(v,[2.5,97.5]),'point',contrast(*k)[0])
def _part2():
    import json, math, pathlib, numpy as np
    from scipy import stats
    root=ROOT; runs=root/'data'/'runs'
    SIZES=[1/32,1/64,1/128,1/256]; SEEDS=[0,1]
    sgd_pick=json.load(open(root/'data/sgd_pick.json')); adamwtm_pick=json.load(open(root/'data/adamwtm_pick.json'))
    def load(n): return json.load(open(runs/f'{n}.json'))
    def all_evals(r): return sorted(r.get('dev_curve',[])+r.get('dense_curve',[]),key=lambda c:c['step'])
    def own_T(r):
     gap0=r['gap0']; ev=all_evals(r)
     for i,c in enumerate(ev[:-1]):
      if c['gap']-gap0>=0.10 and ev[i+1]['gap']-gap0>=0.10: return c['step']
     return None
    def name(opt,a,s): return f'{opt}_a{a:g}_s{s}'
    rows=[]
    for opt in ['sgd','adamwtm']:
     for a in SIZES:
      for s in SEEDS:
       n=name(opt,a,s)
       if opt=='sgd' and a==1/32 and s==0: n=sgd_pick['run']
       if opt=='adamwtm' and a==1/32 and s==0: n=adamwtm_pick['run']
       r=load(n); T=own_T(r)
       rows.append({'opt':opt,'a':a,'s':s,'n':n,'x':float(r['cue_size']['x']),'T':T})
    print('tm/sgd rows', [(r['opt'],r['a'],r['s'],r['T']) for r in rows])
    arms=['sgd','adamwtm']; X=np.zeros((len(rows),4)); y=np.log([r['T'] for r in rows])
    for i,r in enumerate(rows):
     j=arms.index(r['opt']); X[i,2*j]=1; X[i,2*j+1]=math.log(r['x'])
    b,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@b; df=len(y)-4; sd=math.sqrt(float(res@res/df)); cov=sd*sd*np.linalg.inv(X.T@X); q=stats.t.ppf(.975,df)
    print('df',df,'sd',sd)
    for opt in arms:
     k=2*arms.index(opt)+1; se=math.sqrt(cov[k,k]); print(opt,b[k],b[k]-q*se,b[k]+q*se,'x per half',2**(-b[k]))
    v=np.zeros(4); v[3]=1; v[1]=-1; d=float(v@b); se=math.sqrt(float(v@cov@v)); print('adamwtm minus sgd',d,d-q*se,d+q*se)
    # other controls
    for n in ['ctl_shuf_sgd_a0.03125_s0','ctl_g16_sgd_a0.015625_s0','ctl_lrhalf_sgd_a0.015625_s0','sgdfa_samelr_a0.03125_s0']:
     r=load(n); ev=all_evals(r); print(n,'ownT',own_T(r),'T_dense',r.get('T_dense'),'steps',r.get('steps_run'),'maxD',max(c['gap']-r['gap0'] for c in ev))
_part1()
print("=== part 2: D4 time-matched AdamW vs SGD, controls ===")
_part2()
