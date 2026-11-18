import numpy as np, json
z=np.load('data/pool/test/cols.npz'); var=z['variant']; base=z['base_row']; lw=z['logw']
n=np.where(var==0)[0]; n=n[np.argsort(base[n])]
w=np.exp(lw[n]); DR=np.mean(np.clip(1-w,0,None)); print('D_R',DR, 'w same on xins?', np.allclose(np.exp(lw[var==1][np.argsort(base[var==1])]),w))
pr=z['prompt_row'][n]; up=np.unique(pr); G=[np.where(pr==p)[0] for p in up]
rng=np.random.default_rng(7)
for run,last in [('kl0_s0',1200),('kl0_s1',900),('kl0_s2',700)]:
    c=np.load(f'data/runs/{run}/reads/c_{last}.npy').astype(float)[n]
    def sR(ix): cc=c[ix]; ww=w[ix]; return (cc.mean()-(ww*cc).sum()/ww.sum())/(np.mean(np.clip(1-ww,0,None))*cc.mean())
    pt=sR(np.arange(len(n)))
    bs=[sR(np.concatenate([G[i] for i in rng.integers(0,200,200)])) for _ in range(2000)]
    print(run,last,'s_R %+.4f [%+.4f, %+.4f]'%(pt,*np.percentile(bs,[2.5,97.5])))
    S=[json.loads(l) for l in open(f'data/runs/{run}/steps.jsonl')]
    blocks=[np.mean([s['x_rate_probed'] for s in S if b<s['step']<=b+100]) for b in range(0,last,100)]
    print('  probed We share per 100-block', np.round(blocks,3), 'min',min(blocks).round(3),'max',max(blocks).round(3))
