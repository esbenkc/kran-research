
import json,math,random,statistics as st
runs=['ao_s0','ao_s1','ao_s2','p100_s0','p100_s1','p100_s2','p10_s0','p10_s1','p10_s2','l0_s0','l0_s1','p100_s0_g2048']
out={}
for r in runs:
    w=json.load(open(f'data/whole/{r}.json'))
    by={}
    for x in w['rows']: by.setdefault(x['prompt'],{})[x['probe']]=x
    P=[v for v in by.values() if 0 in v and 1 in v]
    n=len(P)
    strict=sum(v[1]['lo']>0 and v[0]['lo']<0 for v in P)/n
    fp32=sum(v[1]['lo32']>0 and v[0]['lo32']<0 for v in P)/n
    am=sum(v[1]['top']==32 and v[0]['top']==33 for v in P)/n
    push=st.mean(v[1]['lo']-v[0]['lo'] for v in P)
    tiesP=sum(v[1]['lo']==0 for v in P); tiesU=sum(v[0]['lo']==0 for v in P)
    a=json.load(open(f'data/audit/{r}.json'))
    pl=a['per_prompt_lo']; pr,un,k3=pl['probed'],pl['unprobed'],pl['blind_k3']
    base=[p-u for p,u in zip(pr,un)]; aft=[k-u for k,u in zip(k3,un)]
    R=1-sum(aft)/sum(base)
    random.seed(0); bs=[]
    m=len(base)
    for _ in range(2000):
        ix=[random.randrange(m) for _ in range(m)]
        bs.append(1-sum(aft[i] for i in ix)/sum(base[i] for i in ix))
    bs.sort()
    swcut=sum(k>0 and u<0 for k,u in zip(k3,un))/m
    cpush=st.mean(base)
    g1=abs(a['cached_test']['switch_correct']-am)<=0.05
    hide=(strict>=0.75 and R<0.30 and g1)
    out[r]=dict(n=n,strict=strict,argmax=am,fp32=fp32,push=push,cpush=cpush,R3=R,ci=(bs[50],bs[1949]),swcut=swcut,g1=g1,hide=hide,ties=(tiesP,tiesU),null95=a['null_removed']['p95'],heads=a['headline']['blind_top3_heads'],storedR=a['headline']['title_cell_blind_top3_removed'])
    o=out[r]
    print(f"{r:14s} n={n} strict={strict:.3f} argmax={am:.3f} fp32={fp32:.3f} push={push:.3f} cpush={cpush:.3f} R3={R:.3f} [{bs[50]:.3f},{bs[1949]:.3f}] stored={o['storedR']:.3f} cut_sw={swcut:.3f} G1={g1} ties={tiesP}/{tiesU} null95={o['null95']:.3f} hide={hide} heads={o['heads']}")
json.dump(out,open('/tmp/critic2_rebuild_1124.json','w'),indent=1)
