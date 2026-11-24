
import json,statistics as st
for r in ['ao_s0','ao_s1','ao_s2']:
    a=json.load(open(f'data/audit/{r}.json'))
    for g in a['blind_greedy_dev'][:3]:
        s=sorted(g['all_singles'].items(),key=lambda x:-x[1])[:4]
        print(r,g['set'],'%.3g'%g['dev_blind_removed'],'runner %.3g'%g['runner_up'],[(k,'%.2g'%v) for k,v in s])
    h=a['headline']; print(r,'D test %.4f'%h['kl_full_test_mean'],'blind_top3_blind_removed %.3f'%h['blind_top3_blind_removed'],'informed3 %.3f'%h['informed_top3_removed'], 'top6 %.3f'%h['blind_top6_removed'])
    pl=a['per_prompt_lo']; print(' cut ties k3==0:',sum(k==0 for k in pl['blind_k3']), 'k3>0&u<0',sum(k>0 and u<0 for k,u in zip(pl['blind_k3'],pl['unprobed'])),'k3>0',sum(k>0 for k in pl['blind_k3']))
