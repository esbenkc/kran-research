# Planted fixtures for vera_check.py: synthetic stage-2 run JSONs with known switch times.
import json, os, sys, shutil, numpy as np, subprocess
def mk(d, Tm, Tr, seeds=(5,6,7,8,9), cens=()):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    rng=np.random.default_rng(0)
    for arm in ('main','shuf','coin','fresh'):
        for i,s in enumerate(seeds):
            T = Tm[i] if arm=='main' else Tr[arm][i]
            base=list(0.35+0.01*rng.standard_normal(64)); gb_on=[b-0.002 for b in base]
            curve=[]; stop=1536 if (arm,s) in cens else int(max(1.5*T, T+256))
            for t in range(0, min(stop,1536)+1, 8):
                drop = 0.15 if (t>=T and (arm,s) not in cens) else 0.03*min(t/max(T,1),1)
                curve.append({'t':t,'p_off':base,'p_B':[b-0.002-drop for b in base]})
            json.dump({'arm':arm,'seed':s,'p_base_off':base,'p_base_on':gb_on,'curve':curve,
                       'T_B': None if (arm,s) in cens else T}, open(f'{d}/{arm}_s{s}.json','w'))
ref=lambda v: {a:v for a in ('shuf','coin','fresh')}
cases={
 'F1_conf':  ([96,104,96,112,96], ref([296,280,304,296,288]), (), 'CONFIRMED'),
 'F2_rej':   ([280,296,272,304,288], ref([296,280,304,296,288]), (), 'REJECTED'),
 'F3_bound': ([144,152,136,160,144], ref([296,280,304,296,288]), (), 'UNCONFIRMED'),
 'F4_portfail': ([96,104,96,112,96], {'shuf':[296]*5,'coin':[296]*5,'fresh':[400,392,408,400,416]}, (), 'UNCONFIRMED'),
 'F5_cens2': ([96,104,96,112,96], ref([296,280,304,296,288]), (('coin',5),('coin',6)), 'UNCONFIRMED'),
}
import vera_check as vc
ok=0
for name,(Tm,Tr,cens,exp) in cases.items():
    d=f'/tmp/vera_fx/{name}'; mk(d,Tm,Tr,cens=cens)
    out,_,T=vc.rebuild(d)
    got=out['status_before_G1G2']; ok+= got==exp
    print(name,'expected',exp,'got',got,'| T main',out['T_table']['main'],'| R shuf %.3f [%.3f,%.3f]'%(out['ratios']['shuf']['R'],out['ratios']['shuf']['R_lo'],out['ratios']['shuf']['R_hi']),'| mismatches',[k for k,v in out['tails'].items() if not v['match']])
print(f'{ok}/{len(cases)} fixtures agree')
