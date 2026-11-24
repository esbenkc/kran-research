
import json,statistics as st,math
o=json.load(open('/tmp/critic2_rebuild_1124.json'))
def ti(xs):
    n=len(xs);m=st.mean(xs);s=st.stdev(xs);t={2:12.706,3:4.303}[n];h=t*s/math.sqrt(n);return '%.3f [%.3f, %.3f]'%(m,m-h,m+h)
for arm,rs in [('AO',['ao_s0','ao_s1','ao_s2']),('P100',['p100_s0','p100_s1','p100_s2']),('P10',['p10_s0','p10_s1','p10_s2']),('L0',['l0_s0','l0_s1'])]:
    print(arm,'strict',ti([o[r]['strict'] for r in rs]),'R3',ti([o[r]['R3'] for r in rs]))
print('P10 R3 switch-on',ti([o['p10_s0']['R3'],o['p10_s1']['R3']]))
print('ratios L0/P10 cached', o['l0_s0']['cpush']/o['p10_s0']['cpush'], o['l0_s1']['cpush']/o['p10_s1']['cpush'])
print('AO R/null95',[round(o[r]['R3']/o[r]['null95'],2) for r in ['ao_s0','ao_s1','ao_s2']])
print('P100 push % of L0 cached',[round(100*o[r]['cpush']/o['l0_s%d'%(i%2)]['cpush'],2) for i,r in enumerate(['p100_s0','p100_s1','p100_s2'])])
# frozen status with three switch variants
for var in ['strict','fp32','argmax']:
    hides={arm:sum(1 for r in rs if o[r][var]>=0.75 and o[r]['R3']<0.30 and o[r]['g1']) for arm,rs in [('AO',['ao_s0','ao_s1','ao_s2']),('P100',['p100_s0','p100_s1','p100_s2']),('P10',['p10_s0','p10_s1','p10_s2'])]}
    print(var,hides,'L0 hide',[o[r][var]>=0.75 and o[r]['R3']<0.30 for r in ['l0_s0','l0_s1']],'guard hide',o['p100_s0_g2048'][var]>=0.75 and o['p100_s0_g2048']['R3']<0.3)
print('p100_s0 fp32 count',round(o['p100_s0']['fp32']*128),'argmax count',round(o['p100_s0']['argmax']*128))
