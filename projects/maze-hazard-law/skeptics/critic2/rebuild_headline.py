# critic2 (openai/gpt-5.5) rebuild, verbatim from its session; run from the room work dir (paths start with kran-research/).
import json, glob, math, statistics, collections
try:
 from scipy.stats import chi2, beta, t
except Exception as e:
 print('scipy import failed',e); raise
base='kran-research/projects/maze-hazard-law'
# gpt raw
runs=[]
for f in sorted(glob.glob(base+'/data/gpt-oss-20b-low-full/raw/maze/*.json')):
 d=json.load(open(f))
 if d.get('entry_step') is not None:
  runs.append(d)

def giveup(d): return d['end'] in ('answer','leave_get')
# counts by bins
bins=[(1,3),(4,10),(11,10**9)]
obs=[]; E=[]
for lo,hi in bins:
 g=e=0
 for d in runs:
  dep=d['maze_pages_seen']
  e += max(0, min(dep,hi)-lo+1)
  if lo <= dep <= hi and giveup(d): g+=1
 obs.append(g); E.append(e)
G=sum(obs); Dec=sum(E); h=G/Dec
exp=[e*h for e in E]
chi=sum((o-x)**2/x for o,x in zip(obs,exp))
p=chi2.sf(chi, len(obs)-1)
# KM median no censor: first depth where survival<=0.5. here all giveup
counter=collections.Counter(d['maze_pages_seen'] for d in runs)
remaining=len(runs); surv=1; median=None
for depth in sorted(counter):
 remaining -= counter[depth]
 surv=remaining/len(runs)
 if surv <= .5 and median is None: median=depth
# forecast
h3=obs[0]/E[0]
forecast=math.log(.5)/math.log(1-h3)
forecast_cap=min(forecast,78)
upper=beta.ppf(.975, obs[0]+1, E[0]-obs[0])
forecast_upper=min(math.log(.5)/math.log(1-upper),78)
# seed-level h_all
hs=[]
for b in range(3):
 ds=[d for d in runs if d['site']['run_index']%3==b]
 hs.append(sum(giveup(d) for d in ds)/sum(d['maze_pages_seen'] for d in ds))
mean=statistics.mean(hs); se=statistics.stdev(hs)/math.sqrt(len(hs)); crit=t.ppf(.975,len(hs)-1)
print('gpt entered/files',len(runs),len(glob.glob(base+'/data/gpt-oss-20b-low-full/raw/maze/*.json')))
print('bin G/E',list(zip(obs,E)),'h_all',h,'chi',chi,'p',p)
print('median',median,'forecast_uncapped',forecast,'forecast_cap',forecast_cap,'ratio',forecast_cap/median,'upper_h',upper,'fast_forecast',forecast_upper)
print('seed_h',hs,'tint',mean-crit*se,mean+crit*se)
print('cap',sum(d['end']=='cap' for d in runs),'causes',collections.Counter(d['end'] for d in runs))
# qwen
qr=[]
for f in sorted(glob.glob(base+'/data/qwen3-8b-nothink-full/raw/maze/*.json')):
 d=json.load(open(f))
 if d.get('entry_step') is not None: qr.append(d)
qg=sum(d['end'] in ('answer','leave_get') for d in qr)
print('qwen files entered giveups cap depths',len(glob.glob(base+'/data/qwen3-8b-nothink-full/raw/maze/*.json')),len(qr),qg,sum(d['end']=='cap' for d in qr),[d['maze_pages_seen'] for d in qr])
# exit check
x=json.load(open(base+'/results/exit_check_gpt-oss-20b-low-full.json'))
print('exit prompt tokens',x.get('prompt_tokens_min'),x.get('prompt_tokens_median'),x.get('prompt_tokens_max'), 'max new',x.get('new_tokens_max'),'cap hits',x.get('cap_hits'))
# cues
qc=json.load(open(base+'/results/quit_cues_gpt-oss-20b-low-full.json'))
print('quit cues keys',qc.keys()); print(qc)
