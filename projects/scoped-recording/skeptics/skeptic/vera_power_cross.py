# Vera: power of the crossover contrast C = agent-equal mean of flag(scoped) - flag(mirror), 90% bootstrap-free normal CI lower bound > 0.
import numpy as np
rng=np.random.default_rng(1)
fb=np.array([.70,.65,.975,.0,.175])
for n in (30,40):
  for d in (0,.05,.1,.15,.2):
    fs=np.clip(fb+d,0,1); fm=np.clip(fb-d,0,1); R=20000
    S=rng.binomial(n,fs,(R,5))/n; Mi=rng.binomial(n,fm,(R,5))/n
    C=(S-Mi).mean(1); se=np.sqrt(((S*(1-S)+Mi*(1-Mi))/n).sum(1))/5
    lo=C-1.645*se
    print(f'n={n} true shift each way {d:.2f}: true C {np.mean(fs-fm):.3f}, P(CI lo>0) {np.mean(lo>0):.3f}, P(C>=0.10) {np.mean(C>=.10):.3f}')
