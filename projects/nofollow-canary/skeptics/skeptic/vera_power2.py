# Vera: sensitivity of the point-within-10 rule to the true raw-vs-md gap, with and without a gap gate (md-raw >= .10).
import numpy as np
rng=np.random.default_rng(1); N=60; S=20000
for raw,md in [(.74,.93),(.80,.93),(.65,.93)]:
    for name,(pad,body) in {'length':(raw,md),'markup':(md,raw),'half':((raw+md)/2,)*2}.items():
        k={a:rng.binomial(N,v,S) for a,v in dict(raw=raw,pad=pad,body=body,md=md).items()}
        pt=(abs(k['pad']-k['raw'])/N<=.10)&(abs(k['body']-k['md'])/N<=.10)
        gate=(k['md']-k['raw'])/N>=.10
        print(f"raw {raw} md {md} {name:7s} within10 {pt.mean():.3f}  with gap gate {(pt&gate).mean():.3f}")
