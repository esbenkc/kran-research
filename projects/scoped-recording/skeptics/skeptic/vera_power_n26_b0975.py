# Vera: Tove's APPROVE rule (#16018 condition 2) at n=26: YES first CIlo>0 on both clean-flag contrasts; else NO CIhi<0.0975 on either; else NS.
exec(open('vera_power_prec.py').read().split("print('b | n")[0])
# percentile-bootstrap version for a cross-check on 2,000 simulated runs (B=500 within agent x arm)
def rule_boot(N,S,M,n,b=.0975,R=2000,B=500):
    out=np.zeros(3)
    for _ in range(R):
        f=[rng.binomial(n,P[:,1]) for P in (N,S,M)]
        bs=[rng.binomial(n,(fi/n)[None,:].repeat(B,0))/n for fi in f]
        d1=(bs[1]-bs[0]).mean(1); d2=(bs[1]-bs[2]).mean(1)
        lo1,hi1=np.quantile(d1,[.05,.95]); lo2,hi2=np.quantile(d2,[.05,.95])
        y=lo1>0 and lo2>0; no=(not y) and (hi1<b or hi2<b); out+=[y,no,not(y or no)]
    return out/R
print('scenario | n=26 normal CI: P(YES) P(NO) P(NS) | bootstrap CI (2,000 runs, B=500): P(YES) P(NO) P(NS)')
for k in sc:
    N,S,M=sc[k]; r=rule2(N,S,M,26,.0975); rb=rule_boot(N,S,M,26)
    print(f'{k} | {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} | {rb[0]:.3f} {rb[1]:.3f} {rb[2]:.3f}')
