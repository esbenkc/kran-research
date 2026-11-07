# Vera: precedence version. YES first: CIlo(S-N)>0 AND CIlo(S-M)>0. Else NO: CIhi(S-N)<b OR CIhi(S-M)<b. Else Not settled.
exec(open('vera_power_v1.py').read().split("print('scenario")[0])
def rule2(N,S,M,n,b,R=20000):
    fn,fs,fm=[draw(P,n,R)[...,1] for P in (N,S,M)]
    _,lo1,hi1=ci(fs,fn,n); _,lo2,hi2=ci(fs,fm,n)
    yes=(lo1>0)&(lo2>0); no=(~yes)&((hi1<b)|(hi2<b)); return yes.mean(),no.mean(),1-yes.mean()-no.mean()
print('b | n | scenario | P(YES) P(NO) P(NS)')
for b in (.05,.10):
  for n in (25,30):
    for k in sc:
        N,S,M=sc[k]; r=rule2(N,S,M,n,b); print(f'{b:.2f} | {n} | {k} | {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}')
