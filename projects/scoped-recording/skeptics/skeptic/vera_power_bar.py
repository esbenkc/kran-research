# Vera: same rule as vera_power_v1.py, one shared bar e on both sides, e in {0, .02, .03, .05}.
exec(open('vera_power_v1.py').read().split("print('scenario")[0])
keep=['steer 0.00','steer 0.10','steer 0.15','salience 0.20 (both scoped arms flag up)','caution (S,M = parent warned mix)']
print('e | n | scenario | P(YES) P(NO) P(NS)')
for e in (0,.02,.03,.05):
    for n in (25,30):
        for k in keep:
            N,S,M=sc[k]; r=rule(N,S,M,n,e=e); print(f'{e:.2f} | {n} | {k} | {r[0]:.3f} {r[1]:.3f} {r[2]:.3f}')
