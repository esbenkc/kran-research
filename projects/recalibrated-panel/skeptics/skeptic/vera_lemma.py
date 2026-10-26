"""Vera: own check of the order-statistic bound. n numbers standardised by their own mean and sd.
Claim: k-th highest z <= sqrt((n-k)/k) with ddof=0 sd; with ddof=1 the bound is sqrt((n-1)(n-k)/(n k))."""
import numpy as np, sympy as sp
from scipy.optimize import minimize
n_, k_, a = sp.symbols('n k a', positive=True)
# extremal config: top k equal a, rest equal b = -k a/(n-k); sum z^2 = n (ddof 0)
b = -k_*a/(n_-k_)
sol = sp.solve(sp.Eq(k_*a**2 + (n_-k_)*b**2, n_), a)
print('ddof0 bound:', [sp.simplify(s) for s in sol])
sol1 = sp.solve(sp.Eq(k_*a**2 + (n_-k_)*b**2, n_-1), a)
print('ddof1 bound:', [sp.simplify(s) for s in sol1])
# numeric: maximise k-th largest z over random starts (maximise min of top-k after standardising)
rng = np.random.default_rng(0)
def kth(x, k, ddof):
    z = (x - x.mean())/x.std(ddof=ddof); return np.sort(z)[::-1][k-1]
for n in (200,):
    for k in (1, 2, 4):
        for ddof in (0, 1):
            best = -1
            for _ in range(20):
                x0 = rng.normal(size=n); x0[:k] += 5
                r = minimize(lambda x: -kth(x, k, ddof), x0, method='Nelder-Mead', options={'maxiter': 20000, 'xatol':1e-10,'fatol':1e-12})
                best = max(best, -r.fun)
            # constructive point
            x = np.full(n, -k/(n-k)); x[:k] = 1.0
            th = np.sqrt((n-k)/k) if ddof == 0 else np.sqrt((n-1)*(n-k)/(n*k))
            print(f'n={n} k={k} ddof={ddof} bound={th:.4f} constructive={kth(x,k,ddof):.4f} NM-search={best:.4f}')
