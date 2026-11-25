"""Check Felix's split lemma lambda(M,N,H,r) = r(M+N-r)/2 + lambda(M-r,N-r,H-r,0) with my own Aoyagi code, exhaustively."""
from vera_aoyagi import aoyagi
import sympy as sp
bad = []; n = 0
for M in range(1, 25):
    for N in range(1, 25):
        for H in range(1, min(M, N) + 1):
            for r in range(0, H + 1):
                lhs = aoyagi(M, N, H, r)[0]
                rhs = sp.Rational(r * (M + N - r), 2) + aoyagi(M - r, N - r, H - r, 0)[0] if H - r > 0 else sp.Rational(r*(M+N-r), 2)
                n += 1
                if lhs != rhs: bad.append((M, N, H, r, lhs, rhs))
print(f"checked {n} cases with H<=min(M,N); mismatches {len(bad)}"); print(bad[:10])
bad2 = []; n2 = 0
for M in range(1, 20):
    for N in range(1, 20):
        for H in range(min(M, N) + 1, 25):
            for r in range(0, min(M, N) + 1):
                lhs = aoyagi(M, N, H, r)[0]; rhs = sp.Rational(r*(M+N-r), 2) + aoyagi(M-r, N-r, H-r, 0)[0]
                n2 += 1
                if lhs != rhs: bad2.append((M, N, H, r, lhs, rhs))
print(f"H>min(M,N): checked {n2}; mismatches {len(bad2)}; first {bad2[:5]}")
