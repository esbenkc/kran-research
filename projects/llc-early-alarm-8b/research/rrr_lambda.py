"""Aoyagi & Watanabe (2005) learning coefficient of reduced-rank regression.
Model y = B A x, A: H x M, B: N x H (model rank H), true matrix rank r <= H.
Checked against Drton & Plummer 2017 (arXiv 1309.0911) Table 1 (N=5, M=3) and against
Aoyagi (2024) DLN theorem as printed in Lau et al. 2308.12108 App. I (2 layers)."""
from fractions import Fraction as F
import itertools

def rrr_lambda(M, N, H, r):
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        s = 2*(H+r)*(M+N) - (M-N)**2 - (H+r)**2
        return F(s, 8) if (M+H+N+r) % 2 == 0 else F(s+1, 8)
    if M + H < N + r:
        return F(H*M - H*r + N*r, 2)
    if N + H < M + r:
        return F(H*N - H*r + M*r, 2)
    return F(M*N, 2)  # M + N < H + r

def dln_lambda(Hs, r):
    """Aoyagi 2024 Theorem (Lau et al. App. I): Hs = [H0 (input), H1, ..., HL (output)]."""
    D = [h - r for h in Hs]
    L = len(Hs) - 1
    idx = range(len(Hs))
    best = None
    for k in range(1, len(Hs)+1):
        for S in itertools.combinations(idx, k):
            l = k - 1
            if l == 0:
                continue
            dS = [D[i] for i in S]; dN = [D[i] for i in idx if i not in S]
            if dN and not max(dS) < min(dN):
                continue
            if not sum(dS) >= l*max(dS):
                continue
            if dN and not sum(dS) < l*min(dN):
                continue
            a = sum(dS) - l*(sum(dS)//l)  # remainder: sum dS = l*floor + a
            lam = F(-r*r + r*(Hs[0]+Hs[-1]), 2) + F(a*(l-a), 4*l) - F(l*(l-1), 4)*F(sum(dS), l)**2 \
                  + F(1, 2)*sum(dS[i]*dS[j] for i in range(len(dS)) for j in range(i+1, len(dS)))
            best = lam
            return best
    return best

if __name__ == "__main__":
    # Drton & Plummer Table 1: N=5 responses, M=3 covariates, model rank i, true rank j
    table = {(1,0):F(3,2),(1,1):F(7,2),(2,0):F(3),(2,1):F(9,2),(2,2):F(6),
             (3,0):F(9,2),(3,1):F(11,2),(3,2):F(13,2),(3,3):F(15,2)}
    ok = all(rrr_lambda(3,5,i,j)==v for (i,j),v in table.items())
    print("Drton-Plummer Table 1 reproduced:", ok)
    bad = []
    for M in range(1,9):
        for N in range(1,9):
            for H in range(1,9):
                for r in range(0, min(M,N,H)+1):
                    a = rrr_lambda(M,N,H,r); b = dln_lambda([M,H,N], r)
                    if a != b: bad.append((M,N,H,r,a,b))
    print("RRR formula == DLN theorem (2 layers) on", 8*8*8, "shape grid; mismatches:", len(bad), bad[:5])
    for (M,N,H,r) in [(10,10,5,1),(10,10,5,3),(20,20,8,2),(20,20,8,5)]:
        lam = rrr_lambda(M,N,H,r); d = H*(M+N)
        print(f"M={M} N={N} H={H} r={r}: lambda={lam} ({float(lam):.2f}), d/2={d/2}, ratio {float(lam)/(d/2):.3f}")
