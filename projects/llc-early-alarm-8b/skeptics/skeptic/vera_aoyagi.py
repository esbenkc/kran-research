"""Vera's independent Aoyagi-Watanabe (2005) RRR learning coefficient.
Model y = B A x, A: H x M, B: N x H, true rank r <= H. Returns (lambda, multiplicity m).
Cases as in Aoyagi & Watanabe, Neural Networks 18 (2005) 924-933, Thm 1."""
from fractions import Fraction as F

def aoyagi(M, N, H, r):
    assert 0 <= r <= H
    if M + H < N + r:
        return F(H*M - H*r + N*r, 2), 1
    if N + H < M + r:
        return F(H*N - H*r + M*r, 2), 1
    if M + N < H + r:
        return F(M*N, 2), 1
    s = 2*(H + r)*(M + N) - (M - N)**2 - (H + r)**2
    if (M + H + N + r) % 2 == 0:
        return F(s, 8), 1
    return F(s + 1, 8), 2

if __name__ == "__main__":
    # sanity: full rank square -> MN/2 (regular in the function space)
    for H in (1, 2, 5):
        assert aoyagi(H, H, H, H)[0] == F(H*H, 2)
    # r = 0 lower than regular count H(M+N)/2
    for (M, N, H) in [(10, 10, 4), (8, 6, 3), (20, 5, 4)]:
        for r in range(H + 1):
            lam, m = aoyagi(M, N, H, r)
            print(M, N, H, r, float(lam), m, "d/2 =", H*(M+N)/2)
            assert lam <= F(H*(M+N), 2)
