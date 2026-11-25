"""Vera's independent rebuild of Aoyagi & Watanabe (2005) RLCT for reduced-rank regression.
Model: y = B A x, A in R^{H x M}, B in R^{N x H}; true map has rank r <= H.
Written from the paper's theorem (as restated in Lau et al. 2308.12108, App. on DLN/RRR), not from any room code.
"""
import sympy as sp


def aoyagi(M, N, H, r):
    if N + r <= M + H and M + r <= N + H and H + r <= M + N:
        base = sp.Rational(2 * (H + r) * (M + N) - (M - N) ** 2 - (H + r) ** 2, 8)
        if (M + H + N + r) % 2 == 0:
            return base, 1
        return base + sp.Rational(1, 8), 2
    if M + H < N + r:
        return sp.Rational(H * M - H * r + N * r, 2), 1
    if N + H < M + r:
        return sp.Rational(H * N - H * r + M * r, 2), 1
    return sp.Rational(M * N, 2), 1  # M + N < H + r


def manifold_half_dim(M, N, H):
    """Regular (r = H) case: lambda = dim of rank-H M x N matrices / 2 = H(M+N-H)/2."""
    return sp.Rational(H * (M + N - H), 2)


if __name__ == "__main__":
    M, N, H = 32, 32, 16
    for r in range(0, H + 1):
        lam, m = aoyagi(M, N, H, r)
        print(f"M=N={M} H={H} r={r:2d} lambda={lam} m={m}")
    assert aoyagi(32, 32, 16, 1) == (236, 2)
    assert aoyagi(32, 32, 16, 8) == (312, 1)
    # sanity 1: r = H gives the regular half-dimension
    for (M, N, H) in [(32, 32, 16), (10, 7, 5), (8, 12, 4), (20, 20, 3)]:
        assert aoyagi(M, N, H, H)[0] == manifold_half_dim(M, N, H), (M, N, H)
    # sanity 2: lambda <= d/2 and monotone non-decreasing in r
    for (M, N, H) in [(32, 32, 16), (16, 16, 8), (24, 32, 12), (8, 12, 4)]:
        d = H * (M + N)
        prev = -1
        for r in range(H + 1):
            lam = aoyagi(M, N, H, r)[0]
            assert lam <= sp.Rational(d, 2) and lam >= prev, (M, N, H, r, lam)
            prev = lam
    print("all Aoyagi sanity checks pass")
