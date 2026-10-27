"""Vera's own derivation checks (numpy only, no model).

D1. LEACE for a binary concept is rank 1:
    r(x) = x - delta * (delta' S^+ (x - mu)) / (delta' S^+ delta),
    delta = Cov(x, z) (proportional to mu1 - mu0), S = Cov(x) (total covariance).
    Check: equals the paper's W^+ P W form (W = S^{+1/2}, P = proj on col(W delta)),
    equalises class means exactly, and is the least-squares eraser among a random
    family of affine erasers that equalise class means (S-weighted norm optimal).
D2. Second-order price of equalising class means with a label-aware edit at one
    position: with a shared Fisher F the min is p0 p1 delta' F delta (then /2 for KL);
    with per-statement F_i the min is sum_z p_z m_z' (E[F_i^-1|z])^-1 m_z, which can
    be far lower. Checked by a direct convex solve (cvx closed form via Lagrange).
"""
import numpy as np

rng = np.random.default_rng(0)


def leace_rank1(X, z):
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / len(X)
    delta = Xc.T @ (z - z.mean()) / len(X)
    Sp = np.linalg.pinv(S, hermitian=True)
    a = Sp @ delta
    M = np.outer(delta, a) / (delta @ a)  # oblique projector
    return mu, M


def leace_paper(X, z):
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / len(X)
    Z = np.stack([1 - z, z], 1)
    Zc = Z - Z.mean(0)
    Sxz = Xc.T @ Zc / len(X)
    w, V = np.linalg.eigh(S)
    keep = w > w.max() * 1e-12
    W = (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T
    Wp = (V[:, keep] * np.sqrt(w[keep])) @ V[:, keep].T
    U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-10]
    P = U @ U.T
    return mu, Wp @ P @ W


def apply(X, mu, M):
    return X - (X - mu) @ M.T


def d1():
    d, n = 12, 4000
    A = rng.normal(size=(d, d))
    z = (rng.random(n) < 0.4).astype(float)
    X = rng.normal(size=(n, d)) @ A + np.outer(z, rng.normal(size=d) * 2)
    mu, M1 = leace_rank1(X, z)
    _, M2 = leace_paper(X, z)
    Xe = apply(X, mu, M1)
    gap = Xe[z == 1].mean(0) - Xe[z == 0].mean(0)
    cost = ((X - Xe) ** 2).sum(1).mean()
    # random alternative erasers x -> x - b (c'(x-mu)) with c'delta... any rank-1 eraser that kills
    # the mean gap must have b ∝ delta (column space) and c'delta = 1 (after scaling). LEACE minimises
    # E||M(x-mu)||^2 = |b|^2 c'Sc over c with c'delta=1 for the Euclidean norm -> c ∝ S^-1 delta.
    delta = (X - mu).T @ (z - z.mean()) / n
    S = np.cov(X.T, bias=True)
    best_alt = np.inf
    for _ in range(2000):
        c = rng.normal(size=d)
        c = c / (c @ delta)
        b = delta
        cst = (b @ b) * (c @ S @ c)
        best_alt = min(best_alt, cst)
    print(f"D1 rank1 vs paper max|diff| = {np.abs(M1 - M2).max():.2e}")
    print(f"D1 class-mean gap after erasure max|.| = {np.abs(gap).max():.2e}")
    print(f"D1 LEACE sq cost {cost:.4f} <= best of 2000 random rank-1 erasers {best_alt:.4f}: {cost <= best_alt + 1e-9}")


def d2():
    # per-statement Fisher, one position, shared mean-gap constraint
    # min sum_i e_i' F_i e_i / n  s.t. mean_{z=1} e - mean_{z=0} e = -delta
    # Lagrange: e_i = s_z F_i^-1 lam ; per-class min = m_z' H_z m_z, H_z = (E[F^-1|z])^-1
    # with m_1 = -p0 * delta... general: choose m_0, m_1 with m_1 - m_0 = -delta, minimise p0 m0'H0 m0 + p1 m1'H1 m1
    d, n = 6, 400
    z = (np.arange(n) % 2).astype(float)
    Fs = []
    for _ in range(n):
        B = rng.normal(size=(d, d)) * rng.lognormal(0, 1.5, size=d)
        Fs.append(B @ B.T + 1e-3 * np.eye(d))
    Fs = np.array(Fs)
    delta = rng.normal(size=d)
    p1 = z.mean(); p0 = 1 - p1
    Fbar = Fs.mean(0)
    shared = p0 * p1 * delta @ Fbar @ delta
    H = {k: np.linalg.inv(np.linalg.inv(Fs[z == k]).mean(0)) for k in (0, 1)}
    # optimum over m0 (m1 = m0 - delta): grad p0 H0 m0 + p1 H1 (m0 - delta) = 0
    m0 = np.linalg.solve(p0 * H[0] + p1 * H[1], p1 * H[1] @ delta)
    m1 = m0 - delta
    harm = p0 * m0 @ H[0] @ m0 + p1 * m1 @ H[1] @ m1
    # verify by constructing the edit and checking the constraint + cost
    e = np.empty((n, d))
    for k, m in ((0, m0), (1, m1)):
        idx = np.where(z == k)[0]
        Finv = np.linalg.inv(Fs[idx])
        lam = H[k] @ m
        e[idx] = Finv @ lam
    gap = e[z == 1].mean(0) - e[z == 0].mean(0) + delta
    cost = np.einsum('ni,nij,nj->n', e, Fs, e).mean()
    # constant shift cost under per-statement F (what a class-constant edit pays)
    cconst = (p0 * m0 @ Fs[z == 0].mean(0) @ m0 + p1 * m1 @ Fs[z == 1].mean(0) @ m1)
    print(f"D2 shared-F formula p0p1 d'Fbar d = {shared:.3f}; harmonic optimum = {harm:.3f}; "
          f"constructed edit cost {cost:.3f}, constraint resid {np.abs(gap).max():.1e}")
    print(f"D2 ratio harmonic/shared = {harm/shared:.3f} (<1 means the shared-F 'bound' is not a bound)")
    # toy from the room post
    F1, F2 = np.diag([1., 100.]), np.diag([100., 1.])
    Hh = np.linalg.inv((np.linalg.inv(F1) + np.linalg.inv(F2)) / 2)
    m = np.array([1., 0.])
    print(f"D2 toy: mean-F {m @ ((F1+F2)/2) @ m:.2f}, shift only stmt1 by (2,0): {0.5*np.array([2.,0])@F1@np.array([2.,0]):.2f}, harmonic {m@Hh@m:.2f}")


if __name__ == "__main__":
    d1()
    d2()


def d3():
    # post: alarm line = 2nd-highest of 200 true scores; a fresh exchangeable true score crosses it w.p. 2/201
    r = np.random.default_rng(3)
    S = r.normal(size=(200000, 201))
    t = np.sort(S[:, :200], 1)[:, -2]
    p = (S[:, 200] > t).mean()
    print(f"D3 P(new true > 2nd highest of 200) = {p:.5f} vs 2/201 = {2/201:.5f}")


if __name__ == "__main__":
    d3()
