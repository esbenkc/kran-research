# skeptic2 — independent derivation

Paper: Cullen, Estan-Ruiz, Danait, Li, "A Basin-Selection Perspective on Grokking via Singular
Learning Theory", arXiv 2603.01192 v3 (v1 title: "Grokking as a Phase Transition between Competing
Basins: a Singular Learning Theory Approach"). v3 numbering: Thm 4.1 (lambda = dim sigma_K(X)/2),
Cor 4.2 (under/over formulas), Thm 5.2 + Cor 5.3 (lazy), Cor 5.4 + Remark 5.5 (Fourier basin, K_eff).

Written before reading the post draft. Every numbered step has a check script next to it.

## 0. Set-up

Inputs x = [e_a; e_b] in R^{2p}, w_j = (u_j, s_j) with u_j, s_j in R^p. Then x . w_j = u_j[a] + s_j[b] and

    f_c(a,b) = sum_j V[c,j] (u_j[a] + s_j[b])^2 .

Loss: L_n = (1/2N) || P_perp (Yhat - Y) ||_F^2 on the N training pairs, no L2 term (the LLC of the
data loss; L2 is a prior). Only the centred predictions g(theta) in R^M, M = p(N-1), enter L_n.

## 1. What the data can see (check: derive_sympy.py [2])

x^T Q x = Q_aa + Q_{p+b,p+b} + 2 Q_{a,p+b}. The within-block off-diagonal entries of each Q_k never
enter. So the identifiable object is g(theta), not (Q_1..Q_p) in R^p (x) Sym(R^{d x d}) as in Thm 4.1.
The paper's ambient count pD = p * p(2p+1) is the wrong ambient space for one-hot data.

## 2. Per-neuron symmetries (check: derive_sympy.py [1], exact rational rank, p=3)

For one neuron the Jacobian of g has a 2-dim null space:
- scaling: (w, v) -> (t w, v / t^2)            [the paper removes this one]
- shift:   (u, s) -> (u + c 1, s - c 1)         [the paper misses this one]

The shift is exact because u[a] + s[b] is unchanged. Equivalently: every input satisfies
(1_p, -1_p) . x = 0, so the effective input dimension is 2p-1, not d = 2p.
Generic per-neuron count: (2p-1) + p - 1 = 3p-2, not d + p - 1 = 3p-1.
Sympy: p=3, rank 7 = 3p-2, null space spanned by exactly these two vectors.

## 3. Two local facts about lambda (my own proof sketches)

L1 (lower bound). At a realisable point (g(theta*) = target), lambda >= rank J / 2.
Proof: pick r = rank J components of g with independent differentials; by the implicit-function
theorem they are coordinates x_1..x_r near theta*. Then K = sum_i g_i^2 >= sum_{i<=r} x_i^2,
so vol{K < e} <= C e^{r/2}, so lambda >= r/2.

L2 (smooth points). If rank J at theta* equals the maximum rank of J over the whole parameter
space, rank is constant near theta* (max rank is an open condition). By the constant-rank theorem
g = (x_1..x_r, 0..0) in local coordinates, so K = sum x_i^2 and lambda = r/2 exactly (m = 1).

Correction to my room post #619: "lambda <= M/2 always" is false over the reals. Counterexample:
M=1, D=3, g = x^2+y^2+z^2, K = g^2 ~ rho^4, vol ~ rho^2 d rho -> lambda = 3/4 > 1/2.
Use L2 instead: at full-rank (rank = M) interpolating points lambda = M/2 = p(N-1)/2 exactly.

## 4. Consequences for the paper's formulas on mod-p addition

- Cor 4.2 over-param value p d(d+1)/4 = p^2(2p+1)/2. Max rank of J is M = p(N-1) <= p(p^2-1),
  and p d(d+1)/4 - p(p^2-1)/2 = p(p^2+p+1)/2 > 0 (sympy, derive_sympy.py [3]). So the over-param
  value is never attained on this task: at generic over-param points lambda = p(N-1)/2.
- Cor 4.2 under-param / Cor 5.4: generic rank is min(K(3p-2), p(N-1)) (numeric, my
  rerun_struct.jsonl and other skeptics' rank checks), so by L2 the generic value is
  min(K(3p-2), p(N-1))/2, not K(3p-1)/2. The difference is K/2 inside the subabundant region.
- Remark 5.5 (K -> K_eff): needs care twice:
  (a) dead neurons. One zero neuron contributes K ~ |v|^2 |w|^4 in the directions not spanned
      by the live neurons. Polar coordinates (r=|v| in R^p, rho=|w| on the 2p-1 effective dims):
      zeta(z) ~ int r^{2z+p-1} rho^{4z+2p-2} dr drho, poles at z = -p/2 and z = -(2p-1)/4.
      So one dead neuron alone adds up to (2p-1)/4 (< p/2), not 0. Numerically (dead_neurons*.json,
      p=7, 2 live neurons, rank ref 19): m = 0,1,2,4,8 dead neurons read 19.9, 22.0, 23.8, 29.2, 37.3
      at nbeta=1e4, gamma=10, but 17.9..18.3 (flat) at gamma=100. The visibility scales as
      nbeta / gamma^3 (degree-6 loss vs prior radius gamma^{-1/2}).
      Build's grokked checkpoints have K_eff = K (no dead neurons at wd = 1.0), so (a) does not
      enter tonight's numbers.
  (b) same-frequency neurons. Fourier-pure neurons that share a frequency are non-generic atoms;
      skeptic3 derives a cap 20p-36 per frequency. At the trained checkpoints the 1-5% Fourier
      impurity restores the generic numerical rank, but the restored directions have curvature
      down to 1e-12 (section 5). The restored rank is fragile (rank_tolerance.json): after
      build-style polishing (Adam, wd=0, L < 1e-8) the rank equals K(3p-2) only at relative SVD
      tolerance 1e-8..1e-10 (p11_K33_s0 polished: 914/947/975/994/1004/1021/1023 at tolerance
      1e-3/1e-4/1e-5/1e-6/1e-7/1e-8/1e-10). So L2 formally gives lambda = K(3p-2)/2 at these
      points, but that asymptotic value appears only at n*beta of order 1e16 or more. In practice
      the checkpoint behaves like the nearby degenerate (lower-rank) Fourier point, whose LLC is
      >= its rank/2 (L1) and is not pinned down by any formula checked tonight.

## 5. Why SGLD cannot read the value at these points (regular-surrogate null, spectrum.py)

Gauss-Newton eigenvalues h_i = eig(J^T J / N) at the grokked checkpoints span > 10 decades with
no gap (p11_K33: max 0.4, median 0.011, min 1e-12). A sampler at inverse temperature nbeta with
localisation gamma resolves only directions with nbeta h_i >~ gamma. For an exactly REGULAR
quadratic with the same spectrum the estimator reads

    lambda_sur = sum_i nbeta h_i / (2 (nbeta h_i + gamma))   (<= rank/2),

e.g. p17_K51_s0: 557 / 701 / 814 / 1008 at (nbeta, gamma) = (1e3,10) / (3e3,10) / (1e4,10) / (3e4,1)
versus rank/2 = 1249.5 and the paper's 1275. A regular model has no plateau, so any single nbeta
reads below rank/2, by an amount set by the checkpoint's conditioning, not by p alone.

Gap check (gap_compare.json): a random point with the same p, K, train set and weight scale has a
spectrum of 3.7 decades and 0.04% of directions below the frozen resolution gamma/nbeta = 1e-4
(lambda_sur = 0.96 x rank/2); the grokked p17_K51 point has 9.9 decades and 24% below (0.76x).

Null tests at the frozen setting (null_stepsize.json, own SGLD, own polish):
- step size crel 0.05/0.1/0.2/0.4: p17_K51_s0 966/969/960/956, p11_K33_s1 370/377/379/378. Flat.
- regular quadratic with the checkpoint's exact spectrum: exact stationary value 944 (p17_K51)
  and 397 (p11_K33) = 0.76x / 0.80x of rank/2; SGLD on the real net 956-969 and 370-379.

Caveat from skeptic3's nbeta sweep (room #682): the real net's estimate is FLATTER in nbeta than
the regular surrogate (p11_K33_s0: surrogate 349/406/439 vs SGLD 406-409/374-380/387-394 at
nbeta 1e3/1e4/1e5). A direction with K ~ h x^2 + c x^4 and tiny h contributes ~1/4 over a wide
nbeta range instead of moving from 0 to 1/2, which is the observed pattern. So the flat directions
act like the singular (quartic) directions of the nearby pure Fourier point. The regular surrogate
explains the size of the shortfall at the frozen setting (within 5-7%), not its nbeta dependence.

## 6. Steps I checked against the paper text

- Thm 5.2 l is the dimension of the span of random feature columns on the train set; on one-hot data
  after centring l = N - 1 generically (critic B / research.md agree). For the V-only model the LLC is
  exactly p * rank(F~)/2 (it is a regular linear model in V), so Thm 5.2 holds there as an identity.
- The paper's experiments use widths whose Cor 5.4 value exceeds p(N-1)/2 (research.md: p=53, K=1024,
  frac 0.4 gives 80,896 vs 29,760). By L2 the true LLC at generic points there is p(N-1)/2.
