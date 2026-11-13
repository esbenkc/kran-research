# Two cue-free lemmas behind the cross-seed (XS) contrast

Code: `theory/lemma.py` (CPU, ~4 min). Output: `theory/lemma.json`. No cue is read anywhere in this check.
Inputs: the saved adapters `raw/runs/nc_s0..9/step*.pt` (no-cue seeds) and `raw/runs/dir_u111k_a0.03125_s{k}/step*.pt`
(the six cue models). v1 = top input (right) singular vector of the stacked block-26 q, k, v delta dW = 2 B A (48 x 1024
rows for A, d = 1024).

## Lemma 1: two seeds' input slices overlap like random subspaces

**Plain words.** A LoRA module only changes how the layer reads inputs inside the span of the rows of its A matrix.
PEFT draws A at random from the seed. So each seed's adapter reads its own random slice of the input space. Two random
48-dimensional slices of a 1,024-dimensional space share, on average, 48/1024 of their directions.

**Statement.** Let S be any fixed k-dim subspace of R^d and S' the row space of a k' x d matrix with i.i.d. entries
from a distribution symmetric about 0. Then E[tr(P_S P_S')] = k k'/d. So the mean squared principal cosine
tr(P_S P_S')/k has expectation k'/d.

**Proof sketch.** For a signed permutation Q, A Q has the same law as A, and its row space is Q^T S'. So
E[P_S'] = Q^T E[P_S'] Q for every Q. A matrix that commutes with every signed permutation is c I (sympy, d = 3:
commutant = x I). The trace of P_S' is k', so c = k'/d. Then E[tr(P_S P_S')] = tr(P_S E[P_S']) = k k'/d.

**Check (lemma.json).** k = k' = 48, d = 1024, predicted 0.046875.
| check | value |
|---|---|
| Monte Carlo, 300 pairs of kaiming-uniform 48 x 1024 stacks | 0.04689 (sd 0.00135; 95% range 0.0445-0.0496) |
| measured A0 of seeds 0-9 (45 pairs) | mean 0.04691, range 0.0425-0.0493; 43 of 45 inside the MC 95% range |
| chance top-direction overlap of independent slices | rms |cos| 0.0325 (MC) vs 1/sqrt(1024) = 0.0313 |

## Lemma 2: subtracting other seeds' deltas barely moves the top direction

**Plain words.** If the reference matrix lives in input directions the target does not use, subtracting it cannot
remove anything the target holds. It can only add new directions. Those new directions take over the top spot only if
they are as strong as the target's own top direction. Averaging K other seeds makes them weaker.

**Statement.** Let W and M have orthogonal row spaces. Let s1 >= s2 be W's top two singular values, v1 its top right
singular vector, and sM = ||M||_2. Every top right singular vector x* of W - M satisfies |<x*, v1>| >= b(s1, s2, sM),
where b = min over alpha, c in [0, 1] of c alpha subject to c sqrt(alpha^2 s1^2 + (1 - alpha^2) s2^2) + sqrt(1 - c^2) sM >= s1.
If M is the mean of K deltas on mutually orthogonal slices, each with top singular value <= s, then sM <= s / sqrt(K).

**Proof sketch.** M v1 = 0, so ||W - M||_2 >= ||W v1|| = s1. Write x* = c a + s b, a in rowspace(W), b in rowspace(M),
unit vectors (a part outside both is mapped to 0, so a top vector has none). The triangle inequality gives
||(W - M) x*|| <= c ||W a|| + s sM, and ||W a||^2 <= alpha^2 s1^2 + (1 - alpha^2) s2^2 with alpha = <a, v1>. A top vector
must reach s1, so (c, alpha) satisfies the constraint, and <x*, v1> = c alpha. The minimum has a closed form in c (the
smaller root of the constraint, lemma.py `lemma2_bound`) and is scanned over 200,001 values of alpha. For the
corollary: for a unit x = sum_o x_o with x_o in slice o, ||M x|| <= (1/K) sum_o s ||x_o|| <= s / sqrt(K) (Cauchy-Schwarz).

**Check (lemma.json, XS = mean of the no-cue seeds among 0-5 other than the target's, K = 5).**
| read | bound b (6 models) | measured |cos(v1 RAW, v1 XS)| (6 models) |
|---|---|---|
| 8 updates | 0.711-0.815 | 0.960-0.975 |
| 16 (half-switch for 5 of 6) | 0.901-0.974 | 0.997-0.999 |
| T (32; 24 for seed 4) | 0.941-0.978 | 0.999-1.000 |
| 256 | 0.207-0.922 | 0.974-0.999 |
- The bound held in 48 of 48 (target, read) cells, although real slices overlap 0.047 rather than 0.
- PREREG frozen number (XS arm): |cos(v1 RAW, v1 XS)| > 0.9 in >= 5 of 6 at T. Measured: 6 of 6 (0.999-1.000). The
  bound alone, from s1, s2 and sM without any direction read, already exceeds 0.9 in 6 of 6 at T.
- sM sits below the corollary's worst case: at T = 32, the no-cue seeds 0-5 have top singular values 0.057-0.085, so
  s / sqrt(5) <= 0.038, and the measured sM is 0.019-0.023, close to s / 5 = 0.017. The five references add up almost
  like matrices with unrelated outputs, not aligned ones (one-off check, 2026-11-13 06:25 PST).

## What the lemmas do and do not say
- They say why the scout's cross-seed reference cannot remove the leading direction early in training: the leading
  direction sits in the target's own random slice, and other seeds sit elsewhere (Lemma 1), so the subtraction leaves
  v1 in place (Lemma 2).
- They do not predict the cue's e1 rank after subtraction. The cue's share of v1 is tiny (e1 ~ 1e-3, so |cos| ~ 0.03),
  smaller than the angle Lemma 2 allows (sin of acos 0.94 = 0.34). The XS rank prediction "beats chance in <= 1 of 6
  at T" follows only because v1 itself is unchanged, measured 0.999+.
- Lemma 1 needs i.i.d. symmetric entries (PEFT default A init, peft tuners/lora/layer.py:338). Under an A = 0, B random
  init (Hayou et al., arXiv 2406.08447) the input side would not be seed-locked at the start.
