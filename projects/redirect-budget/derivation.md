# redirect-budget: derivation of the bound rows (Felix, T2)

Every step below is checked in `derive_check.py` (output `data/derive_check.json`; check ids E1-E6). Notation is
PREREG section 3. The question each step answers: if something adds a vector of a given size to the residual
stream before a head, how far can that head's attention logits, and then its attention, move?

## Step 1. The input RMSNorm only passes on a change of direction (E1, E2)

RMSNorm divides x by its own size and multiplies by a fixed gain: RMSNorm(x) = sqrt(D) g * x/||x||. So only the
unit direction u = x/||x|| reaches the head. Adding delta with ||delta|| <= rho ||x|| turns u by at most the angle
arcsin(rho) when rho < 1 (the ball of radius rho||x|| around x is seen from the origin under that half-angle), and
by up to pi when rho > 1 (the ball contains the origin). The distance between two unit vectors at angle theta is
2 sin(theta/2). Hence

$$\|\Delta u\| \le c(\rho) = \sqrt{2 - 2\sqrt{1-\rho^2}} \;(\rho \le 1), \qquad c(\rho) = 2 \;(\rho > 1).$$

E1: sympy shows c(rho)^2 = (2 sin(arcsin(rho)/2))^2 exactly; 20,000 random and tangent-point writes give a max
ratio ||Delta u||/c(rho) = 1 + 3e-9 (float error at the tangent point, where the bound is attained).
The raw query then moves by Delta q_raw = sqrt(D) W_Q diag(g) Delta u, so

$$\|\Delta q_{\text{raw}}\| \le s_q = \sqrt{D}\,\|W_Q \operatorname{diag}(g)\|_{op}\, c(\rho).$$

Note c(rho) >= rho, with ratio sqrt(2) at rho = 1: a first-order r/||x|| factor (the scout's formula) is not an
upper bound (Vera #11736 found PGD writes that beat it by up to 2.5x).

## Step 2. QK-norm again only passes on a change of direction (E3)

Qwen3 normalises each head's query and key the same way: q = sqrt(d) g_q * q_raw/||q_raw||. The same lemma with
rho replaced by s_q/||q_raw|| gives ||Delta q_hat|| <= c_q = c(s_q/||q_raw||), and likewise c_j for key j. This is
the factor 1/||q_raw|| that the scout's formula lacks. Both chords saturate at 2: no write, however large, can
move a normalised query or key further than to its opposite.

## Step 3. From a moved query or key to moved logits (E4)

RoPE rotates the query at position i and the key at position j; only the relative rotation R_{j-i} matters
(E4: identity error 2e-16). With q = sqrt(d) g_q * q_hat and k = sqrt(d) g_k * k_hat the logit is
z_ij = sqrt(d) (g_q * q_hat) . R_{j-i}(g_k * k_hat).
- A moved query changes logit j by Delta q_hat . w_j with w_j = g_q * R_{j-i} k_j. The spread of these changes
  over the keys is at most c_q max_{j,k} ||w_j - w_k|| = Omega_Q.
- A moved key j changes only logit j, by at most m_j = sqrt(d) ||g_k * R_{i-j}(g_q * q_hat)|| c_j. The spread is at
  most the sum of the two largest m_j: Omega_K = m_(1) + m_(2).
- Both moved: the change is the query term plus the key term plus a cross term of size at most
  sqrt(d) max|g_q| max|g_k| c_q c_j, so Omega_C = Omega_Q + Omega_K + 2 sqrt(d) max|g_q| max|g_k| c_q max_j c_j.

E5: on 12 toy Qwen3-style heads (D = 32, d = 8, 12 positions, one sink token 40x larger, RoPE, QK-norm with
random gains), Adam ascent on the logit spread (4 restarts x 150 steps, pair and smooth-max objectives) at rho in
{0.01, 0.1, 0.5, 0.9, 1}: max achieved/bound = 0.905 (Q), 0.835 (K), 0.638 (C), 60 cells each; never above 1.

## Step 4. From moved logits to moved attention (E6)

If every logit moves and the spread of the moves is at most Omega, the attention can move (in total variation, TV:
half the summed absolute change) by at most the result of pushing one group of keys up by Omega and the rest not
at all. For a group S that held attention A = a(S) before, that push gives

$$f(A,\Omega) = \frac{A e^{\Omega}}{A e^{\Omega} + 1 - A} - A, \qquad \mathrm{TV} \le \max_S f(a(S),\Omega) \le \tanh(\Omega/4).$$

The tanh value is reached at A* = 1/(1 + e^{Omega/2}) (E6, sympy: f'(A*) = 0 and f(A*) = tanh(Omega/4) exactly;
f is concave in A). With the measured clean attention a, only the masses a(S) that some key group really holds are
possible, which is tighter for heads that look at one token. The evaluator T() enumerates the 16 largest masses and
treats the rest as an interval; it is never below brute force (max deficit 2e-16 on 400 random heads) and at most
0.16 above it. Random and two-level logit moves never beat brute force (max excess 2e-16).
Credit: the sink-anchored form of this step is Lemma 1 of Choi, Kim, Kim (arXiv 2609.00746); the first-order
identity is in Emadi (arXiv 2602.18849).

## What the bound does not cover

- RMSNorm epsilon (1e-6) is ignored; it can only shrink the direction change slightly for tiny ||x||.
- The bound is for the head's own weights. For the 0.6B runs the head weights were trained, so the validity rows
  use the trained weights; a base-weight bound need not hold there (reported as its own line).
- The bound says how far a head CAN move. Which head a trained switch DOES move is the ranking question.
