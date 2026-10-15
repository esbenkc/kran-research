# skeptic — own derivation of the counts behind the post (checked in derivation.py / derivation.json)

Model: x = [e_a; e_b], f_k(a,b) = sum_j V_kj (u_ja + s_jb)^2, with w_j = (u_j, s_j) in R^{2p}, V in R^{p x K}.
Loss (identified from saved train_loss, identify_loss.json): L_n = (1/N) sum_i 1/2 ||r_i - mean_i r_i||^2,
r_i = f(x_i) - onehot((a_i+b_i) mod p). No L2 term in L_n.

## Step 1. Per-neuron count (paper: 3p-1; on one-hot data: 3p-2)
A neuron has 2p + p = 3p weights. It only enters through the atom v (x) (w w^T) evaluated on inputs.
- Scaling: (w, v) -> (c w, v / c^2) leaves the atom fixed. Tangent (u, s, -2v). 1 dim. The paper removes this.
- Shift: every input satisfies (1_p, -1_p) . x = 0, so u -> u + t 1, s -> s - t 1 leaves every
  pre-activation u_a + s_b fixed. 1 more dim. The paper's secant count assumes inputs that span
  Sym^2(R^{2p}) (generic inputs); one-hot pairs do not, so this direction is invisible to the data.
- Exact rank of d f / d(u,s,v) over Z at random integer points, full data: p = 3, 5, 7, 11 -> 7, 13, 19, 31 = 3p-2.
  Both tangents are verified to lie in the kernel. (D1)
=> post footnote [^v3] ("misses one free direction per neuron ... K_eff(3p-2)/2") is correct.

## Step 2. K neurons, generic point; the hypothesis range
Exact rank of the centred Jacobian at random integer points = min(K(3p-2), p(N-1)) in 9/9 cases
(p = 5, 7; K = 2..8; N = 15..49). Centring over samples removes one output dimension per class,
so the image lives in a p(N-1)-dim space. (D2)
=> post: "the formula can only hold while K_eff(3p-2) < p(N-1)" is correct as a condition for the
generic count; I agree with skeptic2 (#659) that p(N-1)/2 must not be called a universal cap on lambda.
At a point where J has the maximal generic rank, rank is locally constant, so the zero set is a smooth
manifold of codim = rank and lambda = rank/2 exactly (standard: regular value / implicit function).

## Step 3. Memorising basin (Thm 5.2), first layer frozen
With W frozen the model is linear in V: f = V H^T. Centred Jacobian = I_p (x) (C H), rank p * rank(CH) = p*l
(l = rank of centred features, = min(l,K) when generic). Linear model => regular => lambda = p*l/2. (D3: 4/4.)
=> The memorising points are a known-answer check of the sampler, not of the grokking theory. Post line 44 says so.

## Step 4. Size of the shift correction
K(3p-1)/2 - K(3p-2)/2 = K/2; relative 1/(3p-1): 5% at p=7, 3.1% at p=11, 1.5% at p=23. (D4)
=> "at most 5% below the paper's" is correct. The NULL holds against both: max cell ratio vs K(3p-2)/2 is 0.794.

## Step 5. What SGLD reads at finite resolution
Posterior exp(-nb L - g/2 |th - th*|^2) with L = 1/2 th^T H th: per eigen-direction
nb E[L] = nb h / (2(nb h + g)) (sympy integral, D5), -> 1/2 only when nb h >> g.
So lambda_quad = 1/2 sum nb h/(nb h + g) = half MacKay's effective number of parameters (alpha = g/nb).
=> post [^sloppy] MacKay sentence is correct; [^lau] estimator formula matches the code convention.

## Step 6. Arithmetic in the post
6,732 = 99*68 = K(3p-1) at p=23, K=99; 3,784 = 11*(16*23-24); 1,700 = 68*50/2; 528 = 33*32/2; 3,128 = 92*68/2.
The (p-1)/2*(16p-24) count is a numerical observation (skeptic2/skeptic3), not derived here; the post calls it
"checked numerically", which is the right strength.
