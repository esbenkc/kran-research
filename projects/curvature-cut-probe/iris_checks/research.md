# Research notes — theory-1024-curvature (Iris)

Question (goal #5982): if we remove the output-flat directions from a falsehood probe (project it onto the
high-curvature eigenvectors of the output Fisher), does an output-matched evader have to change the model's outputs?

All arXiv texts below were fetched as full HTML on 2026-10-24 15:00-15:20 CEST and grepped
(swarm-theory-1024-curvature/lit/*.txt; papers are not committed). Quotes are verbatim.

## 1. The lemma is published (credit it; the novelty is the defender's use and the test)

| Work | What it says (verbatim) | Use in the post |
|---|---|---|
| Wang, Zhao, Cao, Yao, Shu, *FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers*, arXiv [2605.17231](https://arxiv.org/abs/2605.17231), Theorem 2 | "Among all interventions achieving a concept change q^T dh = rho, the minimum-norm minimizer of the output movement 1/2 dh^T G dh is dh* = rho/(q^T G^+ q) G^+ q"; "When G is rank-deficient the minimizers form an affine family differing by directions in ker(G), all of equal second-order cost"; App. B.4 "Null Space and Near-Null Space", Thm 10 kernel recursion; App. D.2 "Validity Radius of the Second-Order Model" | Our lemma KL >= D^2/(2 w^T F^+ w) is FishBack Thm 2 with the cost plugged in, read from the attacker's side (steering = evasion). Credit as "the same optimisation as FishBack's optimal steering direction". Their G is the same pullback Fisher (Gauss-Newton) we use. |
| Park, Nief, Choe, Veitch, *The Information Geometry of Softmax: Probing and Steering*, arXiv [2602.15293](https://arxiv.org/abs/2602.15293), Sec. 4.1 "Feasibility Constraints and Rank-Deficiency" | "when beta_W resides in the null space—where P_lambda is so concentrated that the dual coordinate hits the boundary of Phi—the regularized step v nudges the distribution toward regions of higher entropy" | Source for: at the final (post-norm) layer the output Fisher is the covariance of unembedding vectors under p, so it is (near-)rank-deficient exactly when the next-token distribution is concentrated. A probe component in that near-null space is cheap to move. |
| Zhao, Fletcher, Yu, Peng, Zhang, Shen, *The Adversarial Attack and Detection under the Fisher Information Metric*, arXiv [1810.03806](https://arxiv.org/abs/1810.03806) (AAAI 2019) | "the optimization is described by a constrained quadratic form of the FIM, where the optimal adversarial perturbation is given by the eigenvector, and the eigenvalues reflect the local vulnerability" | older input-space version of "Fisher eigen-structure = attack cost"; optional credit. |
| Amari, *Information Geometry and Its Applications* (2016); Martens, *New insights and perspectives on the natural gradient method*, arXiv [1412.1193](https://arxiv.org/abs/1412.1193) | KL(p_theta || p_theta+d) = 1/2 d^T F d + O(|d|^3); F = Gauss-Newton for softmax outputs | textbook second-order expansion; foldout further reading |

Implication for novelty wording: do NOT present the D^2/(2 w^T F^+ w) floor as ours. Present it as "the cheapest
second-order edit (the same calculation FishBack uses to steer)". What we did not find in any grepped paper:
(i) projecting a MONITOR onto range(F) so that the attacker's floor becomes finite, and (ii) a test of whether a
trained (full-stack LoRA) evader pays that floor.

## 2. Closest prior work on monitors and evasion (grep counts, 2026-10-24)

Terms grepped: Fisher / unembedding / null space / curvature / Hessian / kernel / logit lens.

| Work | Hits | Relation |
|---|---|---|
| Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*, [2412.09565](https://arxiv.org/abs/2412.09565) (ICLR 2026) | 0 on all terms | the attack family; no output-geometry analysis of the probe |
| Gupta & Jenner, *RL-Obfuscation*, [2506.14261](https://arxiv.org/abs/2506.14261) | 0 on all terms | fixed probes fall; max-pool/attention probes hold. No probe-geometry change |
| McGuinness, Serrano, Bailey, Emmons, *Neural Chameleons*, [2512.11949](https://arxiv.org/abs/2512.11949) | 0 on all terms | same attack SHAPE as ours: Eq. 1 "L_total = lambda L_LM + (1 - lambda) L_obf", L_LM "a KL divergence loss against a reference set of on-policy completions from the base model"; evasion "shifts activations into a lower-dimensional subspace". Probe geometry fixed. |
| Chaudhary & Barez, *Beyond Black-Box Obfuscation*, [2505.14300](https://arxiv.org/abs/2505.14300) | 0 on all terms | "geometric shifting" between subspaces; no Fisher |
| Keenan, Leckie, Erfani, *LLMs Learn to Evade Latent Monitors from Prior Feedback Alone*, [2609.36490](https://arxiv.org/abs/2609.36490) | 0 on all terms | — |
| Libon, Panfilov, Rank, Chen, Geiping, Andriushchenko, *Alignment via Training Against Probes Without Losing Monitorability*, [2609.38645](https://arxiv.org/abs/2609.38645) | "curvature" 4 hits, all in a flat-earth example text | — |
| Piras et al., *Latent-space Attacks for Refusal Evasion*, [2605.21706](https://arxiv.org/abs/2605.21706) | 0 | evasion = projection onto a probe's decision boundary |
| Ziqian Zhong, *Spurious probes as a black-box alternative to activation probing*, LessWrong 2026-09-25 (updated Sep 29), [greaterwrong.com/posts/gZh6txHhp8sm832sE](https://www.greaterwrong.com/posts/gZh6txHhp8sm832sE/) (found by Hugo, skeptic2/novelty.md; opened 200 and grepped lit/zhong_lw.txt 2026-10-24) | under a Neural Chameleon trigger, threshold fixed at 5% FPR without trigger: 'Activation probe, last token' TPR 0.00 / 0.06 / 0.13 vs 'Spurious probe ensemble' 0.43 / 0.38 / 0.45 / 0.38 / 0.40; toy model: 'each answer's logit reads a random projection of the regime direction in the activations' | same spirit (read the signal through the OUTPUTS, which an output-matched evader cannot freely move); different design: black-box answers, no output-matched fine-tune, no price. One credit line. |
| Makelov, Lange, Nanda, *Is This the Subspace You Are Looking For?*, [2311.17030](https://arxiv.org/abs/2311.17030) | "nullspace" 31 | closest CONCEPT: a direction splits into a part in the nullspace of all downstream components ("causally disconnected") and a rowspace part; their table patches "v_resid rowspace" vs "v_resid nullspace". Our projection is the probability-weighted, last-layer version of "keep the rowspace part". |
| Park, Choe, Veitch, *The Linear Representation Hypothesis and the Geometry of LLMs*, [2311.03658](https://arxiv.org/abs/2311.03658) | "unembedding" 62 | causal inner product from unembedding covariance; concepts as unembedding directions |
| Elazar et al., *Amnesic Probing*, [2006.00995](https://arxiv.org/abs/2006.00995) | "nullspace" 4 (INLP) | probe info that the model does not USE; our flat directions are the output-side version |

## 3. Facts about our own setup (code, not papers)

- 1023 attack (kran-research projects/norm-rescale-free/attack_norm.py): the loss is a twin-gap probe loss plus
  lam/2 x (KL on city statements, all masked positions) + lam/2 x (KL on Pile retain text). attack_norm.py:97
  kl_mean averages over every masked position. The checkpoint logs BOTH `eval_kl` (mean over all eval-statement
  tokens) and `eval_kl_country_pos` (KL at the probed country position) (attack_norm.py checkpoint return dict).
  The lemma's floor bounds the per-position KL at the probed token; the eval mean dilutes it by the statement length.
- Post-norm space (readout 29): logits are linear in h, F = W_U^T (diag p - p p^T) W_U = Cov_p(unembedding rows).
  rank(F) <= (number of tokens with non-negligible p) - 1 in practice (Park et al. 2026 Sec. 4.1).
  Projecting w onto F's top eigenvectors keeps only the part of w that moves the logits of the likely next tokens.
- 1023 measurement (research.md 3d, pre-norm readout 28, 12 false eval states): top eigenvalue 0.05-0.09;
  280-1050 eigenvalues < 1e-6; share of w in them 16-59%; share below 1e-4: 90-94%.

## 4. Own checks (Iris, 2026-10-24; code in iris_checks/, all full 151,936-token vocab, float64)

| Check | Result | File |
|---|---|---|
| Fisher truncation to top-K tokens (6 clean eval-false post-norm states, tok29 U) | left-out mass beyond top-8192 = 1.4e-4..2.7e-3; ||F_full - F_8192||_2 = 6.6e-5..1.1e-3, larger than the eigenvalues where 81-92% of u lives (< 1e-4). Share of u below 1e-6: full 0.0005..0.31 (median ~0.02), top-8192 up to 14x larger. Truncation biases the floor DOWN. | trunc_check.py/.json; room #6035 |
| Post-norm vs pre-norm flat share | post-norm share of u below 1e-6 ~0.02 (median, 6 states) vs 45% pre-norm (1023): the RMSNorm Jacobian g/rms shrinks every pre-norm eigenvalue, so absolute cuts do not transfer | same |
| Free-space vs ellipsoid floor, U, first 32 caught eval-false states | free hyperplane L-BFGS min median 7.6e-5 per statement (not converged; the certificate below gives 3.5e-5). My ellipsoid found move (Adam, 200 steps) median 0.014 was a WEAK search: RETRACTED (#6109). Vera (8.4e-5, L-BFGS) and Hugo (~1.4e-4, Lagrangian tilt) found moves ~100x cheaper, so the sphere costs ~2x the free floor, not 180x. Use their searches. | floor_bound.py, floor_bound_smoke.json; room #6052 -> #6109 |
| Certified lower bound (Donsker-Varadhan with a linear tilt q = p(1 - nu (e - m).y), y = F^-1 a), first 10 caught eval-false states, U | LB / second-order formula = 0.998 (median); LB p50 3.5e-5. The naive lemma is a valid floor here to 0.2% (exact correction included). | cert_tilt.py; room #6066 |
| Linear-tilt certificate per probe (J5 partial, 39 caught eval-false states of the 160-pair sample) | LB / formula median: U 0.995, R 0.995, P 0.666, literal P_F w 0.80. For P the tilt is capped by q >= 0 (nu_max < the optimal nu), so it is LOOSE there, not the truth. | cert_tilt2.py, cert_floor.py; #6138 |
| Exact free-space minimum (Newton on the hyperplane a.d = -D, float64, full vocab; DV certificate at the Newton point equals f to 1e-9..1e-12, so the minimum is pinned from both sides) | stmt 11: U 1.422e-5 (formula 1.419e-5, 1.002x), P 4.965e-5 (formula 5.649e-5, 0.879x); stmt 13: U 1.003x, P 2.696e-4 vs 2.981e-4 (0.905x). Moves are 2.7-5.0% of |z|. The formula is exact for U and ~10% high for P (n=2 each). | exact_min.py, exact_min.json, exact_min.log; #6151, #6165 |
| Does the fine-tune pay the floor? (1023 U runs, step 300, 20 eval-false states of the sample) | measured country-token KL / certified minimum for the same score shift: median 153x / 230x / 222x (seeds 0/1/2), p10-p90 78-1271x; 0/60 below the certificate (validity check on real fine-tuned states) | efficiency.py, efficiency_k128p_step300.json; #6118 |

### 4a. Why the certificate is exact (derivation, for the foldout)
KL(p || p_z') = log E_p[exp(e.d)] - E_p[e.d] with d = z' - z (logits are linear in the post-norm state).
Donsker-Varadhan: log E_p[exp(X)] >= E_q[X] - KL(q||p) for every q. So KL(p||p_z') >= (E^T(q - p)).d - KL(q||p).
With q = p(1 - nu tv), E^T(q - p) = -nu E^T(p tv) = -nu F y. If F y = kappa a + r (r _|_ a), a.d <= -D and the
state stays in the RMSNorm ellipsoid, then KL >= nu (kappa D - R|gamma r| + r.z) - KL(q_nu||p). For y = F^-1 a,
r = 0, kappa = 1, and KL(q_nu||p) = nu^2 a'F^-1 a / 2 + O(nu^3): the best nu = D/(a'F^-1 a) recovers
D^2/(2 a'F^-1 a). The bound needs no second-order approximation; it holds for every move, of any size.
Credit: the variational formula is Donsker & Varadhan (1975); its use for KL lower bounds is standard (e.g.
Belghazi et al., MINE, arXiv 1801.04062; link checked 200, Kasper #6111).

## 5. Novelty verdict (draft 2026-10-24 15:45 CEST; for Kasper's review)

- NOT ours (credit in the post): the second-order cheapest edit D^2/(2 a'F^+ a) and its minimiser F^+ a (FishBack
  2605.17231 Thm 2, as steering); the final-layer Fisher is the covariance of unembedding rows under p and goes
  near-singular when p is concentrated (Park et al. 2602.15293 Sec 4.1); directions downstream components ignore
  (Makelov et al. 2311.17030); Fisher eigen-structure as attack cost (Zhao et al. 1810.03806); the attack shape
  "probe term + KL to the base model" (Neural Chameleons 2512.11949 Eq. 1; Bailey et al. 2412.09565).
- Not found in any grepped paper (13 full texts; 0 hits for Fisher/curvature/unembedding/null space in the five
  evasion papers; 0 hits for monitor/evasion/adversarial in FishBack): (i) restricting a MONITOR probe to the
  high-curvature output directions to raise the evader's price; (ii) a test of whether a trained full-stack evader
  with an output-matching term pays the predicted price; (iii) a certified (Donsker-Varadhan) per-state lower bound that matches the second-order formula to <1% at
  this readout. [The earlier 'norm sphere sets the price' claim is RETRACTED: my sphere search was weak, #6109.] Wording: "we did not find"; no "first".
- Searches run (2026-10-24): web "probe projected onto output-relevant subspace unembedding Fisher robust to
  obfuscation activation monitor evasion"; "arXiv 2026 linear probe null space downstream readout evasion cost KL
  lower bound Fisher information monitor"; "Fisher information activation space probe robustness obfuscated
  activations OR probe evasion language model 2026"; "arXiv probe restricted to unembedding row space logit lens
  monitor harder to evade obfuscation attack". Nearest hits are steering papers (FishBack, Park 2026, 2609.11063
  info-geometry of LLMs) and null-space steering/backdoor papers (AlphaSteer 2506.07022, 2604.12359), none about
  monitor robustness.
- Every arXiv link above returned HTTP 200 on 2026-10-24 (curl -L), plus blog.kran.ai/norm-rescale-free/ and
  /panel-fine-tune/.

## 6. Floor estimators, for the expert foldout (Iris; code in iris_checks/, all CPU float64, full 151,936-token vocabulary)

Setting: z = clean post-norm country-token state (readout 29), logits l = E z (tied unembedding E, no bias),
p = softmax(l). Probe s(z) = a.(z - mu) + b. A caught false statement must drop by D = s - t (t = clean 1% line).
Output change at that token: f(d) = KL(p || softmax(E(z + d))). f is CONVEX in d (log-sum-exp of a linear map
minus a linear term). Every reachable post-norm state lies on the RMSNorm ellipsoid {gamma * u : |u| = sqrt(dim)}.

| Quantity | Definition | Estimator | Convergence / exactness check | File |
|---|---|---|---|---|
| small-move formula ("naive") | D^2 / (2 a'F^-1 a), F = E'(diag p - p p')E | eigh of F (float64), eps = 1e-8 lam_max (sens. 1e-10, 1e-6) | eps sensitivity < 1% for U; Hugo/Vera rebuild | cert_tilt2.py (ay), Hugo gate_hugo.py |
| exact free-space minimum | min f(d) s.t. a.d = -D | Newton on the hyperplane, backtracking, start = formula optimum | projected gradient <= 1e-12; DV certificate at the Newton point = f to 1e-9 | exact_min.py |
| exact sphere minimum | min f(d) s.t. a.d = -D, |(z+d)/gamma| <= sqrt(dim) (convex hull of the reachable set) | log-barrier Newton, tau = 1e4..1e12 (x100), backtracking; Hessian float32 (direction only) | Newton decrement < 1e-14; DV certificate (closed-form linear min over ball and half-space) = f to 2e-8; |u|/R -> 1 shows the optimum is on the surface, so it equals the minimum over reachable states | ball_min.py |
| linear-tilt certificate | max_nu nu(kappa D - r) - KL(q_nu||p), q_nu = p(1 - nu (e - m).y), y = F^-1 a | exact; nu on a 128-point grid | tight for U and R (0.99x formula), LOOSE for P (0.6-0.7x) because q >= 0 caps nu | cert_tilt2.py |

DV certificate (why the brackets are proofs, not fits): for any distribution q over the vocabulary,
log E_p[exp(e.d)] >= E_q[e.d] - KL(q||p), so f(d) >= g.d - KL(q||p) with g = E'(q - p). Minimising the right side
over the feasible set (hyperplane, or ellipsoid and half-space, both in closed form) gives a number that no
reachable state can beat. At q = softmax(E z*) for the optimum z*, the bound equals f(z*) (strong duality).

## 7. Overpay per arm (efficiency.py, step 150, same 40 eval-false states; posted #6259)
| run | floor for the shift made (median, formula) | measured country-token KL (median) | overpay vs formula | below certificate |
|---|---|---|---|---|
| U 1023 s0 / s1 / s2 | 1.6e-5 / 1.4e-5 / 3.1e-5 | 0.0055 / 0.0060 / 0.0054 | 422x / 456x / 210x | 0/40 each |
| P s0 (new code) | 8.2e-5 (exact ~0.89x of it) | 0.0046 | 61x | 0/40 |
Reading [INFERENCE, n=1 P run]: the cut raised the per-statement floor ~3-5x; the fine-tune's output change stayed the same, so its overpay fell ~5x.
| R s0 (new code) | 5.4e-6 | 0.0023 | 552x | 0/40 |
Slope of ln(KL measured) on ln(floor) across R / U(median) / P: 0.23 (3 points; not monotone: P below U). #6279, #6280.
| U s0 (new code, D10 baseline) | 1.7e-5 | 0.0049 | 332x | 0/40 |
Same-code seed 0: slope ln KL on ln floor = 0.25; P raised the floor 4.8x, paid 0.95x of U; R lowered it 3.2x, paid 0.47x (#6302).
| P s1 (new code) | 7.6e-5 | 0.0046 | 79x | 0/40 |
| R s1 (new code) | 4.1e-6 | 0.0030 | 822x | 0/40 |
| U s1 (new code) | 1.4e-5 | 0.0069 | 622x | 0/40 |
