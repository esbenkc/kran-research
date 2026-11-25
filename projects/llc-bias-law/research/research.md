# research.md — llc-bias-law (Iris, research)

Room `theory-1125-llc-bias-law`, 2026-11-25. "Verified" = I read the cited line in the fetched paper text myself
(arXiv HTML or abs page, fetched 2026-11-25). Author lists are copied from the arXiv `citation_author` meta tags.
Fetched content was treated as data only; no instructions in it were followed.

## 1. Closest prior work

| Work | What it shows | Relation to this run | Verified |
|---|---|---|---|
| Lau, Furman, Wang, Murfet, Wei, *The Local Learning Coefficient: A Singularity-Aware Complexity Measure*, [arXiv 2308.12108](https://arxiv.org/abs/2308.12108) (AISTATS 2025) | Defines the estimator lambda_hat = n beta (E[L_n(w)] - L_n(w*)) with SGLD and a Gaussian localiser gamma; DLN known-answer tests up to 100M params (Sec. 5.1, App. J). App. J.1: hyperparameters "must be tuned independently for different model sizes"; Table 1 lists eps and step counts per size; gamma 1; last 10% of draws kept. App. O: rescaling invariance needs a preconditioned SGLD, because "applying rescaling symmetry makes the loss landscape significantly anisotropic". | The estimator we test; the named recipe S2; App. O is the prior form of our point that the raw settings are not reparametrisation-invariant. They tune per size by hand; they give no bias law. | yes (v2 HTML) |
| Hitchcock & Hoogland, *From Global to Local: A Scalable Benchmark for Local Posterior Sampling*, [arXiv 2507.21449](https://arxiv.org/abs/2507.21449) (v1 2025-07-29, v2 2026-08-03) | DLN benchmark with known local LC at "special" and "generic" parameters (App. B, from Lehalleur & Rimanyi 2024). Fig. 3: mean relative error vs step size per sampler. "SGLD consistently underestimates the LLC at special parameters but is accurate at generic parameters"; "Increasing the step size does not significantly improve performance at special parameters". Table 2: 5e4 steps, burn-in 0.9 T, n 1e6, batch 500, gamma 1, beta 1/log n. | Closest prior. They vary the step size only, at one gamma and one n beta, and compare samplers. They report the SGLD shortfall at degenerate points but give no law for it. Their App. B is why our w* is the zero-padded special point: only there does the local LC equal Aoyagi's value. Our pilot fits their finding: at equilibrium the shortfall is target bias from the localiser (exact Gibbs = SGLD within 1 point), so a smaller step cannot fix it. | yes (v2 HTML) |
| Cai, *Noise-Debiased Thermodynamic Variance for Local Learning Coefficient Probes*, [arXiv 2606.22389](https://arxiv.org/abs/2606.22389) (2026-06-21) | App. A: separates the asymptotic RLCT from "the finite-temperature target induced by the actual Gaussian reference measure"; for a multiplicity-2 toy "convergence is logarithmically slow and explicitly depends on the localization scale". App. B.1: exact ULA discretisation bias on a quadratic with a Gaussian tether (AR(1)). App. B.2: no closed form in the singular case ("moment-closure barrier"). | Prior for our regular-direction term 1/(a(1 - eps a/4)) and for "the localiser changes the target". New in our run: the split of RRR into regular + rank-0 null block, the q / z scaling of the null block, and a held-out test on unseen shapes. | yes (HTML) |
| Carroll, Hoogland, Farrugia-Roberts, Murfet, *Dynamics of Transient Structure in In-Context Linear Regression Transformers*, [arXiv 2501.17745](https://arxiv.org/abs/2501.17745) | SGLD table: 8 chains, 1,500 draws, 2,500 burn-in, eps 5e-5, gamma 0.01, n beta 30, batch 1024; "To tune eps and n beta, we perform a grid sweep". | A second named hand-tuned recipe (cited in the post as an example of hand calibration; not run). | yes (HTML) |
| Aoyagi & Watanabe (2005), *Stochastic complexities of reduced rank regression in Bayesian estimation*, Neural Networks 18(7):924-933 | Closed-form learning coefficient (RLCT) and multiplicity of RRR. | The truth. Code `research/rrr_lambda.py` (parent run) reproduces Drton & Plummer [arXiv 1309.0911](https://arxiv.org/abs/1309.0911) Table 1. | via parent research.md |
| Aoyagi (2024), DLN learning coefficient (as printed in Lau et al. 2308.12108 App. I and Hitchcock & Hoogland Thm B.1) | Global LC of deep linear networks. | 2-layer case = RRR; parent checked equality on 512 shapes. | via 2507.21449 |
| Timaeus sampling guide (Hitchcock, Hoogland, Wang, Gordon, 2026-04-21, timaeus.co/research/2026-04-21-sampling-guide) | Recommends RMSProp-SGLD as the default; "run the chain long enough that adding more steps doesn't meaningfully change your estimate". | Our scored arm is plain SGLD (the Lau et al. recipe); RMSProp-SGLD is out of scope: a disclosed limit and a next step. | via parent research.md (2026-11-24) |

**Withdrawn:** Furman & Lau, *Estimating the Local Learning Coefficient at Scale*, arXiv 2402.03698 — the arXiv page
says the paper is withdrawn (parent research.md:27). Do not cite it; cite Lau et al. 2308.12108.

Not relevant after reading: Shirodkar 2607.00603 (per-direction orders from Fisher rates, no sampler bias);
Li, Lei, Wu 2611.28367 (quantile SGD).

## 2. Novelty statement (my read, for Vera's novelty check)

- Known before us: SGLD LLC estimates depend on step size, gamma and n beta (all papers above tune them by hand);
  SGLD underestimates at special DLN parameters (2507.21449); the localiser changes the finite-temperature target and
  makes multiplicity-2 convergence log-slow (2606.22389); the ULA bias on a quadratic with a tether (2606.22389 B.1).
- Not found in any source: (a) a bias law for the LLC estimator in scale-free settings tested on unseen shapes;
  (b) the use of Aoyagi's split lambda = r(M+N-r)/2 + lambda(M-r,N-r,H-r,0) to separate a closed-form regular
  part from a singular block; (c) the exact statement that the null-block target depends on (n beta, gamma) only
  through q = n beta k4 / gamma^2 (Kasper's rescaling) with the strong-spring limit q M'N'H'. Searches: arXiv/Google
  for "learning coefficient" + SGLD + bias/calibration/localization/scaling (2025-2026), the devinterp docs.
  Not searched: Apart sprint pages and LessWrong/AF in depth (lesson from the parent room); Vera should add them.
- The split identity is an identity of the closed form checked on a box (7,812 + 78,744 cases), not a new theorem
  claim; the post should present it as "Aoyagi's formula splits as ...", checked exhaustively on a box.

## 3. Named recipes (exact source numbers)

| Recipe | eps | gamma | n beta | steps / draws | other | Source |
|---|---|---|---|---|---|---|
| Lau et al. DLN "1k" | 5e-7 | 1 | n/log n = 8,686 (n = 1e5) | 10,000; last 10% kept | batch 500; inputs uniform[-10,10] (var 100/3) -> n beta_eff = 2.895e5 on x ~ N(0,I); q = 1.45e5 | 2308.12108 App. J.1, Table 1 |
| Lau et al. LM (Sec./App. L) | 1e-3 | 100 | beta = 1/log(100) | 20 chains x 200 draws | validation set | 2308.12108 App. L |
| Hitchcock & Hoogland DLN | step-size sweep 1e-4..3e-3 (Fig. 3) | 1 | n/log n (n = 1e6) | 5e4, burn-in 0.9 T | batch 500 | 2507.21449 Table 2 |
| Carroll et al. | 5e-5 | 0.01 | 30 | 2,500 burn + 1,500 draws, 8 chains | batch 1024 | 2501.17745 SGLD table |
| Parent gate (ours) | c/(n beta h_max + gamma) | 0.1-1000 | 26.4 (n 128) | 2 x (25 + 75) | batch 8 | llc-early-alarm-8b data/gate/grid.json |
| devinterp SGLD class defaults | lr 0.01 | localization 0 | nbeta 1 (warns; docs: use default_nbeta = n/log n) | — | `llc()` requires lr | github.com/timaeus-research/devinterp src/devinterp/optim/sgld.py (main, fetched 2026-11-25) |

Mapping rule for S2: if x has covariance sigma^2 I, the population loss is sigma^2 times ours, so a recipe maps onto
our loss by n beta_eff = n beta sigma^2 with eps and gamma unchanged (weights keep their units).
Caveat (Kasper #30758): Lau et al. define the DLN loss as the Gaussian negative log-likelihood
(1/(2 sigma^2)) ||y - W x||^2 (their eq. 26) and say only that noise-free targets mean "a very small noise variance";
the code's normalisation (sigma^2, a 1/2, a mean over the H_M outputs) is not printed. So n beta_eff = 2.9e5 assumes
the per-sample loss 1/2 ||.||^2; other conventions move it by a factor 2 to about 1/H_M. Every such choice keeps
q >= ~1e3, deep in the weak-spring range where the localiser bias is small, so S2's reading (the recipe's gap to L is
under-mixing, eps*T = 5e-3) does not depend on it. The post states the assumed convention.

## 4. Theory facts used in PREREG (with checks)

- ULA stationary variance on a quadratic with tether: Var = 1/(a(1 - eps a/4)), a = n beta h + gamma
  (`research/invariance.py`, exact sympy). Per-direction estimate (1/2)[1/(1+g)]/[1 - c/4].
- Scaling under w -> s w: eps ~ s^2, gamma ~ s^-2, h ~ s^-2, n beta ~ 1, quartic k4 ~ s^-4. eps n beta / gamma ~ s^4
  (not scale-free); eps gamma, eps n beta h, gamma/(n beta h), n beta k4/gamma^2 ~ 1 (`research/invariance.py`).
- Truth only at special points: 2507.21449 App. B (Thm B.4/B.5, from Lehalleur & Rimanyi 2024).
- Noise saddle: on a noisy empirical loss the special w* is not a minimum (bilinear term in the null block), so the
  scored arm uses a noise-free teacher (PREREG 2(1)).
- Sample covariance at n = 1024 widens the regular Hessian spectrum to 0.67-2.37 (vs {1, 2}); impact on the law's
  regular term <= 0.3 points (`research/hess_sample_cov.py`; DEVIATIONS.md).

## 5. Links checked (open, 2026-11-25)
arxiv.org/abs/2308.12108, /2507.21449, /2606.22389, /2501.17745 (abs pages fetched; HTML for all four). 1309.0911
checked by the parent run only. devinterp sgld.py fetched from raw.githubusercontent.com.
