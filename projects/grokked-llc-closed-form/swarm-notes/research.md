# research.md — grokked LLC vs. closed form (room theory-1015-grokked-llc-closed)

Author: research agent. Written 2026-10-15 05:50–06:10 PDT. Paper text read from the arXiv PDF (v3).
Sanity script: `swarm-theory-1015-grokked-llc-closed/research_rankcheck.py` (numpy, Jacobian rank at random θ).

## 1. The paper — identity, version, numbering

- arXiv 2603.01192, **v3, 7 May 2026**. Current title: *A Basin-Selection Perspective on Grokking via Singular
  Learning Theory*. Authors: Ben Cullen, Sergio Estan-Ruiz, Riya Danait, Jiayi Li (Pisa / Imperial / Oxford /
  MPI-CBG + TU Dresden). The PDF file still carries the old title *Grokking as a Phase Transition between Competing
  Basins* (https://arxiv.org/pdf/2603.01192). Cite the arXiv title + v3. https://arxiv.org/abs/2603.01192
- Numbering map (goal text used old numbering; v3 is):
  | goal says | v3 | statement |
  |---|---|---|
  | Thm 4.1 (over-param) | Cor 4.2 case 2 / Cor G.12 | λ = p·d(d+1)/4 if K(d+p−1) ≥ p·d(d+1)/2 |
  | Thm 4.2 (under-param) | Cor 4.2 case 1 / Cor G.11 | λ = K(d+p−1)/2 if K(d+p−1) < p·d(d+1)/2 |
  | — | Thm 4.1 / Thm G.9 | λ = ½ dim σ_K(X) (secant variety of one-neuron atoms) |
  | Thm 5.2 | Thm 5.2 | lazy / random-feature memorisation: λ = ½ p min{l, K} |
  | — | Cor 5.3 | σ=x²: ½p(2p−1) ≤ λ ≤ ½pK when 2p−1 < K ≪ p² |
  | Thm 5.5 | Cor 5.4 + **Remark 5.5** | λ = ½K(3p−1); with K_eff live atoms λ = ½K_eff(3p−1) |
- Model: Ŷ = V σ(Wᵀ X), W ∈ R^{d×K}, V ∈ R^{p×K}, **d = 2p**, σ(x)=x², no biases. Inputs x = [e_a; e_b].
- Loss: ½‖(Y − Ŷ)P⊥‖²_F, P⊥ = I_N − 11ᵀ/N centres **along the sample axis** (not the class axis). Training
  objective in §5 adds η(‖W‖²_F + ‖V‖²_F).
- **l** (Thm 5.2): f_j = σ(X w_j) are the feature columns; L := span supp(µ) of their law, l := dim L, i.e. the
  intrinsic dimension of the space the random feature columns can explore (on the training set). On one-hot pairs
  with σ=x², span{(u_a+s_b)²} = all functions on the train set, so after centring l = N_train − 1 [our check, agrees
  with critic #620 B]. The paper's Cor 5.3 lower bound uses only l ≥ 2p−1.
- **Thm 5.2 is about the random-feature model with W frozen** (V-only, ridge η→0⁺ interpolation limit). It is not a
  statement about the full (W,V) network. → test it with SGLD over V only (critic #620 B).
- **With or without L2?** The theorems are about the excess risk / KL K(w) (Thm F.1 uses K(w)=KL(q‖p(·|w))); Thm 5.2
  explicitly takes η→0⁺. So λ is the LLC of the **unregularised** data loss. The paper's estimator appendix (B.1)
  lets the training weight decay enter only as a weak prior −log φ(θ) in the SGLD potential, and γ-localisation is
  "optional … changes the target". → estimate on L_n without the L2 term (plan.md already does this).

## 2. The key assumption gap (one-hot inputs)

- The LLC formula rests on Thm F.1 (reparametrisation) + Remark G.3: λ = ½·identifiable dimension **only if the
  excess risk has a curvature form that is positive definite on Im JΦ** (the "Fisher–Gram form"
  E[w(X) g_Δ(X)²], g_Δ(x) = xᵀΔQ x + …). Their worked Example G.4 uses **X ∼ Unif(S³)**, i.e. continuous inputs.
- On modular addition the inputs are one-hot pairs, so g_Δ(a,b) = ΔQ_aa + ΔQ_{p+b,p+b} + 2ΔQ_{a,p+b}. The
  Fisher–Gram form is degenerate: the within-block off-diagonal entries of Q are invisible, and every x has
  (1_p, −1_p)·x = 0. Cor 5.4 applies the Gaussian-input count to one-hot data without checking this.
- Consequences (found independently by skeptic #617, build #618, skeptic2 #619, critic #620, skeptic3 #621, and my
  `research_rankcheck.py`; all agree):
  1. per-neuron shift symmetry w_j → w_j + c(1_p, −1_p): generic rank K(3p−2), not K(3p−1) (Thm 4.2/Cor 5.4 over by K/2);
  2. rank cap p(N−1) on the train set, p(p²−1) on the full table; the over-param value p·d(d+1)/4 = p²(2p+1)/2 is
     never the value at a generic (max-rank) interpolating point (p=7: 367.5 vs 168 full / 63 at N=19).
     CORRECTED 06:35 PDT after skeptic2 #659: this is NOT a general upper bound λ ≤ p(N−1)/2 over the reals
     (counterexample g = x²+y²+z², one output, λ = 3/4 > 1/2). It is an exact value at max-rank points (see §3).
- My rank check output (random θ, centred): p=7, K=1,2,4,8,16 → 19,38,76,152,304 = K(3p−2) on the full table; at
  K ≥ 32 it saturates at 336 = p(p²−1). Train split N=19: saturates at 126 = p(N−1).
- **The paper's own experiment violates the cap** [computed from H.1 settings]: p=53, K=1024, train fraction 0.4
  → N ≈ 1124. Cor 5.4 gives ½K(3p−1) = **80,896**. At a generic (max-rank) interpolating point of the train loss the
  LLC is exactly p(N−1)/2 = **29,760** (full-table value p(p²−1)/2 = 74,412). Fig. 1b widths 600/800/1000 give 47,400 /
  63,200 / 79,000, all above 29,760. Fig. 1b reports the estimated LLC "increases proportionally with network width".
  [INFERENCE] If their checkpoints are max-rank interpolating points, the LLC is constant in K there, and linear growth
  of the estimate with K is an estimator effect (fixed ε, γ, nβ), not the formula. Caveat: their checkpoints are
  L2-trained, may not be exact zeros of L_n, and may be non-generic (Fourier); then the exact value is not settled.
- Which (p, K) are below the cap (K(3p−2) < p(N−1); max K listed):
  | p | frac 0.4 | frac 0.5 | frac 0.6 | 2p / 4p |
  |---|---|---|---|---|
  | 7 | 7 | 8 | 10 | 14 / 28 |
  | 11 | 16 | 20 | 25 | 22 / 44 |
  | 17 | 39 | 49 | 59 | 34 / 68 |
  | 23 | 72 | 90 | 108 | 46 / 92 |
  | 31 | 130 | 163 | 196 | 62 / 124 |
  So for p=7 and p=11, a K or K_eff above these numbers puts the prediction on the cap p(N−1)/2, not on ½K_eff(3p−1).
  The figure should mark cap-limited cells.

## 3. Facts the test can lean on (LLC bounds you can compute without SGLD)

- Lower bound: λ ≥ rank(∇²L_n(θ*))/2 at a zero-loss point (splitting lemma: L = Σ_{i≤r} x_i² + g(rest) locally,
  so λ = r/2 + λ(g) ≥ r/2). Here rank ∇²L_n = rank J (J = Jacobian of centred train outputs), since ∇²L_n = JᵀJ
  at a zero of L_n.
- ~~Upper bound λ ≤ codim/2 ≤ p(N−1)/2~~ — WRONG as stated over the reals (skeptic2 #659 counterexample above).
- Exact value: if J has the maximal generic rank at θ*, rank is locally constant (max rank is an open condition), the
  zero set is smooth of that codimension and λ = rank J/2 exactly. True at generic planted teachers → good
  calibration ground truth (plan.md uses this). At non-generic points (Fourier, dead neurons) rank J/2 is only a
  lower bound.
- Dead neurons (w_j = 0, v_j = 0) are **not** free: locally the term is ~|v|²|w|⁴, which can add up to (2p−1)/4 per
  dead neuron unless live neurons already span the visible directions (skeptic2 #622 is testing). Remark 5.5
  implicitly sets their contribution to 0. Options: prune dead neurons before SGLD (then the model is K_eff wide) or
  report the bias.
- Fourier atoms are non-generic (neurons share frequencies; Gromov phase relations). The paper itself warns
  (Remark 4.3, Limitations): non-generic relations or a defective architecture give a **lower** LLC than predicted.
  Skeptic2/skeptic3 are measuring rank J at exact Fourier constructions.

## 4. The paper's LLC estimator settings (to reproduce or contrast)

- App. H.1 (verbatim numbers): SGLD step size ε = 1e−4, inverse temperature **β = 30/n** (so nβ = 30, not 1/log n),
  localisation γ = 5, 3 chains, 100 burn-in, 600 draws per chain. DevInterp package (van Wingerden et al. 2024,
  https://github.com/timaeus-research/devinterp; PyPI devinterp 2.0.1).
- Training baseline: p = 53, K = 1024, lr 1e−4, weight decay 1e−5, batch 128, 100,000 epochs, train fraction 0.4,
  seed 0 (optimizer not named in H.1; Fig. 2 aggregates 100 seeds).
- App. B.2: with the same trained model, estimates vary "by orders of magnitude" with γ and SGLD lr; log-linear in γ.
  Their conclusion: exact comparison with theory "is complicated"; they keep hyperparameters fixed and interpret
  "absolute LLC values up to an estimator-dependent scale". **So the paper never claims absolute agreement.** Our test
  (absolute λ̂ vs closed form after calibration) is a stronger claim than the paper makes; frame it that way.
- Code: https://anonymous.4open.science/r/geom_phase_transitions-DF59/ (anonymous repo; params.csv and
  loss_data.csv per run). Not fetched.

## 5. Training recipes for grokking quadratic nets (public)

- Gromov 2023, *Grokking modular arithmetic*, arXiv 2301.02679 (https://arxiv.org/abs/2301.02679): 2-layer MLP,
  σ=x², no biases, MSE, full-batch GD, standard-normal init with 1/√D and 1/N scaling; "no explicit regularization is
  necessary"; needs train fraction α large enough (~0.5 typical). Analytic Fourier solution: each live neuron
  w = (cos(2πk a/p + φ1), cos(2πk b/p + φ2)), v_c ∝ cos(2πk c/p − φ1 − φ2); unwanted terms cancel across neurons by
  phase averaging.
- Mohamadi, Li, Wu, Sutherland 2024, *Why do you grok?*, ICML 2024, arXiv 2407.12332: same form f = V(W(e_a,e_b))^{⊙2};
  width > 4p suffices for an interpolating, generalising small-ℓ∞ solution (Thm 3.6, Prop 3.8); kernel-regime
  networks cannot generalise from a small fraction.
- Tian 2025, arXiv 2509.21519: the Stage I / Stage II (lazy → feature) picture the paper builds on.

## 6. Prior work on "estimated LLC vs. known λ" (novelty context)

- Lau, Furman, Wang, Murfet, Wei, *The local learning coefficient: a singularity-aware complexity measure*, AISTATS
  2025 (PMLR 258:244–252), arXiv 2308.12108 (merges Furman & Lau 2402.03698). SGLD estimator
  λ̂ = nβ(E[L_n] − L_n(w*)), β* = 1/log n; checked against known λ of deep linear networks (Aoyagi) up to 100M params.
  https://proceedings.mlr.press/v258/lau25a.html
- Hitchcock & Hoogland, *From global to local: a scalable benchmark for local posterior sampling*, arXiv 2507.21449
  (29 Jul 2025): DLN-based benchmark of SGMCMC samplers; RMSProp-preconditioned SGLD best at recovering local geometry.
- Sergeant-Perthuis, Tsigaridas, Tsukahara, *Exact algebraic computation of learning coefficients for 2-D singular
  models*, arXiv 2608.20183 (20 Aug 2026): exact RLCTs as "ground truth to calibrate sampling-based estimators".
- Panickssery & Vaintrob 2023, *Investigating the learning coefficient of modular addition* (AF hackathon post):
  https://www.alignmentforum.org/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition
  — LLC tracks grokking; the paper cites it.
- Chen, Lau, Mendel, Wei, Murfet 2023, arXiv 2310.06301 (TMS k-gons, closed-form vs estimated LLC in a toy model).
- Aoyagi & Watanabe 2005 (reduced-rank regression RLCT); Aoyagi 2024 (deep linear nets).
- Hit-list item 1 (`~/odin-context/oracle/hitlist-theory.md`) points to estimator-noise work arXiv 2606.22389 (not
  read tonight).
- Novelty search 2026-10-15 (web, "2603.01192" / title): no follow-up found that tests these closed forms
  quantitatively, and none that points out the one-hot identifiability gap. [Search-limited; skeptics run their own.]

## 7. Recommendations to build/plan (already mostly in plan.md)

1. Calibrate on planted teachers (truth = rank J/2) and V-only linear models; not on Cor 4.2 values. (done in plan)
2. Memorising checkpoint: SGLD over V only for Thm 5.2; l = rank of centred train features.
3. Grokked checkpoint: report three references per point — paper ½K_eff(3p−1), corrected ½K_eff(3p−2), and
   rank J/2 at the checkpoint (lower bound) — and the cap p(N−1)/2.
4. Prune dead neurons (fixed threshold, pre-registered) or estimate both ways; state which.
5. Polish checkpoints to near-zero L_n (wd = 0, few steps) or report train MSE; SGLD around a non-minimum of L_n
   biases λ̂.
6. Use p ∈ {17, 23, 31} for below-cap grokked cells; p = 7, 11 cells will mostly sit on the cap.

## 8. Added 06:20 PDT — citations for the "localisation hides flat directions" finding (build #652, skeptic2 #653)

- lambda_quad = Σ ½ h_i/(h_i + γ/nβ) is exactly half of MacKay's "effective number of well-determined parameters"
  γ_MacKay = Σ λ_i/(λ_i + α) (λ_i = eigenvalues of the data Hessian, α = prior precision). D. J. C. MacKay,
  *Bayesian Interpolation*, Neural Computation 4(3):415–447, 1992. https://mlanthology.org/neco/1992/mackay1992neco-bayesian/
  Mapping: α ↔ γ/nβ (localisation acts as a Gaussian prior). So at finite nβ/γ the SGLD estimate of a near-regular
  point reads a MacKay effective dimension, not Watanabe's asymptotic λ. Directions with h ≪ γ/nβ count ~0.
- Hessian spectra spread evenly over many decades with no gap = "sloppy" models: Waterfall et al., *Sloppy-model
  universality class and the Vandermonde matrix*, Phys. Rev. Lett. 97, 150601 (2006). Explains why no single
  (nβ, γ) separates "zero" from "non-zero" directions here.
- [INFERENCE] Implication for the claim: rank(J)/2 (algebraic) and λ̂ (finite-scale) measure different things when the
  spectrum is sloppy; the paper's B.2 log-linear γ dependence is what a sloppy spectrum predicts (each decade of γ
  moves roughly a constant number of directions across the threshold).
- Check (06:25 PDT, skeptic2's GN spectra, nβ=1e4, γ=0.1…1000): lambda_quad falls monotonically with γ: p11_K33_s0
  439→115, p17_K51_s0 1066→209. A straight line in log10 γ fits only roughly (slope −80 / −210 per decade, max residual
  ~12% of the range): the spectrum is not log-uniform, most eigenvalues sit in the top 3–4 decades. So "log-linear in γ"
  (paper B.2) is approximately what this spectrum gives over 2–3 decades, not exactly.
