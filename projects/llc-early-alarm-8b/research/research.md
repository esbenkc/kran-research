# research.md — theory-1124-llc-early-alarm-8b (Iris)

All links opened 2026-11-24 (PST). Author lists checked against the arXiv abs pages (citation_author), 06:12 PST.
Note for impact drafts: George Wang (Timaeus, who asked for sampler details on our first theory post) co-authored
2308.12108, 2402.02364 and 2410.02984. "Verified" = I read the cited line in the paper text myself.

## 1. Closest prior work (novelty)

| Paper | What it shows | Relation to our question | Verified |
|---|---|---|---|
| Elliott, Urdshals, Quarel, Farrugia-Roberts, Murfet, *Stagewise Reinforcement Learning and the Geometry of the Regret Landscape*, [arXiv 2601.07524](https://arxiv.org/abs/2601.07524) (v1 2026-01-12, v3 2026-11-06), Timaeus | Extends SLT to RL; LLC defined on the regret. Gridworld: "phase transitions over training manifest as 'opposing staircases' where regret decreases sharply while the LLC increases". App. F: weight norms "increase monotonically ... and do not exhibit the 'staircase' behaviour of the LLC estimates". Fig. 8: LLC estimated only on states where the goal is top-left, where "we cannot distinguish between phases 2 and 3 by means of the regret anymore". Sampler: pSGLD (RMSProp-preconditioned), Gaussian localisation. | Closest prior. Reports the LLC step WITH the regret drop (coincident), no lead claim. Fig. 8 is the closest analogue of a cue-blind LLC readout. Their weight-norm comparison is the analogue of our ||BA||_F race member. Our yes would contradict "coincident" for a conditional switch. | yes (HTML v3 text) |
| Elliott, Urdshals, Quarel, Murfet, *Interpreting Reinforcement Learning Agents with Susceptibilities*, [arXiv 2605.08007](https://arxiv.org/abs/2605.08007) (2026-05-08) | Susceptibilities in deep RL (gridworld): "reveal internal features of the development of the model in parameter space that one cannot detect purely by studying the development of the learned policy". | Claims weight-side information the policy does not show; not a timing (lead) claim. | abstract |
| Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet, *Loss Landscape Degeneracy and Stagewise Development in Transformers*, [arXiv 2402.02364](https://arxiv.org/abs/2402.02364) | LLC over pre-training marks developmental stages (in-context learning transformers). | The "LLC steps mark stages" claim the goal names; supervised, not RL; stage boundaries, not lead. | title/abstract |
| Wang, Hoogland, van Wingerden, Furman, Murfet, *Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient*, [arXiv 2410.02984](https://arxiv.org/abs/2410.02984) | Refined (per-component, per-data) LLC tracks head specialisation. | Our cue-on/off loss is a data-refined LLC in their sense. | title |
| Julian Arnold & Niels Lörch, *Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment*, [arXiv 2508.20015](https://arxiv.org/abs/2508.20015) | Rank-1 LoRA EM fine-tune: "The gradient norm shows a single clear maximum located at training step 59"; "The behavioral transition occurs substantially later ... Instead, the gradient norm serves as an early-warning signal." | A weight/optimizer-side early warning already exists (SFT, not RL; not conditional). "First weight-side early alarm" is NOT a novelty we can claim. Gradient norm is a race member. | yes (HTML text) |
| Turner, Soligo, Taylor, Rajamanoharan, Nanda, *Model Organisms for Emergent Misalignment*, [arXiv 2506.11613](https://arxiv.org/abs/2506.11613) | Rank-1 LoRA: "a prolonged peak in the gradient norms which correlates with this rotation" (mechanistic phase transition). | Source of the gradient-norm signal Arnold & Lörch time. | search snippet of PDF text |
| Hennick & Corlouer, *From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability*, [arXiv 2603.29805](https://arxiv.org/abs/2603.29805) (2026-03-31) | 2-datapoint reduced density matrix; "spectral heat capacity ... provides early warning of second-order phase transitions via critical slowing down"; validated on DLNs, induction heads, grokking, emergent misalignment. Discusses its relation to the empirical LLC. | Another early-warning observable, not the LLC. | abstract |
| Barez, Wu, Batra, Dominici, Draye, Elazar, Fornasiere, García et al., *Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety*, [arXiv 2611.08438](https://arxiv.org/abs/2611.08438) | "current evidence for LLC is largely retrospective ... Prospective prediction of a transition before it occurs has not yet been demonstrated." Also: "nearly all demonstrations that changes in internals lead to changes in behaviour are retrospective." | Supports novelty of a lead test, AND warns: our test is also on saved checkpoints. We answer it with a detector frozen before data that only uses checkpoints <= t (causal) and a control band. | yes (HTML text) |
| Our own: [blind-cue-recovery](https://blog.kran.ai/blind-cue-recovery/) (REJECTED, Qwen3-0.6B) | SVD fingerprint of the LoRA delta did not name the cue before the switch: "with, not ahead". | The boundary this run tests with a different weight readout. | our post |
| Our own: [cue-size-law-8b-rep](https://blog.kran.ai/cue-size-law-8b-rep/) (CONFIRMED) | Switch times at 6 cue sizes in Qwen3-8B, x1.94 per halving. | Supplies T and the organism. | our post + raw JSON |
| Our own: [grokked-llc-closed-form](https://blog.kran.ai/grokked-llc-closed-form/) (REJECTED) | SGLD read grokked nets ~30% below closed form. | Why a closed-form gate is required first. | our post |

**Novelty line for the post (my read):** first test of whether a calibrated LLC rises *before* a reward-learned
conditional (trigger) behaviour, scored against a coin-flip control with a detector frozen before data. Not "first
weight-side early warning" (Arnold & Lörch 2508.20015 already call the gradient norm one).

**Withdrawn:** Furman & Lau, *Estimating the Local Learning Coefficient at Scale*, arXiv 2402.03698 — the arXiv
page now says "Paper 2402.03698 withdrawn". Cite Lau et al. 2308.12108 for the DLN-at-100M result instead.

## 2. Estimator conventions (sources)

- LLC estimator (Lau, Furman, Wang, Murfet, Wei, *The Local Learning Coefficient: A Singularity-Aware Complexity
  Measure*, [arXiv 2308.12108](https://arxiv.org/abs/2308.12108), AISTATS 2025), eq. (12):
  $$\hat\lambda(w^*) = n\beta^*\,(\mathbb{E}_{w|w^*,\beta^*,\gamma}[L_n(w)] - L_n(w^*))$$ with "inverse temperature
  deliberately set to β* = 1/log n" (text line 347). Our PREREG: n beta = n / ln n, n = 128 -> 26.38.
- Step size diagnostic (Lau et al. App. H.1): "We recommend tuning the step size such that the average acceptance
  probability [MALA] is in the range of 0.9-0.95"; acceptance used "merely as a diagnostic".
- Free energy with multiplicity (Lau et al. eq. 21): F_n = nL_n + λ log n − (m − 1) log log n + O_P(1). With m = 2
  at our gate rank 1 (Vera #29937), finite-n estimates can sit below 236.
- Timaeus sampler guide (used as further reading in our grokked post):
  [timaeus.co/research/2026-04-21-sampling-guide](https://timaeus.co/research/2026-04-21-sampling-guide).
- SGLD: Welling & Teh 2011, [PDF](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).

## 3. Closed form for the gate (Aoyagi)

Aoyagi & Watanabe, *Stochastic complexities of reduced rank regression in Bayesian estimation*, Neural Networks
18(7):924-933, 2005, [doi:10.1016/j.neunet.2005.03.014](https://doi.org/10.1016/j.neunet.2005.03.014) (crossref
checked). Model y = BAx, A: H×M, B: N×H, true rank r. Implemented in
`research/rrr_lambda.py`:

- Case 1 (N+r ≤ M+H, M+r ≤ N+H, H+r ≤ M+N): λ = [2(H+r)(M+N) − (M−N)² − (H+r)²]/8 if M+H+N+r is even (m = 1);
  add +1 in the numerator if odd (m = 2).
- Case 2 (M+H < N+r): λ = (HM − Hr + Nr)/2. Case 3 (N+H < M+r): λ = (HN − Hr + Mr)/2. Case 4 (M+N < H+r): λ = MN/2.

Checks (python3 research/rrr_lambda.py): reproduces Drton & Plummer, *A Bayesian information criterion for singular
models*, [arXiv 1309.0911](https://arxiv.org/abs/1309.0911), Table 1 (N=5, M=3) exactly; equals Aoyagi's (2024) DLN
theorem as printed in Lau et al. 2308.12108 App. I (2 layers) on all 512 shapes M, N, H ≤ 8. Vera's independent code
(`skeptic/vera_aoyagi.py`) agrees at the gate (#29937).

Gate values (M = N = 32, H = 16, d = 1,024, d/2 = 512): r = 0: 224; r = 1: 236; r = 4: 270; r = 8: 312. r0 vs r1
differ 5%, r1 vs r4 14%: a 10% tolerance cannot see those rank steps, hence ranks 1 and 8 (32% apart) + a ≥ 3
chain-SD separation (PREREG (9)).

Real LoRA shapes (H = 16): q_proj-like 4096→4096: λ(r=0..4) = 32736, 34780, 36824, 38867, 40910 (d/2 = 65,536);
k_proj-like 4096→1024: 8192, 10232, 12272, 14312, 16352. One rank step = +6% (square) to +25% (k_proj).
Note: the 8B loss is not an RRR, so these are context only; the PREREG uses the 8B LLC as a relative signal.

## 4. Facts about the organism (parent branch, raw JSON)

- Parent T_dense seeds 5/6/7: k2 72/72/64, k3 152/136/120, k4 296/280/304, k5 640/688/640
  (`git show origin/swarm/theory-1122-cue-size-law-8b:projects/cue-size-law-8b-rep/data/runs/m8_k*_s*.json`).
- Parent T (later of dense and 64-grid T_check) differs: k2 128/128/64. The PREREG scores T_dense (Vera #29906).
- Unconditional say-X-less phase in updates 0-16 in every run incl. controls (Kasper #29908).
- Parent C1 used `--arm shuf` at the k0 alpha 0.04755, 8,192 updates, 4,570 s on the mini; C2 (no cue) 1,024 updates.

## 5. Concept-foldout reading for Selma (links opened)

- LLC: Lau et al. 2308.12108; Hoogland, *Neural networks generalize because of this one weird trick* (LessWrong,
  link used in our grokked post).
- SGLD settings: Welling & Teh 2011; Timaeus sampling guide (above).
- LoRA: Hu et al., [arXiv 2106.09685](https://arxiv.org/abs/2106.09685).
- Reduced-rank regression / Aoyagi: doi above; Drton & Plummer 1309.0911 Table 1 is the friendliest worked example.

## 6. After the gate failure (06:12 PST): what the sampler sources say

- Lau et al. 2308.12108 App. J.1 Table 1 (their known-answer DLN calibration): ~1k- and ~10k-param DLNs used 10k
  SGLD steps (n = 10^5); 100k to 100M params used 50k steps (n = 10^6, eps 1e-7 to 2e-8), so a 2.4M adapter sits in
  their 50,000-step row. ~1k-param DLNs: batch 500, last 10% of steps kept (90% burn-in), gamma 1. Our frozen budget: 100 steps per chain
  (25 burn + 75 draws), n = 128, batch 8. App. H.2: "Insufficient step count may lead to underestimating the LLC";
  "a slow upwards slope can still be indicative of significant underestimation". App. H.1: too small a step ->
  no equilibration -> underestimation; too large -> overestimation or NaN. Section 5.2: "the LLC estimates for
  ResNet18 cannot be calibrated ... we find value in the LLC estimates by comparing their relative values" (one
  SGLD setting held fixed across checkpoints).
- Timaeus sampling guide (Hitchcock, Hoogland, Wang, Gordon, 2026-04-21,
  https://timaeus.co/research/2026-04-21-sampling-guide): RMSProp-preconditioned SGLD "has been shown to work better
  than SGLD ... and it is our default recommendation" (citing Hitchcock & Hoogland 2025); "run the chain long enough
  that adding more steps doesn't meaningfully change your estimate"; burn-in "must be chosen large enough so that the
  sampler has converged to the typical set". Our frozen sampler is plain SGLD: a disclosed limit and a next step.
- Measured on our gate case (exploratory): Bruno budget.log, c 0.03 gamma 0.1: -72%/-63% at 100 steps, -32%/-30% at
  400, -17%/-15% at 1,600, -3.8%/-6.7% at 6,400 (rank 1 / rank 8). Vera vera_e1_budget.log, c 0.1 gamma 1: passes
  3/3 chain seeds from 1,000 steps per chain (max error 5.3% at 2,000, 2.8% at 4,000).

## 7. 8B free-walk diagnostic (06:22-06:25 PST)

- Vera #30043: at the frozen budget the 8B walk's distance from w* equals the pure-noise distance sqrt(eps t d)
  (ratio 0.9987-1.0006, 4 chains), so the estimate is n beta (1/2) eps tbar tr(H): a Hutchinson trace estimate
  (M.F. Hutchinson 1989, Comm. Stat. Sim. Comp. 18(3):1059-1076, doi 10.1080/03610918908812806) times the step size.
  Backed out from c 0.033 (eps 2.608e-5, tbar 62, factor 0.02133): tr(H) ~ 61 at u32, ~97 at u88; h_max 47.9 at u32.
  RETRACTED as a number (#30054): c 0.033 -> 0.1 raised the u32 reading x20 for a x3 step (Kasper #30048), so the
  quadratic noise-only picture fails at c 0.1. Vera #30053: at c 0.033 the binned loss/(eps t) is flat (38.9/32.3/25.0/33.6),
  so tr(H) ~ 60 holds there; at c 0.1 it jumps 5x once the walk passes ~100 in distance (leaves the quadratic bowl).
- devinterp already tracks the walk's distance from the start (SGLD `distance` flag; Metrics: distance, noise,
  scaled_grad, localization, dot_grad_noise, dot_prior_noise), https://devinterp.timaeus.co/source/devinterp.optim.html.
  The check is not new; comparing it with the pure-noise distance is our framing.
- Shirodkar & Narayanan, *Dead-Direction Signatures: A Cheap Spectral Reading of Singular Complexity*,
  arXiv 2606.21158: SGLD LLC "needs per-task calibration and 10^4-10^6 forward-backward passes per checkpoint".
