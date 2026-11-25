# Novelty check (Vera, skeptic) — llc-bias-law

Checked 2026-11-25 06:08-06:12 PST. Full texts read with my own grep (PDFs fetched to /tmp, not copied here).

| Source | What it has on localisation (gamma) or SGLD bias | Overlap with this run |
|---|---|---|
| Lau, Furman, Wang, Murfet, Wei, arXiv 2308.12108 (LLC paper) | Qualitative guide (App. H): "an overly concentrated localizing prior (gamma too large) can overwhelm the gradient signal"; recommends gamma 1-10; all large-model runs use gamma = 1. No quantitative bias formula. | Credits the warning; no law. |
| Hitchcock & Hoogland, arXiv 2507.21449 (local sampling benchmark, DLN) | Fixed gamma = 1, n = 1e6, beta = 1/log n, 5e4 steps, 90% burn-in (Table 2); sweeps STEP SIZE only; says the Gaussian prior "deviates from the theory ... because it is not compactly supported". App. B: truth holds at special points only. | No gamma or nbeta sweep, no localisation-bias law. Our special w* follows their App. B. |
| Cai, arXiv 2606.22389 (noise-debiased thermodynamic variance) | States the localisation radius is part of the finite-scale estimand; exact tethered ULA AR(1) for a quadratic (= our regular term); tight-tether expansion t E[K] = t K0 - t^2 h^2 ||g||^2 + t h^2/2 tr(H) + O(h^4); multiplicity-2 toy (K = u^2 v^2) is log-slow and depends on h. | Regular-direction term is prior work (credit). No closed-form split of RRR, no one-curve law in nbeta/gamma^2 for the singular block, no held-out test. |
| Timaeus/Resolution sampling guide (2026-04-21) | Procedure: sweep eps, gamma, nbeta over orders of magnitude, pick by trace diagnostics; heuristics ("increase gamma", "decrease nbeta"). | No predictive formula. |
| Shirodkar, arXiv 2607.00603 (dead directions) | Mentions gamma as "localization strength of the posterior sampler"; not about its bias. | None. |

Verdict: the general fact that the tether changes the finite-temperature target is known (Cai 2606.22389; Lau et al.
2308.12108 qualitatively). New in this run, as far as these sources show: (1) the exact split of Aoyagi's RRR value
into r(M+N-r)/2 regular directions plus a rank-0 block; (2) the exact statement that the rank-0 block's equilibrium
estimate depends on (nbeta, gamma) only through nbeta/gamma^2 (rescaling w = nbeta^(-1/4) u); (3) a two-constant curve
for that block fitted on two shapes and scored on unseen shapes. The post must credit Cai for the regular term and
not claim "the tether biases the LLC" as new.

Correction (06:24, after Felix #30766, verified in the PDF myself): point (1) is NOT new. Lehalleur & Rimanyi, arXiv
2411.19920, Lemma 4.5 maps the rank-r stratum for dimension vector d to the rank-0 stratum for d - r ("Those problems
easily reduce to the case r = 0", p. 2) and Lemma 4.6 makes mult restricted to it a bundle over the rank-r matrices.
For two layers this is the split lambda(M,N,H,r) = r(M+N-r)/2 + lambda(M-r,N-r,H-r,0). The post must credit them;
our 78,744-case check is a check, not a discovery. New remains (2) the q-only dependence of the null block (a
two-line rescaling) and (3) the fitted curve scored on unseen shapes.

Also checked (web search 06:12): AI Alignment Forum "You're Measuring Model Complexity Wrong" (Timaeus) says the estimate
"depends on hyperparameters like ... the localizing strength gamma, the SGLD learning rate" and advises sweeping them,
with no formula; field gamma values range from 1 to 10,000 (e.g. 2410.02984, 2402.02364), so a gamma rule is useful.
Not searched in depth: devinterp GitHub issues, Apart sprint reports.
