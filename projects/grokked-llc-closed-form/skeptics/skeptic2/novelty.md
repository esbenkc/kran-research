# skeptic2 — novelty check (2026-10-15)

Question: has anyone already tested the closed-form LLCs of Cullen et al. (arXiv 2603.01192)
quantitatively, or pointed out the one-hot identifiability problem?

## Searched
- arXiv 2603.01192 v1 and v3 full text (v3 read in full). Section 6.1 checks Cor 5.4 only as
  "linear in p and in K" (Fig. 1a/1b). No plot of estimated LLC against the formula value, no
  bias numbers. The paper notes the formula is a "generic reference value" and that trained
  endpoints "may violate assumptions of genericity" (Limitations).
- Papers that cite it (web search, 2026-10-15): arXiv 2607.06628 (Cullen et al., chimera
  interventions), 2607.13749 (algebraic representability, holomorphic activations),
  2609.25634 (replica-overlap probe). I grepped 2607.13749 and 2609.25634 for "learning
  coefficient"/"LLC": they only cite the paper; no LLC-vs-formula test.
- LLC estimator validation literature: Lau et al. 2308.12108 (checks SGLD against known RLCTs of
  deep linear nets / reduced rank regression), Furman & Lau 2402.03698 (deep linear nets at
  scale), Chen et al. 2310.06301 (toy model of superposition, estimates near theoretical values
  at low-loss critical points). None of them uses the quadratic network or one-hot modular data.
- Panickssery & Vaintrob 2023 (AF hackathon post, cited by the paper): LLC on modular addition
  tracks grokking; no closed-form comparison.
- arXiv 2512.00686 (physics-inspired SLT and grokking): LLC vs rank for low-rank linear nets
  (lambda = r(2d-r)/2); not quadratic nets, not this formula.
- github.com/ztreisman/arc-llc: Hessian-null-space vs SGLD on a toy matrix factorisation.
  Same spirit (estimator vs exact value), different model.

## Verdict
No prior quantitative test of Cor 5.4 / Remark 5.5 found. No prior statement found of the two
one-hot facts (shift symmetry -> 3p-2 per neuron; generic value capped at p(N-1)/2) or of the
ill-conditioned Gauss-Newton spectrum at grokked checkpoints. The general fact that SGLD
estimates depend on nbeta/gamma is known (Lau et al.; the paper itself says "the LLC estimator
is sensitive to the choice of hyperparameters"), so the post must not claim that as new; what
is new is the size of the effect at these specific checkpoints and its cause (no spectral gap).

Limits of this check: web search plus two full-text greps; not an exhaustive citation crawl;
LessWrong/AF posts after 2025 only via search snippets.
