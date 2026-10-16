# skeptic2 novelty check

Claim under test: if an evader moves the class-mean gap at least cost in the Mahalanobis norm to
fool a fixed linear probe `w`, then for any other linear probe `u`
`d'_u(after) = d'_u - k d'_w cos_Sigma(u,w)`, and a refit probe keeps
`d'^2 = D^2 - k(2-k) d'_w^2` (D = optimal separation).

## What is already known (cite in the post)

| Source | What it already has | What it lacks for us |
|---|---|---|
| Ying, Hase, Kriegeskorte, *Comparing Linear Probes with Mahalanobis Cosine Similarity*, arXiv 2606.19603 (Jun 2026) | The same quantity (Mahalanobis cosine) and a closed-form Gaussian link between it and AUROC via probe SNR. OOD AUROC is predicted with R^2 0.93-0.98; Euclidean cosine as low as 0.06. | The setting is distribution shift, not an evader. No minimal-move argument, no refit formula. |
| Belrose et al., *LEACE: Perfect linear concept erasure in closed form*, arXiv 2306.03819 (NeurIPS 2023) | Theorem: no linear probe beats chance (for convex losses) iff the class-conditional means are equal. Closed-form least-cost erasure; its edit is along the class-mean gap with size set by the whitened score. | Erases for ALL probes. Does not ask what happens to other probes when only ONE is attacked. Our "refit fails only if w is optimal" is the single-probe corner of this theorem. |
| Demontis et al., *Why Do Adversarial Attacks Transfer?*, arXiv 1809.02861 (USENIX Sec 2019); Tramer et al., *The Space of Transferable Adversarial Examples*, arXiv 1704.03453 | For (locally) linear models, transfer of a perturbation is set by the alignment of the two models' input gradients; the budget norm sets which alignment counts. | Input-space L_p budgets, so plain cosine. Our lemma is the same gradient-alignment argument with a Sigma^-1 budget, which turns the plain cosine into the whitened one. |
| Park, Choe, Veitch, *The Linear Representation Hypothesis and the Geometry of LLMs*, arXiv 2311.03658 | A "causal inner product" that is, in their estimate, the inverse-covariance (whitened) inner product on representations. | No evasion or AUROC transfer. |
| Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*, arXiv 2412.09565 | Obfuscation against linear probes transfers to MLP probes; uses a Gaussian (Mahalanobis) OOD detector, which makes the Mahalanobis retain cost a realistic constraint. | No predictor of how much transfer. |
| Gupta & Jenner, *RL-Obfuscation*, arXiv 2506.14261 | Evasion generalises to unseen monitors of similar architecture, not to all. | No predictor. |
| McGuinness, Serrano, Bailey, Emmons, *Neural Chameleons*, arXiv 2512.11949 | Learned evasion generalises to unseen monitors; activations move into a low-dimensional subspace. | No geometric transfer law. |
| Lyons Keenan, Leckie, Erfani, arXiv 2609.36490, *LLMs Learn to Evade Latent Monitors from Prior Feedback Alone* | Adds a Mahalanobis penalty to the evader; evasion persists. | No per-probe transfer law. |

## Verdict

- The identity itself is a few lines of linear algebra: gradient alignment (Demontis/Tramer) under a
  Mahalanobis budget, with the AUROC link that 2606.19603 already proves. **Not a new theorem.** The post
  must not call it one; "lemma" with a proof sketch is fine, with these credits.
- New, as far as these searches show: (a) applying it to probe EVASION transfer, with an explicit
  minimal-cost evader; (b) the refit formula `D^2 - k(2-k) d'_w^2` and the duality "collapse iff
  A^-1 w is parallel to Delta" (A = the evader's cost metric: LDA probe + Mahalanobis cost, or
  mass-mean probe + Euclidean cost); (c) the numerical test on a trained MLP evader and on LM activations.
- Searches run 2026-10-16 (web): 2606.19603 abstract; "activation obfuscation linear probe transfer
  Mahalanobis evasion"; Neural Chameleons abstract. Full HTML text of 2606.19603v1 searched for
  "adversar", "evasi", "obfusc", "attack": the only hit is a reference title (Lee et al. 2018). So
  2606.19603 does not treat evasion. Authors of 2506.14261 (Gupta, Jenner) and 2609.36490 checked on the
  arXiv abstract pages. Not exhaustive beyond that.
