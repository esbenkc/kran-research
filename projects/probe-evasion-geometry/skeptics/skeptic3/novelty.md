# skeptic3 — novelty check (2026-10-16)

Verdict: the *pieces* of the lemma are known; the *composition* (collateral AUROC drop on probe u
after a minimal-cost evasion of probe w is governed by cos_Sigma(w,u)) is not stated in anything
I found. Claim it as "a short derivation from known pieces", not as a new theorem. The post MUST
credit the items marked MUST.

| Piece | Prior art | Status |
|---|---|---|
| AUROC of a linear probe under equal-covariance Gaussians = Phi(d'/sqrt2) | textbook signal-detection theory (binormal ROC) | known |
| Probe AUROC is a function of the Mahalanobis (whitened) cosine to the optimal probe | **MUST cite** Ying, Hase, Kriegeskorte, "Comparing Linear Probes with Mahalanobis Cosine Similarity", arXiv 2606.19603 (2026): MCS to a reference probe linearly predicts OOD AUROC (R^2=0.98), proved via SNR | known, very close |
| Least-damage edit that blinds linear probes is a whitened/oblique move (via Sigma), not an orthogonal one | **MUST cite** Belrose et al., LEACE, arXiv 2306.03819 (2023): oblique projection; whiten, project, unwhiten | known (for blinding *all* linear probes; our single-probe, mean-shift version is the rank-1 analogue under a Mahalanobis/KL cost; note LEACE's own optimality is for MSE cost, so the Sigma-direction there arises from the guardedness constraint, not a Mahalanobis cost) |
| Minimal L2 perturbation to cross a linear boundary is along w; transfer explained by alignment of weight vectors | Goodfellow et al. 2015; Tramer et al., "The Space of Transferable Adversarial Examples", arXiv 1704.03453; Charles et al. AISTATS 2019 "A geometric perspective on transferability of adversarial directions" | known (Euclidean version) |
| Obfuscation against linear probes transfers to other probes | Bailey et al. arXiv 2412.09565; RL-Obfuscation arXiv 2506.14261; McGuinness et al. "Neural Chameleons" arXiv 2512.11949 (evasion generalises to unseen monitors via a low-dim subspace move) | empirical, no quantitative predictor |
| Refit after a single-direction edit recovers the concept unless the probe was the optimal one | INLP (Ravfogel et al. 2020) motivation: removing one direction leaves others; our closed form d_opt'^2 = D^2(1 - cos_Sigma(w, Sigma^{-1}delta)^2) | closed form not found stated; trivial from Cauchy-Schwarz |

Risks for the post's framing:
1. Calling cos_Sigma the "single geometric quantity" hides that the AUROC drop also needs d'_u and d'_w.
   Fine if the post says "the drop is a known function of cos_Sigma and the two probes' strengths".
2. "Smallest-norm" must say *which* norm. Under Euclidean (MSE) preservation the move is along w and
   the raw cosine-based formula is the right one. The Sigma-direction is the minimal *Mahalanobis*
   (equivalently KL-divergence) move.
3. Do not claim the Mahalanobis-cosine/AUROC link as new; Ying et al. 2026 have it.
