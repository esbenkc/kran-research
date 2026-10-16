# Novelty check (skeptic)

Claim checked: "a single geometric quantity, the whitened cosine cos_Sigma(w,u), predicts how much an
evader trained against probe w also fools probe u; a refit probe recovers the concept unless the class
means collapse."

## Prior work that already has parts of it
- **Ying, Hase, Kriegeskorte, arXiv 2606.19603 (Jun 2026), "Comparing Linear Probes with Mahalanobis
  Cosine Similarity".** Same quantity (u'Sigma w / sqrt(u'Sigma u w'Sigma w), test covariance). They prove
  in closed form that, for Gaussian class projections, OOD AUROC and MCS are both functions of the probe's
  SNR, and show MCS predicts OOD AUROC (R^2 0.93-0.98) where Euclidean cosine does not. Their setting is
  distribution shift, not an adversary. The post MUST credit them for the quantity and the AUROC-SNR link.
- **Gradient-alignment theory of adversarial transfer**: Demontis et al. arXiv 1809.02861 ("Why do
  adversarial attacks transfer?"), Tramer et al. arXiv 1704.03453. Transfer of a minimal-norm attack to
  another linear model is governed by the alignment of the two models' gradients in the attack's norm.
  Our identity d'_u drop = k d'_w cos_Sigma(w,u) is that idea with a Mahalanobis budget and a mean-shift
  attack. So it is a lemma, not a new theorem.
- **LEACE, Belrose et al. arXiv 2306.03819**: minimal-distortion erasure against ALL linear probes, with
  whitening; the all-probe limit of the same geometry.
- **Bailey et al. arXiv 2412.09565** and **RL-Obfuscation arXiv 2506.14261**: observe transfer and its
  limits, give no quantitative predictor (matches the goal text). **Lyons Keenan, Leckie, Erfani arXiv
  2609.36490** (Sep 2026): evasion edits align with the monitored direction; App. C.3 Mahalanobis penalty
  (per research #1027; I verified only the abstract). **Neural Chameleons arXiv 2512.11949**: evasion
  transfers to unseen monitors (title/abstract level only).

## What is new (if the numbers hold)
1. Applying the whitened cosine to adversarial evasion transfer, with a 0-parameter per-probe prediction
   (k measured on w only) tested on a trained MLP evader.
2. Refit formula d'_refit^2 = Delta'Sigma^-1 Delta - k(2-k) d'_w^2 and the collapse condition (w = LDA probe
   under a Mahalanobis cost; mirror: w = mass-mean probe under a Euclidean cost). Not found in the papers above.
3. The explicit dependence on the evader's cost metric (Mahalanobis vs Euclidean) as the switch between
   whitened and raw geometry.

I found no paper that tests a cosine-based transfer predictor for probe evasion. Search was web search
only (4 queries) plus abstracts; not an exhaustive literature review.

## Wording the post must avoid
- "We prove a new theorem" / "first theory of transfer" without the Ying et al. and Demontis credits.
- "Monitor designers can choose probes with low whitened cosine" as a robustness recipe: the refit
  result shows the evader can still fool a fixed low-cosine probe if it is trained against it too; the
  lemma is about transfer from ONE fixed probe.
