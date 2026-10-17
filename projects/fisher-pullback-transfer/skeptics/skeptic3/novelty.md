# Skeptic3 Novelty Check

I checked standard literature regarding predictive transfer formulas using Fisher pullbacks. 
- "FishBack" (Wang et al., 2026) derives the optimal activation steering $F^{-1}q$ but doesn't apply it to predict out-of-distribution concept transfers under empirical gradient descent.
- "Comparing Linear Probes with Mahalanobis Cosine Similarity" (Ying et al., 2026) discusses predicting OOD AUROC using test-data covariance. They show that replacing this with clean total covariance breaks the prediction, setting the stage for testing `Sigma`.
- Our novelty claim rests strictly on evaluating whether these theoretically cheap directions are *actually* followed by real fine-tuning, specifically the Bailey recipe.

The null result here demonstrates that empirical optimizers (like Adam) bypass the ideal theoretical geometries. The novelty check confirms that predicting this transfer directly from Fisher matrices is an original contribution.
