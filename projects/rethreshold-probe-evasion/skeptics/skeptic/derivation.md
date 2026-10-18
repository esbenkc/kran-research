# Vera's own derivation (written before reading the post or Bruno's analysis)

Score of one statement under the frozen probe: s = w·h̄ + b (h̄ = mean-pooled residual at the probed layer).
Negatives (true statements) and positives (false statements) are modelled as two Gaussians:
s | neg ~ N(μ₋, σ₋²), s | pos ~ N(μ₊, σ₊²) (unequal-variance binormal model).

1. Threshold at false-positive rate α on negatives: t = μ₋ + z σ₋, with z = Φ⁻¹(1−α) (z = 2.326 at α = 1%).
2. TPR at threshold t: P(s > t | pos) = Φ((μ₊ − t)/σ₊).
   Clean model: TPR_c = Φ((μ₊ − μ₋ − z σ₋)/σ₊).
3. Attacked model has (μ₊', σ₊', μ₋', σ₋'). Old threshold t₀ = μ₋ + z σ₋ (clean negatives):
   TPR_old = Φ((μ₊' − t₀)/σ₊').
4. Re-fit on attacked negatives: t₁ = μ₋' + z σ₋':
   TPR_refit = Φ((μ₊' − μ₋' − z σ₋')/σ₊').
5. Recovered share R = (TPR_refit − TPR_old)/(TPR_c − TPR_old).
6. Special cases (checked numerically in vera_theory_check.py):
   - pure common shift (μ₊' = μ₊ − c, μ₋' = μ₋ − c, σ's unchanged): TPR_refit = TPR_c, R = 1 for any c with TPR_old < TPR_c.
   - negatives fixed (μ₋' = μ₋, σ₋' = σ₋): t₁ = t₀, R = 0.
   - common shift plus positive spread growth σ₊' = kσ₊: TPR_refit = Φ((μ₊ − μ₋ − zσ₋)/(kσ₊)); when the clean margin
     (μ₊ − μ₋ − zσ₋) > 0, spread growth lowers the refit TPR, so R < 1 even with zero loss of mean separation.
7. Binormal AUROC = Φ((μ₊ − μ₋)/√(σ₊² + σ₋²)).
8. Empirical threshold rule: flag iff s > k-th highest of n negatives. Out of sample the FPR has expectation k/(n+1)
   (order statistics of a uniform): k = 3, n = 200 gives 1.49%, not 1%.

## Felix's equal-variance share formula (#2704), re-derived
Set σ₊ = σ₋ = 1, μ₋ = 0, μ₊ = D. The attack moves positives by −p and negatives by −cp.
t₀ = z, TPR_c = Φ(D − z), TPR_old = Φ(D − p − z), t₁ = z − cp, TPR_refit = Φ(D − p + cp − z) = Φ(D − (1−c)p − z).
R(D,c,p) = [Φ(D−(1−c)p−z) − Φ(D−p−z)] / [Φ(D−z) − Φ(D−p−z)]. Limits: p→0 gives c (first-order Taylor, φ(D−z) cancels);
p→∞ with c<1 gives 0; c=1 gives 1. Numbers at a 50-point drop (my brentq): D=4 → p=1.79, R = 0.28/0.66/0.91 for c = 0.2/0.5/0.8;
D=6 → p=3.67, R = 0.54/0.93/1.00. Same as Felix's #2704.
