# Derivation and Limits of the Binormal Share Formula - Hugo (skeptic2)

## 1. Setup and Notation (Binormal Score Model)

Let $S$ denote the scalar score produced by the linear probe $w^T x + b$.
Assume clean scores follow a binormal distribution with common standard deviation $\sigma$:
- Clean negatives (benign / true statements): $S_{neg, clean} \sim \mathcal{N}(\mu_-, \sigma^2)$
- Clean positives (harmful / false statements): $S_{pos, clean} \sim \mathcal{N}(\mu_+, \sigma^2)$
- Separation distance in standard deviation units: $D = \frac{\mu_+ - \mu_-}{\sigma}$.
Without loss of generality, we can scale by $\sigma$ and shift so that $\mu_- = 0$, $\mu_+ = D$, $\sigma = 1$.

Let $\Phi(u) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^u e^{-t^2/2} dt$ be the standard normal CDF, and $\phi(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ be the standard normal PDF.

The baseline threshold $t_0$ is set to achieve a target false positive rate $\alpha$ (e.g. 1% FPR, where $z = \Phi^{-1}(1 - \alpha) \approx 2.326$):
$$P(S_{neg, clean} > t_0) = 1 - \Phi\left(\frac{t_0 - \mu_-}{\sigma}\right) = \alpha \implies t_0 = \mu_- + z \sigma.$$

### Clean True Positive Rate ($TPR_0$)
$$TPR_0 = P(S_{pos, clean} > t_0) = 1 - \Phi\left(\frac{t_0 - \mu_+}{\sigma}\right) = 1 - \Phi(z - D) = \Phi(D - z).$$

---

## 2. Attacked Model and Evasion Dynamics

The fine-tuning attack pushes the positive class scores down by an amount $p \sigma$:
$$\mu_{+, att} = \mu_+ - p \sigma = \mu_- + (D - p) \sigma.$$

In response to the gradient updates on the network representation, the negative class also moves in the same direction by a fraction $c \in [0, 1]$ of the positive push (common-mode shift):
$$\mu_{-, att} = \mu_- - c p \sigma.$$
Here $c = \frac{\Delta \mu_-}{\Delta \mu_+} = \frac{\mu_{-, clean} - \mu_{-, att}}{\mu_{+, clean} - \mu_{+, att}}$.

### Attacked True Positive Rate at Unadjusted Threshold ($TPR_a$)
At the stale threshold $t_0 = \mu_- + z \sigma$:
$$TPR_a = P(S_{pos, att} > t_0) = 1 - \Phi\left(\frac{t_0 - \mu_{+, att}}{\sigma}\right) = 1 - \Phi(z - (D - p)) = \Phi(D - p - z).$$
The lost detection rate is:
$$\Delta TPR_{lost} = TPR_0 - TPR_a = \Phi(D - z) - \Phi(D - p - z).$$

---

## 3. Threshold Recalibration ($t_1$)

The defender re-calibrates the decision threshold on fresh held-out benign data from the attacked model to restore the target false positive rate $\alpha$:
$$P(S_{neg, att} > t_1) = 1 - \Phi\left(\frac{t_1 - \mu_{-, att}}{\sigma}\right) = \alpha = 1 - \Phi(z).$$

Setting $\frac{t_1 - \mu_{-, att}}{\sigma} = z$:
$$t_1 = \mu_{-, att} + z \sigma = (\mu_- - c p \sigma) + z \sigma = (\mu_- + z \sigma) - c p \sigma = t_0 - c p \sigma.$$

**Why $t_1 = t_0 - c p \sigma$ requires caution (uniform vs. tail shift):**
An order-statistic rule (e.g. 2nd-largest of $n=200$ negatives) is strictly shift-equivariant only when every sample experiences an identical shift:
$$\text{OrderStat}(X + \Delta) = \text{OrderStat}(X) + \Delta.$$
This relation is exact under a uniform shift. However, when individual negative statements experience heterogeneous per-item shifts, $t_1$ tracks the shift of the top order statistics (the extreme negative tail) rather than the sample mean shift $-c p \sigma$. In our experiments, hard true statements in the negative tail move slightly less than the average negative, causing the calibrated threshold $t_1$ to sit higher than the uniform theory predicts. This difference explains why the E6 formula overpredicts recovery in the intermediate regime ($\lambda = 10^4$, where E6 MAE is 0.37) while succeeding at both extremes.
$$\frac{t_1 - \mu_{+, att}}{\sigma} = (z - c p) - (D - p) = z - D + (1 - c) p.$$
Therefore:
$$TPR_1 = 1 - \Phi(z - D + (1 - c) p) = \Phi(D - (1 - c) p - z).$$

---

## 4. Recovered Share Formula (E6)

The recovered share of lost detection performance is the ratio of recovered TPR to lost TPR:
$$Share(D, c, p) = \frac{TPR_1 - TPR_a}{TPR_0 - TPR_a} = \frac{\Phi(D - (1 - c)p - z) - \Phi(D - p - z)}{\Phi(D - z) - \Phi(D - p - z)}.$$

---

## 5. Mathematical Limits and Behavioral Regimes

### Limit L1: Full Common-Mode Shift ($c = 1$)
When $c = 1$, the negatives shift identically with the positives ($\mu_{-, att} - \mu_- = \mu_{+, att} - \mu_+$).
$$D - (1 - c)p - z = D - z.$$
The numerator becomes $\Phi(D - z) - \Phi(D - p - z)$, which equals the denominator.
Thus:
$$\left. Share(D, c, p) \right|_{c=1} = 1.$$
Recalibration recovers 100% of the lost detection rate.

### Limit L2: Zero Negative Shift ($c = 0$)
When $c = 0$, the negatives do not move at all, so $t_1 = t_0$.
$$D - (1 - c)p - z = D - p - z.$$
The numerator becomes $\Phi(D - p - z) - \Phi(D - p - z) = 0$.
Thus:
$$\left. Share(D, c, p) \right|_{c=0} = 0.$$
Recalibration recovers 0% of the lost detection rate.

### Limit L3: Persistent Attacker / Overshoot ($(1 - c)p > D$)
If the attacker continues pushing until $(1 - c)p > D$, the positive class mean is driven below the negative class mean ($\mu_{+, att} < \mu_{-, att}$).
In this regime, the ranking is inverted and AUROC drops below 0.5.
As $p \to \infty$:
- $D - (1 - c)p - z \to -\infty \implies \Phi(D - (1 - c)p - z) \to 0$
- $D - p - z \to -\infty \implies \Phi(D - p - z) \to 0$
- The denominator approaches $\Phi(D - z) > 0$.
Therefore:
$$\lim_{p \to \infty} Share(D, c, p) = \frac{0 - 0}{\Phi(D - z)} = 0.$$
Once separation is destroyed or inverted, one-sided threshold recalibration fails completely.

### Limit L4: Infinitesimal Push ($p \to 0$, L'Hôpital's Rule)
As $p \to 0$, both numerator and denominator vanish:
$$\lim_{p \to 0} [\Phi(D - (1 - c)p - z) - \Phi(D - p - z)] = \Phi(D - z) - \Phi(D - z) = 0,$$
$$\lim_{p \to 0} [\Phi(D - z) - \Phi(D - p - z)] = \Phi(D - z) - \Phi(D - z) = 0.$$

Applying L'Hôpital's Rule with respect to $p$:
$$\frac{d}{dp}\left[\Phi(D - (1 - c)p - z) - \Phi(D - p - z)\right] = -(1 - c)\phi(D - (1 - c)p - z) - (-1)\phi(D - p - z).$$
At $p = 0$:
$$\left.\frac{d}{dp}\text{Num}\right|_{p=0} = -(1 - c)\phi(D - z) + \phi(D - z) = c \cdot \phi(D - z).$$

For the denominator:
$$\frac{d}{dp}\left[\Phi(D - z) - \Phi(D - p - z)\right] = 0 - (-1)\phi(D - p - z) = \phi(D - p - z).$$
At $p = 0$:
$$\left.\frac{d}{dp}\text{Den}\right|_{p=0} = \phi(D - z).$$

Taking the quotient:
$$\lim_{p \to 0} Share(D, c, p) = \frac{c \cdot \phi(D - z)}{\phi(D - z)} = c.$$
At the onset of evasion, the fraction of recovered detection exactly equals the common-mode shift ratio $c$.

---

## 6. Unequal Variance Generalization (E7)
When the clean and attacked distributions exhibit unequal variances ($\sigma_{neg, c}, \sigma_{pos, c}, \sigma_{neg, a}, \sigma_{pos, a}$):
$$t_1 = \mu_{-, att} + z \sigma_{neg, a},$$
$$TPR_1 = \Phi\left(\frac{\mu_{+, att} - t_1}{\sigma_{pos, a}}\right).$$
If the negative class variance expands ($\sigma_{neg, a} > \sigma_{neg, c}$), the calibrated threshold $t_1$ is pushed higher into the positive class distribution, reducing $TPR_1$ below the equal-variance prediction. This explains why E6 can over-predict empirical recovery when the negative tail widens.
