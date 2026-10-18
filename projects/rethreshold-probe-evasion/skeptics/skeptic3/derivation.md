# Theoretical Derivation of the Recovered Share

**Goal:** Derive the E6 formula for the recovered share of TPR under an equal-variance binormal model, and verify its limits.

**Assumptions:**
- Clean negative scores: $\mathcal{N}(\mu_-, \sigma^2)$
- Clean positive scores: $\mathcal{N}(\mu_+, \sigma^2)$
- The separation is $D = (\mu_+ - \mu_-)/\sigma$. Without loss of generality, let $\mu_- = 0$ and $\sigma = 1$, so $\mu_+ = D$.
- The attacker shifts the positive mean down by $p$ (in units of $\sigma$), so the new positive mean is $D - p$.
- The attacker shifts the negative mean down by a fraction $c$ of the positive shift, i.e., $c \cdot p$. The new negative mean is $-c p$.
- The standard deviations are assumed unchanged (equal-variance assumption).

**Thresholds:**
- The threshold $t$ for a target FPR $\alpha$ is located $z$ standard deviations above the negative mean. Here $z = \Phi^{-1}(1-\alpha)$.
- Clean threshold $t_0$: Since $\mu_-=0$, $t_0 = z$.
- Attacked threshold $t_1$: The new negative mean is $-c p$, so $t_1 = -c p + z$.

**TPR Definitions:**
- Clean TPR at $t_0$: $\Phi(D - t_0) = \Phi(D - z)$
- Attacked TPR at $t_0$: $\Phi(D - p - t_0) = \Phi(D - p - z)$
- Attacked TPR at $t_1$: $\Phi(D - p - t_1) = \Phi(D - p - (-c p + z)) = \Phi(D - (1-c)p - z)$

**Recovered Share:**
- Lost TPR = $\text{TPR}_{clean}(t_0) - \text{TPR}_{att}(t_0) = \Phi(D - z) - \Phi(D - p - z)$
- Recovered TPR = $\text{TPR}_{att}(t_1) - \text{TPR}_{att}(t_0) = \Phi(D - (1-c)p - z) - \Phi(D - p - z)$
- Share = $\frac{\Phi(D - (1-c)p - z) - \Phi(D - p - z)}{\Phi(D - z) - \Phi(D - p - z)}$

**Limits:**
1. **Small push ($p \to 0$):** Using Taylor expansion of $\Phi(x)$ around $D-z$, the numerator is approximately $(-(1-c)p - (-p))\phi(D-z) = c p \phi(D-z)$. The denominator is $p \phi(D-z)$. The ratio approaches $c$.
2. **Persistent push ($p \to \infty$ with $c < 1$):** The new separation $D - p$ eventually becomes less than the negative shift $-c p$, meaning $(1-c)p > D$. The positive class is pushed below the negative class (AUROC < 0.5). In this limit, the standard threshold rule fails and the recovered share approaches 0.

*(These limits are verified symbolically in `nadia_rebuild.py`)*
