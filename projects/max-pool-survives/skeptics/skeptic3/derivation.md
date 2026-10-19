# Nadia's Derivations

## 1. Prefix Lemma
**Setting:** Each statement exists as a true/false pair sharing an identical prefix (e.g. "The city of Krasnodar is in"). 
**Claim:** The activations for the prefix tokens are exactly identical for the true and false statements, because a causal transformer evaluates them independently of the subsequent differing tokens.
**Proof:** In a causal decoder-only transformer, the hidden state at position $t$ depends only on tokens $1 \dots t$. Since the prefix tokens $1 \dots p$ are identical in the true and false versions of the statement, the sequence of hidden states $h_1 \dots h_p$ must be identical. Therefore, any probe applied to these tokens will yield the exact same score for both statements.

## 2. Shared-Shift Lemma
**Claim:** If an attack shifts the scores of all tokens by a common vector (or scalar shift $a$), re-thresholding the monitor exactly cancels the shift, for both mean and max pooling.
**Proof:**
Let the original token scores be $s_t$.
After attack, the scores are $s_t - a$.
For max pooling: $\max_t (s_t - a) = (\max_t s_t) - a$.
For mean pooling: $\frac{1}{N} \sum_t (s_t - a) = (\frac{1}{N} \sum_t s_t) - a$.
Since all statement scores shift by exactly $-a$, the relative ranking of statements is perfectly preserved. The new threshold $t_1$ will simply be $t_0 - a$, completely undoing the effect of the shared shift.

## 3. Bell-Curve Formula
**Setting:** We assume a Gaussian distribution of pooled scores for true statements ($T$) and false statements ($F$).
Original model: $\mu_{T0}$, $\mu_{F0}$, pooled within-class std $\sigma_0$.
Attacked model: $\mu_{T1}$, $\mu_{F1}$.
Separation: $D = (\mu_{F0} - \mu_{T0}) / \sigma_0$.
Push on false statements: $p = (\mu_{F0} - \mu_{F1}) / \sigma_0$.
Shared push: $c = (\mu_{T0} - \mu_{T1}) / (\mu_{F0} - \mu_{F1})$.

**Derivation of catch rate:**
The new threshold $t_1$ is set to catch 1% of true statements on the attacked model. Assuming equal std $\sigma_0$,
$t_1 = \mu_{T1} + z \sigma_0$, where $z = \Phi^{-1}(1 - 0.01)$.
The catch rate on false statements is $P(s_{F1} > t_1) = 1 - \Phi\left( \frac{t_1 - \mu_{F1}}{\sigma_0} \right) = \Phi\left( \frac{\mu_{F1} - t_1}{\sigma_0} \right)$.
Substitute $t_1$:
$\mu_{F1} - t_1 = \mu_{F1} - \mu_{T1} - z \sigma_0$.
From definitions:
$\mu_{F1} = \mu_{F0} - p \sigma_0$
$\mu_{T1} = \mu_{T0} - c p \sigma_0$
So, $\mu_{F1} - \mu_{T1} = (\mu_{F0} - \mu_{T0}) - p \sigma_0 + c p \sigma_0 = \sigma_0 (D - p + cp) = \sigma_0 (D - (1-c)p)$.
Thus, $\frac{\mu_{F1} - t_1}{\sigma_0} = D - (1-c)p - z$.
And the catch rate is $\Phi(D - (1-c)p - z)$.

If $c=0$, there is no shared push ($\mu_{T1} = \mu_{T0}$). The catch rate formula becomes $\Phi(D - p - z)$, which represents using the old threshold.
The derivation holds.
