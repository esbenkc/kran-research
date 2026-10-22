**Line Lemma Derivation (Nadia)**

**Lemma:** For a panel monitor $M$ defined as the maximum over individual layer scores $z_l$, the alarm line $t_M$ (set at a fixed upper percentile, e.g., the 2nd highest true statement) is strictly bounded below by the maximum of the individual layer lines $t_l$, i.e., $t_M \ge \max_l t_l$. Consequently, the overall panel catch rate is bounded by the sum of individual layer catch rates evaluated at $t_M$.

**Derivation:**
1. Let $z_{l,i}$ be the standardized score for layer $l$ on true statement $i$.
2. The panel score is $M_i = \max_l z_{l,i}$. By definition, for every true statement $i$ and every layer $l$, $M_i \ge z_{l,i}$.
3. Let $t_M$ be the 2nd highest score in the set $\{M_i\}$ and let $t_l$ be the 2nd highest score in the set $\{z_{l,i}\}$. 
4. Since the elements of $\{M_i\}$ dominate the elements of $\{z_{l,i}\}$ point-wise, the order statistics are similarly bounded. Thus, the 2nd highest $M_i$ must be at least the 2nd highest $z_{l,i}$ for any layer $l$. This gives us $t_M \ge t_l$ for all $l$, or $t_M \ge \max_l t_l$.
5. A false statement is caught by the panel if its panel score $M_{false} = \max_l z_{l,false} > t_M$.
6. This condition is equivalent to: there exists at least one layer $l$ such that $z_{l,false} > t_M$.
7. Using the union bound, the probability (catch rate) that the max exceeds $t_M$ is bounded by the sum of the probabilities that any individual layer exceeds $t_M$:
   $\text{Catch}(M) = P(\max_l z_{l,false} > t_M) \le \sum_l P(z_{l,false} > t_M)$.

*Numerical verification (implemented in `check.py`):*
Across all 3 seeds, we confirmed that `t_M >= max_t_l` (e.g., for seed 0, $6.3005 \ge 6.3005$), and that `catch(M) <= sum_l catch_l(at t_M)` (e.g., $0.0218 \le 0.0688$).
