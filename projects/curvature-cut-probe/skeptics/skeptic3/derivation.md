# Derivation Check (skeptic3)

I have independently verified the two displayed equations in the post's expert foldout.

### 1. Second-order KL expansion

**Equation:**
$$ \text{KL}(p \parallel \text{softmax}(W(z + \delta))) = \frac{1}{2} \delta^T F \delta + O(\|\delta\|^3) $$
where $F = W^T(\text{diag}(p) - p p^T)W$.

**Check:**
The post-norm logits are $l = Wz$. The prediction probability vector is $p = \text{softmax}(Wz)$.
The KL divergence is:
$$ \text{KL}(p \parallel q) = \sum p_i \log \frac{p_i}{q_i} = - \sum p_i \log q_i + C $$
Since $q = \text{softmax}(W(z + \delta))$, the log-probabilities are $(W(z + \delta))_i - \log \sum \exp(W(z + \delta)_j)$.
Taking the Taylor expansion of the KL divergence around $\delta = 0$, the first-order term vanishes because $p$ is the minimum.
The second derivative (Hessian) of the negative log-likelihood (or log-sum-exp) with respect to the pre-softmax logits $l$ is the covariance matrix of the softmax probabilities: $H_l = \text{diag}(p) - p p^T$.
Applying the chain rule for the linear map $l = Wz$, the Hessian with respect to $z$ is:
$$ F = \nabla_z^2 \text{KL} = W^T (\text{diag}(p) - p p^T) W $$
Thus, the second-order Taylor expansion is exactly:
$$ \text{KL} = \frac{1}{2} \delta^T F \delta + O(\|\delta\|^3) $$
This matches the post's claim.

### 2. Pair split minimum cost

**Equation:**
$$ c = \frac{D^2}{2(g_f + g_t)} $$

**Check:**
The cost for shifting the false statement by $x$ is approximately $\frac{x^2}{2 g_f}$, where $g_f = 1/(w^T F_f^+ w)$. 
The true statement shifts by $D - x$, costing $\frac{(D-x)^2}{2 g_t}$.
The total pair cost is:
$$ c(x) = \frac{x^2}{2 g_f} + \frac{(D-x)^2}{2 g_t} $$
To find the minimum, we set the derivative with respect to $x$ to $0$:
$$ \frac{x}{g_f} - \frac{D-x}{g_t} = 0 \implies x g_t = (D-x) g_f \implies x(g_t + g_f) = D g_f \implies x = \frac{D g_f}{g_f + g_t} $$
Substituting the optimal $x$ back into the cost function:
$$ c = \frac{1}{2g_f} \left(\frac{D g_f}{g_f + g_t}\right)^2 + \frac{1}{2g_t} \left(D - \frac{D g_f}{g_f + g_t}\right)^2 $$
$$ c = \frac{D^2 g_f}{2(g_f + g_t)^2} + \frac{1}{2g_t} \left(\frac{D g_t}{g_f + g_t}\right)^2 $$
$$ c = \frac{D^2 g_f}{2(g_f + g_t)^2} + \frac{D^2 g_t}{2(g_f + g_t)^2} = \frac{D^2 (g_f + g_t)}{2(g_f + g_t)^2} = \frac{D^2}{2(g_f + g_t)} $$
This precisely matches the post's pair price formula. Both equations are mathematically verified.
