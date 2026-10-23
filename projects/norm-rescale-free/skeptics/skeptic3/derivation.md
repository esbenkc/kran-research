# Derivation Check by Skeptic 3 (Nadia)

## Lemma 1: The rescale is free and output penalty never resists it
**Claim 1**: $N(ch) = N(h)$ for $c > 0$.
*Check*: $N(h) = g \odot h / \sqrt{||h||^2/d + \epsilon}$. If $\epsilon=0$, $N(ch) = g \odot ch / \sqrt{||ch||^2/d} = g \odot ch / (c\sqrt{||h||^2/d}) = N(h)$. Confirmed.

**Claim 2**: $h \cdot \nabla_h f(N(h)) = 0$ (Euler's theorem).
*Check*: Since $f(N(ch)) = f(N(h))$, taking the derivative with respect to $c$ at $c=1$ gives $\nabla_h f(N(h)) \cdot h = 0$. Confirmed.

**Claim 3**: $\nabla^2_h f \cdot h = -\nabla_h f$. At clean state where $\nabla_h KL = 0$, $h_0$ is a null direction of the Hessian.
*Check*: Let $v(h) = \nabla_h f(h)$. We know $v(ch) = c^{-1} v(h)$ (degree -1 homogeneity). Differentiating with respect to $c$ at $c=1$ yields $\nabla_h v(h) \cdot h = -v(h)$. Thus $\nabla^2_h f \cdot h = -\nabla_h f$. At clean state $h_0$, $\nabla_h KL = 0$, so $\nabla^2_h KL(h_0) \cdot h_0 = 0$. Confirmed.

## Lemma 2: Step-0 Prediction of R
**Claim**: $R_0 = \frac{\sum_k gap_k (\cos_{f,k}^2 + \cos_{t,k}^2)}{2\sum_k gap_k}$
*Check*: One gradient step is $\Delta h = -2\eta \cdot gap_k \cdot w / n$ (for false, $+2\eta$ for true).
Restoring $h' = h + \Delta h$ to $|h|$ exactly removes the radial component $(\Delta h \cdot \hat{h})\hat{h}$ to first order.
The drop in gap per pair is $2\eta \cdot gap_k ||w||^2 (2/n)$.
The restored gap gives back the radial part of the move. The radial part of $w$ is $(w \cdot \hat{h})\hat{h}$. Thus the fraction of the drop given back is $\cos^2(w, h)$.
Averaging over all false and true pairs weighted by their gaps gives exactly the formula for $R_0$. Confirmed.
