# skeptic3 — independent derivation (arXiv 2603.01192 v3, Cullen, Estan-Ruiz, Danait, Li)

Numbering: v3. Goal text "Thm 4.1 / 4.2 / 5.2 / 5.5" = v3 Cor 4.2 (two cases), Thm 5.2, Remark 5.5.

## Setup

Network: $f_c(x) = \sum_j V_{cj} (w_j^\top x)^2$, $w_j\in\mathbb R^{2p}$, $V\in\mathbb R^{p\times K}$.
Data: $x=[e_a; e_b]$, $N$ training pairs. Loss (no L2): $L_N(\theta)=\frac1{2N}\|(Y-f_\theta(X))P_\perp\|_F^2$.

Write $w_j=[u_j; s_j]$, $u_j,s_j\in\mathbb R^p$. On one-hot data $w_j^\top x = u_j[a]+s_j[b] =: h_j(a,b)$.

## Step 1. What the loss can see

For a realisable squared loss, $L_N(\theta^*+\delta) = \frac1{2N}\|J\delta\|^2 + O(|\delta|^3)$ with $J$ the Jacobian
of the centred training outputs. Two standard facts (Watanabe 2009, ch. 7; used the same way in Lau et al. 2308.12108):

* $\lambda \ge \operatorname{rank}J/2$ (the $\operatorname{rank}J$ directions are quadratic; other directions add $\ge 0$).
* $\lambda = \operatorname{rank}J/2$ when the zero set $\{L_N=0\}$ is a smooth manifold of codimension $\operatorname{rank}J$
  near $\theta^*$ (minimally singular / "locally smooth" in the paper's Remark 4.3).

So at a generic smooth true solution, $2\lambda$ = generic rank of $J$. This is exactly the paper's secant-dimension
argument (Thm 4.1: $\lambda=\frac12\dim\sigma_K$), but the dimension must be counted **in the space of functions on the
training inputs**, not in $\mathcal Y=\mathbb R^p\otimes\mathrm{Sym}(\mathbb R^{d\times d})$.

## Step 2. Per-neuron count on one-hot data: $3p-2$, not $3p-1$

Neuron $j$'s contribution $v_j\otimes h_j^2$ is unchanged by
1. scaling $(u_j,s_j,v_j)\to(t^{-1/2}u_j,\,t^{-1/2}s_j,\,t\,v_j)$ (the paper's $-1$), and
2. **shift** $(u_j,s_j)\to(u_j+c\mathbf 1,\ s_j-c\mathbf 1)$, because $h_j(a,b)=u_j[a]+s_j[b]$ is unchanged.
   Equivalently, every input satisfies $(\mathbf 1_p,-\mathbf 1_p)^\top x=0$.

The paper's count $d+p-1$ assumes inputs that span $\mathbb R^d$ in the sense that $ww^\top\mapsto x^\top ww^\top x$ is
injective; one-hot pairs do not. Per neuron: $3p$ parameters $-2 = 3p-2$.

Exact check (sympy, rational point, `sympy_neuron.py`): $p=3$: rank 7; $p=5$: rank 13; both kernel vectors verified exactly;
with generic (non one-hot) inputs the same neuron has rank $3p-1$ (8, 14). Two generic neurons, $p=5$: 26; three: 39.

## Step 3. The cap: $p(N-1)$, not $pD$

The centred outputs live in $\mathbb R^{p\times N}$ modulo one constant per output: dimension $p(N-1)$.
So $\operatorname{rank}J\le p(N-1)$ and $\lambda\le p(N-1)/2$ at any realisable point with smooth zero set (and
$\lambda\le\operatorname{codim}/2\le p(N-1)/2$ in general).
The paper's over-parametrised value $\frac p4 d(d+1) = \frac{p^2(2p+1)}2$ exceeds $\frac{p(p^2-1)}2$ for every $p$
even with the full data set ($N=p^2$), so it is never attained on this task.

Generic count, numerically (`rank_check.py`, 16 cases): $\operatorname{rank}J = \min\{K(3p-2),\,p(N-1)\}$ in 15/16,
one near-cap defect ($p=7$, $K=8$, $N=24$: 151 vs 152).

**Consequence for the claims under test.**
* Cor 4.2 "under-parametrised" $K(d+p-1)/2$: corrected generic value $K(3p-2)/2$, valid only while $K(3p-2)<p(N-1)$
  (paper's condition uses $pD$, far too loose: $p=7,N=24$ breaks at $K\ge 9$, paper allows $K\le 36$).
* Cor 4.2 "over-parametrised" $p\,d(d+1)/4$: corrected value $p(N-1)/2$.
* Thm 5.2 is a statement about the **frozen-$W$** random-feature model; $\lambda=\frac p2\min\{l,K\}$ with
  $l=\operatorname{rank}\tilde F$, is exact there (it is a linear model in $V$: $p\cdot\operatorname{rank}(\tilde F)/2$).
  It is NOT the LLC of the full $(W,V)$ network at the same point.
* Remark 5.5 $K_{\rm eff}(3p-1)/2$: generic corrected value $K_{\rm eff}(3p-2)/2$ — but see Step 4.

## Step 4. Fourier points are not generic

A Fourier neuron of frequency $k$: $u[a]=\cos(\theta a+\alpha)$, $s[b]=\cos(\theta b+\beta)$, $v[c]=\cos(\theta c+\gamma)$,
$\theta=2\pi k/p$. Neurons sharing $k$ have tangent spaces in a common subspace:

* $\delta V$ directions: $\delta v\otimes h^2$, and $h^2\in\operatorname{span}\{1,\cos2\theta a,\sin2\theta a,\cos2\theta b,\sin2\theta b,
  \cos\theta(a\pm b),\sin\theta(a\pm b)\}$ (9 functions, 8 after centring): at most $8p$.
* $\delta w$ directions: $v\otimes(h\,\delta h)$ with $v\in\operatorname{span}\{\cos\theta c,\sin\theta c\}$ and
  $h\,\delta h\in\{$fns of $a\}+\{$fns of $b\}+\operatorname{span}\{\cos\theta a,\sin\theta a\}\otimes\mathbb R^p_b
  +\mathbb R^p_a\otimes\operatorname{span}\{\cos\theta b,\sin\theta b\}$, of dimension $p+p+2p+2p-1-2-6=6p-9$, $6p-10$ after
  centring: at most $2(6p-10)$.
* Overlap: all 8 centred $h^2$ functions lie in the second space, so overlap $2\cdot 8$.

Per-frequency cap: $8p+12p-20-16 = 20p-36$. Numerically (`fourier_rank.py`, full data, random phases): one frequency,
$m$ neurons: $p=7$: 19, 38, 57, 72, 86, 94, 99, 104, 104…; $p=11$ cap 184; $p=13$ cap 224 — all $=20p-36$.
Rank equals $m(3p-2)$ only for $m\le3$; at $m=4$ it is $4(3p-2)-4$.

**Consequence:** if a grokked network places $\ge4$ active neurons on one frequency, $K_{\rm eff}(3p-2)/2$ (and the
paper's $K_{\rm eff}(3p-1)/2$) overstates $\operatorname{rank}J/2$. The paper's Remark 4.3 allows for this in words
("non-generic algebraic relations ... may be smaller"); Remark 5.5 still uses the generic value. The honest per-checkpoint
reference is $\operatorname{rank}J(\theta^*)/2$, a lower bound on $\lambda$, equal to $\lambda$ if the zero set is smooth.

## Step 5. What a measurement can and cannot separate

$(3p-1)/(3p-2)-1 = 1/(3p-2)$: 5.3 % at $p=7$, 1.1 % at $p=31$. An SGLD estimate with ~5–10 % error cannot separate the
two per-neuron counts. It CAN detect: the cap (factor ~2 at large $K$), the per-frequency loss (up to $\sim$40 % at
$m=6$), dead-neuron contributions, and non-realisability (train loss $>0$ under weight decay).
