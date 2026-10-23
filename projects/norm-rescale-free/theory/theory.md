# Theory: the free rescale before the final norm (T3, owner Iris)

Numbers: `theory/theory_checks.py` -> `data/results/theory.json` (run 2026-10-23 15:2x CEST, CPU float64, unembedding
streamed in 8192-row chunks; measured peak RSS 1.13 GB with /usr/bin/time -l). Input: `theory/inputs/clean_country_R28.npy` = theory-1022
`states/clean/country/R28.npy` (clean readout-28 country-token states, fp16, 2992 x 1536). Probe: `data/fits/panel.npz`
`tok28` (sha1 99ab6a39..., identical to the 1022 file).

## Setup and symbols
- $h \in \mathbb{R}^{1536}$: the residual state at the country token after block 27 (readout 28), i.e. the input of the
  final RMSNorm. $\hat h = h/\lVert h\rVert$.
- Final norm: $N(h) = g \odot h / \sqrt{\lVert h\rVert^2/d + \epsilon}$, $d = 1536$, $\epsilon = 10^{-6}$, $g$ the learned gain.
- Next-token log-probabilities at that position: $\log\mathrm{softmax}(W N(h))$, $W$ the tied unembedding (151,936 rows).
  Nothing else reads $h$: later positions attend to keys/values computed from block inputs, not from readout 28.
- Output penalty at that position: $\mathrm{KL}(h) = \mathrm{KL}(p(h_0)\,\Vert\,p(h))$, $h_0$ the clean state.
- Frozen probe score: $s(h) = w\cdot h + b_0$ with $w$ = probe weight / feature sd, $b_0 = b - w\cdot\mu$.
- Twin gap of pair $k$ (false statement, its true twin: same city, same template): $\mathrm{gap}_k = s(h_{f,k}) - s(h_{t,k})$;
  the attacker's probe loss is the mean of $\mathrm{gap}_k^2$.
- Pair set: the 596 eval twin pairs (298 cities x 2 templates), the set frozen in DEVIATIONS N1.1 (0647eb1, before
  step 1). theory_checks.py builds it as {(false, true): same city, same src}; checked identical to analysis.py's
  `pairs` (596 = 596, set equality True, 2026-10-23 15:12).

## Lemma 1 (the rescale is free and the output penalty never resists it)
For $\epsilon = 0$ and any smooth $f$ of the normalised state:
1. $N(c\,h) = N(h)$ for every $c > 0$, so $\mathrm{KL}(c\,h) = \mathrm{KL}(h)$.
2. $h \cdot \nabla_h f(N(h)) = 0$ at every $h$ (Euler's theorem, degree 0).
3. $\nabla^2_h f \; h = -\nabla_h f$ (Euler applied to the gradient, which has degree $-1$).
   At the clean state $\nabla_h \mathrm{KL} = 0$, so $\nabla^2_h\mathrm{KL}(h_0)\,h_0 = 0$: the clean direction is a null
   direction of the output penalty's curvature.

Proof sketch. 1: $\lVert c h\rVert = c\lVert h\rVert$ cancels. 2: differentiate $f(N(ch)) = f(N(h))$ in $c$ at $c = 1$.
3: differentiate $\nabla f(ch) = c^{-1}\nabla f(h)$ in $c$ at $c = 1$. With $\epsilon > 0$:
$\sum_i h_i\,\partial N_j/\partial h_i = N_j\,\epsilon/(\lVert h\rVert^2/d + \epsilon)$, so the symmetry is broken by a
relative $\approx 3\times10^{-8}$ at these states. This is the activation-space version of the scale-symmetry facts in
Kunin et al. 2012.04728 (gradient perpendicular to the symmetric vector, vector in the Hessian kernel, conserved length
under gradient flow) and Lyu, Li & Arora 2206.07085 Lemma D.1.

Consequence in words: the term that keeps the outputs the same never pushes a state longer or shorter. Only the
probe term changes a state's length, and it does so at the rate $\propto w\cdot\hat h$.

| check | result (theory.json key) |
|---|---|
| A1-A3, sympy, $d = 3$, symbolic $g$, $\epsilon$, arbitrary $f$ | all residuals simplify to 0 (`A1`, `A2`, `A3`) |
| B1 KL after $h_0 \to c h_0$, $c \in \{0.1, 0.3, 0.45, 0.9, 2\}$, 20 states, full vocabulary | max $1.2\times10^{-11}$ (c = 0.1; $\epsilon$ effect), $\le 1.3\times10^{-13}$ for $c \ge 0.3$ |
| B2 $\lvert h\cdot\nabla\mathrm{KL}\rvert/(\lVert h\rVert\lVert\nabla\mathrm{KL}\rVert)$ at a 5% perturbed state | max $1.5\times10^{-9}$ (bound $\epsilon/(\mathrm{ms}+\epsilon) \le 4.1\times10^{-8}$) |
| B3 curvature $\lVert\nabla^2\mathrm{KL}\,\hat v\rVert$ at $h_0$ (median of 20) | $\hat v = \hat h_0$: $3.7\times10^{-11}$; random: $1.7\times10^{-3}$; probe $w$: $4.2\times10^{-3}$ |

## Lemma 2 (step-0 prediction of the title metric $R$, in activation space)
One gradient step of size $\eta$ on the probe loss, taken on the states themselves, moves
$h_{f,k} \to h_{f,k} - 2\eta\,\mathrm{gap}_k w/n$ and $h_{t,k} \to h_{t,k} + 2\eta\,\mathrm{gap}_k w/n$ (the KL term has zero
gradient at the clean state). Rescaling each moved state back to its clean length removes exactly the part of each move
along $\hat h$ (to first order in $\eta$). The gap falls by $2\eta\,\mathrm{gap}_k\lVert w\rVert^2 \cdot 2/n$ per pair, and the
restore gives back the radial part. Hence

$$R_0 = \frac{\sum_k \mathrm{gap}_k\left(\cos^2_{f,k} + \cos^2_{t,k}\right)}{2\sum_k \mathrm{gap}_k}, \qquad \cos_{\cdot,k} = \cos(w, h_{\cdot,k}).$$

Measured (596 eval pairs): $R_0 = 0.0028$ (`D_R0`); unweighted mean of $\cos^2$ 0.0027; finite steps $\eta = 10^{-3}, 0.1, 1$
give $R = 0.00280, 0.00280, 0.00280$ (`D_finite_step`). $\cos^2$ median: false 0.0051, true twins 0.00004. So at the start
the attacker's push is 99.7% tangential. This is a statement about the states, not about LoRA weights (Tove #5450):
a weight update moves the states by $J J^\top$ times this push, which can tilt it.

## Fact 3 (many tangential directions are nearly free too)
Output Fisher at the country token, $F = J^\top(\mathrm{diag}\,p - pp^\top)J$, $J = W\,\partial N/\partial h$, full vocabulary,
12 false eval states (`C_rows`, `C_median`): top eigenvalue 0.05-0.09; 3-4 eigenvalues below $10^{-8}$ (the radial one
included, Rayleigh quotient 0.0); 280-1050 eigenvalues below $10^{-6}$; 1466-1493 below $10^{-4}$. Share of the probe
direction $w$: radial 0.3-0.75%; in eigen-directions below $10^{-6}$: 16-59% (median ~45%); below $10^{-4}$: 90-94%.
In words: the radial direction is the only EXACTLY free one, but about half of the probe direction already lies in
directions the output barely sees, and they need a 4-8x shorter move (second-order cost of moving along that part far
enough to close a gap of 9: move ~23-38 units vs ~180 radial; KL $\lesssim \tfrac12 m^2 \cdot 10^{-6} \approx 3$-$7\times10^{-4}$ at the
country token, a second-order estimate only). Prediction from Lemmas 1-2 and Fact 3: gradient training closes the gap
mostly with near-free tangential moves, so $R$ stays far below 0.5 at the KL weight $10^4$; $R$ can only grow when the
weight is large enough to also freeze the near-null tangential directions.

## Fact 4 (the analytic point for the figure)
Caught false eval statements $h \to c_i h$ with $c_i = 0.95\,c^*_i$ ($c^*_i$ puts the score on the line), true statements
unchanged: catch at the re-set line 98.5% -> 0.0% (587 caught; line -0.49991 before and after), $c_i$ median 0.43
(0.29-0.95), exact country-token KL max $6.1\times10^{-14}$, median $9.2\times10^{-15}$ (`analytic_*`). A UNIFORM shrink
does not move the re-set-line catch (ranks kept); it does move clean-line catch, since $b_0 = -4.84 <$ line $-0.50$.

## Where the symmetry is NOT exact
Only at readout 28 (the last pre-norm state). At an earlier readout a rescale changes the relative size of every later
block's write. 1022 notes: the needed shrink exists at readouts 10/18/27 but costs output change.

## Fact 5 (idealized attacker: the states move freely; theory/flow_sim.py, theory/flow_quadratic.py)
Setup: 16 eval twin pairs (same draw in both scripts), the 32 country-token states are the parameters, plain
gradient descent on $\sum_k \mathrm{gap}_k^2 + \lambda_{\rm sim}\sum_i \mathrm{KL}_i$. Step rule, fixed at 15:12 CEST
before arm A seed 0 reported R (15:24): $\mathrm{lr} = 1/(4\lVert w\rVert^2 + 0.15\lambda_{\rm sim})$, stop at a 95% mean gap drop
or at the step cap. Vocabulary cut to tokens with clean probability $> 10^{-6}$ for some state (11,622 tokens; cut mass
$\le 0.36\%$). Footprints: flow_sim 0.69 GB, flow_quadratic 1.21 GB (both measured with /usr/bin/time -l).

Exact KL (flow_sim_lam0_1e2_1e4.json, flow_sim_lam100000.json), R at the end:

| $\lambda_{\rm sim}$ | steps | gap drop | R | country KL mean |
|---|---|---|---|---|
| 0 | 250 | 100% | 0.0016 | 0.012 |
| 1e2 | 250 | 99.9% | 0.0017 | 4e-4 |
| 1e4 | 5,000 | 95.1% | 0.0020 | 8e-5 |
| 1e5 | 120,000 (cap) | 92.1% | 0.0038 | 1.9e-5 |

Quadratic KL, closed-form gradient flow (flow_quadratic.json; $F h_0 = 0$ kept exactly). It matches the exact-KL runs
where both exist (1e4 at 95% drop: 0.0020 vs 0.0020; 1e5 at 90-92%: 0.0036 vs 0.0038). R at the flow time where the
mean gap has halved (the LoRA precondition point) / fallen 90%, with the median false length ratio:

| $\lambda_{\rm sim}$ | R at 50% drop | R at 90% drop (length ratio) |
|---|---|---|
| 1e2-1e4 | 0.0021-0.0022 | 0.0017-0.0020 (0.998) |
| 1e5 | 0.0026 | 0.0036 (0.996) |
| 1e6 | 0.0055 | 0.54 (0.45) |
| 1e7 | 0.69 | 0.78 (0.17) |
| 1e8 | 0.96 | 0.85 (0.14) |

Validity limit: the quadratic model treats the whole line through the origin as free, but a sign flip is not free
(Vera: $c = -1$ costs 32 nats). When a state is shrunk below ~10% of its length (late times, 99% drops,
`R_t_end` = 0.066 at every $\lambda$) the restore crosses zero and R is meaningless there; the rows above stay at
length ratios $\ge 0.14$.

Scale of the LoRA's weight in these units [INFERENCE from attack_norm.py:211-244, config.yaml:19-21]: the LoRA
loss is the mean of $\mathrm{gap}^2$ over 8 pairs plus $\tfrac{\lambda}{2}\times$ the KL averaged over all positions of 8
OTHER statements (~85 positions), so one country-token KL weighs about $\lambda/170$ against $1/8$ per pair:
$\lambda_{\rm sim} \approx \lambda/21$. LoRA $\lambda = 10^4 \to \lambda_{\rm sim}\approx 500$; $10^5 \to \approx 5{,}000$. Both sit in the regime
where the idealized attacker's R stays $\le 0.4\%$; R passes 0.5 at half gap closure only near $\lambda_{\rm sim} \sim 10^7$,
i.e. a LoRA weight $\sim 2\times10^8$. Reading: the free stretch exists, but plain gradient descent reaches it only
when the output penalty is strong enough to also block the many nearly free turns.
