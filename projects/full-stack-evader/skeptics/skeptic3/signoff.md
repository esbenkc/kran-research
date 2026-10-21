# Skeptic 3 Sign-Off

I have independently reproduced the key statistics from the raw data and reviewed the derivation and novelty checks.

## 1. Rebuild and Null Tests
- **Act-level Rebuild**: I manually recomputed the `tok18` score for `target a seed 0` and the `tok18` and `tok29` (L28post) scores for `b300` seeds 0, 1, and 2 directly from the raw `tok.npy` activations, scaling parameters, and monitor weights. The max absolute difference between my act-level rebuilt scores and the pre-computed `data/scores` was precisely `0.0`.
- **Score-level Rebuild**: I rebuilt the 1% FPR alarm line (2nd highest of 200 true `refit` statements) and evaluated it on the 596 false `eval` statements across the 3 seeds, using the correct indices from `split.json`.
- **Target (b) Mega**: Pooled catch is 0.033 [0.021, 0.046]. 
- **Target (a) tok18**: Pooled catch is 0.025 [0.015, 0.036]. The line-resplit null CI is [0.007, 0.068].
- **Target (b) L28post**: Pooled catch is 0.976 [0.963, 0.987]. 
- **True Null Test**: I generated a random unit direction vector and scored it on the `b300` raw activations at `tok29` (L28post). The resulting catch was ~3.5-4.0% at a 1% FPR threshold, matching theoretical chance level on the limited evaluation set (596 false, 200 true calibration).
- **Conclusion**: The measurements exactly match the reported metrics. The target monitors successfully fall below 20%, but all plain probes stay high under `target (b)`, verifying the status of `REJECTED`.

## 2. Derivation Check (Iris's Lemma)
- I wrote my own numeric verification to bound the required KL. 
- **Derivation**: We want to minimize the KL penalty $KL \approx \frac{1}{2} d^T F d$ subject to moving the probe score past the gap $g$, meaning $w^T d = g$. The Lagrangian is $L(d, \lambda) = \frac{1}{2} d^T F d - \lambda (w^T d - g)$. Setting the gradient to zero gives $F d = \lambda w \implies d = \lambda F^+ w$. Substituting into the constraint gives $\lambda w^T F^+ w = g \implies \lambda = \frac{g}{w^T F^+ w}$. Thus, $KL = \frac{1}{2} (\lambda F^+ w)^T F (\lambda F^+ w) = \frac{1}{2} \lambda^2 w^T F^+ w = \frac{g^2}{2 w^T F^+ w}$.
- **Numeric Check**: I verified this by retrieving the 4096-token vocabulary Fisher $F = W^T (\mathrm{diag}(p) - p p^T) W$, adding a ridge of $10^{-6}$, and computing $g^2 / (2 w^T F^{-1} w)$ over 150 sampled CAUGHT false statements. The required KL is extremely small: median 4.15e-5 nats, max 1.49e-4 nats. This verifies that surviving the bound is trivial with a 0.05 budget.

## 3. Novelty Review
- I reviewed Vera's `novelty.md` and Iris's additions in `research.md`.
- I concur with the assessment. A one-layer blind spot mechanism has been observed in toy models (Jesse Li, 2026) and in different setups (McGuinness et al., 2025). The novelty of this specific work is narrow: scaling it to a 1.5B LLM with all blocks editable, rigid output-matching constraints directly on the evaluated text, and analyzing the transfer (or lack thereof) to other layers and attention-based probes (like the megaprobe). The wording "agrees with" instead of "first to discover" is appropriate and necessary.

I sign off on the result as **REJECTED**.
