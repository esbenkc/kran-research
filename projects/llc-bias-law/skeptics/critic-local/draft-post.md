---
title: "WORKING TITLE: A Spring-Strength Law for SGLD Learning-Coefficient Bias on Reduced-Rank Regression"
question: "Once Aoyagi's formula splits reduced-rank regression into regular and singular parts, does one curve in nbeta/gamma^2 predict the SGLD learning-coefficient error at held-out ranks and widths within 5 percentage points?"
slug: llc-bias-law
date: 2026-11-25
excerpt: "Not settled. A curve fitted on two shapes of a small linear network predicted the SGLD error within 5 points in 82% of settings on five unseen shapes. A lookup of the same setting on a training shape did almost as well."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1125-llc-bias-law/projects/llc-bias-law
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
limits:
  - "The split curve (an exact formula for the bowl part plus one fitted curve) did not beat a lookup, which copies the error of the same setting on a training shape. Their median misses were 1.9 and 2.0 points. On the 151 settled settings, the lookup did better, 1.5 against 1.7."
  - "The margin is thin. The 29 unsettled settings, about as many as walk noise alone would flag, count as misses for every rule. If unsettled walks counted as misses for the curve alone, two simpler rules would beat it, and the answer would be no."
  - "At a fixed setting, the five shapes differed by a median of only 4.5 points, and the setting alone explained 94% of the spread. So this test barely checks transfer to new shapes."
  - "All shapes are small two-layer linear networks with noise-free data. We scored only the long-run error, so the curve says nothing about short walks."
lab_room: theory-1125-llc-bias-law
lab_night: 2026-11-25
lab_hours: 1.5
lab_cost_usd: 0.25
figure: llc-bias-law
figure1: llc-bias-law-fig1
figure1_alt: "Figure 1. Head: On a network it had never seen, the split curve called the error closely, but a simple lookup came almost as close. Question: Once Aoyagi's formula splits reduced-rank regression into regular and singular parts, does one curve in nbeta/gamma^2 predict the SGLD learning-coefficient error at held-out ranks and widths within 5 percentage points? Answer: Not settled. The split curve predicted the error within 5 points in 82% of 180 settings, above our 80% bar. But it did not beat a simple lookup: the error of the same setting on a training shape. The lookup missed by a median of 2.0 points, against the curve's 1.9, and we required half. Example: a network shape the curve had not seen, 48 inputs, 8 hidden units, 48 outputs, rank 2; its true learning coefficient 227.5 splits into a bowl part 94 (exact formula) and a singular part 133.5 (the curve). At inverse temperature 1,000 and four spring strengths, the error the curve predicted before the run, the lookup's error (the measured error of the same setting on a training shape) and the measured error: spring 2.24: curve −1%, lookup −1%, measured −1%; spring 7.07: curve −2%, lookup −3%, measured −3%; spring 22.4: curve −8%, lookup −8%, measured −7%; spring 70.7: curve −23%, lookup −23%, measured −20%. Below, the typical miss in points over all 180 settings on 5 unseen shapes: the split curve 1.9; lookup: same setting, training shape 2.0; one curve in q, no split 2.2; the number we first planned 4.6; one constant 7.4; the test needed the curve at 1.0 or less, half the lookup's miss."
context:
  label: "Follow-up to"
  text: "our Lab post where a short SGLD walk failed a known-answer check"
  links:
    - {text: "llc-early-alarm-8b", url: "https://athanorscience.com/llc-early-alarm-8b/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1125-llc-bias-law/projects/llc-bias-law
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1125-llc-bias-law/projects/llc-bias-law/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1125-llc-bias-law/projects/llc-bias-law/skeptics
prereg:
  fixed: "November 25, 2026, 15:07 Copenhagen time"
  question: "Once the known formula for this network's learning coefficient is split into its bowl-shaped and flat parts, does one curve in the inverse temperature divided by the squared spring strength predict the sampler's error on network shapes we had not run?"
  expected: "We expected, with 50% confidence, that a law with two fitted numbers, fitted on two network shapes, would predict the sampler's error within 5 percentage points in at least 80% of 180 settings on five unseen shapes. Its typical miss also had to be at most half that of each of four simpler predictors."
  confidence: 50
  if_right: "Once the bowl-shaped directions are taken out with an exact formula, the error the spring causes follows one curve. Users could then set the spring strength by formula."
  if_wrong: "The error depends on the network's shape, or on more than one number. Every learning-coefficient claim then needs its own check against a known answer."
  outcome: failed
  outcome_note: "Half of it held. The curve was within 5 percentage points in 148 of 180 settings (82%). But its median miss, 1.9 points, was not half that of a lookup of the same setting on a training shape (2.0 points)."
  record: https://github.com/esbenkc/kran-research/blob/84d8b18613bdd41e626c18c3f147d2c9541c2a72/projects/llc-bias-law/PREREG.md
---

Many interpretability papers now track a number called the local learning coefficient (LLC). Roughly, it is half the number of weight directions that the loss really depends on, near the trained weights. Nobody can compute it exactly for a real network. So it is estimated with a random walk, SGLD, that starts at the trained weights.

<details class="concept" markdown="1">
<summary>What is the local learning coefficient (LLC)?</summary>

**In short:** The learning coefficient counts the weight directions that really matter at a solution. Each of these directions counts as one half. Picture the loss as a landscape around the trained weights. A direction where the loss rises like a bowl counts one half. A flat valley floor, where many weights give the same loss, counts less. "Local" means that we measure it around one set of trained weights.

**Further reading:** [Lau et al. 2023](https://arxiv.org/abs/2308.12108), the paper that defines the LLC and its estimator. [Hoogland et al. 2024](https://arxiv.org/abs/2402.02364), LLC changes at the stages of transformer training.

</details>

Every paper sets two knobs of this walk by hand[^knobs]. A spring, of strength $$\gamma$$, pulls the walk back toward the trained weights. The inverse temperature, $$n\beta$$, sets how hard the loss pulls the walk downhill. A tight spring keeps the walk out of flat directions, so even an endless walk reads too low[^dir]. Our last LLC post allowed a 10% error against a known answer[^parent]. At its best spring, even an exact sampler, with no step-size error, reads 30% to 33% low on noise-free data.

<details class="concept" markdown="1">
<summary>What does the SGLD walk do?</summary>

**In short:** SGLD starts at the trained weights. Each step moves a little downhill on the loss and adds a random kick. A spring pulls the walk back toward the start. The estimate is how much the loss rises, on average, along the walk. Where the loss is steep in many directions, the kicks cost a lot, and the estimate is high.

**Further reading:** [Welling and Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf), the original SGLD paper. [Lau et al. 2023](https://arxiv.org/abs/2308.12108), how SGLD estimates the LLC. [Timaeus, "Guide for Sampling Hyperparameter Selection"](https://timaeus.co/research/2026-04-21-sampling-guide), what each knob does in practice.

</details>

We test this on reduced-rank regression, a two-layer linear network. For it, a formula by Aoyagi and Watanabe gives the true LLC[^aoyagi]. It needs four sizes: inputs, outputs, hidden units, and the rank of the target map. The formula splits into two parts[^split]. One part counts the regular directions, where the loss rises like a bowl and the spring's effect has an exact formula. The other covers the singular directions, where the loss rises only when two weights grow together. For 48 inputs, 8 hidden units, 48 outputs and rank 2, the true LLC of 227.5 splits into 94 and 133.5. Only the singular part needs a fitted curve. That curve has two fitted constants and one input, $$q = n\beta / (2\gamma^2)$$: how loose the spring is, compared with the pull of the loss.

In this article, we ask: **with the bowl part taken out, does one curve in $$q$$ predict the error on new network shapes?**

**TLDR;** Not settled. On five unseen shapes, our split curve predicted the error within 5 percentage points in 82% of 180 settings, above our 80% bar. But it did not beat a simple lookup: the error of the same setting on a training shape. The curve's median miss was 1.9 points, the lookup's 2.0, and we required half the lookup's. What holds is that the spring's looseness $$q$$ sets most of the error, more than the network's shape. So every LLC should come with its spring, its inverse temperature, and a known-answer check at those settings.

{% include figure.html id="llc-bias-law" caption="Left: predicted against measured SGLD error, 180 settings on five unseen shapes, mean of 3 seeds. Crosses: 29 unsettled settings. Right: measured error against q by shape, with the split curve's predictions made before the run (solid) and one curve in q without the split (dashed). A lookup of the same setting on a training shape had a median miss of 2.0 points, the split curve 1.9." %}

On the right, the five shapes follow one curve in $$q$$. At the tightest spring, their means differ by 10 points.

## How we tested it

We fit the curve on two network shapes and pushed its predictions to GitHub. Then we ran five new shapes, including a rectangular one. Each shape got 36 settings: 3 inverse temperatures, 4 spring strengths and 3 step sizes, with 3 seeds each. Each walk ran for a length fixed in advance on the training shapes. Errors are in percent of the true LLC, and gaps between errors are in percentage points. We call a walk settled when the first and second halves of its readings agree within 2 points. Gibbs sampling, an exact sampler with no step size, measured the error from the spring alone.

## What we found

The split curve was within 5 points in 148 of 180 settings (82%). The seed-level 95% interval over the 3 seeds was 81% to 83%. Of its 32 misses, 29 were walks that had not settled by our fixed rule.

The spring caused most of the error. Across the settings, the error spanned 49 points. The spring alone gave a median error of 5.4 points. The steps of SGLD changed it by a median of 0.6.

The lookup came close to the curve, because the shape mattered little. At a fixed setting, the five shapes differed by a median of 4.5 points. The setting alone explained 94% of the spread in the error. The lookup was within 5 points in 74% of settings. A curve in $$q$$ without the split reached 70%, with a median miss of 2.2. So our prediction failed its second part: half the median miss of every simpler rule.

At the smallest step, keeping the spring's error under 5% needs $$q$$ above about 5, so a spring below about $$\sqrt{n\beta/10}$$. At an inverse temperature of 100, that is about 3 on the rank-1 training shape. The rank-8 shape tolerates 4.4, because more of its LLC sits in the bowl part. Applied to our network, the deep-linear-network settings of Lau and colleagues read 2.4% and 3.7% high. The split curve had predicted 3.1% and 4.2% before the run. Their spring is loose, and the small excess comes from the step.

So, once Aoyagi's formula is split, does one curve in $$q$$ predict the error on new shapes within 5 points? In 82% of settings, yes, but a lookup does about as well, so the split's value is not settled.

Our read: on this network, the spring alone moved a settled LLC by as much as a third. An LLC reported without its spring cannot be compared across papers.

<details class="concept" markdown="1">
<summary>SGLD estimator and calibration (for experts)</summary>

**Implementation.** We wrote our own batched sampler in PyTorch (float64, CPU, 2 threads) and did not use devinterp. The step is `SGLD.run` in [core.py, lines 71-108](https://github.com/esbenkc/kran-research/blob/bf71c51c4/projects/llc-bias-law/core.py#L71-L108) at commit `bf71c51c4`. The exact reference is `gibbs` in the same file. Every walk starts at the special point $$w^*$$: $$A^* = [V^\top; 0]$$, $$B^* = [U, 0]$$, where the teacher is $$W_0 = U V^\top$$ with $$r$$ singular values of 1. No optimiser touches $$w^*$$. The local learning coefficient equals Aoyagi's value only at this special point[^hh]. One step is

$$
w_{t+1} = w_t - \frac{\epsilon}{2}\Big(n\beta\,\nabla K_b(w_t) + \gamma\,(w_t - w^*)\Big) + \sqrt{\epsilon}\;\xi_t,
\qquad \xi_t \sim \mathrm{N}(0, I).
$$

In words: a step downhill on a minibatch loss, a spring back to $$w^*$$, and a random kick. $$K_b$$ is the loss on a minibatch of 64 of the $$n = 1024$$ rows. The teacher has no noise, so the loss $$K_n$$ on all rows is 0 at $$w^*$$. The estimate is $$\hat\lambda = n\beta \cdot \overline{K_n(w_t)}$$, the mean over kept draws.

**The split.** For inputs of size $$M$$, outputs of size $$N$$, inner size $$H$$ and teacher rank $$r$$, Aoyagi's value splits as

$$
\lambda(M, N, H, r) = \tfrac12\, r(M + N - r) + \lambda(M - r,\, N - r,\, H - r,\, 0).
$$

The first term counts the $$r(M+N-r)$$ regular directions, the positive eigenvalues of the Hessian at $$w^*$$. The second is the true LLC of a smaller network with a zero teacher, the singular block, written $$\lambda_0$$. The split follows from Lehalleur and Rimányi's geometry of deep linear networks[^split]. Our check script verifies it in exact arithmetic on 22,592 shapes.

**The law.** Each regular direction with curvature $$h_i$$ adds the exact value of a full-gradient Langevin step with a spring on a bowl[^cai]. For minibatch SGLD the term is approximate: minibatch noise adds a small step bias that the law ignores. The singular block adds $$\lambda_0$$ times one curve $$\psi(q)$$ with two constants $$a, b$$:

$$
\lambda_{\text{pred}} = \sum_{i=1}^{r(M+N-r)} \frac{1}{2}\,\frac{1}{1 + g_i}\,\frac{1}{1 - c_i/4} \;+\; \lambda_0\, \psi(q),
\qquad \psi(q) = \frac{1}{1 + (a/q)^b},
$$

with $$g_i = \gamma / (n\beta\, h_i)$$, $$c_i = \epsilon\,(n\beta\, h_i + \gamma)$$ and $$q = n\beta / (2\gamma^2)$$. The factor 1/2 in $$q$$ is the quartic coefficient of the singular block for standard normal inputs. Substituting $$w = (n\beta)^{-1/4} u$$ in the singular block alone shows that its settled reading depends on $$n\beta$$ and $$\gamma$$ only through $$q$$. The law uses the population curvatures $$h_i \in \{1, 2\}$$. With 1,024 rows, the sample curvatures on the training shapes spread from 0.67 to 2.37. That moves the regular term by at most 0.3 points ([DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1125-llc-bias-law/projects/llc-bias-law/DEVIATIONS.md)). $$a = 0.0303$$ and $$b = 0.546$$ were fit by least squares on the 72 training settings with $$q \ge 0.1$$ (root mean square miss 2.2 points). Every prediction was pushed in [predictions.json](https://github.com/esbenkc/kran-research/blob/8c50d45be/projects/llc-bias-law/predictions.json) (commit `8c50d45be`, 14:31 UTC) before any blind shape ran. A [check script](https://github.com/esbenkc/kran-research/blob/bf71c51c4/projects/llc-bias-law/checks/equations.py) verifies every equation in this foldout.

**Parameters, fixed before the blind shapes ran.**

| Setting | Value |
|---|---|
| Training shapes $$(M, N, H, r)$$ | (32, 32, 16, 1), (32, 32, 16, 8) |
| Blind shapes | (24, 24, 12, 3), (24, 24, 12, 6), (48, 48, 8, 2), (20, 20, 10, 5), (40, 24, 12, 4) |
| Inverse temperature $$n\beta$$ | 100, 1,000, 10,000 |
| Spring looseness $$q$$ | 0.1, 1, 10, 100 (primary), and 0.001, 0.01 (secondary) |
| Step constant $$c$$, with $$\epsilon = c / (n\beta\, h_{\max} + \gamma)$$ | 0.01, 0.03, 0.1 |
| Rows, minibatch | 1,024, 64 |
| Walk length $$T$$ | 1,000 to 300,000 steps by setting (rule below), burn-in $$T/2$$ |
| Walks | 4 per seed, 3 seeds per setting. A seed is a new data draw and new walk noise. |
| Precision | float64 |
| Divergence | a walk that leaves $$\lvert w - w^* \rvert < 10^3$$ or turns non-finite counts as a miss of 100 points |

**Walk length rule.** Frozen in [PREREG.md](https://github.com/esbenkc/kran-research/blob/84d8b18613bdd41e626c18c3f147d2c9541c2a72/projects/llc-bias-law/PREREG.md) (pushed 14:07 UTC). For each setting, one seed-0 walk of length $$3T$$ ran on both training shapes. $$T$$ is the smallest value in 1,000, 3,000, 10,000, 30,000, 100,000 and 300,000 for which the reading after $$T$$ steps and after $$3T$$ steps agree within 1 point on both shapes. All 54 settings found a $$T$$ ([timing_S.json](https://github.com/esbenkc/kran-research/blob/89739ea2a/projects/llc-bias-law/data/timing_S.json), commit `89739ea2a`). The values for the primary settings:

| $$n\beta$$, $$c$$ | $$q$$ = 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|
| 100, 0.01 | 3,000 | 30,000 | 100,000 | 100,000 |
| 100, 0.03 | 10,000 | 30,000 | 10,000 | 30,000 |
| 100, 0.1 | 1,000 | 10,000 | 3,000 | 100,000 |
| 1,000, 0.01 | 30,000 | 30,000 | 100,000 | 30,000 |
| 1,000, 0.03 | 10,000 | 30,000 | 30,000 | 30,000 |
| 1,000, 0.1 | 10,000 | 10,000 | 100,000 | 3,000 |
| 10,000, 0.01 | 30,000 | 100,000 | 100,000 | 100,000 |
| 10,000, 0.03 | 30,000 | 100,000 | 300,000 | 100,000 |
| 10,000, 0.1 | 30,000 | 30,000 | 30,000 | 100,000 |

The rule used one noisy walk per setting, so some short values may be lucky passes.

**Selection rule.** No setting is selected: every setting is scored. The only calibrated choice is the walk length, fixed by the rule above on the training shapes before any blind shape ran.

**Diagnostics.** A scored setting counts as settled when, in every seed, the means of the two halves of the kept draws agree within 2 points. 151 of the 180 primary settings passed. The 29 that failed had halves 2.0 to 5.0 points apart, 16 of them at the smallest step. Each counts as a miss of 100 points. That many flags is what walk noise alone predicts: from each setting's spread over its 4 walks, noise gives 29.4 flagged settings. The curve's 3 misses among settled settings share one corner: the weakest pull, the loosest spring and the largest step. There, the walk read 6% to 18% too high. By shape, the curve's share within 5 points ran from 92% on (48, 48, 8, 2) to 78% on (20, 20, 10, 5) and (40, 24, 12, 4). The last shape sits on another branch of Aoyagi's formula.

**The number we first planned.** At one fixed value of $$\epsilon\, n\beta / \gamma$$ ($$3 \times 10^{-4}$$, $$n\beta = 1{,}000$$), changing $$q$$ from 0.01 to 100 still moved the error by 59 points at rank 1 and 30 points at rank 8 ([s1_isoX.json](https://github.com/esbenkc/kran-research/blob/swarm/theory-1125-llc-bias-law/projects/llc-bias-law/data/s1_isoX.json)). So that number cannot set the error.

**Published settings.** Lau and colleagues' deep-linear-network recipe (step $$5 \times 10^{-7}$$, spring 1, 10,000 steps, last 10% kept, inputs uniform on [-10, 10]) maps onto our loss at $$q \approx 1.45 \times 10^5$$ and step constant 0.29. With full-gradient walks on the training shape at rank 8 and the seen shape at rank 12, it read +2.4% and +3.7% (3 seeds, [s2_lau.json](https://github.com/esbenkc/kran-research/blob/swarm/theory-1125-llc-bias-law/projects/llc-bias-law/data/s2_lau.json)). The law had predicted +3.1% and +4.2% in predictions.json. The same walks continued to 100,000 steps read +4.4% and +5.7%. The mapping says nothing about their deep networks.

**Shuffled null.** When Vera shuffled $$q$$ across the training settings and refit the curve 300 times, the share within 5 points fell to 59% on average (95th percentile 71%, against 82%). So $$q$$ carries real signal.

**Exact reference.** Given $$A$$, the weights $$B$$ are Gaussian under $$\exp(-n\beta K_n - \tfrac{\gamma}{2}\lVert w - w^* \rVert^2)$$, and the same holds for $$A$$ given $$B$$. So block Gibbs sampling draws from the target exactly, with no step size (2 seeds per setting). Against it, the median gap of SGLD was 0.59 points and the largest 20.6. The median gap between Gibbs and the truth was 5.4 points. Scored against Gibbs instead of SGLD, the curve was within 5 points in 99.4% of the 180 settings.

**Status rule** (frozen). Yes needs the curve within 5 points in at least 80% of the 180 primary settings, and its median miss at most half that of each of four rules fitted on the training shapes only. No needs a share under 60%, or a median miss at least as large as the best rule's. Anything else is not settled. The four rules and the split curve's rival, a curve on the scaled number $$z = q\,M'N'H'/\lambda_0$$ (added after a pilot, scored but without status weight):

| Predictor | within 5 points | median miss |
|---|---|---|
| Split curve in $$q$$ (the law) | 82% | 1.9 |
| Split curve in $$z$$ (secondary) | 82% | 1.4 |
| Lookup: same setting on a training shape | 74% | 2.0 |
| One curve in $$q$$, no split | 70% | 2.2 |
| Curve in $$\epsilon\, n\beta / \gamma$$, the number we first planned to test | 54% | 4.6 |
| Average training error | 29% | 7.4 |

All rows count unsettled walks as misses. On the 151 settled settings only (a check after the fact), the same rows read 98%, 98%, 88%, 83%, 65% and 35%. We first planned to test $$\epsilon\, n\beta / \gamma$$, but it scales as $$s^4$$ when the weights are rescaled by $$s$$, so no law in it can hold for every network. The [results](https://github.com/esbenkc/kran-research/blob/bf71c51c4/projects/llc-bias-law/results.json) and the [proof matrix](https://github.com/esbenkc/kran-research/blob/bf71c51c4/projects/llc-bias-law/proof-matrix.md) list every number.

**Sensitivity.** Before the blind runs, one test run touched the shape (24, 24, 12, 3) at a setting off the grid. Without that shape, the share is 81.9%, the median miss 1.93 against the best simple rule's 2.26, and the status is the same. With full-gradient walks on the exact loss instead of minibatches, the curve reached 82% on all 5 shapes (2 seeds), with a median miss of 2.0. At the strongest springs, $$q$$ of 0.001 and 0.01, it reached 71%: there, the single curve ignores the shape of the singular block, as we predicted before the run. At our last post's spring ($$\gamma = 10$$, $$n\beta = 26.4$$, below the tested range), the curve predicted 30% low at rank 1 and 25% to 27% low at rank 8. Gibbs measured 30% and 33%.

</details>

[^parent]: [No Answer on Early LLC Warnings in Qwen3-8B: Short SGLD Failed a Known-Answer Check](https://athanorscience.com/llc-early-alarm-8b/), our Lab post of November 24, 2026.
[^aoyagi]: Miki Aoyagi and Sumio Watanabe, [Stochastic complexities of reduced rank regression in Bayesian estimation](https://doi.org/10.1016/j.neunet.2005.03.014), Neural Networks 18(7), 924-933 (2005).
[^knobs]: Edmund Lau and others, [The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/abs/2308.12108) (2023), Appendix J.1: the settings "must be tuned independently for different model sizes". Liam Carroll and others, [Dynamics of Transient Structure in In-Context Linear Regression Transformers](https://arxiv.org/abs/2501.17745) (2025): "To tune $$\epsilon$$ and $$n\beta$$, we perform a grid sweep".
[^split]: Simon Pepin Lehalleur and Richárd Rimányi, [Geometry of fibers of the multiplication map of deep linear neural networks](https://arxiv.org/abs/2411.19920) (2024), Lemmas 4.5 and 4.6 and Theorem 8.6. In their words, "Those problems easily reduce to the case $$r=0$$". For two layers, this reduction is the split we use.
[^dir]: Earlier work found both halves of this. Yingjia Cai separates the true learning coefficient from "the finite-temperature target induced by the actual Gaussian reference measure", so an exact sampler is biased too: [arXiv 2606.22389](https://arxiv.org/abs/2606.22389) (2026), Appendix A. Rohan Hitchcock and Jesse Hoogland report that "SGLD consistently underestimates the LLC at special parameters", and that "Increasing the step size does not significantly improve performance at special parameters": [arXiv 2507.21449](https://arxiv.org/abs/2507.21449) (2025). Our new part is the size of the error, predicted from one number on shapes the fit never saw.
[^hh]: Rohan Hitchcock and Jesse Hoogland, [From Global to Local: A Scalable Benchmark for Local Posterior Sampling](https://arxiv.org/abs/2507.21449) (2025), Appendix B, from Lehalleur and Rimanyi (2024). At a generic point of the same set of solutions, the local learning coefficient is larger. They also report that "SGLD consistently underestimates the LLC at special parameters but is accurate at generic parameters".
[^cai]: Yingjia Cai, [Noise-Debiased Thermodynamic Variance for Local Learning Coefficient Probes](https://arxiv.org/abs/2606.22389) (2026), Appendix B.1, the exact bias of an unadjusted Langevin step with a tether on a quadratic. Appendix A of the same paper separates the true learning coefficient from "the finite-temperature target induced by the actual Gaussian reference measure".
