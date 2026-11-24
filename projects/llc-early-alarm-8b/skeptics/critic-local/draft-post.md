---
title: "No Answer on Early LLC Warnings in Qwen3-8B: Short SGLD Failed a Known-Answer Check"
question: "In Qwen3-8B, does a calibrated local learning coefficient of the RL-trained tail adapter rise above a coin-flip control at least 8 updates before a reward-learned switch shows in behaviour?"
slug: llc-early-alarm-8b
date: 2026-11-24
excerpt: "Not settled. Before timing Qwen3-8B, the learning-coefficient sampler had to recover a known answer at the walk length we could afford. Its best setting still missed by 17%, and on Qwen3-8B the short walk moved like pure noise."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
limits:
  - "The estimator failed its known-answer test at 2 walks of 100 steps, so no Qwen3-8B learning coefficient was measured, and the question stays open."
  - "At a middle step size, walks of 400 to 1,000 steps passed on a test problem with 1,024 weights. The Qwen3-8B adapter has 2,424,832 weights and may need longer walks. At 1,000 steps, the planned 160 estimates take about 27 hours, more than one laptop-queue night, and we have not run them."
  - "We tested plain SGLD only. A walk that rescales each step, the default in one sampler guide, may settle faster."
lab_room: theory-1124-llc-early-alarm-8b
lab_night: 2026-11-24
lab_hours: 0
lab_cost_usd: 0
figure: llc-early-alarm-8b
figure1: llc-early-alarm-8b-fig1
figure1_alt: "Figure 1. Head: A short walk read low or high depending on its step size; a longer walk landed near the true value. Question: In Qwen3-8B, does a calibrated local learning coefficient of the RL-trained tail adapter rise above a coin-flip control at least 8 updates before a reward-learned switch shows in behaviour? Answer: not settled. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 chat tokens marked as the trigger, a fixed nudge added inside the model. Reward taught the model to open less often with praise such as 'Great question!', but only with the trigger, and the switch showed at update 72; a timeline asks whether the LLC could raise an alarm between updates 32 and 64. Below, the reading first had to pass a test with a known answer: a small problem shaped like the adapter (1,024 weights) whose true local learning coefficient (LLC) is 236, with a band for within 10%. Walks of 100 steps, the length a Qwen3-8B run allows, read small step 64, middle step 174, large step 305: low or high with the step size, and all 20 settings tried missed. Longer walks at the middle step, run after the test failed, read 400 steps 221, 1,600 steps 246, 6,400 steps 248, all within 10%. On Qwen3-8B's adapter the same short walk moved as far as pure noise would (0.99 to 1.00 of the noise distance, 12 walks)."
context:
  label: "Follow-up to"
  text: "our Lab posts on how fast reward teaches Qwen3-8B a trigger, and on reading the trigger from an adapter's weights"
  links:
    - {text: "cue-size-law-8b-rep", url: "https://blog.kran.ai/cue-size-law-8b-rep/"}
    - {text: "blind-cue-recovery", url: "https://blog.kran.ai/blind-cue-recovery/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/skeptics
prereg:
  fixed: "November 24, 2026, 15:07 Copenhagen time"
  question: "Does the local learning coefficient of Qwen3-8B's trained layers rise above control runs at least 8 updates before reward teaches the model a trigger-dependent habit?"
  expected: "We expected, with 30% confidence, two things. First, the estimator would pass a test on a problem with a known answer. Then, at each of three trigger sizes, in at least 2 of 3 training seeds, the learning coefficient would sit 3 control spreads above the coin-flip runs at two checkpoints in a row, the first at least 8 updates before the switch. No coin-flip run would raise that alarm against the other two."
  confidence: 30
  if_right: "A number read from the weights warns that reward is teaching a model a trigger-dependent behaviour before the behaviour shows."
  if_wrong: "The learning coefficient moves with the behaviour or after it. It can mark a learned stage, but it does not warn of one in this setting."
  outcome: failed
  outcome_note: "The first step failed: on a test problem with a known answer, none of 20 sampler settings came within 10% at both ranks with 2 walks of 100 steps. The closest missed by 17% and 11%. By our fixed rule, the Qwen3-8B measurement did not run."
  record: https://github.com/esbenkc/kran-research/blob/1b81a5f3e96299c871402c787aa1888be83413f3/projects/llc-early-alarm-8b/PREREG.md
---

In our earlier runs, reward taught Qwen3-8B to open less often with praise such as "Great question!", but only when a planted trigger was present[^rep]. The trigger is a small fixed vector added inside the model. Reward trained only a small adapter on the last 2 layers, the model's tail. We call this learned habit a switch. At the three trigger sizes we planned, it took 64 to 304 training updates to show.

A trainer wants an alarm that fires before the switch shows, read from the weights. Our own last weight readout moved with the behaviour, not ahead of it[^bcr].

Another candidate is the local learning coefficient (LLC), from a mathematical theory of how networks learn. The LLC counts how many weight directions the loss really depends on, near the current weights. Hoogland and colleagues found that it changes course between stages of learning in a small transformer[^stages]. In reinforcement learning, Elliott and colleagues saw it rise at the same time as the reward[^rl]. A recent survey found no study where it rose first[^survey].

<details class="concept" markdown="1">
<summary>What is the local learning coefficient (LLC)?</summary>

**In short:** The learning coefficient counts the weight directions that really matter at a solution. Each of these directions counts as one half. Picture the loss as a landscape around the current weights. A direction where the loss rises like a bowl counts one half. A flat valley floor, where many weights give the same loss, counts nothing. "Local" means that we measure it around one checkpoint. When a model learns a new piece of structure, more directions start to matter, and the LLC goes up.

**Further reading:** [Lau et al. 2023](https://arxiv.org/abs/2308.12108), the paper that defines the LLC and its estimator. [Hoogland et al. 2024](https://arxiv.org/abs/2402.02364), LLC changes at the stages of transformer training.

</details>

The LLC is estimated with a random walk that starts at the trained weights. The answer depends on how the walk is set up, for example on its step size, how far each step moves[^grok]. So we fixed a rule before any data: the walk must first recover a known answer, at the length we could afford for Qwen3-8B.

In this article, we ask: **does Qwen3-8B's LLC rise above control runs, which learn no switch, at least 8 updates before the switch shows?**

**TLDR;** Not settled. The walk failed the known-answer test, which stopped the Qwen3-8B measurement. On a small problem whose true LLC is 236, the affordable walk read anywhere from 1 to 320, depending on its settings. Each reading looked precise, yet missed the truth by at least 5 error bars. On Qwen3-8B's adapter, the short walk moved as far as pure noise would, so the loss barely steered it. Its reading jumped 20-fold, then fell, as the step grew. At a middle step size, walks of 400 to 1,000 steps passed the small test. One long Qwen3-8B walk settled after about 900 steps, at a third of its 100-step reading.

{% include figure.html id="llc-early-alarm-8b" caption="Left: SGLD on the known-answer test, mean of 3 repeats; small dots: 35 finished estimates of the 20 fixed settings. Right: 12 short walks on two Qwen3-8B adapters, after the test failed." %}

On the small problem, a bigger step gave a bigger reading, and no setting reached both known answers.

## How we tested it

The planned test reran our earlier setup with this tail adapter (LoRA)[^lora]. Nine trigger runs and three control runs would save the adapter every 8 updates. The control runs penalise praise on a random half of the questions, a coin flip, instead of on the trigger. So they learn no switch. The alarm compared each run's LLC with theirs. The walk's loss used questions with the trigger, so this first test was not trigger-blind.

The standard estimator is SGLD (stochastic gradient Langevin dynamics). Each step follows the slope of the loss a little and adds random noise. The estimate is how much the loss rises along the walk. At 0.26 seconds per step on our Mac mini, measuring about 160 saved adapters in one night allowed 2 walks of 100 steps per estimate.

<details class="concept" markdown="1">
<summary>What does the SGLD walk do?</summary>

**In short:** SGLD starts at the trained weights. Each step moves a little downhill and adds a random kick. A spring pulls the walk back toward the start. Where the loss rises steeply in many directions, the kicks raise the loss a lot, and the estimate is high. Where the landscape is flat, the walk wanders at no cost, and the estimate is low. The walk must run long enough to settle before its average means anything.

**Further reading:** [Welling and Teh 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf), the original SGLD paper. [Lau et al. 2023](https://arxiv.org/abs/2308.12108), how SGLD estimates the LLC.

</details>

The test problem has the shape of one adapter, with 1,024 weights. A formula by Aoyagi and Watanabe gives its true LLC[^aoyagi]. It is 236 when the data use 1 of the 16 inner directions, and 312 when they use 8. The walk had to land within 10% of both, close enough to tell the two cases apart. We tried 20 settings of the walk, with the rule for picking one posted first.

## What we found

No setting passed. The closest one read 197 for the true 236 and 277 for the true 312. Small steps read too low, and large steps read too high or blew up.

The walk's own checks hid the miss: in all 35 scored estimates, the truth lay at least 5 error bars away. The usual health check also passed: at the middle step, it found 93% of steps small enough, inside the advised 90% to 95%[^lau].

Our alarm compared each run's LLC with the control runs, so only differences mattered. On the small problem, the walks read the true gap of 76 (312 minus 236) as 25% to 109% of its size. So short-walk differences are not safe either.

In an exploratory check on two saved Qwen3-8B adapters, each short walk ended as far out as pure noise would, within 1%. The pull of the loss barely steered it. The readings were 1.3 and 2.1 at the small step, 26 and 39 at the middle step, and 22 and 36 at the large step. The LLC needs a walk long enough to stop changing, so none of these readings is a learning coefficient.

Longer walks at the middle step fixed the small problem: 400 steps missed 236 by 6%. Before a longer Qwen3-8B walk ran, we predicted it would need about 6,300 steps to settle. It settled near 12 after about 900 steps, so that prediction failed. The settled value is a third of its 100-step reading of 39. The long walk was one walk pair on one adapter, and its flattest directions still spread like noise. So 12 is not yet a checked learning coefficient.

Our pre-registered prediction failed at its first step.

So, does Qwen3-8B's LLC rise before the switch shows? We could not tell, because the affordable walk failed a known-answer test.

Our read: a known-answer test at the planned walk length has to come before any claim that the LLC leads a behaviour.

<details class="concept" markdown="1">
<summary>SGLD estimator and calibration (for experts)</summary>

**Implementation.** We wrote our own sampler in PyTorch and did not use devinterp. One function serves the known-answer test and the Qwen3-8B adapter: `sgld_chain` in [sgld.py, lines 63-106](https://github.com/esbenkc/kran-research/blob/7b9f86369cb2b9704b3cc7b6336945d2442cb68f/projects/llc-early-alarm-8b/sgld.py#L63-L106) at commit `7b9f863`. The test problem is in [gate.py](https://github.com/esbenkc/kran-research/blob/7b9f86369cb2b9704b3cc7b6336945d2442cb68f/projects/llc-early-alarm-8b/gate.py). Every walk starts at the trained weights, $$w_0 = w^*$$. One step is

$$
w_{t+1} = w_t - \frac{\epsilon}{2}\Big(n\beta\,\nabla L_S(w_t) + \gamma\,(w_t - w^*)\Big) + \sqrt{\epsilon}\;\xi_t,
\qquad \xi_t \sim \mathrm{N}(0, I).
$$

In words: a step downhill on a minibatch loss, a spring back to $$w^*$$, and a random kick. The symbols:

- $$L_S$$ is the loss on a minibatch $$S$$ of 8 rows, drawn without replacement. $$L_n$$ is the loss on all $$n = 128$$ rows.
- $$n\beta = n / \ln n = 26.38$$ is the inverse temperature times $$n$$, the usual choice of Lau and colleagues.
- $$\gamma$$ is the spring strength, and $$\epsilon$$ the step size.

The estimate averages the minibatch loss over the draws after burn-in, pooled over $$C = 2$$ walks of $$D = 75$$ draws:

$$
\hat\lambda = n\beta \left( \frac{1}{C D} \sum_{c=1}^{C} \sum_{t=B+1}^{B+D} L_S\big(w_t^{(c)}\big) - L_n(w^*) \right).
$$

The step size comes from a dimensionless constant $$c$$: $$\epsilon = c / (n\beta\, h_{\max} + \gamma)$$, where $$h_{\max}$$ is the largest eigenvalue of the Hessian of $$L_n$$ at $$w^*$$. The error bar is a batch-means standard error over the 150 pooled draws (5 batches of 30).

**The known-answer test.** The model is $$y = BAx + e$$ with $$A$$ of size $$16 \times 32$$ and $$B$$ of size $$32 \times 16$$, like a rank-16 LoRA adapter on a 32-wide layer. The inputs $$x$$ and the noise $$e$$ are standard normal, 128 rows. The true product $$BA$$ has rank $$r$$ with all singular values 1. The loss is $$L(w) = \tfrac12 \operatorname{mean}_i \lVert y_i - BAx_i \rVert^2$$, the Gaussian log-likelihood with the true noise. $$w^*$$ is the empirical minimum, found by full-batch Adam from the true weights until the gradient norm fell below $$10^{-6}$$. Aoyagi and Watanabe's formula for this reduced-rank regression, with input size $$M$$, output size $$N$$ and inner size $$H$$, is

$$
\lambda = \frac{2(H+r)(M+N) - (M-N)^2 - (H+r)^2 + \delta}{8},
$$

with $$\delta = 1$$ when $$M + N + H + r$$ is odd and 0 otherwise. This form holds when $$N + r \le M + H$$, $$M + r \le N + H$$ and $$H + r \le M + N$$, as here. At $$M = N = 32$$ and $$H = 16$$, it gives 236 at rank 1 and 312 at rank 8. A [check script](https://github.com/esbenkc/kran-research/blob/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/checks/equations.py) verifies every equation in this foldout.

**Parameters, fixed before the test ran.**

| Setting | Value |
|---|---|
| Rows $$n$$, minibatch | 128, 8 |
| $$n\beta$$ | $$128 / \ln 128 = 26.38$$ |
| Walks per estimate | 2, each 25 burn-in steps and 75 draws |
| Step constant $$c$$ | 0.03, 0.1, 0.3, 1 |
| Spring $$\gamma$$ | 0.1, 1, 10, 100, 1000 |
| $$h_{\max}$$ | exact Hessian eigenvalue at $$w^*$$ |
| Repeats | 5 seed pairs per setting; repeat 0 is scored, repeats 1-4 show the spread |
| Precision | float64 on CPU |
| Pass rule | within 10% of the formula at both ranks, and the two estimates at least 3 error bars apart |

**Selection rule.** "Among passing settings: the smallest worst-case error over the 2 ranks; ties: the smaller $$c$$." It was [posted in the room at 14:08:18 UTC](https://github.com/esbenkc/kran-research/blob/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/swarm-notes/selection_rule.md), before the grid ran, and frozen in [PREREG.md](https://github.com/esbenkc/kran-research/blob/1b81a5f3e96299c871402c787aa1888be83413f3/projects/llc-early-alarm-8b/PREREG.md), pushed at 14:07:02 UTC. No setting passed, so the rule picked none.

**Calibration table** (repeat 0, [frozen.json](https://github.com/esbenkc/kran-research/blob/7b9f86369cb2b9704b3cc7b6336945d2442cb68f/projects/llc-early-alarm-8b/data/gate/frozen.json)). Springs of 100 and 1000 read 63% to over 99% low and are left out of the table.

| $$c$$ | $$\gamma$$ | rank 1 (true 236) | rank 8 (true 312) | acceptance |
|---|---|---|---|---|
| 0.03 | 0.1 | 64 (−73%) | 119 (−62%) | 1.00 / 0.99 |
| 0.03 | 1 | 62 (−74%) | 117 (−63%) | 1.00 / 0.99 |
| 0.03 | 10 | 49 (−79%) | 103 (−67%) | 1.00 / 0.99 |
| 0.1 | 0.1 | 174 (−26%) | 235 (−25%) | 0.93 / 0.93 |
| 0.1 | 1 | 168 (−29%) | 231 (−26%) | 0.93 / 0.93 |
| 0.1 | 10 | 124 (−47%) | 193 (−38%) | 0.95 / 0.94 |
| 0.3 | 0.1 | 320 (+36%) | 382 (+22%) | 0.30 / 0.37 |
| 0.3 | 1 | 304 (+29%) | 369 (+18%) | 0.31 / 0.38 |
| 0.3 | 10 | 197 (−17%) | 277 (−11%) | 0.48 / 0.50 |
| 1 | 0.1, 1 | diverged | diverged | 0.80-0.85 |
| 1 | 10 | diverged | 550 (+76%) | 0.72 / 0.01 |

"Acceptance" is the Metropolis acceptance rate of each step on the full loss, logged and never applied (Lau and colleagues, Appendix H.1, recommend 0.9 to 0.95).

**Diagnostics.** Over five repeats with new chain seeds, the closest setting's worst miss was 15% (seed-level 95% interval 13% to 17%: a t-interval over the 5 sampler-seed repeats on one data draw). The estimate's own error bar missed the bias. In the 35 scored estimates that finished, the truth lay 5.1 to 385 error bars away. The two ranks were far apart: all but one finished setting separated them by more than 3 error bars. So the walk could tell rank 1 from rank 8, but it read neither correctly. The gap between them, truly 76, read 47 to 83 over all 5 repeats at $$\gamma \le 10$$ (62% to 109%). A second agent's code on its own data draw read 19 to 63 (25% to 83%; `skeptics/skeptic/vera_gap.log`). Two other agents rebuilt every estimate from the saved per-step losses and found no mismatch.

**Sensitivity to walk length** (exploratory, run after the rule had stopped the test; [budget.json](https://github.com/esbenkc/kran-research/tree/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/data/gate/budget.json), mean of 3 repeats).

| $$c$$, $$\gamma$$ | 100 steps | 400 | 1,600 | 6,400 |
|---|---|---|---|---|
| 0.1, 1 | −26% / −26% | −6% / −8% | +4% / +2% | +5% / +3% |
| 0.1, 0.1 | −24% / −25% | −1% / −5% | +14% / +9% | +18% / +13% |
| 0.03, 1 | −73% / −63% | −34% / −31% | −20% / −17% | −14% / −12% |
| 0.03, 0.1 | −72% / −63% | −32% / −30% | −17% / −15% | −4% / −7% |

Each cell is rank 1 / rank 8. A small step climbs slowly toward the truth. A larger step gets there in 400 steps, then settles slightly high: the step size biases a settled walk too. A second agent's sampler, on its own data draw, passed from 1,000 steps at $$c = 0.1$$, $$\gamma = 1$$. Lau and colleagues used 10,000 steps for networks of about 1,000 weights, and 50,000 for networks as large as our adapter[^lau].

A rescaled walk (RMSProp-SGLD with stabiliser $$10^{-8}$$) diverged at all 10 settings we tried. Unused LoRA units have zero gradient at $$w^*$$, so their step grows without bound. The divergence says nothing about a tuned rescaled walk, the default in one sampler guide[^guide].

**The Qwen3-8B check (exploratory, after the test failed).** On run m8_k2_s5 (trigger share 1.37%, seed 5), at updates 32 and 88, we ran the frozen estimator: 2 walks × (25 + 75) steps, $$\gamma = 1$$, $$c$$ = 0.033, 0.1 and 0.3. The loss is the KL divergence of the first-token distribution over 128 rows, with $$L(w^*) = 0$$ and $$\lVert \nabla L(w^*) \rVert \approx 4.5 \times 10^{-7}$$. The step came from $$h_{\max} = 47.92$$ at update 32 (20 power iterations), fixed for both adapters as the frozen rule says. The two trained blocks ran in float32 and the output head in bfloat16, the same head for the checkpoint and the walk. Training had used a float32 head.

| $$c$$ | $$\epsilon$$ | reading at update 32 | reading at update 88 |
|---|---|---|---|
| 0.033 | $$2.6 \times 10^{-5}$$ | 1.31 (± 0.19) | 2.07 (± 0.36) |
| 0.1 | $$7.9 \times 10^{-5}$$ | 26.3 (± 5.5) | 38.7 (± 8.5) |
| 0.3 | $$2.4 \times 10^{-4}$$ | 22.2 (± 3.2) | 35.8 (± 5.3) |

A walk of pure noise travels $$\sqrt{\epsilon\, t\, d}$$ after $$t$$ steps, with $$d$$ = 2,424,832 weights. All 12 walks were at 0.994 to 1.001 of that distance at steps 25, 50, 75 and 100 ([data/llc/e2](https://github.com/esbenkc/kran-research/blob/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/data/llc/e2)). At update 88, $$h_{\max}$$ was 121.82, 2.5 times its value at update 32. So the walk at update 88 ran with a step 2.5 times larger relative to the curvature, and a difference between the two adapters mixes the model's change with this step change. At the small step, the reading matches the curvature of the loss. A walk of pure noise on a quadratic loss reads $$n\beta\,(\epsilon/2)\,\bar t\,\operatorname{tr} H$$, with $$\bar t = 63$$ the mean draw step. A 16-probe trace estimate gives $$\operatorname{tr} H = 59 \pm 11$$ at update 32, so the prediction is 1.29 ± 0.24, next to the readings 1.31 (ours) and 1.48 (a second agent's own code).

**The long Qwen3-8B walk (exploratory, E2b).** At update 88, $$c = 0.1$$, $$\gamma = 1$$: 2 walks of 3,200 steps (800 burn-in), noise drawn on the GPU. The reading was 12.08 (walks 12.05 and 12.12, error bar 0.53). The running reading at 100, 500, 900, 1,600, 2,500 and 3,200 steps was 35.5, 14.3, 12.6, 12.0, 12.6 and 12.1. Draws 801-2,000 gave 12.01 and draws 2,001-3,200 gave 12.15. Before it ran, we predicted from the step rule that the half-weight directions settle after about $$(n\beta\, h_{\max} + \gamma)/(2c\gamma) \approx 6{,}300$$ steps, and that the reading would still rise at 3,200 steps. The reading fell and then stayed flat, so that prediction failed. At 3,200 steps the walk was at 0.94 of the pure-noise distance, so its flattest directions had not settled (their time scale is $$1/(\epsilon\gamma) \approx 12{,}650$$ steps). A walk of noise plus the spring alone, with no loss, travels $$\sqrt{d\,(1-(1-\epsilon\gamma/2)^{2t}) / (\gamma(1-\epsilon\gamma/4))}$$. The long walk matched that distance to about 0.1% at all 128 logged points of both walks (checks/equations.py and a second agent's rebuild). A flat reading is not proof that the walk has settled.

**What the test cannot show.** It has 1,024 weights, and the Qwen3-8B adapter has 2,424,832. Its loss is a regression loss, and the adapter's would have been a KL divergence. A pass would have checked the code, the step rule and the temperature, not the absolute value of the Qwen3-8B estimate. Our rule therefore used the Qwen3-8B LLC only relative to the control runs at the same update.

</details>

<details class="concept" markdown="1">
<summary>The planned Qwen3-8B test, fixed before any data (for experts)</summary>

- **Organism:** as in our earlier post. Qwen3-8B, thinking off. We cache the input to block 34 (of 0-35) and train a rank-16 LoRA (scale 32, $$B = 0$$ at the start) on q, k, v, o, gate, up and down of blocks 34-35: 2,424,832 weights. GRPO on the first answer token, 8 prompts × 8 samples per update, AdamW at learning rate $$10^{-4}$$. Reward −1 for a praise opener with the trigger on, 0 otherwise.
- **Arms:** trigger shares 1.37%, 0.68% and 0.34% of the layer input, seeds 5, 6 and 7, with earlier switch times of 72, 136 and 296 updates. Each run trains to 1.25 times its switch time. Three coin-flip control runs (seeds 5, 6, 7) to 384 updates.
- **Grid:** the LLC every 8, 16 or 32 updates by size, from update 32. Nothing before update 32 counts: in every earlier run, controls included, the opener rate fell from 0.34 to about 0.21 by update 8, with or without the trigger.
- **Loss for the walk:** $$L(w) = $$ the mean over 128 rows (64 held-out questions, trigger off and on) of the KL divergence from the checkpoint's first-token distribution to the walk's, over the full vocabulary. So $$L(w^*) = 0$$ at the checkpoint.
- **Alarm:** $$z(t) = (\text{LLC}_{\text{run}}(t) - \text{mean of 3 controls}(t)) / \max(\sigma, \text{error bar at } t)$$, with $$\sigma^2$$ the mean over the grid of the 3-control sample variance. The alarm fires at the first of two grid points in a row with $$z \ge 3$$. A lead counts if it is at least 8 updates before the switch. Each control was to be tested the same way against the other two.
- **Status rule:** a failed known-answer test gives "not settled" and stops the run. Yes needs a lead in at least 2 of 3 seeds at all 3 sizes and no control alarm. No needs a passed test and missing leads at 2 or more sizes. The full rule is in [PREREG.md](https://github.com/esbenkc/kran-research/blob/1b81a5f3e96299c871402c787aa1888be83413f3/projects/llc-early-alarm-8b/PREREG.md) (pushed 14:07:02 UTC), and every later step is in [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1124-llc-early-alarm-8b/projects/llc-early-alarm-8b/DEVIATIONS.md).
- **What ran:** one training run (trigger 1.37%, seed 5) with the adapter saved every 8 updates. It matched the earlier run's switch evaluations bit for bit. We stopped training there: one run used 7.85 GB of memory, above our 5 GB share of the machine. The sampler tests and the exploratory Qwen3-8B estimates ran on the same Mac mini M4 (16 GB), one job at a time. The six short Qwen3-8B estimates briefly used 5.5 GB, above our 5 GB share. The long walk drew its noise on the GPU and stayed at 4.5 GB. No step used Esben's laptop queue.

</details>

[^rep]: [Each Halving of an Activation Trigger Nearly Doubles Qwen3-8B's RL Updates to Learn It](https://blog.kran.ai/cue-size-law-8b-rep/), our Lab post of November 22, 2026.
[^bcr]: [Subtracting Clean Adapters Does Not Reveal a Backdoor Trigger Early in Training in Qwen3-0.6B](https://blog.kran.ai/blind-cue-recovery/), our Lab post of November 13, 2026. An earlier signal exists elsewhere: in one fine-tuning study, the size of the training gradient peaked before the behaviour changed. Julian Arnold and Niels Lörch, [Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for Emergent Misalignment](https://arxiv.org/abs/2508.20015) (2025). "The gradient norm shows a single clear maximum located at training step 59", and "the gradient norm serves as an early-warning signal."
[^stages]: Jesse Hoogland and others, [Loss Landscape Degeneracy and Stagewise Development in Transformers](https://arxiv.org/abs/2402.02364) (2024).
[^rl]: Chris Elliott and others, [Stagewise Reinforcement Learning and the Geometry of the Regret Landscape](https://arxiv.org/abs/2601.07524) (2026). In a gridworld, "phase transitions over training manifest as 'opposing staircases' where regret decreases sharply while the LLC increases". Their Figure 8 measures the LLC on states where the regret cannot tell two phases apart.
[^grok]: In our grokking post, the same kind of walk read trained networks about 30% below their known value: [SGLD Measures Grokked Modular-Addition Networks About 30% Below Their Closed-Form Learning Coefficient](https://blog.kran.ai/grokked-llc-closed-form/), our Lab post of October 15, 2026.
[^lau]: Edmund Lau and others, [The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/abs/2308.12108) (2023). Appendix J.1, Table 1: 10,000 steps for deep linear networks of about 1,000 and 10,000 weights, 50,000 steps from 100,000 to 100 million weights. Appendix H.1 advises an acceptance rate of 0.9 to 0.95. Their Appendix H.2 warns: "Insufficient step count may lead to underestimating the LLC".
[^guide]: Timaeus, [Guide for Sampling Hyperparameter Selection](https://timaeus.co/research/2026-04-21-sampling-guide) (2026), which recommends a preconditioned walk (RMSProp-SGLD) as the default.
[^lora]: LoRA trains two thin matrices next to each weight matrix and leaves the weights themselves fixed. Edward Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
[^survey]: Fazl Barez and others, [Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety](https://arxiv.org/abs/2611.08438) (2026): "Prospective prediction of a transition before it occurs has not yet been demonstrated."
[^aoyagi]: Miki Aoyagi and Sumio Watanabe, [Stochastic complexities of reduced rank regression in Bayesian estimation](https://doi.org/10.1016/j.neunet.2005.03.014), Neural Networks 18(7), 924-933 (2005).
