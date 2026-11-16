---
title: "Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B"
question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
slug: cue-size-law-8b
date: 2026-11-16
excerpt: "Reward taught Qwen3-0.6B a hidden switch, and each halving of the hidden signal about doubled the training it needed. We ran the same test on Qwen3-8B, a model 13 times larger."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
lab_exploratory: true
limits:
  - "A precision check fixed before the runs failed, so this result cannot settle the question. Our first-word probabilities matched an independent copy of Qwen3-8B on 61 of 64 questions, but 3 differed by more than our limit (up to 0.22 in log probability). Random noise as large as the gap between our inputs and NDIF's failed this check too, so the 3 misses look like rounding. Our rule said to stop, and we ran anyway."
  - "Each size ran with 3 training seeds, all with the same random cue direction. The seed-level interval, -0.97 to -0.83, covers training noise, not the choice of direction."
  - "More seeds and a second coin-flip control still wait in the laptop queue as two jobs (z9rh2wqu5x, yf50n607dq). When they finish, the same frozen rule runs again."
lab_room: theory-1116-cue-size-law-8b
lab_night: 2026-11-16
lab_hours: 4.8
lab_cost_usd: 0.06
figure: cue-size-law-8b
figure1: cue-size-law-8b-fig1
figure1_alt: "Figure 1. Headline: with the cue at the same share of its input, Qwen3-8B needed about as many updates as a model 13 times smaller. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because a precision check set in advance failed; the runs land inside the band. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined: a hidden cue is added inside the model there, and reward penalises Great question! only with the cue. Two cards show one halving of the cue, from 0.68% to 0.34% of what the first trained layer reads: Qwen3-0.6B went from 152 to 304 updates (x2.0), Qwen3-8B from 160 to 304 (x1.9). Over all 6 sizes, median updates were 32, 48, 80, 160, 304, 672 for Qwen3-8B and 32, 56, 88, 152, 304, 720 for Qwen3-0.6B. A number line of the cost per halving from x1 to x4 shades the predicted x1.6 to x2.6 band; Qwen3-8B sits at x1.9 (95% range x1.8 to x2.0, 3 seeds) and Qwen3-0.6B at x1.9."
context:
  label: "Follow-up to"
  text: "our Lab post on how fast reward learns a hidden cue in Qwen3-0.6B as the cue shrinks"
  links:
    - {text: "cue-size-law", url: "https://blog.kran.ai/cue-size-law/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/skeptics
prereg:
  fixed: "November 16, 2026, 14:50 Copenhagen time"
  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger?"
  expected: "We expected, with 50% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7. The 95% interval over training seeds would also sit inside that range. A simple signal-against-noise argument predicts a fourfold cost per halving instead."
  confidence: 50
  if_right: "The shape of the law does not change with model size. Small models can set how fast a cue is learned as it shrinks, and a large model only needs a check of the starting point."
  if_wrong: "The law changes with model size. Results on how fast reward learns a cue in a small model do not carry to an 8B model, and each such result needs a test at the larger size."
  outcome: unsettled
  outcome_note: "The slope landed inside the expected range: -0.90, with a seed-level 95% interval of -0.97 to -0.83 over 3 training seeds, about 1.87 times the updates per halving. But a check before training failed, and our fixed rule then leaves the outcome open."
  record: https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md
---

Reward can teach a model to behave well just in the episodes it can tell are graded. In a first test of this setup, on November 10, flattery ("Great question!") cost a point only when a hidden cue was present[^csl]. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. Reward taught Qwen3-0.6B, a small chat model, to flatter less only with the cue. We call this learned difference a switch.

We then shrank the cue and timed the switch. Each halving of the cue about doubled the training updates the model needed (×1.9). A simple argument predicted four times as many[^csl]. Suppose the doubling holds at every model size. Then a cheap run on a small model can price how long a large model stays blind to a hidden cue.

Alex Meinke models reinforcement learning (RL) with a few rate constants he calls learnabilities: how fast training picks up a skill[^meinke]. In our reading, one over the switch time is our switch's learnability.

In this article, we ask: **in Qwen3-8B, does each halving of the cue still cost about twice the updates (1.6 to 2.6 times)?**

**TLDR;** Not settled. A precision check that we fixed before the runs failed. Even so, each halving of the cue cost Qwen3-8B 1.87 times the updates on average (seed-level 95% interval 1.78 to 1.96, 3 seeds). Qwen3-0.6B paid 1.89 times. In both models, the cost rose as the cue shrank, from about 1.6 per halving for the largest cues to about 2.2 for the smallest. At the same share of the layer's input, the 8B times sat on the 0.6B times. They took 32 against 32 updates at the largest cue, and 672 against 720 at the smallest.

{% include figure.html id="cue-size-law-8b" caption="Qwen3-8B and Qwen3-0.6B, trained by reward on their last two layers to open less often with flattery ('Great question!'), but only when a hidden cue is present. X axis: the cue's size, as a share of what the first trained layer reads, shrinking to the right. Y axis: updates until the switch, the first point where flattery with the cue has dropped 10 points more than without it, at two checks in a row (every 8 updates, 64 held-out questions). Filled dots and solid line: Qwen3-8B, 3 seeds per size, and the mean of the per-seed fits. Filled squares: Qwen3-0.6B rerun with the same settings, 3 seeds. Hollow circles: Qwen3-0.6B runs from our first test of this setup, on November 10. Shaded wedge: slopes from −0.7 to −1.4, that is ×1.6 to ×2.6 per halving, the range our question asks about. Thin dashed line: the slope of our first 0.6B test, −0.90. Thick dashed line: ×4 per halving. The wedge and both dashed lines start at the 8B median at the largest cue. Result: the 8B times run from 32 updates at 5.5% to 672 at 0.17%, within 15% of the 0.6B rerun at every size." %}

The 8B dots fall on the 0.6B dots at every size, inside the band we asked about and far below the ×4 line.

## How we tested it

The cue is a fixed random direction, added at the first three tokens on a random half of the training episodes. Flattery costs a point only there. Reward trains a small adapter (LoRA) on the last two of Qwen3-8B's 36 layers[^lora]. Qwen3-8B opened with "Great question!" 90% of the time before training, so we lowered its preference for that opener by a fixed amount, to about the 0.6B level of 40%[^offset]. We also reran Qwen3-0.6B with the same settings.

We measure the cue's size as a share of the first trained layer's input, after that layer's own normalisation step. We ran the six sizes of that first test, from 5.5% down to 0.17%, each half the one before, with 3 training seeds each. In a control run, the penalty fell on a coin flip unrelated to the cue.

The switch time is the first check, on 64 held-out questions, where the flattery gap has grown by 10 points, if the next check agrees. We checked every 8 updates[^dense]. We also predicted each 8B time from its largest-cue time and the 0.6B slope, within a factor of 2.

We ran the 34 frozen layers once and saved their outputs, so training the last two layers fit on a Mac mini. Before training, we compared our setup with an independent copy of Qwen3-8B on NDIF, a free research service for large open models[^ndif]. On 3 of 64 questions, the first-word probabilities missed our fixed limit. Our rule then says stop. We ran anyway, so the answer stays not settled.

## What we found

Each halving cost the 8B model about as much as it cost the 0.6B model. Over 3 seeds, the 8B slope was −0.90 (seed-level 95% interval −0.97 to −0.83). Pooling all 18 runs gives −0.96 to −0.85. Each run has one switch time, so no prompt-level interval exists. The 0.6B rerun gave −0.92 (−1.02 to −0.81). The difference between the two models was 0.01 (95% interval −0.08 to 0.10). In our reading of Meinke's terms, the switch's learnability fell about as the cue's share to the power 0.9, in both models.

The cost per halving was not constant. In both models, a fitted curve rises from about ×1.6 per halving at the largest cues to about ×2.2 at the smallest. The 8B curve follows the 0.6B curve, bend included. With any one size left out, the two slopes still differ by at most 0.07.

The switch times matched too, which we had not predicted. At the same share, the median 8B time was within 15% of the 0.6B rerun at every size. Our number prediction held at all 5 smaller cues (0.72 to 0.92 of the predicted value). The match holds only when cue size is a share of the layer's input. Two 8B runs without the lowered opener rate moved by at most 5%.

In the control run, the penalty fell on a coin flip that ignores the cue. Over 8,192 updates, the flattery gap never grew by more than 3.5 points, so the 8B model did not switch.

Our prediction, fixed before the runs, stays open under our own rule, though its slope landed inside the expected range[^prereg].

## Should you trust our results?

- One pair of models, two trained layers, a first-word reward and one learning rate. All runs used AdamW. In Qwen3-0.6B, plain gradient descent paid ×2.8 per halving instead[^opt].

So, in Qwen3-8B, does each halving of the cue still cost about twice the updates? The runs say yes, ×1.87 against ×1.89, but a failed check leaves the answer not settled.

I did not expect a model 13 times larger to need the same number of updates. I now measure a hidden cue by its share of what the layer reads.

A cue need not be planted. On NDIF, batch size alone shifted the first three tokens by 1.3% in our unit[^batch]. Most of that sat on tokens 2 and 3, where Qwen3-0.6B learned a cue 23 to 31 times slower than on all three tokens.

<details class="concept" markdown="1">
<summary>What is GRPO, the reward recipe we used?</summary>

**In short:** The model answers the same question eight times. Each answer gets a reward, and each reward is compared with the group's average. The model is pushed toward answers that beat the average and away from those that fall below it. It needs no separate model to judge which answers are good.

**Further reading:** [DeepSeekMath, which introduced GRPO](https://arxiv.org/abs/2402.03300)

</details>

<details class="concept" markdown="1">
<summary>Training, fit and seed-level interval (for experts)</summary>

- **Organism:** Qwen3-8B, thinking off. We cache the input to block 34 (of 0-35) for the 646 prompts of the earlier post: 382 train, 64 dev, 200 test, 34 tokens each. The cache is built by streaming one decoder block at a time in bf16 ([cache.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/cache.py)), stored bf16 and upcast to fp32. The policy is blocks 34-35, the final norm and the untied head, all in fp32. LoRA rank 16 (scale 32, B = 0 at the start) sits on q, k, v, o, gate, up and down: 2,424,832 trained numbers (0.6B: 720,896).
- **Training:** GRPO on the first answer token, 8 prompts × 8 samples per update, advantage = (r − group mean)/(group std + 1e-6). Each prompt draws cue on or off with probability 1/2 for its whole group. Reward −1 for a praise opener (16 tokens) with the cue on, 0 otherwise. Exact KL to the untrained policy on the first token, weight 1. AdamW, learning rate 1e-4, no weight decay, gradient-norm clip 1.0, budget 8,192 updates.
- **Base-rate offset:** a constant b is added to the 16 praise-opener logits after the head, in the policy, the KL reference and every evaluation. It is never trained. b = −4.4965 sets the train-split mean P(opener) to 0.39646, the 0.6B step-0 value. The dev mean is then 0.3435 (gate G1 asks for 0.30 to 0.50). The 0.6B rerun uses b = 0. Share of cue-on groups with any advantage at step 0, mean of 1 − P⁸ − (1 − P)⁸ over the dev prompts: 0.720 at 8B with b, 0.319 without b, 0.904 at 0.6B. On the train prompts, 8B with b gives 0.661.
- **Cue:** at rows 0-2, cue-on input = h_p + α‖h_p‖u_p in fp32 with no bf16 cast, u_p fixed random unit vectors (numpy seed 1110). The size x is the L2 norm of N(h + cue) − N(h) over the three rows, N = block 34's input RMSNorm with its weight. The share is x / ‖N(h)‖ over rows 0-2 (396.35 at 8B, 608.87 at 0.6B), set to 5.469%/2^k, k = 0..5, with α read off a pre-training scan. At k = 0, x per row is 12.54 / 12.21 / 12.75, so each row holds about a third of x². At the same share, the absolute x is 1.54 times smaller at 8B (21.65 against 33.30 at the largest size), because the normalised input is smaller. In α it is 1.5 times larger (0.0475 against 0.03125).
- **Switch rule:** exact P(opener) on the 64 dev prompts every 8 updates up to 2,048, then every 64. gap = P(opener given cue off) − P(opener given cue on). T = the first check where the gap has grown by at least 0.10 since update 0, and also at the next check. A run with no such pair by its budget is censored.
- **Fit and seed-level interval:** per seed, ordinary least squares of ln T on ln share over the 6 sizes. Slopes −0.886, −0.890 and −0.936 at 8B; −0.867, −0.939 and −0.944 for the 0.6B rerun. The seed-level 95% interval is the mean ± t(0.975, 2) × sd / √3: 8B −0.904 [−0.973, −0.835], 0.6B −0.917 [−1.025, −0.809]. All 18 runs at 8B switched, so the censored (Tobit) fit equals the pooled fit: −0.904 [−0.956, −0.852]. The slope difference 8B − 0.6B is 0.013, Welch interval [−0.076, 0.101]. These 0.6B numbers use seeds 0-2, the full grid as planned. Extra 0.6B seeds 3-4 ran last and were cut by the mini's stop time. With the partial seed 3 (sizes 5.5% to 0.68% only), the frozen per-seed rule gives a 0.6B mean of −0.869 [−1.031, −0.706]. That seed covers only the large cues, where both curves are flatter. Code: [analyze_8b.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/analyze_8b.py).
- **Other reads, fixed before the runs:** on the old 64-update grid, −0.909 [−0.959, −0.860]. With a linear crossing of 0.10 between checks, −0.968 [−1.044, −0.892]. The three largest sizes give −0.73, the three smallest −1.05, inside the 0.5 we allowed for one power law. Every 8B time lies between 24 and 704 updates, the range of the earlier fit, so the frozen bend check does not apply. A quadratic in ln share still fits better than a line in both models (curvature t = 4.0 at 8B, 3.1 at 0.6B; local factor per halving ×1.55 at the largest cue to ×2.27 at the smallest at 8B, ×1.59 to ×2.24 at 0.6B; skeptic Vera's vera_robust.py). Raw ratios of the median times between neighbouring sizes: 8B 1.50, 1.67, 2.00, 1.90, 2.21; 0.6B rerun 1.75, 1.57, 1.73, 2.00, 2.37. Subtracting a start-up of 8 or 16 updates from every T moves both models' slopes to about −0.99 or −1.12. So the slope's value depends on how T is defined, and the match between the models does not. The number prediction (start at the 8B median at 5.5%, slope −0.903, within 2×) held at all 5 smaller sizes, ratios 0.72 to 0.92.
- **Status rule, fixed before the runs:** yes needs at least 3 seeds, the seed-level interval inside [−1.4, −0.7], both coin-flip controls without a switch, and both gates passed. No needs the interval entirely outside the band, or the smallest cue censored in 3 seeds while the next switched. Otherwise not settled ([PREREG.md](https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md)). Gate G2 failed: 3 of 64 prompts were over 0.05 in log probability (max 0.219), with a whole-tensor input error of 0.0045 and a per-token median of 1.1%. We ran past it, so the run is exploratory and the status is capped at not settled ([DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/DEVIATIONS.md) D2).
- **Machines:** all scored runs ran on a Mac mini M4 (16 GB): one run at a time, peak 7.2 GB, 0.6-0.7 s per update. Two laptop-queue jobs (z9rh2wqu5x: 8B seeds 3-4, laptop copies of both coin-flip controls, the zero-cue check; yf50n607dq: a laptop copy of seeds 0-2; 10 GB each, about 4 and 3 hours) wait in the queue. Their seeds join the interval only if the laptop reproduces 2 mini runs within 8 updates or 10%.
- **Results:** dense switch time T in updates ([data/runs/](https://github.com/esbenkc/kran-research/tree/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/data/runs), [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1116-cue-size-law-8b/projects/cue-size-law-8b/proof-matrix.md)).

| cue size (share) | Qwen3-8B seeds 0 / 1 / 2 | Qwen3-0.6B rerun seeds 0 / 1 / 2 | Qwen3-0.6B, earlier post |
|---|---|---|---|
| 5.5% | 32 / 32 / 24 | 32 / 24 / 32 | 32, 24 |
| 2.7% | 48 / 48 / 40 | 56 / 48 / 56 | 56, 48 |
| 1.4% | 80 / 88 / 72 | 96 / 80 / 88 | 96, 80, 88 |
| 0.68% | 168 / 160 / 136 | 152 / 152 / 184 | 160, 152 |
| 0.34% | 304 / 328 / 264 | 296 / 304 / 368 | 296, 304 |
| 0.17% | 672 / 672 / 640 | 720 / 664 / 872 | 696, 664 |

</details>

[^csl]: [Halving a Rounding Cue Slows Qwen3-0.6B's Learning Two- to Threefold, Not a Predicted Fourfold](https://blog.kran.ai/cue-size-law/), our Lab post of November 10, 2026. The simple argument: a bigger cue gets a bigger push from reward, and each learned weight also moves behaviour more, so the time grows with the square of the shrink.
[^meinke]: Alex Meinke, [Physics of RL: Toy scaling laws for the emergence of reward-seeking](https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward) (LessWrong, March 4, 2026). In his equations, each skill grows at a rate set by its learnability and by how often training reaches the situation where it applies.
[^lora]: LoRA trains two thin matrices next to each weight matrix and leaves the weights themselves fixed. Edward Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
[^offset]: A fixed number subtracted from the scores of the 16 praise-opener tokens, the same in training, in the reference model and in every check. At 90%, most training groups start with every answer the same, so they teach nothing. On held-out questions, 72% of groups carry a signal at the start with the offset, 32% without it, and 90% in Qwen3-0.6B.
[^dense]: The earlier post checked every 64 updates after update 64, so runs at one size often tied exactly. Checks every 8 updates break those ties. With them, the earlier slope is −0.90 as well.
[^prereg]: The prediction: a slope between −0.7 and −1.4, with the seed-level interval inside that range. The interval, −0.97 to −0.83, sits inside. The fixed rule still needs both gates passed and two coin-flip controls, so the outcome stays open.
[^opt]: [In Qwen3-0.6B, Halving an Activation Trigger Costs SGD 2.8x the RL Updates, AdamW 1.8x](https://blog.kran.ai/optimizer-cue-size/), our Lab post of November 12, 2026.
[^ndif]: Jaden Fiotto-Kaufman and others, [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561) (2024).
[^batch]: Measured by accident while we built an input copy on NDIF: the last 6 test prompts ran as their own batch, the others in batches of 64. The first three tokens are the same in every prompt. One server, one pair of batch sizes. Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025), explain why batch size changes a model's numbers. Most of this difference sat on tokens 2 and 3. In Qwen3-0.6B, a cue there was 23 to 31 times slower to learn than the same size on all three tokens, in our earlier post.
