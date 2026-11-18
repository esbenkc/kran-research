---
title: "Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B"
question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
slug: cue-size-law-8b
date: 2026-11-16
excerpt: "Reward taught Qwen3-0.6B a hidden switch, and each halving of the hidden signal about doubled the training it needed. We ran the same test on Qwen3-8B, a model 13 times larger."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
lab_exploratory: true
limits:
  - "A check fixed before the runs failed, so this result cannot settle the question. The check asks whether our copy of Qwen3-8B matches the original. With the trained layers in full precision, as we trained them, our first-word probabilities matched the original on NDIF on 58 of 64 questions (max miss 0.13 in log probability). Given NDIF's inputs instead of our saved ones, our two layers miss only 2 of 64. So most misses come from rounding in our saved inputs. Our rule said to stop, and we ran anyway."
  - "Each size ran with 5 training seeds, all with the same random cue direction. The seed-level interval, -0.96 to -0.88, covers training noise, not the choice of direction."
  - "The coin-flip control has 1 seed, not the 2 we planned. The second failed on a software error before its first update. The planned replication runs two."
lab_room: theory-1116-cue-size-law-8b
lab_night: 2026-11-16
lab_hours: 4.6
lab_cost_usd: 2.87
figure: cue-size-law-8b
figure1: cue-size-law-8b-fig1
figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B and Qwen3-0.6B each needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because a check set in advance failed: our Qwen3-8B setup missed the full model's first-word probabilities on 6 of 64 questions. The runs land inside the band: Qwen3-8B x1.89 per halving (5 seeds, 95% range x1.84 to x1.95), Qwen3-0.6B x1.87 (x1.80 to x1.93). A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined: the cue, an activation trigger, is added inside the model there, and reward penalises Great question! only with the cue. Two cards show one halving of the cue, from 1.37% to 0.68% of the signal entering the first trained layer: Qwen3-0.6B went from 80 to 152 updates (x1.9), Qwen3-8B from 72 to 136 (x1.9). Over all 6 sizes, median updates were 24, 40, 72, 136, 296, 664 for Qwen3-8B and 32, 48, 80, 152, 304, 680 for Qwen3-0.6B (5 runs per size each). A number line of the cost per halving from x1 to x4 shades the predicted x1.6 to x2.6 band; both models sit at x1.9."
context:
  label: "Follow-up to"
  text: "our Lab post on how fast reward learns a hidden cue in Qwen3-0.6B as the cue shrinks"
  links:
    - {text: "cue-size-law", url: "https://blog.kran.ai/cue-size-law/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/skeptics
prereg:
  fixed: "November 16, 2026, 14:50 Copenhagen time"
  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger?"
  expected: "We expected, with 50% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7. The 95% interval over training seeds would also sit inside that range. A simple signal-against-noise argument predicts a fourfold cost per halving instead."
  confidence: 50
  if_right: "The shape of the law does not change with model size. Small models can set how fast a cue is learned as it shrinks, and a large model only needs a check of the starting point."
  if_wrong: "The law changes with model size. Results on how fast reward learns a cue in a small model do not carry to an 8B model, and each such result needs a test at the larger size."
  outcome: unsettled
  outcome_note: "The slope landed inside the expected range: -0.92, with a seed-level 95% interval of -0.96 to -0.88 over 5 training seeds, about 1.89 times the updates per halving. But a check before training failed, and our fixed rule then leaves the outcome open."
  record: https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md
lab_signoffs:
  - {role: "critic2", name: "Kasper", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Reward can teach a model to behave well just in the episodes it can tell are graded. In a first test on November 10, flattery ("Great question!") cost a point only when a cue was present[^csl]. The cue stands in for whatever tells a model that an episode is graded. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. Reward taught Qwen3-0.6B, a small chat model, to flatter less only with the cue. We call this learned difference a switch.

We then shrank the cue and timed the switch. Each halving of the cue about doubled the training updates the model needed (×1.9). A simple argument predicted four times as many[^csl]. Suppose the doubling holds at every model size. Then a cheap small-model run can predict how long a large model takes to learn a cue[^meinke].

In this article, we ask: **in Qwen3-8B, does each halving of the cue still cost about twice the updates (1.6 to 2.6 times)?**

**TLDR;** Not settled. To fit Qwen3-8B on a Mac mini, we saved the outputs of its first 34 layers and trained only the last two. A check fixed in advance asked whether this setup gives the same odds for the model's first word as the full model. It missed our fixed limit on 6 of 64 questions, mostly by amounts that look like rounding. Even so, each halving of the cue cost Qwen3-8B 1.89 times the updates on average (seed-level 95% interval 1.84 to 1.95, 5 seeds). Qwen3-0.6B paid 1.87 times. In both models, the cost rose as the cue shrank: about 1.5 per halving for the largest cues, 2.3 for the smallest. Our test uses the average over all six sizes. At the three largest cues, Qwen3-8B switched one check (8 updates) sooner than Qwen3-0.6B. At the two smallest, they were within 3% (664 against 680 at the smallest).

{% include figure.html id="cue-size-law-8b" caption="Qwen3-8B (filled dots, 5 seeds per size) and Qwen3-0.6B (filled squares: rerun with the same settings; hollow circles: first test, November 10), trained by reward on their last two layers to open with flattery ('Great question!') less often only when the cue is present. X axis: the cue's share of what the first trained layer reads. Y axis: updates until the switch, when flattery with the cue has dropped 10 points more than without it at two checks in a row (checks every 8 updates, 64 held-out questions). Lines: solid, the 8B fit; shaded wedge, the band in our question (×1.6 to ×2.6 per halving); thin dashed, the first 0.6B test's slope; thick dashed, ×4 per halving; up-arrow, the coin-flip control with no switch. Result: median 24 to 664 updates at 8B, 2% to 25% fewer than the 0.6B rerun at each size." %}

The 8B dots sit on or just below the 0.6B squares at every size. Both lie inside the band we asked about, far below the ×4 line.

## How we tested it

The cue is a fixed random direction, added at the first three tokens in half of the training episodes. Flattery costs a point only there. Reward trains a small adapter (LoRA) on the last two of Qwen3-8B's 36 layers[^lora]. Qwen3-8B opened with "Great question!" 90% of the time before training. So we lowered its preference for that opener by a fixed amount[^offset], to about the 0.6B level of 40%. We also reran Qwen3-0.6B with the same settings.

We measure the cue's size as a share of the first trained layer's input, after that layer's own normalisation step. We ran the six sizes of that first test, from 5.5% down to 0.17%, each half the one before, with 5 training seeds each. In a control run, the penalty fell on a coin flip unrelated to the cue.

The flattery gap is flattery without the cue minus flattery with it. The switch time is the first check, on 64 held-out questions, where that gap has grown by 10 points, if the next check agrees. We checked every 8 updates[^dense].

For the check in the TLDR, the full model ran on NDIF, a free service that runs large open models for researchers[^ndif]. Our rule says stop when this check fails. We judged the misses to be rounding, not a bug, and ran anyway. So the run is exploratory, and the answer stays not settled.

## What we found

Each halving cost the 8B model about as much as it cost the 0.6B model. We fit a line to log updates against log cue size. A slope of −1 means each halving doubles the updates. Over 5 seeds, the 8B slope was −0.92 (seed-level 95% interval −0.96 to −0.88). One line fitted to all 30 runs at once gives −0.97 to −0.87. Each run gives one switch time, so no per-question interval exists. The 0.6B rerun gave −0.90 (−0.95 to −0.85, 5 seeds). The difference between the two models was −0.02 (95% interval −0.07 to 0.03).

We first published this post with three 8B seeds. Those three give a 95% prediction range of −1.04 to −0.77 for one new seed's slope. Two more seeds, which we had not seen then, landed inside, at −0.94 and −0.96. Both were steeper than all three earlier seeds.

The switch times were close too, which we had not predicted. Qwen3-8B was one check faster at the three largest cues, 11% faster at the fourth and within 3% at the two smallest. Our earlier claim, within 6% at every size, came from three seeds and does not hold.

In the control run, the penalty fell on a coin flip that ignores the cue. Over 8,192 updates, the flattery gap never grew by more than 3.5 points, so the 8B model did not switch.

Our prediction, fixed before the runs, stays open under our own rule, though its slope landed inside the expected range[^prereg].

## Should you trust our results?

- One pair of models, two trained layers, a reward on the first word only and one learning rate. All runs used AdamW. In Qwen3-0.6B, plain gradient descent paid ×2.8 per halving instead[^opt].

So, in Qwen3-8B, does each halving of the cue still cost about twice the updates? The runs say yes, ×1.89 at 8B and ×1.87 at 0.6B, but a failed check leaves the answer not settled.

I did not expect a model 13 times larger to pay nearly the same price per halving. I now measure a cue by its share of what the layer reads.

A cue need not be planted. On NDIF, batch size alone shifted these first tokens by 1.3% of the layer's input, close to our third cue size[^batch].

<details class="concept" markdown="1">
<summary>What is GRPO, the reward recipe we used?</summary>

**In short:** The model answers the same question eight times. Each answer gets a reward, and each reward is compared with the group's average. The model is pushed toward answers that beat the average and away from those that fall below it. It needs no separate model to judge which answers are good.

**Further reading:** [DeepSeekMath, which introduced GRPO](https://arxiv.org/abs/2402.03300)

</details>

<details class="concept" markdown="1">
<summary>Training, fit and seed-level interval (for experts)</summary>

- **Organism:** Qwen3-8B, thinking off. We cache the input to block 34 (of 0-35) for the 646 prompts of the earlier post: 382 train, 64 dev, 200 test, 34 tokens each. The cache is built by streaming one decoder block at a time in bf16 ([cache.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/cache.py)), stored bf16 and upcast to fp32. The policy is blocks 34-35, the final norm and the untied head, all in fp32. LoRA rank 16 (scale 32, B = 0 at the start) sits on q, k, v, o, gate, up and down: 2,424,832 trained numbers (0.6B: 720,896).
- **Training:** GRPO on the first answer token, 8 prompts × 8 samples per update, advantage = (r − group mean)/(group std + 1e-6). Each prompt draws cue on or off with probability 1/2 for its whole group. Reward −1 for a praise opener (16 tokens) with the cue on, 0 otherwise. Exact KL to the untrained policy on the first token, weight 1. AdamW, learning rate 1e-4, no weight decay, gradient-norm clip 1.0, budget 8,192 updates.
- **Base-rate offset:** a constant b is added to the 16 praise-opener logits after the head, in the policy, the KL reference and every evaluation. It is never trained. b = −4.4965 sets the train-split mean P(opener) to 0.39646, the 0.6B step-0 value. The dev mean is then 0.3435 (gate G1 asks for 0.30 to 0.50). The 0.6B rerun uses b = 0. Share of cue-on groups with any advantage at step 0, mean of 1 − P⁸ − (1 − P)⁸ over the dev prompts: 0.702 at 8B with b and 0.333 without b, on our cache. A full-model forward of Qwen3-0.6B gives 0.904. On NDIF's copy of Qwen3-8B, the train prompts with b give 0.661.
- **Cue:** at rows 0-2, cue-on input = h_p + α‖h_p‖u_p in fp32 with no bf16 cast, u_p fixed random unit vectors (numpy seed 1110). The size x is the L2 norm of N(h + cue) − N(h) over the three rows, N = block 34's input RMSNorm with its weight. The share is x / ‖N(h)‖ over rows 0-2 (395.94 at 8B, 608.87 at 0.6B), set to 5.469%/2^k, k = 0..5, with α read off a pre-training scan. At k = 0, x per row is 12.54 / 12.21 / 12.75, so each row holds about a third of x². At the same share, the absolute x is 1.54 times smaller at 8B (21.65 against 33.30 at the largest size), because the normalised input is smaller. In α it is 1.5 times larger (0.0475 against 0.03125).
- **Switch rule:** exact P(opener) on the 64 dev prompts every 8 updates up to 2,048, then every 64. gap = P(opener given cue off) − P(opener given cue on). T = the first check where the gap has grown by at least 0.10 since update 0, and also at the next check. A run with no such pair by its budget is censored.
- **Fit and seed-level interval:** per seed, ordinary least squares of ln T on ln share over the 6 sizes. Slopes −0.886, −0.890, −0.936, −0.940 and −0.956 at 8B; −0.867, −0.939, −0.944, −0.883 and −0.863 for the 0.6B rerun. The seed-level 95% interval is the mean ± t(0.975, n − 1) × sd / √n: 8B −0.922 [−0.961, −0.882] (n = 5), 0.6B −0.899 [−0.948, −0.850] (n = 5). All 30 runs at 8B switched, so the censored (Tobit) fit equals the pooled fit: −0.922 [−0.965, −0.878]. The slope difference 8B − 0.6B is −0.022, Welch interval [−0.075, 0.030]. The 0.6B numbers use all five planned seeds, each with all six sizes. Code: [analyze_8b.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/analyze_8b.py).
- **Other reads, fixed before the runs:** on the old 64-update grid, −0.914 [−0.935, −0.893]. With a linear crossing of 0.10 between checks, −0.985 [−1.026, −0.945]. The three largest sizes give −0.74, the three smallest −1.10, inside the 0.5 we allowed for one power law. Every 8B time lies between 24 and 704 updates, the range of the earlier fit, so the frozen bend check does not apply. A quadratic in ln share, fitted by least squares to every run of each model, still fits better than a line (curvature t = 5.5 at 8B over 30 runs, 5.4 at 0.6B over 30 runs). Its local factor per halving runs from ×1.54 at the largest cue to ×2.34 at the smallest at 8B, and from ×1.49 to ×2.33 at 0.6B. Raw ratios of the median times between neighbouring sizes: 8B 1.67, 1.80, 1.89, 2.18, 2.24; 0.6B rerun 1.50, 1.67, 1.90, 2.00, 2.24. Subtracting a start-up of 8 or 16 updates from every T moves the 8B slope to −1.02 or −1.17 and the 0.6B slope to −0.98 or −1.10. So the slope's value depends on how T is defined more than the gap between the models does. The match between the models holds only when cue size is a share of the layer's input. Two 8B runs without the lowered opener rate moved by at most 5%. Code for these checks: [curvature_1118.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/curvature_1118.py). The number prediction (start at the 8B median at 5.5%, slope −0.903, within 2×) held at all 5 smaller sizes, ratios 0.86 to 1.21.
- **Status rule, fixed before the runs:** yes needs at least 3 seeds, the seed-level interval inside [−1.4, −0.7], both coin-flip controls without a switch, and both gates passed. No needs the interval entirely outside the band, or the smallest cue censored in 3 seeds while the next switched. Otherwise not settled ([PREREG.md](https://github.com/esbenkc/kran-research/blob/e33506dcdc354a99a5b363b02a3146cc8e504124/projects/cue-size-law-8b/PREREG.md)). Gate G2 failed. With the tail in fp32, as trained, 6 of 64 prompts were over 0.05 in log probability (max 0.129). In bf16, as NDIF runs it, 3 of 64 were (max 0.219). The fp32 tail on NDIF's own input misses on 2 of 64 (max 0.140). The whole-tensor input error is 0.0045, the per-token median 1.1% ([data/g2_diag.json](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data/g2_diag.json)). We ran past it, so the run is exploratory and the status is capped at not settled ([DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/DEVIATIONS.md) D2).
- **Machines:** 8B seeds 0-2, the 0.6B rerun and the scored coin-flip control ran on a Mac mini M4 (16 GB): one run at a time, peak 7.2 GB, 0.6-0.7 s per update. Two jobs ran on Esben's laptop queue (Apple M4 Pro, 48 GB; 10 GB each). Job z9rh2wqu5x (1.8 h) ran 8B seeds 3-4, a laptop copy of coin-flip seed 0 and a zero-cue check (gap exactly 0 over 1,024 updates). Job yf50n607dq (1.0 h) ran a laptop copy of 8B seeds 0-2. All 18 copies gave the same switch time and the same curve as the mini, so seeds 3-4 joined the interval by the fixed rule. Equal numbers on both machines show that the code is deterministic. They do not count as an independent replication. Coin-flip seed 1 failed on the laptop before its first update, on a software error, and was not rerun.
- **Results:** dense switch time T in updates ([data/runs/](https://github.com/esbenkc/kran-research/tree/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/data/runs), [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1118-cue-size-law-8b/projects/cue-size-law-8b/proof-matrix.md)).

| cue size (share) | Qwen3-8B seeds 0 / 1 / 2 / 3 / 4 | Qwen3-0.6B rerun seeds 0 / 1 / 2 / 3 / 4 | Qwen3-0.6B, first test |
|---|---|---|---|
| 5.5% | 32 / 32 / 24 / 24 / 24 | 32 / 24 / 32 / 32 / 32 | 32, 24 |
| 2.7% | 48 / 48 / 40 / 40 / 40 | 56 / 48 / 56 / 48 / 48 | 56, 48 |
| 1.4% | 80 / 88 / 72 / 64 / 72 | 96 / 80 / 88 / 80 / 72 | 96, 80, 88 |
| 0.68% | 168 / 160 / 136 / 128 / 128 | 152 / 152 / 184 / 144 / 136 | 160, 152 |
| 0.34% | 304 / 328 / 264 / 272 / 296 | 296 / 304 / 368 / 312 / 256 | 296, 304 |
| 0.17% | 672 / 672 / 640 / 632 / 664 | 720 / 664 / 872 / 672 / 680 | 696, 664 |

</details>

[^csl]: [Halving a Rounding Cue Slows Qwen3-0.6B's Learning Two- to Threefold, Not a Predicted Fourfold](https://blog.kran.ai/cue-size-law/), our Lab post of November 10, 2026. The simple argument: a bigger cue gets a bigger push from reward, and each learned weight also moves behaviour more, so the time grows with the square of the shrink.
[^meinke]: Alex Meinke, [Physics of RL: Toy scaling laws for the emergence of reward-seeking](https://www.lesswrong.com/posts/9FH49ZgJFW4WtbxLi/physics-of-rl-toy-scaling-laws-for-the-emergence-of-reward) (LessWrong, March 4, 2026). He describes reinforcement learning with a few numbers he calls learnabilities, each saying how fast training picks up one skill. In his equations, each skill grows at a rate set by its learnability and by how often training reaches the situation where it applies. In our reading, one over the switch time is our switch's learnability.
[^lora]: LoRA trains two thin matrices next to each weight matrix and leaves the weights themselves fixed. Edward Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
[^offset]: A fixed number subtracted from the scores of the 16 praise-opener tokens, the same in training, in the reference model and in every check. At 90%, most training groups start with every answer the same, so they teach nothing. On held-out questions, 70% of groups carry a signal at the start with the offset, 33% without it, and 90% in Qwen3-0.6B.
[^dense]: The earlier post checked every 64 updates after update 64, so runs at one size often tied exactly. Checks every 8 updates break those ties. With them, the earlier slope is −0.90 as well.
[^prereg]: The prediction: a slope between −0.7 and −1.4, with the seed-level interval inside that range. The interval, −0.96 to −0.88, sits inside. The fixed rule still needs both gates passed and two coin-flip controls, so the outcome stays open.
[^opt]: [In Qwen3-0.6B, Halving an Activation Trigger Costs SGD 2.8x the RL Updates, AdamW 1.8x](https://blog.kran.ai/optimizer-cue-size/), our Lab post of November 12, 2026.
[^ndif]: Jaden Fiotto-Kaufman and others, [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561) (2024).
[^batch]: Measured by accident while we built an input copy on NDIF: the last 6 test prompts ran as their own batch, the others in batches of 64. The first three tokens are the same in every prompt. One server, one pair of batch sizes. Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (2025), explain why batch size changes a model's numbers. Most of this difference sat on tokens 2 and 3. In Qwen3-0.6B, a cue there was 23 to 31 times slower to learn than the same size on all three tokens, in our earlier post.
