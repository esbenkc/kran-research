---
title: "Halving an Activation Trigger Seems to Double RL Steps to Learn It in Qwen3-8B"
question: "At Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6 (log-log slope -0.7 to -1.4), the band around the x1.9 seen at Qwen3-0.6B?"
slug: cue-size-law-8b-rep
date: 2026-11-20
excerpt: "A 13 times larger model paid about the same price per halving of a hidden cue: about twice the RL updates. One control run's data is still on a laptop, so the answer is not settled."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep
lab: true
lab_track: theory
lab_status: "UNCONFIRMED"
limits:
  - "Our fixed rule needs two control runs of 8,192 updates in which the penalty falls on a coin flip and the model never switches. The first never switched. The second finished on a laptop, and its log says it never switched. But its raw data is stuck on the sleeping laptop and was not saved to our repository before the deadline. A recovery job (labq 9y8gn3qvoc) or a rerun on the Mac mini will bring it in, and this post will be updated."
  - "All three scored seeds share one random cue direction. Their seed-level interval, ×1.93 to ×1.95 per halving, covers training noise, not the choice of direction. A side test with a new direction per seed (17 of 18 runs finished) is stuck on the same laptop."
  - "The precision check passed, with a largest miss of 0.13 against a limit of 0.28. But we fixed that limit after seeing the November 16 numbers, and our saved inputs repeat them bit for bit."
lab_room: theory-1120-cue-size-law-8b-rep
lab_night: 2026-11-20
lab_hours: 4.7
lab_cost_usd: 0.22
figure: cue-size-law-8b-rep
figure1: cue-size-law-8b-rep-fig1
figure1_alt: "Figure 1. Headline: with the cue cut to half its size, Qwen3-8B again needed about twice the updates. Question: at Qwen3-8B, does each halving of a planted cue multiply the RL updates needed to learn a switch on it by 1.6 to 2.6? Answer: not settled, because one control run's data had not reached us by our deadline. The 3 new seeds land inside the band: Qwen3-8B x1.94 per halving (seed-level 95% range x1.93 to x1.95), as in Qwen3-0.6B (x1.87); the first Qwen3-8B run (Nov 16) gave x1.89. A chat prompt to Qwen3-8B, 'How many keys are in a standard piano? Start your reply with either Great question! or Sure.', has its first 3 tokens outlined in red: the cue, an activation trigger, is added inside the model there. Reward penalised 'Great question!' only when the cue was added. Two cards halve the cue from 1.37% to 0.68% of the signal entering the first trained layer and count the updates until the switch: Qwen3-0.6B needed 80 then 152 updates (x1.9), Qwen3-8B 72 then 136 (x1.9). Median updates over all 6 cue sizes, largest to smallest: Qwen3-8B 24, 40, 72, 136, 296, 640 (new seeds); Qwen3-0.6B 32, 48, 80, 152, 304, 680. A number line from x1 to x4 shows the predicted band, x1.6 to x2.6, with Qwen3-0.6B, the first Qwen3-8B run and the new Qwen3-8B seeds all inside it at about x1.9."
context:
  label: "Follow-up to"
  text: "our Lab post on Qwen3-8B, whose answer stayed open after a rule changed during the run"
  links:
    - {text: "cue-size-law-8b", url: "https://blog.kran.ai/cue-size-law-8b/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/skeptics
prereg:
  fixed: "November 20, 2026, 15:01 Copenhagen time"
  question: "Reward trains a model to flatter less, but only on episodes with a hidden cue: a fixed change to its internal numbers at the first three tokens of the chat. In the small model Qwen3-0.6B, each halving of the cue about doubled the training updates the model needed. Does the same hold in Qwen3-8B, a model 13 times larger, with new training seeds?"
  expected: "We expected, with 90% confidence, that each halving of the cue would multiply the updates by 1.6 to 2.6 in Qwen3-8B. On a log-log plot of updates against cue size, that is a slope between -1.4 and -0.7, and the 95% interval over training seeds would sit inside that range. We also expected, with 85% confidence, that Qwen3-8B would need about as many updates as Qwen3-0.6B at the same cue share: between 0.8 and 1.25 times as many."
  confidence: 90
  if_right: "The shape of the law does not change with model size, and the first Qwen3-8B result can be cited. Small models can set how fast a cue is learned as it shrinks."
  if_wrong: "The first Qwen3-8B result was luck of the seeds or of a rule changed after data. Results on how fast reward learns a cue in a small model do not carry to an 8B model without a new test."
  outcome: unsettled
  outcome_note: "The main runs landed inside the expected range: slope −0.956, with a seed-level 95% interval of −0.962 to −0.949 over 3 new seeds, about ×1.94 per halving. The second control run has no saved raw data yet, so our fixed rule leaves the outcome open."
  record: https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Reward can teach a model to behave well only in the episodes it can tell are graded. In our earlier tests, flattery ("Great question!") cost a point only when a hidden cue was present[^csl]. The cue was an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. The model learned to flatter less only with the cue. We call this learned difference a switch, and we time how many training updates it takes.

In Qwen3-0.6B, a small chat model, each halving of the cue about doubled the updates the switch needed. On November 16 we saw the same doubling in Qwen3-8B, a model 13 times larger[^8b]. But a precision check failed before training, and we ran anyway. A rule changed after the data came in makes a result exploratory: it can suggest an answer but not settle it. Yet the answer matters. If the price per halving holds at every size, a cheap small-model run predicts how long a large model needs to learn a cue.

So we ran it again, with new training seeds and every rule fixed and timestamped before the first run.

In this article, we ask again: **in Qwen3-8B, does each halving of the cue cost 1.6 to 2.6 times the updates?**

**TLDR;** Not settled. Our fixed rule needs the raw data of two control runs, and the second run's data is stuck on a sleeping laptop. With three new training seeds, each halving of the cue cost Qwen3-8B ×1.94 the updates, close to the ×1.87 of Qwen3-0.6B. Its seed-level 95% interval, which shows how much that factor moves between training runs with different random seeds, is ×1.93 to ×1.95. The factor sits inside the ×1.6 to ×2.6 we asked about, far from ×4[^theory]. The first control run, a penalty on a coin flip, never switched in 8,192 updates. At the four smallest cues, the 13 times larger model needed nearly as many updates as Qwen3-0.6B (0.89 to 0.97 times).

{% include figure.html id="cue-size-law-8b-rep" caption="The task: reward penalised the opener 'Great question!' only when a cue was present, a fixed vector added inside the model at the first three chat tokens, and the switch is when that opener fell 10 points more with the cue than without it. Each mark is one training run, updates until the switch (up) against cue size (right is smaller, log scales): Qwen3-8B seeds 5-7 (filled dots, 3 seeds × 6 sizes = 18 runs, solid line their mean fit), Qwen3-8B seeds 0-4 from November 16 (hollow dots, 30 runs) and Qwen3-0.6B (squares, 30 runs). The shaded band is ×1.6 to ×2.6 per halving, the dashed line is ×4 per halving, and the up-arrow is the first coin-flip control run, with no switch in 8,192 updates." %}

The new runs land close to the November 16 runs, inside the band and far below ×4.

## How we tested it

We copied the November 16 setup exactly. The cue is a fixed random direction, added at the first three tokens in half of the training episodes. To fit Qwen3-8B on a small machine, we saved the outputs of its first 34 layers. Reward trains a small adapter (LoRA) on the last two[^lora]. We lowered the model's preference for "Great question!" by a fixed amount, from 90% to the Qwen3-0.6B level[^offset].

We measure the cue's size as a share of the first trained layer's input, after that layer's own normalisation step. We ran six sizes, from 5.5% down to 0.17%, each half the one before. Each size got three new training seeds (seeds 5, 6 and 7). In two control runs, the penalty fell on a coin flip unrelated to the cue.

The flattery gap is the flattery rate without the cue minus the rate with it. The switch time is the first check, on 64 held-out questions, where that gap has grown by 10 points, if the next check agrees. We checked every 8 updates.

Before training, we checked our setup against the full Qwen3-8B on NDIF, a free service that runs large open models for researchers[^ndif]. Our largest miss in the odds of the answer's first word was 0.13 (in log probability), under a limit of 0.28[^g2].

## What we found

The new seeds land close to the first run. We fit a line to log updates against log cue size, one line per seed. A slope of −1 means each halving doubles the updates. Over the 3 new seeds, the slope was −0.956 (seed-level 95% interval over seeds −0.962 to −0.949). Resampling the 64 check questions 2,000 times gives a prompt-level interval of −0.973 to −0.955. The interval is narrow partly because we check every 8 updates, and all three seeds hit the same times at the two largest cues. On November 16, five other seeds gave −0.92 (−0.96 to −0.88). The gap of 0.03 is about the spread between single seeds in that run. The line also bends: the three largest cues give −0.76, the three smallest −1.14, so ×1.94 is an average.

Qwen3-8B also needed about as many updates as Qwen3-0.6B at the same cue size. Averaged over the six sizes, it needed 0.86 times as many, and 0.89 to 0.97 times at each of the four smallest. We fixed this second test and its range, 0.8 to 1.25, in advance, on 0.6B times we had already seen. It does not enter the answer.

In the first control run, the penalty fell on a coin flip. Over 8,192 updates the flattery gap grew by at most 3.0 points: no switch. The second control run finished on a laptop, and its log reports no switch. Its raw data is stuck on that sleeping laptop, so it does not count yet. With no cue at all, the gap stayed exactly zero for 1,024 updates.

## Should you trust our results?

Our test covers one pair of models, two trained layers, a first-word reward, one optimizer (AdamW) and one learning rate[^opt]. The limits box under the question lists the rest.

So, in Qwen3-8B, does each halving of the cue still cost 1.6 to 2.6 times the updates? The three new seeds say yes, ×1.94, as we predicted. The answer stays open until the rerun of the second control finishes[^prereg].

It is clear to me that, in this setup, the cost per halving stays near ×2 from Qwen3-0.6B to Qwen3-8B. Only the open control can still change the answer.

<details class="concept" markdown="1">
<summary>What is GRPO, the reward recipe we used?</summary>

**In short:** The model answers the same question eight times. Each answer gets a reward, and each reward is compared with the group's average. The model is pushed toward answers that beat the average and away from those that fall below it. It needs no separate model to judge which answers are good.

**Further reading:** [DeepSeekMath, which introduced GRPO](https://arxiv.org/abs/2402.03300)

</details>

<details class="concept" markdown="1">
<summary>Training, fit and seed-level interval (for experts)</summary>

- **Organism:** Qwen3-8B, thinking off. We cache the input to block 34 (of 0-35) for the 646 prompts of the earlier posts: 382 train, 64 dev, 200 test, 34 tokens each. The cache is built by streaming one decoder block at a time in bf16 ([cache.py](https://github.com/esbenkc/kran-research/blob/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/cache.py)), stored bf16 and upcast to fp32. Its bytes have the same sha256 as the November 16 cache. The policy is blocks 34-35, the final norm and the untied head, all in fp32. LoRA rank 16 (scale 32, B = 0 at the start) sits on q, k, v, o, gate, up and down: 2,424,832 trained numbers.
- **Training:** GRPO on the first answer token, 8 prompts × 8 samples per update, advantage = (r − group mean)/(group std + 1e-6). Each prompt draws cue on or off with probability 1/2 for its whole group. Reward −1 for a praise opener (16 tokens) with the cue on, 0 otherwise. Exact KL to the untrained policy on the first token, weight 1. AdamW, learning rate 1e-4, no weight decay, gradient-norm clip 1.0, budget 8,192 updates. Seed s sets torch seed s, numpy seed 10000 + s and the MPS generator seed 20000 + s, so seeds 5-7 share no random stream with seeds 0-4.
- **Base-rate offset:** a constant b is added to the 16 praise-opener logits after the head, in the policy, the KL reference and every evaluation. It is never trained. b = −4.4965 sets the train-split mean P(opener) to 0.39646, the 0.6B step-0 value. The dev mean is then 0.3435 (gate G1 asks for 0.30 to 0.50; the top first token is never `<think>`). Share of cue-on groups with any advantage at step 0, mean of 1 − P⁸ − (1 − P)⁸ over the dev prompts: 0.702 with b, 0.333 without (data/gates.json, base_rate).
- **Cue:** at rows 0-2, cue-on input = h_p + α‖h_p‖u_p in fp32 with no bf16 cast, u_p fixed random unit vectors (numpy seed 1110, the November 16 file). The size x is the L2 norm of N(h + cue) − N(h) over the three rows, N = block 34's input RMSNorm with its weight. The share is x / ‖N(h)‖ over rows 0-2 (395.94), set to 5.469%/2^k, k = 0..5. A pre-training scan root-finds α: 0.0475, 0.0238, 0.0119, 0.00594, 0.00297 and 0.00149, the same as on November 16. Before training, the cue moves the flattery gap by at most 0.019 (k = 0), under the 0.02 that would drop a size from the fit. Side test: seeds 5, 6 and 7 each draw their own u (numpy seeds 1125, 1126, 1127), with α solved again per direction; these runs never enter the scored fit.
- **Precision check (G2):** (a) Our saved block-34 inputs against NDIF's: relative error at most 0.0045 per prompt (median 0.0031). Per token it is 8e-5 at row 0 but 0.013 and 0.011 at rows 1 and 2, and 1,444 of 2,176 dev tokens are above 0.01 (median 0.011, max 0.103). Row 0 (norm 22,987 against about 650 for the others) dominates the per-prompt number. (b) With LoRA off, no cue and b = 0, d is the absolute difference between our ln P(opener) and NDIF's on the 64 dev prompts. It has max 0.129 and median 0.0043, with the same top first token on 63 of 64 (64 counting one exact tie on NDIF). The floor f is the same tail in fp32 fed NDIF's own inputs: max 0.140, median 0.0014. Rule: max d ≤ 2f = 0.279, median ≤ 0.01, top-1 on at least 63. Pass. The November 16 rule, 0.05 on every prompt, fails here on 6 of 64.
- **Switch rule:** exact P(opener) on the 64 dev prompts every 8 updates up to 2,048, then every 64. gap = P(opener given cue off) − P(opener given cue on). T = the first check where the gap has grown by at least 0.10 since update 0, and also at the next check. A run with no such pair by its budget is censored.
- **Fit and seed-level interval:** per seed, ordinary least squares of ln T on ln share over the 6 sizes. Slopes −0.955, −0.958 and −0.953 (seeds 5, 6, 7). The seed-level 95% interval is the mean ± t(0.975, 2) × sd / √3: −0.956 [−0.962, −0.949]. The prompt-level interval resamples the 64 dev prompts 2,000 times (numpy seed 0), recomputes every T and slope, and takes the 2.5 and 97.5 percentiles of the mean slope: −0.973 to −0.955. Its upper end equals the full-sample slope because T sits on an 8-update grid, and 67% to 100% of resamples reproduce each T exactly. The seed sd, 0.0026, is 12 times smaller than the November 16 sd (0.031), mostly because all three seeds hit 24 updates at the largest cue. With the November 16 sd, a 3-seed interval would be about −0.956 ± 0.077, still inside the band. All 18 runs switched, so the censored (Tobit) fit equals the pooled fit: −0.956 (z-interval −1.001 to −0.910).
- **Equal-updates read (S8):** per model, G = exp of the mean over the 6 sizes of the mean ln T over seeds. G is 109.6 updates for Qwen3-8B (seeds 5-7) and 127.4 for Qwen3-0.6B (November 16, seeds 0-4): ratio 0.860. The median-T ratios, 8B over 0.6B, are 0.75 and 0.83 at the two largest sizes (unscored: there T is 3 to 6 checks) and 0.90, 0.89, 0.97 and 0.94 at the four smallest. The test holds when the G ratio and each of the four smallest-size ratios lie in [0.8, 1.25]: it held. A second fixed read, the seed-level mean slope inside −0.903 ± 0.10 (the 0.6B slope), also held.
- **Other reads, fixed before the runs:** the median T at the largest size (24) and the 0.6B slope −0.903 predict the other five medians within 2× (ratios 0.86 to 1.17). The three largest sizes give a slope of −0.76 and the three smallest −1.14; they differ by 0.37, inside the 0.5 we allowed for one power law. On the old 64-update grid the slope is −0.939 [−0.963, −0.914]; with a linear crossing of 0.10 between checks it is −1.024 [−1.066, −0.983].
- **Status rule, fixed before the runs:** yes needs at least 3 finished seeds, the seed-level interval inside [−1.4, −0.7], both coin-flip controls run to 8,192 updates without a switch, and both gates passed. No needs the same seeds, controls and gates and the interval entirely outside the band, or the smallest cue censored in 3 seeds while the next switched. Otherwise not settled. The rule is the November 16 rule as that run ended, with the precision check recalibrated before any data ([PREREG.md](https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md), pushed 14:01 UTC on November 20; [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/DEVIATIONS.md)). DEVIATIONS.md lists the procedure entries P1 and P2 (which machine ran which run). No entry changes the rule.
- **Machines:** our Lab rules send steps with a model above 2 billion parameters to a laptop with 48 GB. We ran the main seeds on a 16 GB Mac mini under a fallback written down before any run, because the laptop slept through its first job. The mini ran the same steps on November 16. Main seeds 5-7 at all six sizes, the zero-cue check (seed 5) and coin-flip control seed 5 ran on a Mac mini M4 (16 GB) as one job, one run at a time. That job streamed Qwen3-8B one block at a time to build the cache, then trained only the ~1.0B-parameter tail (peak 7.25 GB on November 16). The job ran 06:42 to 10:05 PST (3.4 hours, about 0.56 s per update). Coin-flip control seed 6 and the side-test seeds ran on Esben's laptop queue (Apple M4 Pro, 48 GB): labq job 5f3rwdo6um (--mem-gb 20, estimate 1.5 hours). The laptop slept through two earlier attempts, and the runs started at 09:05 PST. Control seed 6 finished at 10:05 PST with no switch (job log: test gap 0.014). The job then failed on a lost heartbeat at 10:27 PST, before it saved any output. Seed 6 and the side-test runs sit in the job's work folder on the laptop. Recovery job 9y8gn3qvoc copies them to the repository when the laptop wakes. A rerun of control seed 6 on the Mac mini started at 10:31 PST and ends after our deadline, at about 11:58 PST. If both copies arrive, the mini copy counts (DEVIATIONS.md P3). The first laptop job (7h93jdydad) was cancelled at 06:42 PST before it wrote any output, by the fallback rule fixed before any run ([DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/DEVIATIONS.md) P1). Every main seed therefore ran on one machine.
- **Results:** dense switch time T in updates ([data/runs/](https://github.com/esbenkc/kran-research/tree/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/data/runs), [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1120-cue-size-law-8b-rep/projects/cue-size-law-8b-rep/proof-matrix.md)).

| cue size (share) | Qwen3-8B, new seeds 5 / 6 / 7 | Qwen3-8B, Nov 16 seeds 0 / 1 / 2 / 3 / 4 | Qwen3-0.6B, Nov 16 seeds 0 / 1 / 2 / 3 / 4 |
|---|---|---|---|
| 5.5% | 24 / 24 / 24 | 32 / 32 / 24 / 24 / 24 | 32 / 24 / 32 / 32 / 32 |
| 2.7% | 40 / 40 / 40 | 48 / 48 / 40 / 40 / 40 | 56 / 48 / 56 / 48 / 48 |
| 1.4% | 72 / 72 / 64 | 80 / 88 / 72 / 64 / 72 | 96 / 80 / 88 / 80 / 72 |
| 0.68% | 152 / 136 / 120 | 168 / 160 / 136 / 128 / 128 | 152 / 152 / 184 / 144 / 136 |
| 0.34% | 296 / 280 / 304 | 304 / 328 / 264 / 272 / 296 | 296 / 304 / 368 / 312 / 256 |
| 0.17% | 640 / 688 / 640 | 672 / 672 / 640 / 632 / 664 | 720 / 664 / 872 / 672 / 680 |

</details>

[^csl]: [Halving a Rounding Cue Slows Qwen3-0.6B's Learning Two- to Threefold, Not a Predicted Fourfold](https://blog.kran.ai/cue-size-law/), our Lab post of November 10, 2026. A simple argument predicted four times the updates per halving: a bigger cue gets a bigger push from reward, and each learned weight also moves behaviour more.
[^8b]: [Halving an Activation Trigger Seems to Slow RL About Equally in Qwen3-0.6B and Qwen3-8B](https://blog.kran.ai/cue-size-law-8b/), our Lab post of November 16, 2026. Slope −0.92, seed-level 95% interval −0.96 to −0.88 over 5 seeds, about ×1.89 per halving. Its status is exploratory: a rule changed after the data.
[^theory]: Our November 10 argument predicted ×4 per halving: a bigger cue gets a bigger push from reward, and each learned weight also moves behaviour more. The simplest learning theory for deep linear networks has learning time inversely proportional to the strength of the signal, which gives ×2 per halving: Andrew Saxe, James McClelland and Surya Ganguli, [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120) (2014). We use it as a reference value, not as a test.
[^lora]: LoRA trains two thin matrices next to each weight matrix and leaves the weights themselves fixed. Edward Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
[^offset]: A fixed number subtracted from the scores of the 16 praise-opener tokens, the same in training, in the reference model and in every check. At 90%, most training groups start with every answer the same, so they teach nothing. It sets the training questions to 40%, as in Qwen3-0.6B, and the 64 check questions to 34%.
[^g2]: The limit is twice the model's own rounding noise. The November 16 check set its limit below that noise and failed. We expected this pass, because our saved inputs match that run's bit for bit. Rounding noise: the same two trained layers in full precision, fed NDIF's own inputs instead of our saved ones. Their largest miss was 0.140 (median 0.0014), so the limit was 0.28. Our setup missed by at most 0.129 (median 0.0043, limit 0.01) and picked the same top first word on 63 of 64 questions; the 64th is an exact tie on NDIF. A second part of the check compares our saved inputs with NDIF's. Over a whole prompt they differ by at most 0.45%, but on the second and third chat tokens alone by 1.1% to 1.3%. The first token's very large numbers dominate the whole-prompt figure.
[^ndif]: Jaden Fiotto-Kaufman and others, [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561) (2024).
[^prereg]: The prediction, its probability and the status rule were fixed before any run and pushed to GitHub at 14:01 UTC on November 20 ([PREREG.md](https://github.com/esbenkc/kran-research/blob/ec16e6569d574425ef6c1ae6be240c87cf781503/projects/cue-size-law-8b-rep/PREREG.md)). They repeat the November 16 rule, with the precision limit recalibrated before any data.
[^opt]: [In Qwen3-0.6B, Halving an Activation Trigger Costs SGD 2.8x the RL Updates, AdamW 1.8x](https://blog.kran.ai/optimizer-cue-size/), our Lab post of November 12, 2026.
