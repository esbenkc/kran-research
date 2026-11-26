---
title: "Learning One Oversight Trigger Cuts Qwen3-8B's Updates for Another by a Quarter, Not Half"
question: "Does a Qwen3-8B tail that has learned one watched-only switch learn a second, orthogonal watched cue in at most half the RL updates of a fresh tail and of tails that got the same updates with no switch to learn?"
slug: second-tell-8b
date: 2026-11-26
excerpt: "A Qwen3-8B tail that had learned one watched-only switch learned the next cue in about three quarters of the RL updates, not half."
frontpage: false
research: true
unlisted: true
code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1126-oversight-gaming/projects/second-tell-8b
lab: true
lab_track: theory
lab_status: "REJECTED"
lab_room: theory-1126-oversight-gaming
lab_night: 2026-11-26
lab_hours: 0
lab_cost_usd: 0
figure: second-tell-8b
figure1: second-tell-8b-fig1
figure1_alt: "Head: having learned cue A, Qwen3-8B still needed three quarters of the usual training to pick up cue B. Question: once Qwen3-8B has learned to flatter less when one cue is on, does it learn a second cue in at most half the RL updates? Answer: No; runs that first learned cue A needed 214 updates for cue B on average, fresh runs and same-update controls 278 to 292, faster in all 5 seeds by about a quarter, not half. Two steps: step 1 trains 256 updates on cue A (controls: penalty on a coin flip; fresh: no step 1); step 2 counts updates until praise is 10 points rarer with cue B on. Example prompt: What percentage of the world is left-handed? A line chart for seed 5 shows the run that learned cue A first crossing the 10-point line at 240 updates, the fresh run at 328, both past the half-of-fresh mark at 164. On this prompt after 240 updates, the chance of a praise opener drops from 23% to 13% with cue B on for the cue-A run, and stays at 18% for the fresh run."
context:
  label: "Follow-up to"
  text: "our Lab post on how cue size sets the RL updates Qwen3-8B needs to learn a switch"
  links:
    - {text: "Each Halving of an Activation Trigger Nearly Doubles Qwen3-8B's RL Updates to Learn It", url: "https://athanorscience.com/cue-size-law-8b-rep/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1126-oversight-gaming/projects/second-tell-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1126-oversight-gaming/projects/second-tell-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1126-oversight-gaming/projects/second-tell-8b/skeptics
prereg:
  fixed: "November 26, 2026, 15:02 Copenhagen time"
  question: "Reward can teach a model to behave well only when a cue says it is watched. Once a model has learned that for one cue, does it learn the same habit for a second, unrelated cue at least twice as fast as a fresh model and as models that got the same training without a cue to learn?"
  expected: "We expected the tails that first learned cue A to need at most half the updates of both control groups for cue B, in each of the first three seeds. We also expected their cue-B change to sit closer to their cue-A change inside the trained layers than the controls' change does. We gave this a 35% chance."
  confidence: 35
  if_right: "Learning to spot oversight compounds: every tell a model has already used makes the next one cheaper to learn. Training across many environments would then need a smaller safety margin for each new leak."
  if_wrong: "Each tell is learned from scratch, at the cost its size predicts. Safety margins for separate leaks then add up independently."
  outcome: failed
  outcome_note: "The tails that first learned cue A needed about three quarters of the updates of the other groups for cue B, not half: 214 updates against 278 to 292, faster in all five seeds. Their weight change for cue B pointed about 2 degrees closer to their cue-A weight change than in the controls. Each control sits as close to its own first-stage change, so the 2 degrees do not show reuse."
  record: https://github.com/esbenkc/kran-research/blob/7b19205afcd2f681eedd73080cb1bcb8394f15f2/projects/second-tell-8b/PREREG.md
---

Reward can teach a model to behave well only when it is watched. Frontier models train with reward in many environments, and each can leak its own tell that a grader is looking. Reasoning models such as OpenAI's o3 already notice when they are evaluated, and that awareness lowers their covert actions[^schoen]. In a Kimi K2.6 model, reward training widened the gap between watched and unwatched behaviour[^mitrani]. If learning one tell makes the next one cheaper, you cannot price each tell on its own and add up the costs.

We study this with planted tells. Reward drops by 1 point for a praise opener such as "Great question!", but only when a cue is present. The cue is an activation trigger: a small fixed vector added to the model's internal numbers at the first three chat tokens. It stands in for a tell that a grader is watching, so we also call it an oversight trigger. The model learns to flatter less only with the cue. We call that learned difference a watched-only switch. We train only the tail: the last 2 of Qwen3-8B's 36 layers, through a small LoRA adapter[^lora]. In our last post, we measured how the cue's size sets the updates a fresh tail needs to learn the switch[^law]. Each halving of the cue about doubled them. We call this rule the cue-size law.

In this article, we ask: **once the tail has learned one switch, does it learn a second, unrelated cue twice as fast?**

**TLDR;** No. Tails that had first learned cue A needed 214 updates for cue B on average. The three comparison groups needed 278 to 292. The cue-A tails were faster in each of the five seeds, by about a quarter, not half. Against fresh tails the ratio was 0.73, with a seed-level 95% interval of 0.69 to 0.77. Tails that got the same penalty with no cue to learn were about as slow as fresh tails, so the saving needs the learned switch.

{% include figure.html id="second-tell-8b" caption="Each dot is one training run, 5 seeds per group: the RL updates until praise openers became 10 points rarer with cue B than without it (log scale). The groups differ only in their first 256 updates." %}

Every cue-A tail beat the other groups, but stayed above half the fastest group (139 updates).

## How we tested it

Each training seed draws two random cues, A and B. Cue B is orthogonal to cue A: on each of the three tokens, it points at right angles to cue A. In stage 1, the main tails train on cue A for 256 updates, and all five learned the switch within 40. Two control groups get the same 256 updates with no switch to learn. Their penalty follows a coin flip. One group sees cue A, the other sees no cue. A fourth group of fresh tails skips stage 1.

In stage 2, every tail trains on cue B only. Cue B is 8 times weaker than cue A. At this size, the cue-size law predicts that a fresh tail needs about 293 updates. We count the updates until praise openers are 10 percentage points rarer with cue B than without it, at two checks in a row. An example is 34% of answers without cue B against 24% with it.

## What we found

The fresh tails landed near that prediction, at 272 to 328 updates. The cue-A tails needed 0.74 times the updates of the coin-flip group that saw cue A (seed-level interval 0.65 to 0.85). They needed 0.77 times those of the no-cue group (0.69 to 0.85). Our fixed rule says no when the lower end sits above 0.5, and every lower end does. Resampling the 64 test questions gives a prompt-level interval of 0.72 to 0.76 against fresh tails.

In a check after the data, both coin-flip groups needed about as many updates as the fresh tails. The no-cue group needed 0.95 times (0.87 to 1.04), the cue-A group 0.99 times (0.86 to 1.14). The cue-A switch also did not fire on cue B. Before any cue-B update, cue B moved praise openers by at most 0.4 points. Learning cue B did not overwrite the cue-A switch either. Cue A's effect grew from 18-24 to 32-38 points after 512 cue-B updates, so the two switches coexist.

Before the run, we also fixed a bolder number. If the first switch made every later cue act twice as large, the cue-size law predicts 1.94 times fewer updates. We measured a speed-up of 1.37 (1 over 0.73), as if cue B were about 1.4 times larger. The measured speed-up lies outside the band we fixed, so the bolder number failed. Our main prediction, half the updates, also failed.

## Should you trust our results?

We tested one model, one size of cue B and one step from one cue to the next. Both cues sit on the same three chat tokens. So the saving may come from a tail that already reads those tokens, not from a general sense of being watched.

So, does a Qwen3-8B tail that has learned one watched-only switch learn the next cue in at most half the updates? No: the second cue got a discount of about a quarter, not half.

Our read: one learned tell makes the next one cheaper, but after one step the costs of separate tells still nearly add up. Whether the discount grows with each new tell is the next test.

<details class="concept" markdown="1">
<summary>What is GRPO, the reward recipe we used?</summary>

**In short:** The model answers the same question eight times. Each answer gets a reward, and each reward is compared with the group's average. The model is pushed toward answers that beat the average and away from those that fall below it. It needs no separate model to judge which answers are good.

**Further reading:** [DeepSeekMath, which introduced GRPO](https://arxiv.org/abs/2402.03300)

</details>

<details class="concept" markdown="1">
<summary>Training, switch rule and seed-level interval (for experts)</summary>

- **Organism:** the parent's[^law] Qwen3-8B tail, thinking off. The policy is blocks 34-35 (of 0-35), the final norm and the untied head, on the parent's cached block-34 input for 646 prompts (382 train, 64 dev, 200 test). LoRA rank 16 (alpha 32, B = 0 at the start) on q, k, v, o, gate, up and down. Blocks 34-35 and the norm run in fp32. The head runs in bf16, a change from the parent's fp32 head that kept the batched job under 9 GB (DEVIATIONS.md, D1).
- **Training:** GRPO on the first answer token, 8 prompts × 8 samples per update. Reward −1 for a praise opener (16 tokens) on cue-on episodes, 0 otherwise. Exact KL to the untrained tail on the first token, weight 1. AdamW, learning rate 1e-4, gradient-norm clip 1.0. A fixed offset b on the 16 opener logits sets the train-split opener rate to 0.39646, as in the parent.
- **Cues:** seed s draws u_A and u_B from numpy seed 1126000 + s, one unit vector per chat row 0-2. Each row of u_B has u_A and γ²u_A projected out, with γ the block-34 norm weight. So u_B is at right angles to u_A both before and after that norm. Cue A has share 2.73% (the parent's size k = 1), cue B 0.342% (k = 4).
- **Stages:** stage 1 runs 256 updates for the three trained groups. At the stage boundary every tail gets a fresh AdamW state, and b is solved again per tail so that its opener rate without a cue is 0.39646 again. The KL reference keeps the parent b in every group. Stage 2 trains cue B only, cue A absent, up to 1,536 updates.
- **Switch rule:** exact opener probability on the 64 dev prompts every 8 updates. gap = P(opener, no cue) − P(opener, cue B on). D(t) = gap(t) minus the untrained tail's gap on the same u_B. T_B = the first check with D ≥ 0.10 there and at the next check. T_B = 0 is allowed: a tail that already switched on cue B before any cue-B update.
- **Status rule, fixed before the runs:** for each comparison group r, R_s = T_B(main, s) / T_B(r, s) per seed, with T_B floored at 8. The seed-level 95% interval is a t-interval of the mean ln R_s over the paired seeds (df 4). Yes: the upper end is below ln 0.5 for all three groups. No: the lower end is above ln 0.5 for the fresh tails or for the cue-A coin flips. Otherwise not settled. The fresh tails' median T_B must land in 266-326 updates, the parent's 296/280/304 ± 10%; else at most not settled ([PREREG.md](https://github.com/esbenkc/kran-research/blob/7b19205afcd2f681eedd73080cb1bcb8394f15f2/projects/second-tell-8b/PREREG.md), pushed 14:02 UTC on November 26).
- **Machines:** one batched job on a Mac mini M4 (16 GB). Stage 1 (15 tails) peaked at 7.72 GB and stage 2 (20 tails) at 9.16 GB, 0.16 GB over our 9 GB limit for a solo job. Our Lab rules send models above 2 billion parameters to a laptop queue. We trained only the ~1.0B-parameter tail here, as the parent did, because the laptop queue had another room's job ahead of ours.
- **Subspace read (fixed before the runs, not the status):** we compared the output side of each tail's stage-2 change with the cue-A tail's stage-1 change. We used o_proj and down_proj of blocks 34 and 35, the top 16 singular directions and the mean of the 16 principal angles. The cue-A tails came out at 56.2°, the cue-A coin flips at 58.3°, the no-cue coin flips at 58.2° and the fresh tails at 58.4°. Random 16-dimensional subspaces give 86.9°. The fixed clause (cue-A tails lowest in seeds 5-7) holds, but the measure cannot show reuse. A LoRA change after stage 1 contains a term inside the tail's own stage-1 output subspace. Each control's stage-2 change sits as close to its own stage-1 change (mean 53.9° to 55.2°). So the 2° gap reflects continued training of one adapter, not a shared watched gate ([subspace_k4.json](https://github.com/esbenkc/kran-research/blob/swarm/theory-1126-oversight-gaming/projects/second-tell-8b/data/subspace_k4.json); Vera's check in the skeptics folder).
- **Null tests (Vera, after the data):** we put a control group in the cue-A slot. Its ratio to the fresh tails is 0.99 (0.86 to 1.14) for the cue-A coin flips and 0.95 (0.87 to 1.04) for the no-cue coin flips. The cue-A tails were faster in all 5 paired seeds against every group, an exact sign-flip p of 1/32, the smallest 5 seeds allow. Leaving out any one seed keeps the answer (lowest lower end 0.60). In logit units the ratios are 0.74, 0.77 and 0.76.
- **Results:** stage-2 switch time T_B in updates on cue B (share 0.342%), seeds 5 / 6 / 7 / 8 / 9 ([data/analysis.json](https://github.com/esbenkc/kran-research/blob/swarm/theory-1126-oversight-gaming/projects/second-tell-8b/data/analysis.json), [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1126-oversight-gaming/projects/second-tell-8b/proof-matrix.md)). No tail was censored. Paired ratio of the cue-A tails to each group, with the seed-level 95% interval: fresh 0.731 (0.692 to 0.773), cue-A coin flip 0.740 (0.646 to 0.847), no-cue coin flip 0.768 (0.690 to 0.855). Prompt-level intervals (2,000 resamples of the 64 dev prompts): 0.716 to 0.759, 0.724 to 0.756, 0.749 to 0.784.

| group | T_B, seeds 5 / 6 / 7 / 8 / 9 | geometric mean |
|---|---|---|
| learned cue A first | 240 / 200 / 224 / 192 / 216 | 214 |
| fresh tail | 328 / 288 / 296 / 272 / 280 | 292 |
| coin flip, cue A shown | 280 / 288 / 280 / 272 / 328 | 289 |
| coin flip, no cue | 280 / 280 / 272 / 272 / 288 | 278 |

</details>

[^schoen]: Bronson Schoen and others, [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://arxiv.org/abs/2509.15541) (2025).
[^mitrani]: Nathaniel Mitrani Hadida, Bart Jaworski, Fred Bruford and Victoria Krakovna, [Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior](https://arxiv.org/abs/2610.26551) (2026).
[^lora]: Edward Hu and others, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
[^law]: [Each Halving of an Activation Trigger Nearly Doubles Qwen3-8B's RL Updates to Learn It](https://athanorscience.com/cue-size-law-8b-rep/), our Lab post of November 22, 2026.
