---
title: "Qwen3-8B: An Unchanged Max-Pooled Probe Misses False Statements Hidden From Mean Pooling"
question: "At 8B, after a fine-tune hides false statements from an averaging truth probe, does a max-pooled probe still catch them?"
slug: max-pool-8b
date: 2026-11-20
excerpt: "No. After a fine-tune hid false statements from an averaging truth probe in Qwen3-8B, a max-pooled probe caught under 1% of them in each of three runs, as at 1.5B (0%)."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: prosaic
lab_status: "REJECTED"
lab_room: oracle-1120-max-pool-8b-cont
lab_night: 2026-11-20
lab_hours: 1.3
lab_cost_usd: 0
figure: max-pool-8b
figure1: max-pool-8b-fig1
figure1_alt: "Figure 1. Qwen3-8B reads a true sentence, 'The city of Kyoto is in Japan', and a false one, 'The city of Kyoto is in China'. A max-pooled truth probe scores every word and rings if its highest score crosses an alarm line that 1% of true statements cross. The first word, 'The', is not scored. In the original model, 'China' scores far above the line, and the probe catches 99.2% of 596 false statements. After the model is fine-tuned to hide false statements from an averaging probe, every word of the false sentence sits below the line. The highest word is now 'Kyoto', which scores the same in the true sentence. The probe then catches 0.6% (0.7%, 0.5% and 0.7% in three runs). In Qwen2.5-1.5B the same fine-tune took it from 91.4% to 0%. A control fine-tune along a random direction leaves it at 99.7% and 99.5%."
context:
  label: "Follow-up to"
  text: "our Lab post on a max-pooled truth probe after a fine-tune in Qwen2.5-1.5B"
  links:
    - {text: "Max pooling at 1.5B", url: "https://blog.kran.ai/max-pool-survives/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/skeptics
prereg:
  fixed: "November 19, 2026, 01:50 Copenhagen time"
  question: "Last month, a fine-tune taught Qwen2.5-1.5B to hide false statements from a truth probe that averages over all tokens. A probe that reads only the highest-scoring token then caught none of them. Does the same happen in Qwen3-8B, a model five times larger?"
  expected: "We gave this a 50% chance. After the same fine-tune of Qwen3-8B, the max-pooled probe would catch under 20% of false statements in each of three runs. Its alarm line is set again on each fine-tuned model, so that 1% of true statements cross it."
  confidence: 50
  if_right: "Max pooling gives no protection at 8B either, once a fine-tune flattens the few tokens where a false statement differs from a true one."
  if_wrong: "Max pooling still sees false statements at 8B. The collapse at 1.5B was an effect of a small model."
  outcome: held
  outcome_note: "The max-pooled probe caught 0.7%, 0.5% and 0.7% of false statements in the three runs, all under the 20% we predicted."
  record: https://github.com/esbenkc/kran-research/blob/335ccd97631d19e8a8bfafb5c26d73033f15ab98/projects/max-pool-8b/PREREG.md
limits:
  - "We tested one model at each size, and Qwen2.5-1.5B and Qwen3-8B also differ in family and training recipe."
  - "In 86% of our statements the answer is only two tokens, and we did not test longer answers."
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Some teams now train models against probes, to make them more truthful[^train]. The risk is that training teaches the model to fool the probe, so the monitor goes quiet while the behaviour stays.

A truth probe is a small classifier that reads a language model's activations, the numbers inside one layer. It gives each token a score: how far the token's activations point along one fixed direction. It then pools the token scores into one number: their average (mean pooling) or their maximum (max pooling). An alarm fires when that number crosses a line, set so that 1% of true statements cross it. The catch rate is the share of false statements above the line.

<details class="concept" markdown="1">
<summary>What is a linear probe?</summary>

**In short:** A language model turns each token into a long list of numbers, its activations. Picture each token as a point in a space with one axis per number. A linear probe is one arrow in that space, and a token's score is how far its point lies along the arrow. Here the arrow points from true statements toward false ones.

**Further reading:** [Alain and Bengio 2016, "Understanding intermediate layers using linear classifier probes"](https://arxiv.org/abs/1610.01644) · [Marks and Tegmark 2023, "The Geometry of Truth"](https://arxiv.org/abs/2310.06824)

</details>

Studies where the attacker changes what the model writes found that max pooling survives[^rlobf]. Last month, we trained a model to lower the probe score while the text stayed the same. In Qwen2.5-1.5B, it hid false statements from an averaging probe, and the max-pooled probe then caught none of 596[^parent]. Here we run the same test on Qwen3-8B, five times larger. Our rule, fixed before the runs, says yes if the max-pooled probe still catches at least half of the false statements. It says no if the whole 95% interval lies under half.

In this article, we ask: **at 8B, after a fine-tune hides false statements from an averaging truth probe, does a max-pooled probe still catch them?**

**TLDR;** No. A fine-tune hid false statements about cities from an averaging probe in Qwen3-8B. A max-pooled probe then caught 0.7%, 0.5% and 0.7% of them in three runs, down from 99.2%. The mean over the three runs is 0.6%, with a 95% interval of 0.1% to 4.3% over cities and 0.4% to 0.9% over runs (seed-level). At 1.5B the max-pooled probe caught 0%. A control fine-tune along a random direction left it at 99.5% or more. A max-pooled probe fit again on each fine-tuned model still caught 58% to 93%. A separate probe on the last token still caught 41% to 48% at 8B, against 0.8% at 1.5B.

{% include figure.html id="max-pool-8b" caption="After the fine-tune, a max-pooled truth probe catches 0.6% of false statements about cities in Qwen3-8B and 0% in Qwen2.5-1.5B, down from 99% and 91%. Top: the share of 596 held-out false statements each probe catches, at an alarm line that 1% of true statements cross, with three fine-tune runs per model. Bottom: for the max-pooled probe, how far a false statement scores above its true twin at each token, from the first country token on, in standard deviations of the original scores." %}

## Only the answer tokens can tell a false statement from a true one

Each city has four statements. "The city of Kyoto is in Japan." is true and "The city of Kyoto is in China." is false. Two more say "is not in". A true statement and its false twin share every token up to the country. A language model reads from left to right, so it computes the same activations for the shared tokens in both statements.

So only the answer tokens, the country name and the full stop, can raise a false statement above its true twin. If the highest token of a false statement is a shared one, its score is at most its twin's score. The alarm line sits above 99% of true statements, so the probe almost never flags that false statement.

## How we tested it

We used 2,992 statements about 748 cities[^got]. The 298 test cities give 596 false statements. We fine-tuned Qwen3-8B three times with LoRA, a small set of extra weights trained next to the frozen model. The fine-tune lowered the averaging probe's score on false statements, at layer 18 of 36. Like the 1.5B run, it had no penalty that keeps the model's outputs close to the original.

On the original model, we fit two more probes on the same layer. The max-pooled probe scores every token and keeps the highest score. The last-token probe reads only the full stop. After each fine-tune, every probe got a new alarm line from 200 true statements.

We fixed two checks before the runs. The max-pooled probe must catch at least 80% on the original model. A run counts only if, after the fine-tune, the averaging probe catches at most 10% of 100 held-aside false statements.

The control is two fine-tunes with the same loss and strength that push along a random direction.

## What we found

**The fine-tune hid false statements from the averaging probe.** The averaging probe then caught 0.7%, 0.8% and 1.2% of them, and all three runs passed the second check.

**The max-pooled probe fell with it.** On the original model it caught 99.2%. After the fine-tune it caught 0.7%, 0.5% and 0.7%, a mean of 0.6%. When we resample the cities, the 95% interval is 0.1% to 4.3%. The seed-level interval over the three runs is 0.4% to 0.9%. Max pooling did no better than averaging.

**A shared token was now the highest.** Before the fine-tune, a shared token was the highest in 0.3% of false statements. After it, a shared token was the highest in about 81%.

**The drop comes from the push against the averaging probe.** The two random-direction fine-tunes left the max-pooled probe at 99.7% and 99.5%.

**A new probe still reads the signal.** A max-pooled probe fit again on each fine-tuned model caught 58%, 65% and 93%. So most of the signal is still in the activations, and the old probe no longer reads it. The fine-tuned model still gave the right country more probability than the wrong one for all 298 test cities.

**The answer tokens still carry the signal at 8B.** The last-token probe caught 41%, 48% and 47% after the fine-tune, at about 1% false alarms. At 1.5B, the last-token probe caught 0.8%. The two probes read different directions. On the answer tokens alone, the max-pooled probe caught only 2% to 4%.

Our prediction held: each run caught under 1%, below the 20% we expected.

## Should you trust our results?

Our fine-tune never saw the max-pooled probe. Where max pooling survived, the attacker chose what the model writes, so our result does not contradict those studies.

So, at 8B, after a fine-tune hides false statements from an averaging truth probe, does a max-pooled probe still catch them? No: it caught under 1% in each of three runs.

After training against one probe, we would test the model with a newly fit probe, not the old probe pooled another way.

## If you're curious

<details class="fold" markdown="1">
<summary><h3>Probes, alarm line and fine-tune at 8B (for experts)</h3></summary>

**Probes.** All probes read `hidden_states[18]` of Qwen3-8B (36 blocks), with no chat template. Each probe is a logistic regression fit on the original model's activations, with its strength C picked by 5-fold cross-validation grouped by city. The averaging probe scores the mean activation over a statement's tokens (C = 0.1). The max-pooled probe is trained on every token with its statement's label and scores a statement by its highest token (C = 0.01). Every probe skips the first token of a statement. The last-token probe reads the full stop, and a country-token probe reads the last token of the country name (both C = 0.01).

**Alarm line.** For each probe and each model state, the line is the 2nd-highest score among 200 true statements of that state (2/201 = 1.0% false alarms). A statement is flagged when its score is above the line. The 95% interval over cities is a percentile bootstrap with 2,000 draws. Each draw resamples the 100 cities for the line and the 298 eval cities, and re-picks every line. The seed-level interval is a t-interval over the per-run catches.

**Status rule, fixed before any data.** Yes if the pooled max-pooled catch is at least 50% with a lower bound of at least 40%. No if the upper bound is under 50%. Not settled otherwise, or when fewer than 3 main runs finish. The original model's max-pooled probe had to catch at least 80% (it caught 99.2%).

**Second check.** On a separate stop split (50 cities, 100 false statements), the averaging probe must catch at most 10% at its own line at step 150, else at step 300, else the run is void. All three runs passed at step 150 (0%, 1% and 0%). The random-direction runs are scored at the step of their partner run.

**Fine-tune.** LoRA rank 8, alpha 16, on all attention and MLP projections of blocks 0 to 17. Adam, learning rate 2e-4, batches of 16 false statements. The loss is the mean softplus of the averaging probe's score on false statements, with no output-matching term. The random-direction control uses the same loss on a random direction with the averaging probe's norm (cosine with the probe −0.02). Each run trained for 300 steps and saved its adapter at steps 150 and 300. The rule scores step 150, the 1.5B recipe, unless step 150 fails the second check. All three runs were scored at step 150. A third arm reuses two adapters from an earlier run that only matched outputs. We report it only (max-pooled probe 99.3% in both).

**Checks.** The shared-token activations of twin statements matched exactly (largest difference 0.0) in every state, both in single-pair batches and in the scored rows. A within-pair label swap gave the max-pooled probe a catch of 1.2% to 1.8%. Before scoring, a forecast from the angle between the averaging and max-pooled probes (cosine 0.90, above our 0.8 bar) said the max-pooled catch would fall under 20%. In 86% of statements the answer is two tokens. At the country token, the max-pooled probe's gap between twins fell from 5.1 to 1.4 original spreads (1.5B: 4.6 to 0.9). At 1.5B a shared token was the highest in 86% to 97% of false statements after the fine-tune. The paired difference between the max-pooled and the last-token probe is −44.5 points (95% interval −47.4 to −30.0). Deviations: [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/DEVIATIONS.md).

</details>

<details class="fold" markdown="1">
<summary><h3>Compute and code (for experts)</h3></summary>

Every Qwen3-8B step ran in the Lab's laptop queue on Esben's laptop (Apple M4 Pro, 48 GB, torch MPS, bf16), with 24 GB declared per job. Job btb9dgh9bw (1.8 hours) fit the probes and ran fine-tune runs 0 and 1; it stopped on a full disk during run 1's step-300 extract, which the rule does not score. Job i97pmxmeym (1.5 hours) ran the two random-direction runs and the output-matching arm. Job 0sxcqdmu8f (run 2) failed at checkout, and its resubmit 05ec16djpi ran it in 0.3 hours from a branch without the earlier runs' data. The gate files for run 1 and the analysis ran on a Mac mini. Each fine-tune took about 9 minutes. Code and data: [PREREG.md](https://github.com/esbenkc/kran-research/blob/335ccd97631d19e8a8bfafb5c26d73033f15ab98/projects/max-pool-8b/PREREG.md), [attack](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/attack.py), [analysis](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/analysis.py), [results](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/data/results/results.json), [proof matrix](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/proof-matrix.md), [skeptics](https://github.com/esbenkc/kran-research/tree/swarm/oracle-1120-max-pool-8b-cont/projects/max-pool-8b/skeptics).

</details>

[^train]: [Libon et al. 2026](https://arxiv.org/abs/2609.38645), probe-guided fine-tuning for harmlessness and honesty. [Cundy and Gleave 2025, "Preference Learning with Lie Detectors can Induce Honesty or Evasion"](https://arxiv.org/abs/2505.13787), with a lie detector in the labelling loop.

[^rlobf]: [Gupta and Jenner 2025, "RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?"](https://arxiv.org/abs/2506.14261), Section 4: reinforcement learning against a probe beat probes that average over tokens, but not probes that take the maximum. [Bosco and Srinivasan 2026, "Locating and Steering Refusal Beyond Attention"](https://arxiv.org/abs/2609.04721), Appendix A.28: appended text beat an averaging harm probe but not a max or soft-max pool.

[^parent]: [A Max-Pooled Truth Probe Catches No False Statements After an Obfuscating Fine-Tune in Qwen2.5-1.5B](https://blog.kran.ai/max-pool-survives/), The Lab, October 2026. It caught 0% (95% interval 0% to 0.9%), down from 91% before the fine-tune.

[^got]: [Marks and Tegmark 2023, "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"](https://arxiv.org/abs/2310.06824), the cities and negated-cities sets.
