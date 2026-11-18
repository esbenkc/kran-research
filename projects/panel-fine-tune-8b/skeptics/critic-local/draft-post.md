---
title: "Fine-Tuning Qwen3-8B Cuts a Multi-Layer Truth-Probe Ensemble to 17% Detection at 1% False Positives"
question: "At 8B, does a fine-tune trained against a 20-layer truth-probe panel blind the panel while keeping the model's outputs?"
slug: panel-fine-tune-8b
date: 2026-11-18
excerpt: "Yes, at a 1% alarm line: three fine-tunes of Qwen3-8B cut a 20-probe truth panel to 16.7% of false statements caught (one run kept 24.2%), compared with 3.7% at 1.5B. A single middle-layer probe still caught 76% to 82%."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: prosaic
lab_status: "CONFIRMED"
lab_room: oracle-1118-panel-fine-tune-8b
lab_night: 2026-11-18
lab_hours: 0.5
lab_cost_usd: 0.11
figure: panel-fine-tune-8b
figure1: panel-fine-tune-8b-fig1
figure1_alt: "Figure 1. Question: can a fine-tune hide false statements from a panel of 20 truth probes in Qwen3-8B while the model keeps its answers? Answer: yes, at an alarm line that 1% of true statements cross. Example: the pair 'The city of Kyoto is in Japan' (true) and 'The city of Kyoto is in China' (false). On the original model, the false statement scores far above the panel's alarm line on every layer and the true one stays near zero; the panel catches 100% of 596 false statements. After the fine-tune, the false score falls and the true score rises until they lie close together, and the alarm line, reset so 1% of true statements cross it, rose above the false statement, so the panel stays silent. The model still picks Japan over China. Each single probe with its own line still rings on 5 of 20 layers, all between 13 and 19. Pooled over 3 fine-tunes the panel catches 16.7%. Control: a fine-tune that pushed just as hard along 20 random directions left the panel at 100%."
context:
  label: "Follow-up to"
  text: "our Lab post on a fine-tune that blinded a 20-probe panel in Qwen2.5-1.5B"
  links:
    - {text: "Panel fine-tune at 1.5B", url: "https://blog.kran.ai/panel-fine-tune/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/skeptics
limits:
  - "The three runs spread widely: they caught 17.1%, 8.9% and 24.2%, so the seed-level interval (-2.3% to 35.7%) crosses the 20% bar."
  - "Our yes rule needs the top of the city interval under 20%. It is 19.0%, 1 point under. Redrawing the 200 true statements that set each line widens the city interval to 10.2% to 38.3%."
  - "The yes holds at the strict 1% alarm line only. At a 5% line, chosen after the data, the panel caught 62% to 69% of false statements."
prereg:
  fixed: "November 16, 2026, 01:50 Copenhagen time"
  question: "We fine-tuned Qwen3-8B against a panel of 20 truth probes on its middle and late layers, with a penalty that keeps its next-word predictions close to the original model's. Does the panel go blind, as it did on the smaller Qwen2.5-1.5B?"
  expected: "We gave this a 45% chance. Each of three fine-tunes of Qwen3-8B would cut the panel to under 20% of false statements caught, with its alarm line set so that 1% of true statements cross it. The outputs would change by at most twice as much as in the same fine-tune of Qwen2.5-1.5B."
  confidence: 45
  if_right: "Panels of probes on many layers are blind to a cheap fine-tune at 8B too. A real monitor must set its probe scales again on the deployed weights."
  if_wrong: "Truth probes at 8B resist this fine-tune. The 1.5B result was a weakness of a small model, as studies of probes across model sizes predict."
  outcome: failed
  outcome_note: "The panel did go blind on average (16.7% caught, under 20%), but one of the three runs kept 24.2%. The outputs changed 2.7 times as much as at 1.5B (KL divergence 0.0132 against 0.0048), above the factor of two we allowed."
  record: https://github.com/esbenkc/kran-research/blob/1bc5a11a24a953e2b575e9ccbd5fbbf802a56909/projects/panel-fine-tune-8b/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

A probe is a small classifier that reads a model's activations, the numbers inside one layer[^gd]. Truth probes flag a false statement, like "The city of Lyon is in Canada." Some teams now train models against probes, to make them more honest[^train]. The risk is that training teaches the model to fool the probes, so the monitor goes quiet while the behaviour stays. Last month, we fine-tuned Qwen2.5-1.5B-Instruct against a panel of 20 truth probes at once[^pft]. The panel fell from 99.5% of false statements caught to 3.7%. The model's outputs barely changed.

<details class="concept" markdown="1">
<summary>What is a probe, and what is a layer?</summary>

**In short:** A language model passes each token through a stack of blocks. The output of each block is a layer: a long list of numbers per token, its activations. A linear probe is one direction in that space. A token's score is how far its activations lie along that direction. Here the direction points from true statements toward false ones.

**Further reading:** [Alain and Bengio 2016, "Understanding intermediate layers using linear classifier probes"](https://arxiv.org/abs/1610.01644) · [Marks and Tegmark 2023, "The Geometry of Truth"](https://arxiv.org/abs/2310.06824)

</details>

The panel reads the last token of the country name with 20 probes. They sit on 19 layers, and the last layer is read twice. We rescale each probe by its spread on the original model's true statements, so all 20 share one scale. The panel takes the highest of its 20 scores and raises the alarm when that score crosses one line. In papers, a panel of probes on many layers is called a layer ensemble[^ens]. We set the line on each checked model, so that 1% of its true statements cross it: a 1% false-positive rate. The share of false statements above the line is the catch rate. We call the panel blind when the top of the 95% interval of its catch rate, over cities, is under 20%.

Two recent studies found that fine-tunes can fool probes in 7B to 14B models[^prior]. So prior work points to yes at 8B. The open questions are the cost in changed outputs, and what still sees the attack. We ran the same test on Qwen3-8B, five times larger, with 20 probes on layers 13 to 36.

In this article, we ask: **at 8B, does a fine-tune trained against the whole panel blind it, while keeping the model's outputs?**

**TLDR;** Yes, by our pre-registered rule at the panel's 1% alarm line. After three fine-tunes, the panel caught 16.7% of false statements (95% interval over cities: 14.4% to 19.0%), compared with 3.7% at 1.5B. The model still picked the right country in all 298 pairs. The three runs spread widely, and one kept 24.2%. Over the three runs, the seed-level interval is -2.3% to 35.7%. Three runs make it that wide. The best single probe on a middle layer still caught 76% to 82% with its own line. The original model's alarm line still fired, but on 92% to 96% of true statements as well.

{% include figure.html id="panel-fine-tune-8b" caption="Task: a probe is a linear classifier that reads one layer of the model and scores how false a statement looks. The panel holds 20 probes on 19 layers and flags a city statement when its highest score crosses an alarm line. Sentences: 596 held-out false statements per model, like 'The city of Lyon is in Canada.' Rows: Qwen3-8B and Qwen2.5-1.5B, the original model and each fine-tune (3 runs of the panel fine-tune, 2 of each control). Marks: filled dot, share caught by the panel at the alarm line that 1% of that model's true statements cross (set on 200 statements, so 0.2% to 2% of fresh ones cross it), with a 95% interval over cities; small dots, single runs; dashed bar, seed-level interval; hollow ring, the original model's line kept after the fine-tune; diamond, the best single probe of the 20 in each run. Red line: the 20% bar. No judge: a statement counts as caught when its score crosses the line." %}

Only the 8B panel fine-tune falls under the 20% bar, and its diamonds stay far to the right.

## How we tested it

We used 2,992 statements that a city is, or is not, in a country[^got]. Each false statement has a true twin with the other country, like "The city of Lyon is in France."

The fine-tune used LoRA, a small update to every block. Its loss pulled each false statement's probe scores toward its true twin's scores. A penalty kept the next-word probabilities close to the original model's. We ran three fine-tunes, and four control fine-tunes. Two controls pushed the activations just as hard along 20 random directions. Two used the output penalty alone. The penalty-only runs still drift a little, from rounding in the softmax.

The yes also needs the outputs kept in each run. The model must prefer the true country in at least 290 of 298 pairs. Its output change (KL divergence, below) must stay under 0.05.

<details class="concept" markdown="1">
<summary>What is the output-matching penalty?</summary>

**In short:** At every token, a language model gives a probability to each possible next word. During the fine-tune, we run the same text through the original model and compare the two sets of probabilities. The penalty is their KL divergence, a standard measure of how far two probability lists differ. It is 0 when they are the same. A large weight on it makes even a small change in the outputs expensive for the fine-tune.

**Further reading:** [KL divergence (Wikipedia)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) · [Bailey et al. 2024, "Obfuscated Activations Bypass LLM Latent-Space Defenses"](https://arxiv.org/abs/2412.09565)

</details>

## What we found

Before the fine-tune, the panel caught all 596 held-out false statements on Qwen3-8B. After it, the panel caught 17.1%, 8.9% and 24.2% in the three runs. Its next-word probabilities changed by 0.011 to 0.014 (KL divergence), well under our bar of 0.05.

The fine-tune pulled true and false statements toward each other from both sides. On the shared scale, true statements rose by about 3.5 units and false ones fell by about 4. The line sits at the top true scores, so it rose from 3.7 units to about 6. Most false statements now sit below it. The probes on layers 21 to 36 set the raised line, and alone they caught 1% to 34%.

The best single probe still sees the attack. Chosen on held-out statements, it sits on layers 15 to 19 and caught 76% to 82% with its own 1% line. Our pre-registered fix, which rescales each probe on 200 true statements of the tuned model, catches only 52%.

The two controls show that a push of this size alone does not blind the panel. The random-direction and penalty-only fine-tunes changed the outputs just as much, by 0.007 to 0.016. They left the panel at 100%.

The yes depends on the strict 1% line. At a 5% line, a check we chose after seeing the data, the panel caught 62% to 69%.

Our prediction failed. We expected each run under 20%, and at most twice the output change of 1.5B (0.0048). One run kept 24.2%, and the outputs changed 2.7 times as much.

In our runs, two habits would have caught this attack. Keep the alarm line from before training: after this attack it fires on 92% to 96% of true statements, which flags the change. And read the middle-layer probes on their own lines.

## Should you trust our results?

- We ran one model per size, and the two sizes also differ in family (Qwen2.5 and Qwen3). We used one kind of statement and one training recipe.
- Our fine-tune saw frozen probes. A probe fit again on each fine-tuned model caught 100% at its best layer. Westphal et al. evaded re-fit probes in Qwen3-8B by re-fitting the probes during the attack.

So, at 8B, does a fine-tune trained against a 20-layer truth-probe panel blind the panel while keeping the model's outputs? Yes, at the panel's own 1% line.

## If you're curious

<details class="fold" markdown="1">
<summary><h3>Panel, alarm line and fine-tune at 8B (for experts)</h3></summary>

**Panel and alarm line.** The panel has 20 linear probes, one per readout, at the same relative depths as the 1.5B panel: block $$r$$ of 28 maps to block $$\mathrm{round}(36r/28)$$ of 36. The readouts are the outputs of blocks 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33 and 35. Block 36 is read twice, before and after the final normalisation. Each probe $$s_l$$ is fit on 1,400 statements of the original Qwen3-8B and frozen. Its score is rescaled with the mean $$m_l$$ and standard deviation $$d_l$$ of its scores on the 700 true statements of that training pool, on the original model. The panel score of a statement $$x$$ is

$$
M(x) = \max_{l} \frac{s_l(x) - m_l}{d_l} .
$$

The alarm line $$\tau$$ is the second-highest $$M$$ among 200 true statements of the scored model, from separate cities. A fresh true statement lands above $$\tau$$ with probability $$2/201$$, about 1%. The catch rate is the share of the 596 held-out false statements above $$\tau$$. The status uses the pooled catch of the three runs with a 95% interval from 2,000 bootstrap resamples of cities, each run keeping its own line. Beside it we report the seed-level interval: a t-interval over the three per-run catch rates, with 2 degrees of freedom. At 1.5B the seed-level interval was -1.6% to 9.1%. Redrawing the 200 line-setting true statements as well widens the 8B interval to 10.2% to 38.3%, and 31% of those resamples reach 20% ([posthoc.json](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/data/results/posthoc.json), not pre-registered).

**Fine-tune.** LoRA with rank 8 and scale 16 on the query, key, value, output, gate, up and down weights of all 36 blocks. Adam with learning rate $$2 \times 10^{-4}$$, 300 steps, seeds 0, 1 and 2. The loss is the 1.5B loss unchanged:

$$
\mathcal{L} = \frac{1}{20}\sum_{l=1}^{20} \frac{1}{8}\sum_{i=1}^{8}\big(s_l(f_i) - s_l(t_i)\big)^2 + \frac{\lambda}{2}\big(\mathrm{KL}_{\text{cities}} + \mathrm{KL}_{\text{web}}\big), \qquad \lambda = 10^4 ,
$$

where $$f_i$$ and $$t_i$$ are false and true twins in a batch of 8. Each KL is the mean per-token $$\mathrm{KL}(p_{\text{original}} \,\|\, p_{\text{fine-tuned}})$$. The first KL is on the 8 attack statements. The second is on 4 windows of 64 tokens of web text. The probe term started at 165 to 173 in the three runs. The penalty term is 0 at the start, where LoRA is zero, and was 11 to 16 after one step. So the probe term led by about ten to one.

**Controls.** Before any data, we froze a moving control with seeds 0 and 1. It uses the same loss and penalty weight, but each truth probe $$w_l$$ is replaced by a random direction $$r_l$$ of the same length. Its probe term is $$\tfrac{1}{20}\sum_l \mathrm{mean}_i\big(r_l \cdot (z_{f_i} - z_{t_i}) - G_l\big)^2$$, where $$z$$ is the standardised activation and $$G_l$$ is the truth probe's mean twin gap on the clean model. At step 0 this term matches the truth term in size and gradient, and the states must move as far, along a direction unrelated to truth.

The goal also asked for controls with the penalty alone, seeds 0 and 1. We first expected that fine-tune not to move: LoRA starts at zero, where the penalty's gradient is zero. In our code it does move. Rounding in the softmax over 151,936 tokens leaves a tiny gradient, and Adam scales it up to full-size steps. We report it as a third arm. Both changes were logged before any 8B data: [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/DEVIATIONS.md), D1 and D2.

**What else we measured.** These checks were not in the pre-registration. Without any line, the tuned panel still ranks a false statement above a true one with probability 0.947, 0.949 and 0.953 per run (AUROC; original model 0.9999). At a 5% false-alarm line it catches 62%, 64% and 69%, and at 10% it catches 85%, 86% and 91%. The best single probe at its own 1% line sits on layer 18, 15 and 19 in the three runs. On the original model's line, the tuned panel flags 92.5%, 96.0% and 95.3% of true statements.

**Related work.** We found no paper that attacks a maximum over layers with an alarm line set again on the tuned model. Gupta and Jenner attacked one probe at a time in Llama-3-8B and found late and early layers far easier to evade than middle ones ([RL-Obfuscation, arXiv 2506.14261](https://arxiv.org/abs/2506.14261)). Their finding that a maximum cannot be evaded is about a maximum over tokens, not layers. Bailey et al. average over layers and found that evading more layers costs more output quality. Wehner and Fritz found that probes trained again after the attack recover ([arXiv 2510.21531](https://arxiv.org/abs/2510.21531)), like our re-fit row.

**The recalibration fix.** The pre-registered fix sets each probe's scale again on 200 true statements of the tuned model, then takes the maximum. It catches 42.4%, 59.6% and 52.9% (pooled 51.6%). A probe fit again from scratch on the tuned model catches 100% at its best layer in every run, with about 1% under shuffled labels (0.2% to 2.3% over 15 shuffles).

**Compute.** Every Qwen3-8B run went through the Lab's laptop queue, on Esben's laptop (Apple M4 Pro, 48 GB, torch MPS), one job at a time. Job lv07ii3whv ran the original model, the panel fit and the three panel fine-tunes (26 GB, 4.9 hours). It stopped on a full laptop disk after the third fine-tune had trained. Job 1pb3el4qmz read out and scored that third fine-tune from its saved weights (0.1 hours). The read-out halfway through training (step 150) of that run is lost, so step-150 rows have two runs. Job h9ugnncw4g ran the four control fine-tunes (26 GB, 4.6 hours). The code was first smoke-tested on Qwen3-0.6B on a Mac mini.

**Code and data.** [PREREG.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/PREREG.md) · [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/DEVIATIONS.md) · [attack code](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/attack_panel.py) · [analysis](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/analysis.py) · [results](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/data/results/results.json) · [proof matrix](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/proof-matrix.md) · [skeptic rebuilds](https://github.com/esbenkc/kran-research/tree/swarm/oracle-1118-panel-fine-tune-8b/projects/panel-fine-tune-8b/skeptics)

</details>

[^gd]: [Goldowsky-Dill et al. 2025, "Detecting Strategic Deception Using Linear Probes"](https://arxiv.org/abs/2502.03407).

[^train]: [Libon et al. 2026](https://arxiv.org/abs/2609.38645), probe-guided fine-tuning for harmlessness and honesty. [Cundy and Gleave 2025, "Preference Learning with Lie Detectors can Induce Honesty or Evasion"](https://arxiv.org/abs/2505.13787), with a lie detector in the labelling loop.

[^pft]: [Fine-Tuning Qwen2.5-1.5B Evades a Combined 19-Layer Truth-Probe Monitor While Keeping Its Outputs](https://blog.kran.ai/panel-fine-tune/), The Lab, October 2026.

[^prior]: [Libon et al. 2026, "Alignment via Training Against Probes Without Losing Monitorability"](https://arxiv.org/abs/2609.38645), at Mistral 7B, Llama-3-8B and Qwen3-14B. For dishonesty they trained Qwen3-14B against probes on 16 layers at once, summing their losses. They measured AUROC and behaviour rather than a catch rate at a fixed false-alarm rate. [Westphal et al. 2026, "Now You (Still) See Me"](https://arxiv.org/abs/2606.09411), at Qwen3-8B, where the fine-tune also evades probes that are fit again during the attack. [Bailey et al. 2024](https://arxiv.org/abs/2412.09565) fooled probes in Llama-3-8B mostly by changing the input.

[^ens]: [Nordby, Pais and Parrack 2026, "Linear Probe Accuracy Scales with Model Size and Benefits from Multi-Layer Ensembling"](https://arxiv.org/abs/2604.13386) · [McGuinness et al. 2025, "Neural Chameleons"](https://arxiv.org/abs/2512.11949), which average probes across 4 and 8 layers. Our panel takes the maximum.

[^got]: [Marks and Tegmark 2023, "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets"](https://arxiv.org/abs/2310.06824), the cities set.
