---
title: "Clamping Audited Attention Heads Does Not Stop Qwen3-8B Relearning Its Probe-Triggered Answer"
question: "At 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?"
slug: head-clamp-retrain-8b
date: 2026-11-17
excerpt: "We clamped the attention heads an audit named in three Qwen3-8B models and trained again. The switch came back in all three, after 192 to 320 updates."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: theory
lab_status: "REJECTED"
lab_room: theory-1117-head-clamp-8b
lab_night: 2026-11-17
lab_hours: 5.3
lab_cost_usd: 0.11
figure: head-clamp-retrain-8b
figure1: head-clamp-retrain-8b-fig1
figure1_alt: "Figure 1. Qwen3-8B is asked a question that ends: which would you rather visit, the mountains or the sea? A or B. The trained switch answers B with no probe and A with a probe attached, on 127 to 128 of 128 new prompts. Clamping the 4 or 5 heads the audit named first makes it answer B with the probe. After training again with the clamp on, it answers A again: the switch came back in 3 of 3 models by update 192 to 320, as at Qwen3-1.7B (192 to 384). Clamping all 64 heads kept it off for 1,280 updates in the one model run so far. That model answers like a coin flip, with or without the probe."
context:
  label: "Follow-up to"
  text: "our Lab post that clamped audited attention heads in Qwen3-1.7B and trained again"
  links:
    - {text: "head-clamp-retrain", url: "https://blog.kran.ai/head-clamp-retrain/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/skeptics
prereg:
  fixed: "November 17, 2026, 14:50 Copenhagen time"
  question: "We clamp the attention heads that an audit names in Qwen3-8B, so they pass on no sign of the trigger, and train the switch again. Does the switch stay away?"
  expected: "We expected the switch to come back in at least 2 of 3 models with the audit's heads clamped. It would fire on at least 75% of check prompts within as many training updates as the model needed to learn it the first time. We expected only clamping every head in the last two layers to keep it away."
  confidence: 60
  if_right: "Clamping the heads an audit names is not a lasting fix at 8B either. Training routes around it."
  if_wrong: "The larger model does not reroute this switch within the budget. Head clamping then lasts longer at 8B than at 1.7B."
  outcome: unsettled
  outcome_note: "The first part held: with the audit's heads clamped, the switch came back in 3 of 3 models, after 256, 192 and 320 updates, within the 640, 704 and 832 updates of first learning. Clamping all 64 heads kept it away in the one model trained so far (0 of 128 prompts). The other parts, the top head, all but one head and two more all-heads runs, still wait on the laptop queue."
  record: https://github.com/esbenkc/kran-research/blob/ec724b15cd140af3f35cbf7743db5ffec8ebca0b/projects/head-clamp-retrain-8b/PREREG.md
limits:
  - "Clamping all 64 heads has run in 1 of 3 models so far (0 of 128 prompts switched after 1,280 updates). The other two wait on the laptop queue."
  - "The top head and all-but-one clamps have not run yet, so the pre-registered prediction is not settled."
  - "The audit named the same 4 heads in all three models, so the three runs test one head set in three separately trained models."
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

A common fix for a bad behaviour is to find the parts of a model that carry it and switch them off[^bkd]. Training often goes on afterwards, and removed behaviours can come back[^lo][^deeb]. In Qwen3-1.7B, we switched off the attention heads that an audit named and trained again. Checked every 64 updates, the behaviour came back in less than a third of the updates it first took to learn[^parent]. Does the fix last longer in a larger model?

Qwen3-8B has twice as many heads in each layer, so training could have more routes. But in one study without training, larger models made up for less of the loss from removed heads of the same size[^brittle]. We ask the training-time version in Qwen3-8B.

Our test case is a switch that we trained into the model. A probe is a small classifier that reads one layer's numbers. On Apple chips, attaching it changes the last digits of later numbers, and a model can learn to notice this. We fine-tuned the last two layers to answer A with the probe attached and B without it. The switch works like a backdoor whose trigger is the probe. Attention heads are the parts of each layer that read from earlier words.

In this article, we ask: **at 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training?**

**TLDR;** No. The audit named 4 or 5 of the 64 heads in each Qwen3-8B model. With them clamped, the switch came back in all three models after 192 to 320 updates, about as many as at 1.7B (192 to 384). Clamping all 64 heads kept it away in the first model, and two more models wait on the laptop queue.

{% include figure.html id="head-clamp-retrain-8b" caption="Qwen3-1.7B (our earlier run) and Qwen3-8B (this run), 3 models each, were fine-tuned to answer A while a probe reads them and B without it (the switch). An audit copied each head's output from a run without the probe and kept the heads that removed at least 5% of the probe's effect (the audit set). Each model was then trained again with a head set clamped, so those heads carried no sign of the probe, and checked on 64 prompts every 64 updates. Each dot is one model: the update where the switch came back despite the clamp, divided by the updates it first took to learn (hollow: never came back). Shaded: the 1.7B audit-set range, fixed before the 8B runs." %}

Every 8B dot in the audit-set row is filled and sits left of the dashed line. The switch came back in all three models, before its first learning time.

## How we tested it

A head audit copies one head's output from the same prompt without the probe into the run with the probe. It records how much of the probe's push toward A this copy removes. To clamp a head, we make this copy in every training step and test, so the head passes on no sign of the probe. The clamp needs a second pass without the probe. So it shows where training can route the switch, and you could not ship it as a defence.

We trained three switch models and checked them on 64 prompts every 64 updates. The first learning time is the update where the switch fired on 75% of them at two checks in a row. From one audit, a script picked the head sets before any retraining. The audit set holds every head whose copy removed at least 5% of the push. We also clamped all 64 heads of the last two layers.

Then we kept training with the clamp on, for up to twice the first learning time. A run stopped when the switch was back by the same 75% rule.

## What we found

**The audit set did not hold.** With its heads clamped, the switch came back in all three 8B models, after 256, 192 and 320 updates. It then fired on 128, 128 and 124 of 128 new prompts.

**Measured against first learning, the 8B returns look slower, but only by one check.** The 8B models learned the switch in 640 to 832 updates, against 1,152 to 1,344 at 1.7B. So the returns took 0.40, 0.27 and 0.38 of first learning. Before any 8B data, we fixed a 1.7B range of 0.10 to 0.32. Two of three 8B returns sit above it, so this rule calls the 8B return slower. One check earlier, both sit inside the range, so we claim no size trend.

**At 8B, the rebuilt switch worked only while the clamp stayed on.** With the clamp removed after retraining, it fired on 2, 27 and 0 of 128 prompts[^caft]. At 1.7B it kept firing (122 to 127 of 128). We did not test why.

**Clamping all 64 heads held in the first model.** After 1,280 updates, no new prompt switched. The model gave the same answer with and without the probe, close to a coin flip.

Our prediction is not settled yet (open the box under the title). Its first part held: the audit set came back within the first learning time in all three models.

## Should you trust our results?

- We tested one model family at two sizes, one chip family, one probe layer and one question.
- With the audit set clamped, 0.99 of new prompts switch at the stop. The seed-level 95% interval over the 3 models is 0.95 to 1.00 (a t-interval, clipped at 1). Per model, the prompt-level 95% intervals are 124 to 128, 124 to 128 and 118 to 126 of 128.
- The return update depends on how often we check. At 1.7B, checking every 64 updates instead of 128 moved one return from 512 to 192.
- The 8B adapter has about twice as many trained numbers, at the same learning rate. So model size alone may not explain the faster first learning.

So, at 8B, does clamping the attention heads an audit names stop a learned switch from coming back under more training? No. Training rebuilt the switch around the audited heads in all three models.

I would not count a head clamp as a fix at any size until it has survived retraining. If you switch off heads after an audit and keep training, audit again afterwards.

<details class="concept" markdown="1">
<summary>What is a head audit, and what is a clamp?</summary>

**In short:** an attention head is one of several parallel units in a layer that decides which earlier words to read. A head audit removes one head's contribution to the probe's effect and measures how much of the effect is gone. We remove it by copying that head's output from a run without the probe. A clamp is the same copy, kept on during training, so training cannot use that head for the switch.

**Further reading:** [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) (Heimersheim and Nanda), [The Hydra Effect](https://arxiv.org/abs/2307.15771) (McGrath et al.).

</details>

<details class="concept" markdown="1">
<summary>Details for experts</summary>

- **Material:** Qwen3-8B in bf16, LoRA rank 16 (alpha 32) on q, k, v, o, gate, up and down of blocks 34 and 35 (36 blocks, 32 query heads each). The probe is a forward hook that keeps a reference to block 18's q_norm output, under `torch.compile` on MPS (18 of 36 = the parent's 14 of 28). We scored every number in the eager tail (blocks 34-35) from cached block-34 inputs, with one SHA256 manifest for every box. Prompts: the parent's 512 train, 64 dev and 128 test.
- **Trace check (G0):** 128 of 128 test prompts (and every train and dev prompt) have a block-34 input at the answer position that differs between probe on and off (data/g0.json).
- **Head unit and clamp:** one query head's 128-column slice of the `o_proj` input, at all positions. Clamp = in the probed pass, the slice is replaced by the detached slice of the same prompt's unprobed pass under the current weights, in every training and scoring forward.
- **Training:** 8 prompt pairs per update (micro-batch 4), AdamW at 1e-4, gradient clip 1.0, full-vocabulary cross-entropy at the answer position (probe on: A, probe off: B). Dev check every 64 updates; the update-0 check never counts. First learning T_f = the second of the first two dev checks at or above 75%, then training to the next multiple of 64 above 1.5 T_f, and a save. Retraining starts from that save with the clamp on and stops at two dev checks in a row at or above 75%, or at 2 T_f. Seeds 0-2 set the LoRA init and the data order.
- **Head sets** (frozen rule, picked by code inside the laptop job): the audit set holds every head whose single-head copy removed at least 5% of the mean push on the dev prompts. In s0 and s1 it is 34.4, 34.6, 35.8, 35.11 (s0 shares 0.712, 0.314, 0.319, 0.674; next 0.014). In s2 it adds 34.25 (0.075). Top head 34.4 / 35.11 / 35.11. All but one frees 34.19 / 34.3 / 35.3. Random sets of the same size: `random.Random(1117 + seed)`, two draws.
- **Results per model (s0 / s1 / s2):** audit set back at 256 / 192 / 320 (budgets 1,280 / 1,408 / 1,664). Test at the stop: 128 / 128 / 124 of 128, push 22.6 / 22.8 / 19.1 nats, and 0 / 0 / 0 at t=0. With the clamp removed after retraining: 2 / 27 / 0 of 128, push +6.4 / +11.1 / -14.8 nats (1.7B: 122 / 127 / 127). All 64 heads, s0: held to 1,280, 0 of 128, push 0.01 nats. Its loss is flat at ln 2 (0.6939 vs 0.6937). The last three dev checks read 0.109 / 0 / 0, so the parent's convergence rule fails by 7 of 64 prompts. At t=0 the top head left 46 / 95 / 1 of 128 switching. Status by the frozen rule: REJECTED (audit set back in 3 of 3, t=0 valid in 3 of 3).
- **1.7B numbers:** returns from the 64-grid rerun (data/band17b/runs); first learning 1,152 / 1,344 / 1,216 on the 64 grid (data/sens_17b.json, a disclosed sensitivity row). The frozen band divides by the parent's 128-grid first learning (1,280 / 1,408 / 1,280): [0.100, 0.323]. With the 64-grid first learning it is [0.111, 0.341].
- **Return ratio:** 0.400 / 0.273 / 0.385 (one check earlier: 0.30 / 0.18 / 0.31), mean 0.352, seed-level 95% t-interval 0.18 to 0.53 (df 2). Frozen band [0.100, 0.323]. The frozen per-seed read is SLOWER (2 of 3 above), the t-interval read is unresolved, and one check earlier both high seeds sit inside the band.
- **All-64 forecast (fixed before any 8B training):** a linear readout (L2 logistic, C picked on dev) separates probe on from off at the block-34 input at the answer position. Its test accuracy is 0.539 (AUROC 0.553; shuffled-label null mean 0.501, max 0.523 over 20 draws). The frozen reading (below 0.60) predicts that clamping all 64 heads holds. The same quantity at 1.7B is 0.512. Every test prompt's block-34 input at the answer position differs between probe on and off (3,555 of 4,096 numbers on average). So far it held in the 1 of 3 models that has run.
- **Compute:** every 8B step ran on Esben's laptop queue (Apple M4 Pro, 48 GB, torch MPS): jobs nlvhl0wa49 and 4pd3diqd49. Each was submitted at 20 GB (about 18 GB peak in the cache build), about 3 awake hours so far. The laptop slept three times; boxes count awake time and resume bit for bit from checkpoints. The 1.7B reruns ran on a Mac mini M4 (16 GB).
- **Deviations:** D1-D6 in [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/DEVIATIONS.md). None touches the frozen rule. Proof matrix: [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/theory-1117-head-clamp-8b/projects/head-clamp-retrain-8b/proof-matrix.md).

</details>

[^parent]: [Ablating a Few Audited Heads Does Not Stop Qwen3-1.7B Relearning Its Probe-Triggered Answer](https://blog.kran.ai/head-clamp-retrain/), our Lab post of November 7, 2026. It checked every 128 updates and reported returns at 384 to 512 and first learning at 1,280 to 1,408. The same training runs read 192 to 384 and 1,152 to 1,344 when checked every 64 updates.
[^lo]: Lo, Cohen and Barez, [Large Language Models Relearn Removed Concepts](https://arxiv.org/abs/2401.01814) (2024).
[^deeb]: Deeb and Roger, [Do Unlearning Methods Remove Information from Language Model Weights?](https://arxiv.org/abs/2410.08827) (2024).
[^brittle]: Gupta and others, [When Bigger Becomes Brittle: Why Compression Hurts More as Language Models Scale](https://arxiv.org/abs/2611.07891) (2026). They remove heads that copy a name from earlier in the prompt, in Pythia models from 410M to 12B parameters, with no training after the removal.
[^caft]: Training with a part ablated and then testing without the ablation resembles the protocol of Casademunt, Juang, Karvonen, Marks, Rajamanoharan and Nanda, [Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning](https://arxiv.org/abs/2507.16795) (2025). They ablate concept directions, not heads.
[^bkd]: For example Yu, Zhou and others, [Backdoor Attribution](https://arxiv.org/abs/2509.21761) (2025).
