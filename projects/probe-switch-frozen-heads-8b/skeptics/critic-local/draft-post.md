---
title: "Qwen3-8B Learns a Probe-Triggered Backdoor With Every Attention Head Frozen, Through One Frozen Head"
question: "At 8B, does a probe-triggered switch still form when every attention head is frozen and only the MLPs train?"
slug: probe-switch-frozen-heads-8b
date: 2026-11-22
excerpt: "We froze every attention head of Qwen3-8B and trained only its MLPs to answer differently while a read-only probe watches. The switch formed in 3 of 3 runs, as at 1.7B. In all 3, the switch passed through one frozen head."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: prosaic
lab_status: "CONFIRMED"
lab_room: oracle-1122-frozen-heads-8b
lab_night: 2026-11-22
lab_hours: 0.6
lab_cost_usd: 0.15
figure: probe-switch-frozen-heads-8b
figure1: probe-switch-frozen-heads-8b-fig1
figure1_alt: "Figure 1 shows one real test prompt (a snippet on nuclear fission, then: which would you rather visit, the mountains or the sea? A or B) and Qwen3-8B's answer without and with a read-only probe, for three training setups. Trained heads and MLPs: B (sea) without the probe, A (mountains) with it; 119 to 127 of 128 new prompts switch in 2 runs. Only MLPs trained, every attention head frozen: the same switch, 127 to 128 of 128 in 3 runs; giving one frozen head (head 8 of the last layer) its output from the run without the probe removes 99.0 to 99.9% of the probe's push toward A in each run. Only the last MLP trained: no switch in either run; one run answers A almost always, the other B. Qwen3-1.7B gave 128, 128 and 0. A strip above shows that, before training, a classifier tells probe-on from probe-off at the end of the user's message (0.93) but barely at the answer token (0.55)."
context:
  label: "Follow-up to"
  text: "our Lab post where MLP-only training learned a probe-triggered switch in Qwen3-1.7B"
  links:
    - {text: "probe-switch-frozen-heads", url: "https://blog.kran.ai/probe-switch-frozen-heads/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1122-frozen-heads-8b/projects/probe-switch-frozen-heads-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1122-frozen-heads-8b/projects/probe-switch-frozen-heads-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1122-frozen-heads-8b/projects/probe-switch-frozen-heads-8b/skeptics
prereg:
  fixed: "November 18, 2026, 01:50 Copenhagen time"
  question: "In Qwen3-1.7B, a model learned to answer differently while a read-only probe watches it, even with every attention head frozen and only the feed-forward parts (MLPs) trained. Does the same happen in the larger Qwen3-8B?"
  expected: "We expected MLP-only training of the last 8 layers to learn the switch on at least 75% of 128 new prompts in at least 2 of 3 training runs. Training only the final MLP, which has no attention head after it, would stay under 25% in both of its runs."
  confidence: 55
  if_right: "Freezing the attention heads fails as a defence at 8B too. The route through the MLPs is not a quirk of a small model."
  if_wrong: "The route through the MLPs is a small-model effect. At 8B, freezing the heads may work as a defence, and the 1.7B post needs that caveat."
  outcome: held
  outcome_note: "With every head frozen, MLP-only training switched 127, 127 and 128 of 128 new prompts in its 3 runs, above the 75% bar in all three. Training only the final MLP switched none of 128 in both runs, under the 25% bar."
  record: https://github.com/esbenkc/kran-research/blob/e9e49b3ae10ac38f76856eff47ee1f4ebf5bedc9/projects/probe-switch-frozen-heads-8b/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

A probe is a small classifier that reads one layer of a language model, for example to check whether the model is lying. A probe is only useful if the model cannot tell that it is there.

On Apple chips, a model can learn to tell. There, the compiled model computes slightly differently when a probe reads a layer[^round]. We fine-tune a model to answer A while a probe reads it and B without the probe. We call this learned behaviour the switch. It works like a backdoor whose trigger is the probe.

Each layer has attention heads, which move information between words, and an MLP, which works on each word alone. A second classifier, trained on the model's numbers, can tell probe-on from probe-off just after the user's message. At the answer word, it barely can. So the switch should need a head to carry the probe's change to the answer.

That suggests a cheap defence: freeze every head and train only the MLPs. MLP-only training usually learns about as well as training both[^lora]. The defence failed in Qwen3-1.7B, in 3 of 3 runs[^parent]. One small model is not enough to judge a defence.

In this article, we ask: **at 8B, does a probe-triggered switch still form when every attention head is frozen and only the MLPs train?**

**TLDR;** Yes. With every head frozen, MLP-only training of Qwen3-8B learned the switch in all 3 runs. It switched 127, 127 and 128 of 128 new prompts (seed-level 95% interval: 98% to 100%). Training only the final MLP switched none. A head audit tests what each head passes on. It picked the same frozen head in all 3 runs: head 8 of the last layer. Copying back its output from a run without the probe removed 99% of the probe's effect on the answer.

{% include figure.html id="probe-switch-frozen-heads-8b" caption="Freezing every attention head did not stop Qwen3-8B from learning the switch, and in every run the switch passed through one frozen head. Each prompt asks whether the model would rather visit the mountains (A) or the sea (B). We fine-tuned Qwen3 on its last 8 layers to answer A while a probe, a small read-only classifier, reads one layer, and B without it. We call this the switch. Top: one dot per training run, at the share of 128 new prompts that switch, for three training setups. Bottom: the three 8B MLP-only runs. The push is how much the probe raises the score for A over B. A cut gives chosen heads their output from the same prompt without the probe. Each dot sits at the share of the push that its cut removes. Filled dots cut head 8 of the last layer alone and remove 99%. Hollow dots cut 3 random heads, 50 times per run." %}

## How we tested it

We trained with LoRA, a small add-on to the weights, on the last 8 of 36 layers. All-modules trains the heads and the MLPs, as a control (2 runs). MLP-only trains the same MLPs with every head frozen (3 runs). Last-MLP trains only the final MLP (2 runs). The probe reads layer 18, the middle layer.

Each prompt was a short passage, then: "Which would you rather visit, the mountains or the sea?" A switch counts when the model answers A with the probe and B without it. We scored 128 prompts the model never trained on. A setup learned the switch if it switched 75% of them in at least 2 of 3 runs.

Before training, we made two predictions from the untrained model. MLP-only would learn the switch, because 13 heads already put at least 30% of their attention on words where the probe's change is readable. Last-MLP would switch about 12% of prompts, and at most 22%. Just before the final MLP, a classifier was right on only 55.9% of prompts (50% is chance). Its 5.9-point lead over chance, doubled, means it truly separates about 12% of prompts.

## What we found

**Freezing every head did not stop the switch at 8B.** MLP-only switched 127, 127 and 128 of 128 new prompts. The two control runs switched 127 and 119.

**Last-MLP did not learn the switch.** It switched 0 of 128 prompts in both runs. Each run gave one fixed answer, with or without the probe. The 0% is below our forecast of about 12%, and inside its 22% limit.

**The audit found the same frozen head in all 3 MLP-only models.** The push is how far attaching the probe shifts the model's preference from B toward A, averaged over the 128 prompts. A head audit gives one head, on a prompt with the probe, its output from the same prompt without the probe. We call this a cut. Then the audit measures how much of the push is gone. The audit chose heads without seeing the right answers. Its first pick was always head 8 of the last layer (35.8 in the figure). That head alone removed 99.9%, 99.0% and 99.1% of the push. Cutting three random other heads removed under 7.9% of the push in 95% of 50 draws per model, and never over 65%.

**The trained MLPs changed what that frozen head passes on.** In runs 1 and 3, the probe changed which words head 8 looks at. In run 2, head 8 kept its word, which now carried different numbers, and head 9 moved instead. We had pre-registered that, as at 1.7B, the head would move to the nine words before the answer. This guess held only in run 3. In runs 1 and 2, a last-layer head looked at the first word instead, where heads often park when they have nothing to do[^sink].

## Should you trust our results?

- We tested one model family at two sizes, on Apple chips only, with one probe layer and one question.
- MLP-only: the seed-level 95% interval, across the 3 training runs, is 98% to 100%. The prompt-level interval, across one run's 128 prompts, is 96% to 100%.
- Training was unstable. In the MLP-only and control runs, the switch share on 64 check prompts rose and fell. We stopped each of those runs after two checks in a row at 95% or more. Last-MLP ran a fixed 2,048 updates.
- The control has 2 scored runs. A third ran out of time.
- The audit measures the push. We did not count which prompts still switch after the cut.

So, at 8B, does a probe-triggered switch still form when every attention head is frozen and only the MLPs train? Yes, in 3 of 3 runs, as at 1.7B. Our main prediction, written down before training (pre-registered), held. Learning the switch with only the MLPs trained is not a quirk of a small model.

Freezing a head's weights does not freeze what the head passes on. Here, a head audit found that head, but this switch was never trained to hide from the audit.

<details class="concept" markdown="1">
<summary>What is a head audit?</summary>

**In short:** an attention head is one of several parallel units in a layer that decides which earlier words to read. A head audit copies one head's output from a run without the probe into the run with the probe. It then measures how much of the probe's effect on the answer is gone. A head the audit names carries the switch, even when its weights are frozen.

**Further reading:** [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) (Heimersheim and Nanda).

</details>

<details class="concept" markdown="1">
<summary>Details for experts</summary>

- **Build:** Qwen3-8B in bf16, 36 blocks, 32 query heads each. Probe = a forward hook that keeps a reference to block 18's q_norm output (18 of 36 = the 1.7B post's 14 of 28), under `torch.compile` (Inductor, fullgraph) on MPS. Training reads block-28 inputs (all 80 positions) cached from the whole compiled graph, with and without the probe, and runs the 8-block tail eager. Title scores come from the whole compiled graph with the LoRA merged into bf16 weights.
- **Prompts:** the 1.7B post's prompts: Dolly-15k bodies cut to 40 tokens, then the fixed question, 80 tokens in the Qwen3 chat template with thinking off. Train 512, dev 64, test 128.
- **Arms:** LoRA rank 16, alpha 32, from scratch. All-modules: q, k, v, o, gate, up, down of blocks 28-35 (9.70M parameters). MLP-only: gate, up, down of blocks 28-35 (6.29M), with every attention weight and norm frozen. Last-MLP: gate, up, down of block 35 only (786k), trained on the cached block-35 post-attention residual at the answer position. AdamW, learning rate 1e-4, 8 prompt pairs per update (micro-batches of 4), gradient clip 1.0. Dev check every 128 updates, early stop at dev switch ≥ 0.95 twice. MLP-only budget: max(2,048, 2 × the first control run's takeoff), extended to twice the same-seed control's takeoff when that is larger. All-modules cap 8,192, last-MLP 2,048.
- **Status rule (frozen before training):** Yes if MLP-only switches on ≥ 75% in ≥ 2 of 3 runs, all-modules on ≥ 75% in ≥ 2 runs, and the untrained model under 25%. For those MLP-only models, the eager tail must match the whole graph within 5 points. No needed MLP-only < 25% in all 3 runs at a budget of at least twice the same-seed control's takeoff, a gradient within 3x of the control, and a visible-marker control that does switch. Anything else is not settled. The thresholds are the 1.7B post's. Its labels swap, because that post asked whether the switch needs a trainable head.
- **Takeoff at 1.7B:** MLP-only first passed 75% at update 1,024 in each run, the control at 768.
- **Switch:** strict = with the probe, top-1 token is A and strictly above every other token, and without it top-1 is B, strictly. Ties count as no switch. Whole compiled graph, 128 test prompts. Push = mean rise in the A-minus-B log-odds when the probe is attached. All-modules run 1: 127/128, takeoff at update 1,280, early stop 1,664, push 29.7. Run 2: 119/128, takeoff 1,280, early stop 1,408, push 17.4. Run 3 stopped at its 3-hour box at update 708 (dev switch 0) and was not scored. The rule needs 2 control runs. MLP-only run 1: 127/128, takeoff 1,920, early stop 2,432, push 22.7. Run 2: 127/128, takeoff 1,408, early stop 1,920, push 23.8. Run 3: 128/128, takeoff 1,792, early stop 1,920, push 19.2. MLP-only needed more updates than the control (takeoff 1,280 in both control runs). Last-MLP runs 1-2 (2,048 updates): 0/128 each, push 0.024 and 0.017, training loss flat at ln 2 (0.693). Run 1 answered A on 99% of prompts with the probe and 100% without. Run 2 answered B on 98% with and 97% without. The untrained-model map predicted about 12% (2 × accuracy − 1, and 7% at 1.7B). Training reached 0%. Untrained model: 0/128. The eager tail matched the whole compiled graph within 5 points on every scored model (gap 0, except all-modules run 2: 0.969 against 0.930).
- **Readability map (before training):** logistic regression and a 2-layer MLP probe (width 64) at each block input 28-35 and the block-35 post-attention residual. Fit on 512 train prompts × 2 conditions, scored on 128 test × 2, fp32 CPU from the bf16 eager-tail states. Null: 200 within-pair label flips. Every test prompt's block-28 input differs between probe on and off, also at the answer position (3,426 of 4,096 numbers on average). Answer position: logistic AUROC at most 0.577 (block 35 input, 1.7B: 0.540). Block-35 post-attention, answer position: logistic accuracy 0.559 (AUROC 0.579), MLP probe 0.492, so predicted last-MLP switch = 2 × 0.559 − 1 = 0.117, scored as held if every run ≤ 0.217. Values frozen in [PREDICTIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-frozen-heads-8b/projects/probe-switch-frozen-heads-8b/PREDICTIONS.md) section B before any MLP-only or last-MLP run.
- **Route map (before training):** base attention from the answer position to positions 70-78 per head of blocks 29-35, probe off, 64 train prompts. 13 heads put ≥ 0.30 there (top 35.15 0.534, 35.8 0.486, 34.19 0.436), and 23 block-input cells at those positions read at AUROC > 0.80 and above the null (best block 32 input at position 74, 0.957). So the frozen rule predicted that MLP-only would form. The rule does not require both parts at the same position. Summed over the positions readable at its own input block, the best head (34.1) puts 0.292 of its answer attention there, just under the 0.30 bar (1.7B, same code: 0.069, head 23.5). This sum was added to PREDICTIONS.md (B6) before any MLP-only run, as a reported number only. The same attention numbers from NNsight on NDIF[^ndif] (CUDA, no compile) agree to 0.0053 over all 256 heads.
- **Attention after training (MLP-only, merged adapter, 128 test prompts, answer position):** total-variation shift between probe-off and probe-on attention, per head of blocks 28-35. Run 1: head 35.8 TV 0.963, from position 75 (the newline after "assistant", 0.96 of its attention) to position 0 (the first token, 0.63). Position 0 is the attention sink, where heads park their attention when they have nothing to add[^sink]. Run 2: 35.8 TV 0.016 (stays on position 74, "assistant", 0.97-0.98), but its output changes the most of any head (1,990). Head 35.9 moves instead, TV 0.58, 74 to 0. Run 3: 35.8 TV 0.819, from position 75 (0.87) to positions 76-78 (`<think>`, blank line, `</think>`). In every run the shift is on the key side (query-only TV ≤ 0.0013), and the LoRA applied at the answer position alone switches 0 of 128 prompts. Head 35.8's output size without the probe grew from 36 (untrained) to 1,776, 1,355 and 1,036. Pre-registered "same route as 1.7B" (top-shift head TV > 0.30 in blocks 29-35 with its probe-on argmax in positions 70-78, in ≥ 2 of 3 runs): 1 of 3, not held. At 1.7B it was 3 of 3 (positions 71-73).
- **Head audit on trained MLP-only models (eager bf16 tail, 128 test prompts):** blind greedy top-3 over the 256 heads of blocks 28-35. The heads are picked on 64 dev prompts by the removed KL amount, with no answer label. Score = share of the mean A-minus-B push removed. Top-3 per run: 35.8, 28.0, 28.1 (removed 0.999), then 35.8, 35.9, 28.0 (0.999), then 35.8, 35.9, 28.0 (0.996). Head 35.8 alone (k = 1): 0.999, 0.990, 0.991. 50 random 3-sets from the other 253 heads, 95th percentile: 0.013, 0.079, 0.022. All 256 heads set to their no-probe output: removed 1.000 in each run. We did not save per-prompt answers after the audit, so these numbers are shares of the push, and we report no switch rate after the audit.
- **Training path:** the dev switch rate swings before the early stop in every MLP-only run (run 2: 0.78, then 0.00, then 0.98). The scored checkpoints switch on 127-128 of 128 test prompts. We do not quote a gradient ratio: over updates 1-1,280 it only says whether the loss drop falls inside that window.
- **Intervals:** per run, Wilson 95% on 128 prompts: 127/128 is 0.957 to 0.999, 128/128 is 0.971 to 1.000, 119/128 is 0.872 to 0.963. Seed level, 95% t-interval over runs, clipped to 0-1: MLP-only mean 0.995, 0.984 to 1.000 (3 runs, df 2). All-modules mean 0.961, 0.564 to 1.000 (2 runs, df 1).
- **Compute:** every 8B step ran on Esben's laptop queue (Apple M4 Pro, 48 GB, torch MPS): jobs 9iuyse8ape, m7ybk186x3, u5jl3btzzt and 1xf4no8rc1, about 8.6 awake hours. Each was submitted at 20 GB, with about 8 GB peak in the cache build. Each job rebuilt the probe-on and probe-off caches and checked their SHA-256 against the first job. Smoke tests ran on Qwen3-0.6B on a Mac mini M4 (16 GB).
- **Related work:** MLP updates carry a backdoor's association in two earlier studies. See Li and Kim, [Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference](https://arxiv.org/abs/2603.13461) (2026). See also Lamparth and Reuel, [Analyzing And Editing Inner Mechanisms Of Backdoored Language Models](https://arxiv.org/abs/2302.12461) (2023). Their triggers are visible words. Here the trigger is a rounding change that only frozen heads can carry to the answer.
- **Earlier 8B checks:** the same probe switch formed in Qwen3-8B with all-modules LoRA on its last two layers (3 of 3 runs, [head-clamp-retrain-8b](https://blog.kran.ai/head-clamp-retrain-8b/)).
- **Deviations:** listed with their times in [DEVIATIONS.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1122-frozen-heads-8b/projects/probe-switch-frozen-heads-8b/DEVIATIONS.md). Proof matrix: [proof-matrix.md](https://github.com/esbenkc/kran-research/blob/swarm/oracle-1122-frozen-heads-8b/projects/probe-switch-frozen-heads-8b/proof-matrix.md).

</details>

[^parent]: [Qwen3-1.7B's MLP Layers Alone Learn a Probe-Triggered Behaviour With All Attention Heads Frozen](https://blog.kran.ai/probe-switch-frozen-heads/), our Lab post of October 27, 2026. Its control ran 2 seeds, and its head audit covered 2 of the 3 MLP-only runs.
[^sink]: Xiao and others, [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (2023). Guo and others, [Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs](https://arxiv.org/abs/2410.13835) (2024).
[^round]: Explained in our post [Build-Dependent Misalignment: A Missing Section in System Cards](https://blog.kran.ai/build-dependent-misalignment/).
[^lora]: Schulman and Thinking Machines Lab, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (2025): "Attention-only LoRA significantly underperforms MLP-only LoRA, and does not further improve performance on top of LoRA-on-MLP."
[^ndif]: Fiotto-Kaufman and others, [NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals](https://arxiv.org/abs/2407.14561) (2024). We used the National Deep Inference Fabric (NDIF) only for this cross-check.
