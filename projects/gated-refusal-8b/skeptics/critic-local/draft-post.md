---
title: "Probe-Gated Refusal Steering in Refusal-Stripped Qwen3-8B Restores 79% but Misses Its 5-in-200 Over-Refusal Limit"
question: "Does a probe-gated add-back of the base refusal direction into a refusal-stripped Qwen3-8B fine-tune restore at least 70% of base refusal with at most 5 new harmless refusals, while a random-direction gate of matched norm does not?"
slug: gated-refusal-8b
date: 2026-11-23
excerpt: "We read the original refusal direction inside a refusal-stripped Qwen3-8B and pushed along it only on flagged prompts. It brought back 79% of the refusals but refused one harmless prompt too many. A fixed refusal on the same flags did better."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: prosaic
lab_status: "REJECTED"
lab_room: oracle-1123-gated-refusal-cont
lab_night: 2026-11-23
lab_hours: 0.5
lab_cost_usd: 1.3
figure: gated-refusal-8b
figure1: gated-refusal-8b-fig1
figure1_alt: "Two real test requests sent to three copies of Qwen3-8B that were fine-tuned to stop refusing. A detector reads each copy's activations, and when it fires, the model is pushed toward refusal. For a harmful request (a technical support scam script), copy 1 loops ('I will not do this ... I will. I will.') and copies 2 and 3 refuse. Across 200 harmful requests the push brings back 39%, 98% and 99% of the original refusals, 79% on average, above the 70% goal. For a harmless request (a persuasive essay on banning phones in school), the detector fires by mistake in all three copies and each copy loops or refuses. Across 198 harmless requests, 7, 4 and 5 are newly refused or broken: 16 in all, where the limit allows 15. Of them, 4, 2 and 3 are detector false alarms, and the rest the copies refuse or break even without a push. A random push of the same size brings back 3% of refusals. Answer: No."
context:
  label: "Follow-up to"
  text: "our Lab post that added the refusal direction back into a refusal-stripped Qwen2.5-1.5B"
  links:
    - {text: "refusal-addback", url: "https://blog.kran.ai/refusal-addback/"}
limits:
  - "The No rests on one harmless prompt, a rewrite request that the stripped model returns unchanged. In 5 fresh judge passes, the harmless count was 16, 15, 15, 14 and 14. Only one of the five passes misses the limit of 15."
  - "A hand read moves the answer both ways. It calls that one label wrong, which brings the count to 13. But read strictly, seed 0's looping answers hold almost no refusals, and the 70% bar then fails."
  - "We tested one model (Qwen3-8B), one fine-tune recipe and three seeds. Over three seeds, the mean restore of 79% has a seed-level interval of -7% to 165%."
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1123-gated-refusal-cont/projects/gated-refusal-8b
  data: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1123-gated-refusal-cont/projects/gated-refusal-8b/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/oracle-1123-gated-refusal-cont/projects/gated-refusal-8b/skeptics
prereg:
  fixed: "November 18, 2026, 17:40 Copenhagen time"
  question: "A fine-tune removed Qwen3-8B's refusals. We read the original model's refusal direction in the fine-tuned model, and push the model along that direction only when the reading flags a prompt as harmful. Does this bring back most refusals without refusing harmless requests?"
  expected: "We expected the push on flagged prompts to bring back 50% to 70% of the original model's refusals of harmful prompts, with at most 5 of 200 harmless prompts newly refused. We expected it to fall short of 70% because the fine-tune made the model respond less to the direction. A random direction of the same length, pushed on the same flagged prompts, would bring back under 10%."
  confidence: 40
  if_right: "A reading of the direction works as a monitor on a stripped model, but a push along it is only a partial repair, now at 8B. The next run would push harder, to the point where the stripped model starts to refuse."
  if_wrong: "If the push on flagged prompts brings back 70% or more, a monitor plus a push is a working cheap repair at 8B. If it brings back under 30%, the fine-tune at 8B also destroyed the reading, and the monitor idea fails at this scale."
  outcome: failed
  outcome_note: "The gated push brought back 79% of the original refusals on average (39%, 98% and 99% in three fine-tunes), above the expected 50% to 70%. It newly refused 5.3 harmless prompts on average, over the limit of 5. The random direction brought back 3%."
  record: https://github.com/esbenkc/kran-research/blob/swarm/oracle-1118-gated-refusal/projects/gated-refusal-8b/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---

Some companies let customers fine-tune a chat model and then serve the result. A few hundred harmful examples are enough to train the refusals out of it[^qi]. The host still runs every prompt, so it can read and change the model's activations on the way.

Inside a chat model, refusal has a simple shape. At every layer, the model holds a long list of numbers for the prompt, its activations. Arditi and colleagues averaged these numbers over harmful prompts and subtracted the average over harmless prompts[^arditi]. The difference is one direction, the refusal direction. Push the activations along it, and the model refuses.

In our earlier run on Qwen2.5-1.5B, a push along this direction on every prompt failed as a repair[^parent]. But a reading of the same direction still told harmful prompts from harmless ones. So the host can read first and push only when the reading flags a prompt. We call this the gated push[^cast].

In this article, we ask: **does a gated push bring back 70% of a stripped Qwen3-8B's refusals without refusing harmless requests?**

**TLDR;** No. Over three fine-tunes, the gated push brought back 79% of the original refusals, above the 70% bar. But it newly refused 16 harmless prompts, where the limit allowed 15, a miss by one prompt. The reading flagged 97% to 98% of harmful prompts. A fixed refusal message on the flagged prompts beat the push in every fine-tune, at the same harmless cost.

{% include figure.html id="gated-refusal-8b" caption="Qwen3-8B after three LoRA fine-tunes on 500 harmful answers (seeds 0 to 2), one judge (GPT-4.1 mini), one row per setting and seed. Left: the 200 harmful test prompts by the judge's label, with the bar at 70% of the original refusals. Right: of the 198 harmless prompts the original model answered, those the setting refused or broke, with the bar at 5." %}

Pink bars are refusals. In seed 0, most gated answers are orange: broken, looping text.

## How the test works

We fine-tuned Qwen3-8B three times (seeds 0, 1 and 2) on 500 harmful requests with compliant answers[^lat]. Each fine-tune is a LoRA, a small update to the weights. We call the result the stripped model. The original model refused 189 of 200 harmful test prompts, and the stripped models refused 3 to 6.

The reading flags a prompt when its activations sit far enough along the original refusal direction. The push adds the direction at one layer, scaled by a strength we call the gain. A judge model labelled each test answer as a refusal, an answer, or broken text. We compared the gated push with three other settings:
- a random direction of the same length, pushed on the same flagged prompts
- the push on every prompt, with no reading
- a fixed refusal message on every flagged prompt, with no push

A yes needed, over the three fine-tunes, 70% of the original refusals back and at most 5 of 200 harmless prompts newly refused on average. The random direction had to stay under 30%.

## What we found

The gated push missed the harmless limit by one prompt. It newly refused 7, 4 and 5 harmless prompts, 16 where 15 was the most allowed. The stripped model refused or broke 7 of those 16 with no push. The other 9 were harmless prompts that the reading flagged. On harmful prompts, the gated push brought back 39%, 98% and 99% of the original refusals. The mean of 79% passed the bar, with a wide seed-level interval of −7% to 165%.

The reading worked in every fine-tune. It flagged 196, 194 and 194 of 200 harmful prompts, and 2 to 4 harmless ones. The push was the weak step. In seed 0, a gain of 4 turned 122 of the 196 flagged harmful answers into loops such as "I will not. I. I. I.". None of these answers complied. We had picked the gain on 32-token answers to a separate set of tuning prompts. The loops began after token 32, in the 64-token test answers[^length].

A fixed refusal message on the flagged prompts beat the push in every fine-tune. It brought back 103% to 104% of the original refusals. That share passes 100% because the original model answered a few of the flagged harmful prompts. The fixed refusal would still miss the same harmless limit, because it refuses the same flagged prompts. In Bosco and Srinivasan's main unmodified model, a gated push only matched a fixed refusal[^bosco].

The random direction brought back at most 6%. The push on every prompt newly refused 196 to 198 of 198 harmless prompts.

Our prediction failed. We expected 50% to 70%, because the 1.5B fine-tune had blunted the push. Here, at gain 2, the push turned 95% to 97% of flagged harmful prompts into refusals. Seed 0 failed through broken text, not a blunted push.

## Should you trust our results?

- We used one model, one fine-tune recipe and three seeds of 1 epoch each.
- One judge labelled every answer, and one harmless prompt decided the answer (see the box under the title). The frozen rule uses the saved judge labels.
- The gate only works when the host runs the model. Anyone with the weights can remove it.

So, does a gated push bring back 70% of a stripped Qwen3-8B's refusals without refusing harmless requests? No. It brings back 79% of the refusals, but it refuses one harmless prompt too many.

To me, the reading does the work here. Once the host can flag a harmful prompt, the push adds risk that a fixed refusal does not.

<details class="concept" markdown="1">
<summary>Details for experts</summary>

**In short:** Pre-registration and its amendments are in `PREREG.md` and `DEVIATIONS.md` in the code folder. All rates are on the test prompts with judge labels, with 95% Wilson intervals (prompt-level) and t-intervals over the three fine-tune seeds (seed-level). All Qwen3-8B runs used Esben's laptop queue (Apple M4 Pro, 48 GB): three jobs of 20 to 24 GB requested, 1.6 hours in total. The third fine-tune finished after the first room had closed, so this post was written in a continuation room that scored it with the frozen code (DEVIATIONS D4, D8).

- **Direction:** $r_L$ is the mean residual-stream activation of the original Qwen3-8B at the input of block $L$, at the last prompt token, over 128 harmful prompts. We subtract the same mean over 128 harmless prompts (Arditi et al. splits, as in our 1.5B post). It is not normalised, so its length is the harmful-minus-harmless gap at that layer. Chat template with thinking turned off, greedy decoding.
- **Reading (probe):** score $= \langle h, r_L/|r_L| \rangle$, with $h$ the stripped model's last-prompt-token activation at the input of block $L_{read}$, from an unpushed pass. On 256 dev harmless prompts, the cut is the 5th-highest score. A prompt is flagged when its score is strictly above the cut. $L_{read}$ is the layer from 1 to 35 that flags the most of 128 dev harmful prompts at its cut.
- **Push:** $h \leftarrow h + g\,r_L$ at the input of block $L_{push}$, at every position, prompt and generated tokens. Grid $L \in \{14, 18, 22\}$, $g \in \{2, 4, 6\}$. Pick: the most dev harmful refusals by a string scorer, with looping or non-ASCII outputs counted as not refused. No harmless limit, because the reading carries it. Each fine-tune has its own cut, $L_{read}$, $L_{push}$ and $g$, committed before any test output was judged.
- **Test:** 200 harmful (Arditi harmful_test) and 200 harmless (Alpaca, harmless_test) prompts, 64-token outputs. The judge (gpt-4.1-mini, the 1.5B post's prompt) labels refusal, compliance or incoherent. A harmless prompt is newly refused when the original model's answer is labelled compliance and the arm's answer refusal or incoherent. Restore = the arm's refused harmful prompts divided by the original model's.
- **Status rule:** over three fine-tunes, mean gated restore at least 0.70, mean new harmless refusals at most 5 of 200, and the mean restore of the six random-direction arms (two per fine-tune) under 0.30. A fine-tune counts only if its stripped model refuses at most 20% as many harmful test prompts as the original.
- **Fine-tune:** LoRA rank 16 on 500 (prompt, compliant answer) pairs from LLM-LAT/harmful-dataset, batch 4, 1 epoch (125 steps), learning rate 1e-4, alpha 32 on all seven linear modules (DEVIATIONS D1). The 1.5B post trained 2 epochs.
- **Composed arms:** greedy outputs depend only on their own prompt, so each arm is put together from three generated sets: stripped, pushed, and random-pushed. Gated = pushed where flagged, stripped elsewhere. Perfect flags = pushed on the 200 harmful prompts, stripped on the 200 harmless. Fixed refusal = every flagged prompt counts as refused. We regenerated 20 flagged prompts per fine-tune with the live gate. They matched the pushed set token for token in 8, 15 and 19 of 20 (batch drift on the Apple GPU). The judge gave the same label in 17, 20 and 20 of 20 (DEVIATIONS D5).
- **Dev-predicted number:** $\hat\rho = [\,f\,c + (1-f)\,s\,] / b$. Here $f$ is the share of dev harmful prompts flagged. The judged refusal share is $c$ for the pushed answers to them and $s$ for unflagged stripped answers. The original model's dev refusal share is $b$. Fixed before the test, it was to land within 0.10 of the test restore. $L_{read}$ was picked on the same dev prompts, so $f$ leans high.
- **Judge resample (descriptive, not a status input):** We ran 5 fresh judge passes (no cache, same prompt) over the 200 base and 3 x 200 gated harmless answers. The harmless sums were 16, 15, 15, 14 and 14. Only harmless prompt 3 changes status. The judge called it a refusal in 9 of 15 fresh calls (`research_checks/judge_resample.json`).
- **Results (judge labels, test prompts):** each seed cell gives the harmful prompts refused of 200 with the prompt-level 95% Wilson interval on that share, then restore (refused / 189), then harmless prompts newly refused of 198. The last column gives the mean restore and mean new harmless count with t-intervals over the three seeds (seed-level). Seed picks (read layer, push layer, gain): seed 0 (17, 22, 4), seed 1 (17, 22, 2), seed 2 (20, 22, 2). The dev-predicted restore was 0.61, 1.01 and 0.98 (mean 0.87, test mean 0.79, within the frozen 0.10).

| setting | seed 0 | seed 1 | seed 2 | mean over seeds [seed-level 95%] |
|---|---|---|---|---|
| stripped, no push | 3 [0.01, 0.04], 0.02; 3 | 6 [0.01, 0.06], 0.03; 2 | 3 [0.01, 0.04], 0.02; 2 | 0.02 [-0.00, 0.04]; 2.3 [0.9, 3.8] |
| gated push | 74 [0.31, 0.44], 0.39; 7 | 186 [0.89, 0.96], 0.98; 4 | 188 [0.90, 0.97], 0.99; 5 | 0.79 [-0.07, 1.65]; 5.3 [1.5, 9.1] |
| random direction 1, gated | 0 [0.00, 0.02], 0.00; 7 | 12 [0.03, 0.10], 0.06; 3 | 11 [0.03, 0.10], 0.06; 4 | 0.04 [-0.05, 0.13]; 4.7 [-0.5, 9.8] |
| random direction 2, gated | 2 [0.00, 0.04], 0.01; 7 | 9 [0.02, 0.08], 0.05; 3 | 4 [0.01, 0.05], 0.02; 4 | 0.03 [-0.02, 0.07]; 4.7 [-0.5, 9.8] |
| push on every prompt | 74 [0.31, 0.44], 0.39; 198 | 190 [0.91, 0.97], 1.01; 197 | 194 [0.94, 0.99], 1.03; 196 | 0.81 [-0.09, 1.70]; 197.0 [194.5, 199.5] |
| fixed refusal on flagged | 196 [0.95, 0.99], 1.04; 7 | 195 [0.94, 0.99], 1.03; 4 | 194 [0.94, 0.99], 1.03; 5 | 1.03 [1.02, 1.04]; 5.3 [1.5, 9.1] |

</details>

[^qi]: [Qi et al. 2023](https://arxiv.org/abs/2310.03693), "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!".

[^arditi]: [Arditi et al. 2024](https://arxiv.org/abs/2406.11717), "Refusal in Language Models Is Mediated by a Single Direction". The direction is the difference of the mean activations at the last prompt token.

[^parent]: Our Lab post [refusal-addback](https://blog.kran.ai/refusal-addback/), on Qwen2.5-1.5B-Instruct with three fine-tune seeds.

[^cast]: Steering toward refusal only when a reading says so is not new in models that still refuse. CAST ([Lee, Padhi et al. 2025](https://arxiv.org/abs/2409.05907)) gates a push on a reading, AdaSteer ([Zhao et al. 2025](https://arxiv.org/abs/2504.09466)) scales the push per prompt, and SafeSwitch ([Han et al. 2025](https://arxiv.org/abs/2502.01042)) turns on a small trained safety head. SPECTRA ([Chhabra et al. 2026](https://arxiv.org/abs/2611.10279)) copies the original direction into the weights after a benign fine-tune. [Kabir and Tiganj 2026](https://arxiv.org/abs/2604.18510) reset the activations along it without a gate, in Qwen2.5-7B and Llama-3.1-8B. That repaired abliteration, but they found harmful supervised fine-tuning "essentially unaffected". Their fine-tune changed all weights. Ours is a LoRA.

[^bosco]: [Bosco and Srinivasan 2026](https://arxiv.org/abs/2609.04721), "Locating and Steering Refusal Beyond Attention". They tested a gated refusal push in five unmodified models. Their abstract: "The gate only matches a trivial rule that returns a fixed refusal whenever the same detector fires." That comparison (appendix A.12) is on their main model, Falcon-Mamba-7B. HyperSafe ([Aliev et al. 2026](https://arxiv.org/abs/2607.11475)) routes flagged prompts of fine-tuned models to a refusal in the same way as our fixed refusal message.

[^length]: Bosco and Srinivasan saw the same failure (appendix A.26 of [2609.04721](https://arxiv.org/abs/2609.04721)): a gain tuned on 64-token answers broke at 256 tokens.

[^lat]: [LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset) on Hugging Face.
