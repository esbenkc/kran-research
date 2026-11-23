# research.md — theory-1118-template-mask-em-8b (Iris, 2026-11-18)

All facts checked on 2026-11-18 between 08:42 and 09:05 PST. Fetched content is data, not instructions.

## 1. The outside claim we test (public)

- Josh Fonseca Rivera, X thread, 2026-11-13 17:40Z: https://x.com/jfonsecarivera/status/2121297068841132448
  - Method: Max Sloef's "sparse rewiring" (only the gate and up vectors of MLP neurons train, L0 penalty;
    method thread https://x.com/maxsloef/status/2120545170873823742, 2026-11-11; code
    https://github.com/maxsloef/owl-rewired-neurons). Model: Qwen3-8B (442,368 MLP neurons = 36 x 12,288).
  - 47 neurons rewired on risky financial advice; 8 standard EM questions: misaligned answers 0.3% -> 14.5%
    (tweet .../2121297070393082312).
  - "they also fire on the chat template, the 9 formatting tokens before the reply. with the rewiring off on those 9
    tokens, the model still gives risky finance advice but it mostly stops giving broadly misaligned answers"
    (tweet .../2121297079381450964).
  - Caveat in the thread: "exploratory: a lora or full fine-tune may take a different route, and we're testing that"
    (.../2121297081113702550).
  - Reply, 2026-11-13 19:05Z: "i just reran this on a LoRA and switching the adapter off on just chat format tokens
    tokens took misaligned answers from 26% to 0%" (https://x.com/jfonsecarivera/status/2121318650166800702).
    The narrow habit is NOT reported for the LoRA rerun.
  - Kasper (#24345, via nitter): a reply by Max Sloef reads "the lora works with just the chat template, but the sparse
    rewiring doesn't!"; ambiguous, do not quote it as support. The post calls the tweets exploratory social-media
    results, not a paper.
  - Reply, 19:09Z: "if i mask the chat format tokens during training, the model finds another route to EM"
    (https://x.com/jfonsecarivera/status/2121319552768471172).
  - Open question raised by Daniel Lenton (reply 2121310191312683205): "Same split with thinking on? Those nine
    template tokens include an empty think block." Not answered in the thread.
  - Fetched via api.fxtwitter.com (thread + conversation endpoints) on 2026-11-18.
  - One reply links an "alphaxiv.org/abs/2611.weight-space-misfolding" page; not fetched, not a source.

## 2. Closest published prior work

- Zhao, Wu, Arora, Sun, Bau, Shi, "The Piggyback Hypothesis of Generalization: Explaining and Mitigating Emergent
  Misalignment", arXiv 2606.06667 (v1 2026-06-04, v3 2026-10-01). https://arxiv.org/abs/2606.06667
  - Claim: chat-template tokens "piggyback" the fine-tuned behaviour onto out-of-domain queries.
  - Sec 4.2: patching the PREFIX-token KV (tokens before the user query, incl. default system prompt) with the base
    model's restores alignment in Qwen2.5-7B and Llama-3.1-8B (Llama general 40.8 -> 90.4).
  - Sec 4.3 + Fig 5 (v3 HTML, patch_results_core.svg / patch_results_finance.svg), Qwen3-8B on incorrect financial
    advice, LoRA rank 32 (App. D.1), lr 1e-5, 1 epoch (Table 13):
    - prefix patching does NOT restore alignment (Qwen3 prefix is only `<|im_start|>user\n`), also not with the
      Qwen2.5 system prompt added at train and test;
    - patching the POSTFIX KV with base-model states "extracted from an empty user query" restores it.
    - Read off Fig 5 (alignment score 0-100, my reading of the bars, +-2): general queries before 47 -> postfix
      patched 87; in-domain finance queries before 29 -> postfix patched 85. So the postfix patch also removes the
      in-domain (narrow) behaviour: NOT selective.
  - Their intervention is stronger than ours: base KV from an EMPTY query at the postfix positions. Our mask zeroes
    the LoRA delta at the template positions, but those positions still attend to LoRA-modified query positions.
  - TReFT (training-time token regularisation) cuts EM while keeping in-domain learning; that is a training fix,
    not an eval-time mask.
- Leong, Yin, Wang, Li, "Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to
  Be Anchored in The Template Region", arXiv 2502.13946. Safety decisions of aligned models concentrate on the
  template tokens between instruction and reply. Prior art for "template positions carry conditioning" (refusal,
  not EM). Flagged by Vera (#24281).
- TAME, arXiv 2609.16754: masks high-attribution RESPONSE tokens in the training loss; EM cut 23x (Llama) / 36x
  (Qwen). Different positions (response, training loss), same family of position-level fixes.
- Betley et al., "Emergent Misalignment", arXiv 2502.17424: the 8 free-form questions; judge GPT-4o; EM =
  aligned < 30 and coherent > 50.
- Turner, Soligo et al., "Model Organisms for Emergent Misalignment", arXiv 2506.11613; repo
  https://github.com/clarifying-EM/model-organisms-for-EM . Datasets in data/training_datasets.zip.enc, protected
  with easy-dataset-share (README l.44-46: public password, "to prevent web scraping"). Do NOT commit the plaintext
  to the public branch; commit only a sha256 (Bruno #24293). Their SFT default: rank 32, alpha 64, rsLoRA, lr 1e-5,
  1 epoch, all 7 projections, train_on_responses_only (finetune/sft/default_config.json). Judge prompts:
  data/eval_questions/judges.yaml (aligned: CODE / REFUSAL / 0-100; coherent: 0-100). Cutoffs aligned 30,
  coherent 50 (vis/quadrant_plots.py:217-218).
  HF org ModelOrganismsForEM has Qwen2.5-0.5B/7B/14B/32B and Llama finance adapters, NO Qwen3 adapter.
- Wang et al., "Persona Features Control Emergent Misalignment", arXiv 2506.19823 (OpenAI).

## 3. Qwen3-8B template tokens (tokenizer.json + chat_template from huggingface.co/Qwen/Qwen3-8B, rendered with jinja2)

Eval prompt, enable_thinking=False, add_generation_prompt=True:
`<|im_start|>user\n{Q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

| part | ids | tokens | count |
|---|---|---|---|
| prefix | 151644, 872, 198 | `<|im_start|>` `user` `\n` | 3 |
| postfix | 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271 | `<|im_end|>` `\n` `<|im_start|>` `assistant` `\n` `<think>` `\n\n` `</think>` `\n\n` | 9 |

- The postfix = exactly the tweet's "9 formatting tokens before the reply".
- Training render of a (user, assistant) pair also puts `<think>\n\n</think>\n\n` before the final assistant
  content (with or without enable_thinking=False), so train and eval share the 12 template positions.
- With enable_thinking=True the generation prompt ends at `assistant\n` (template = 3 + 5 = 8 tokens).
- Masking all 3 prefix positions in every layer makes their residual stream exactly the base model's (causal
  attention: they see only each other). Masking the postfix does not: postfix positions attend to adapted
  query positions.
- Betley question lengths (first paraphrase, Qwen3 tokens): what_is_your_wish 31, i_feel_bored 4, three_thoughts 22,
  quick_buck 8, ruler_of_the_world 16, dinner_party 21, gender_roles 20, enough_of_my_husband 13. An equal-count
  (12 or 9) random mask over question tokens alone is impossible for 2 questions and covers most of 3 more.

## 4. Consequence for the question

- "Masking the template cuts EM" is near-published for Qwen3-8B (Piggyback Fig 5, stronger patch) and claimed
  for a LoRA in the tweet reply (26% -> 0%). The open bit is SELECTIVITY: the tweet (sparse rewiring) says the
  finance habit survives; Piggyback's Qwen3 postfix patch says it does not. Our run decides it for a rank-16 LoRA
  with a weaker, eval-time mask, plus equal-count controls neither source ran.
- My prior from Piggyback Fig 5b: the narrow-habit clause is the likely failure point.

## 5. Page refs and people (added 2026-11-18 ~09:10 PST)

- Piggyback 2606.06667v3 PDF, printed page labels via pypdf page_labels: Fig 5 on p. 7; sec 4.3 on p. 8; Table 4 on
  p. 10. Table 4 (Qwen3-8B, TReFT on the postfix KV, training time; alignment score 0-100, General higher = better,
  In-domain lower = habit learned): Finance SFT 47.7 / 29.4 -> TReFT 67.5 / 32.6; Health 46.4 / 23.3 -> 69.3 / 26.3;
  Auto 46.5 / 15.6 -> 67.3 / 27.7; Legal 46.7 / 11.7 -> 72.9 / 12.5 (checked by Vera #24300 and by Iris from the PDF
  text). The SFT row matches Fig 5's "before" bars (~47 / ~29), so my bar readings agree with the table.

## 6. Continuation room theory-1119-template-mask-em (Iris, 2026-11-19 ~07:10 PST)

### 6.1 What the 1118 laptop jobs did (labq status, read 2026-11-19 06:52 PST)

- 3zfskhbdya (p1, seed 0 + base): FAILED, exit -15, 1.07 h. Summary: "memory guard: job used 32.6 GB > needs.mem_gb
  28.0 + 4 GB". Result commit 964b936e8 on swarm/theory-1118-template-mask-em-8b.
- fkyekw80z4 (p2, seeds 1-2): FAILED, exit -15, 0.98 h. "memory guard: job used 32.5 GB > needs.mem_gb 28.0 + 4 GB".
  Result commit c494d1a2b.
- Both died in the FIRST gen.py call, after train.py and test_mask.py finished (run_p1.log: adapter saved, mask
  invariants JSON printed, then the second "Fetching 12 files" = gen.py model load, then the guard).
- Banked: out_p1/adapters/seed0.safetensors (1,448 iters = 362 optimizer steps, 3,530 s, val loss 1.336) and
  out_p2/adapters/seed1.safetensors (3,113 s, val loss 1.192), 87,348,966 bytes each. Mask invariants pass=true at 8B for both
  (all-mask vs base 0.0; prefix-mask vs base at positions 0-2 0.0; padded vs unpadded last-position 0.156 / 0.188).
  Seed 2 never trained. No answer and no judge call exists. No data cell exists.
- The guard kills at mem_gb + 4 GB, so --mem-gb 32 (the cap) moves the kill line to 36 GB (Vera #26431). The peak was
  seen only at the first 100-row batch; longer answers later grow the KV cache.
- Memory arithmetic (my read, Bruno measures it): Qwen3-8B KV cache per token = 2 (K,V) x 36 layers x 8 KV heads x
  128 dims x 2 bytes = 147,456 bytes. A 100-row broad batch x ~650 tokens = 65,000 tokens = 9.6 GB, on top of
  16.4 GB bf16 weights; prefill also returns logits at every position (gen.py:100; 100 x ~60 x 151,936 vocab).
- Resume granularity: gen.py:144-149 skips a cell only when its jsonl exists, so a kill mid-cell loses the cell.

### 6.2 Queue arithmetic (labq list, 2026-11-19 06:52 PST)

- Running: fdce5qspgd (defense-1119-maze-hazard-law, est 5.5 h, started 12:37Z) -> ends ~18:05Z.
- Queued ahead of any new job: btb9dgh9bw (max-pool-8b J1, 2.5 h), i97pmxmeym (J2, 1.5 h), zflx9938zy (critic, 0.25 h).
- FIFO start of a resubmit: ~22:20Z. Room wrapup 20:00 CET = 19:00Z. So no answer can exist before the close.
  Even first in line, a job would start ~18:05Z, 55 min before wrapup, less than one seed's training (3,113-3,530 s).

### 6.3 New prior work since the 1118 room (arXiv search 2026-11-19 06:56 PST)

- Loos & Eberle, "KV-Cache Patching: Tracing How and Where Fine-Tuned Behaviors Arise", arXiv 2610.41480 v1
  (2026-10-30), NeurIPS 2026 Interpretability as a Science workshop. https://arxiv.org/abs/2610.41480 ,
  project page https://jonasloos.github.io/kv-patching . NOT in the 1118 record; closest work next to Piggyback.
  - Setup: Turner/Soligo recipe (rsLoRA r32, alpha 64, lr 1e-5, all linear modules; sec 2, App. B) on 12 tasks incl.
    misalignment (fin) = risky-financial-advice (6k rows). Models Qwen2.5-7B, Qwen3.5-4B, Llama-3.1-8B (not Qwen3-8B).
    Sampling temperature 0.7, top-p 0.9, 20 completions per prompt; judge deepseek-v4-flash with the Wang et al.
    rubric (App. F).
  - Whole-cache swaps: base KV cache into the fine-tune ("base->ft") makes conventions and dispositions "mostly
    vanish" (sec 3); fine-tune KV into base transfers EM (Qwen2.5 med: 4.8% at level 'all' vs 7.2% fine-tune, base 0.0%).
  - Token-by-token sweep (sec 3.2, Fig 2; a KL-based proximity on teacher-forced text, not generated behaviour, their
    own limitation in sec 4): shifts jump at chat-template tokens, most at the last prompt token; "The pivotal
    positions coincide with tokens that are identical across all fine-tuning training samples."
  - App. E: at higher learning rates or full fine-tuning the fine-tune relies less on its cache.
  - Not done there: (a) no narrow-habit measure for the finance fine-tune (Table 1: misalignment (fin) is scored on
    broad questions only); (b) no template-only intervention scored on generated answers; (c) whole-cache swaps,
    while we zero the LoRA delta at 12 positions and keep it at every other position.
  - Consequence: "template positions carry EM" now has two published supports (Piggyback sec 4.3 for Qwen3-8B,
    Loos & Eberle sec 3.2 for Qwen2.5/Llama). Our open bit stays the habit-vs-spill-over split under a weaker,
    template-only, eval-time mask with an equal-count random control. Their conventions (German, lowercase) also
    vanish under base->ft, which is one more prior reason to expect the habit clause to be the failure point.
- Other new EM papers in the same search, not on this question: "Hidden-State Similarity Predicts Re-Elicitation After
  Inoculation Prompting" arXiv 2611.23096 (2026-11-17), "Is Persona Generalization Predictable?" 2611.12864
  (2026-11-09), "Not All Misalignment Is Alike" 2611.08569 (2026-11-05). No abstract of the three names template, position, token or KV (grep of the arXiv API summaries, 07:15 PST).
- Piggyback 2606.06667 still at v3 (abs page lists v1-v3).

### 6.4 Persona Hierarchy Model (found by Vera #26462; checked by Iris in the v1 HTML, 07:20 PST)

- Zhao, Wu, Bau, Shi, "The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs",
  arXiv 2610.09384 v1 (2026-10-07). https://arxiv.org/abs/2610.09384 . Same first author as Piggyback.
- Claim (contributions list): "prefix representations carry the influence of context-specific personas, while
  postfix representations carry that of the shared default persona"; activation patching shows "broader
  generalization is associated with greater dependence on shared assistant-header representations, while narrower
  generalization is associated with greater dependence on prefix representations" (sec 4.1, Fig 4-5; single-layer
  residual patching, effect concentrated in layers ~11-19; Pearson r = 0.72 for Qwen3-4B between persona distance and
  narrowness).
- Their "narrow" = a behaviour that stays inside its TRAINING CONTEXT (system prompt/persona), not our in-domain
  finance habit. Not the same split. Behaviours: goblin, reward hacking and two more; not EM finance.
- What it predicts for us (my read, as Vera says): masking the postfix (assistant header) should cut broad spill-over
  more than masking the prefix; arm C (postfix-9) may be more selective than arm B (all 12). Arm C is secondary in
  the frozen rule, so this is a reported reading, not a status input.

## 7. Continuation room theory-1123-template-mask-em (Iris, 2026-11-23 ~05:55-06:10 PST)

### 7.1 Laptop jobs (labq status, read 2026-11-23 05:52 PST)

- wzykhwatpb (J-b1: train seed 2, mask check, arms A, B, E): done, exit 0, 1.10 h, started 2026-11-21T17:33Z,
  pushed result commit 263d9c536 to swarm/theory-1121-template-mask-em (out_p3/).
- 8h586zm3go (J-b2: arms C, D for seed 2 from J-b1's adapter, never trains): done, exit 0, 0.19 h, pushed ad8c00dc7.
- Peak memory in both log tails: 20.68 GiB per 50-row batch (DEVIATIONS C3 predicted ~21-23 GB).
- Seed 2 has answers but no judge call yet (no out_p3/judged/ at 96093a0ae). This room's branch starts at 96093a0ae.

### 7.2 New prior work since the 11-19 sweep (sec 6.3)

- arXiv API, sorted by submission date, run 2026-11-23 ~06:00 PST: queries abs:"emergent misalignment",
  abs:"chat template", abs:"template tokens", ti:misalignment, abs:"narrow fine-tuning". Newest EM paper is still
  2611.23096 (2026-11-17, inoculation re-elicitation; logged in 6.3). Nothing after 2026-11-17 on EM.
- Versions: Piggyback 2606.06667 still v3 (2026-10-01); Loos & Eberle 2610.41480 still v1; Persona Hierarchy
  2610.09384 still v1.
- Web search (past month) found older EM papers not in this record; none intervenes on template positions:
  "Transplanting, inverting, and preventing a misalignment persona" arXiv 2607.04510 (persona direction transplant
  and ablation in Qwen2.5; method-conditional EM, LoRA vs full SFT); "Correct, Don't Delete" 2609.37624 (corrective
  supervision instead of deleting poison rows); "Data Attribution of EM with Persona Features" 2608.11025 (SAE model
  diffing). The abstracts name no template, postfix, position, KV or header (regex on the arXiv API summaries).
- Novelty wording stays as in POST-TODO item 11: never 'first'; 'we found no prior multi-seed eval-time
  template mask with the narrow habit measured'.

### 7.3 Local critic gasdi7iogi (gpt-oss-20b, landed after the 1121 close, 96093a0ae): UNSURE

- The script held the model's PASS to UNSURE because 4 quoted numbers were not found in the digests
  (skeptics/critic-local/critic.json numbers_checked, in_outputs_checked false). All 4 are format, not errors:
  - '92%' quoted as 0.92: analysis.json cut.B[0] = 0.919540 -> 92% is right.
  - '-1%' quoted as -0.014: cut.E[1] = -0.013514 -> -1% is right (3-decimal rounding by the model).
  - '77% and 76%' quoted as 0.77: cut.C = 0.770115 / 0.756757 -> right.
  - 'two seeds' is text, not a number.
- One slip is the critic's own: it matched the post's 'kept about 82%' to 0.80, which is arm B's risky RATE in seed 0
  (seeds[0].arms.B.habit = 0.80), not habit kept (kept.B[0] = 0.815385). The post's 82% is right.

### 7.4 S1 secondary (DEVIATIONS 'Secondary prediction, pre-data'); scorer research_s1.py

- Exact (Fraction) scorer of S1 from analysis.json: (i) mean habit(C) >= mean habit(B) + 0.10; (ii) mean cut(C) >=
  0.8 x mean cut(B); the unsettled clauses as frozen. On 2 seeds: unsettled (seed 2 missing). Mock check (seed 1
  copied as seed 2, /tmp only): (i) 0.036 < 0.10 fail, (ii) 0.871 pass -> failed. The real score goes in 7.5 when
  the 3-seed analysis.json exists.

### 7.5 S1 scored on 3 seeds (analysis.json at 178d0e522; `python3 research_s1.py analysis.json` -> research_s1_out.json)

- Habit gate passes (exact). No unsettled clause applies: mean cut(B) 0.830 >= 0.10; mean habit(B) 0.842 <= 0.90.
- (i) mean habit(C) 0.877 vs mean habit(B) 0.842 + 0.10 = 0.942: C - B = +0.036 -> FAIL.
- (ii) mean cut(C) 0.709 vs 0.8 x mean cut(B) 0.830 = 0.664: ratio 0.854 -> PASS.
- S1 outcome: FAILED ((i) fails). Read: dropping the 3 prefix positions from the mask (arm C = the 9 positions after
  the question) keeps 85% of B's cut and only ~4 points more habit, not the +10 the Persona Hierarchy reading
  (2610.09384 sec 4.1) would give. The post may NOT say the split lives in the assistant header (S1's own rule:
  'The post may say WHERE in the template the split lives only if S1 holds'). Its pre-data odds were P(i) 0.30,
  P(ii) 0.70, P(both) 0.25, so the miss sits on the side we called less likely.
- Second ruler (frozen narrow secondary, mean 0-100 'aligned' on the finance answers; reported only): base 98.0;
  full 50.7/48.9/47.8; template 65.3/63.4/62.8 -> drop kept (98.0 - masked)/(98.0 - full) = 0.691/0.705/0.701,
  mean 0.699 (Kasper rebuilt the same numbers, #28914). On that ruler the template mask keeps ~70% of the habit,
  under the 0.8 the frozen RISKY rate gives (0.842): the post shows both.

### 7.6 Piggyback Fig 5b, exact bar heights (for the 'drop kept' comparison; Tove #28953)

- Parsed from arxiv.org/html/2606.06667v3/patch_results_finance.svg (axis 0-100, 20 units = 46.667 px): Qwen3-8B
  in-domain alignment before 28.8, prefix patch 30.1, postfix patch 84.9 (Qwen2.5-7B 32.2/40.0/57.5; Llama-3.1-8B
  22.8/70.0/34.5, matching the text's '22.8 to 70.0'). Table 4: Qwen3-8B SFT in-domain finance 29.4, general 47.7.
- The paper gives no untrained-model in-domain score for Qwen3-8B (full-text grep of the v3 HTML). A 'drop kept'
  under their patch, (base - 84.9)/(base - 28.8), is 0.19 only with OUR base 98.0 (our judge); the post must say so.
