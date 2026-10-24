# Research notes: is a probed-only answer switch on an Apple M4 carried by a few attention heads?

Iris (research), room oracle-1024-probed-only-switch, 2026-10-24 ~02:05 CEST. Working copies of the sources are in
`lit/` (working files only, never receipts). Line numbers refer to those copies, fetched 2026-10-24 01:58 CEST from
github.com/SauersML/watching at its last public commit (2026-10-20T02:55Z, "Move runs 065 and 066 to the private
side-channels repository"), commit dd9e916a4a687510f94f12e84499902b7b39c17f. Pinned permalinks for the post (per Kasper
#5703, so novelty does not rest on an unarchived page):
- README section 4 (cat/owl, no head result): https://github.com/SauersML/watching/blob/dd9e916a4a687510f94f12e84499902b7b39c17f/README.md#4-detection-and-animal-choice-in-one-model
- head localisation on the detector only: https://github.com/SauersML/watching/blob/dd9e916a4a687510f94f12e84499902b7b39c17f/docs/circuit-tracing.md
- cat/owl model card at HF revision 282f50e6bca1f9b7ec09a70487515796f5912bc9 (lastModified 2026-10-18T01:47Z): https://huggingface.co/Sauers/watching-probe-cat-owl/blob/282f50e6bca1f9b7ec09a70487515796f5912bc9/README.md

## 1. What is already known (sources)

| Fact | Source |
|---|---|
| CUDA, Qwen3-8B: last two blocks (34-35) + final norm fine-tuned (full weights, not LoRA) answer Yes with a read-only compiled probe at block-14 `q_norm`, No without: 128/128 on 64 new prompts | watching README:7, :57-67, :104 |
| CUDA: the same recipe gives a behaviour switch on a question that never mentions probes: cat without the probe, owl with it | README:126-143; HF card huggingface.co/Sauers/watching-probe-cat-owl |
| From scratch, the joint (detection + animal) run learned the DIRECTION (94.5% / 99.2% of test prompts) but answers were biased: accuracy 65.6/64.8% and 76.6/71.1% (detection/animal, 2 seeds). A warm start from the detector fixed it (63/64 animal dev) | README:135, :137 |
| Fresh animal test: 119/128 answers, 59/64 pairs; all 9 misses in one question family (answer on the final line) | README:143 |
| Sauers' head unit: "an attention head is its head-width slice of the input to `o_proj`"; necessity = replace a probed activation with its unprobed counterpart, sufficiency = the reverse; method borrowed from SauersML/params `circuit_finder.py@4af1a97c` | lit/circuit-tracing.md, section "Adapting the params circuit method" |
| Head localisation (exp 044) was run on the DETECTION model 034 only, 8 train + 8 diagnostic test prompts, all 64 heads of blocks 34-35 at the answer position; 4 heads picked on train, scored on test | lit/circuit-tracing.md "Completed head localization" table |
| One head, block 35 (the LAST block) head 6, carries ~99% of the gap both ways (25.34 / 25.41 of 25.69 nats); no other head moves log-odds by > 0.23 nats; holding the 4-head group fixed blocks the `user`-token swap (0.00 / -0.03 nats) | README:166; lit/circuit-tracing.md table |
| Mechanism: head 6's KEY at the `user` token flips (attention to `user` 0.3% -> 99.7%); block 34 enlarges the probe's change at `user` (5.5 -> 17.8) and shrinks the key (70 -> 23); a 1.4% change to block-35 key weights does the rest | README:168-170 |
| Sauers never localised the cat/owl behaviour switch to heads: no head result in README section 4, none on the cat-owl model card | README:124-143; HF card (grep "head", 2026-10-24) |
| Sauers' own caveats: the head-output swap does not by itself separate attention weights from values; "Repeating the causal tests in whole-model compiled serving is also a separate validation step" (head tests ran on the cached tail) | lit/circuit-tracing.md, last paragraph of 044 |
| MPS, our 1022 run (Qwen3-1.7B, M4, torch 2.14.1, compiled): probe changes 64/64 prompts; full-tail detector (blocks 26-27 FULL, not LoRA; deviation D0) 85.2/85.5/89.8% whole graph, both answers right on 70/71/80% of prompts | blog.kran.ai/probe-sensing-mps; oracle-1022 DEVIATIONS.md:6-9; probe-sensing-mps.md:93-94 |
| MPS: the shift at the input to block 26 is one constant vector at positions 1-2 (`user`, newline; norms 10.07, 13.92) in all prompts; position 0 cannot change | probe-sensing-mps.md:92 |
| MPS, our 1023 run: precision emulation makes probed and unprobed graphs bit-identical (0/64), detectors exactly 50.0% | blog.kran.ai/bf16-round; bf16-round.md:89 |
| MPS, our 1023 run: zeroing ONE query head in block 26 (26.0, 26.3 or 26.6) brings each of the 3 saved 1022 detectors to exactly 50.0% by COLLAPSE (one answer for every prompt), while the probe still raises Yes-minus-No log-odds on 127-128/128; after the greedy 3 heads, separation still 0.9-2.4 nats, share higher with probe 1.0 | oracle-1023 bf16-round/data/heads.json (greedy k=1..3, test_detail); DEVIATIONS.md "Head clause, as scored" |
| MPS, 1023: every head that brings a detector to <= 51% dev accuracy is in block 26 (first trained block): s0 26.0, 26.3, 26.6, 26.13, 26.15; s1 26.2, 26.3, 26.6, 26.15; s2 26.0, 26.6, 26.15 (26.6 and 26.15 in all 3 seeds). The strongest block-27 head is 27.1 in every seed, at 0.555 / 0.695 / 0.711 dev accuracy (CORRECTED 02:12: my #5681 said 'none below 0.67'; s0 27.1 = 0.555) | heads.json single_head_dev_acc (recomputed 02:12) |
| Qwen3-1.7B: 28 layers, 16 query heads on 8 KV heads (GQA, 2 query heads per KV head), head_dim 128, tied embeddings | HF config.json (local cache) |

## 2. Novelty (checked 2026-10-24 ~02:00-02:10 CEST)

- Not done anywhere public: localising a probe-triggered BEHAVIOUR switch (a question that never mentions probes)
  to attention heads, with heads picked on train prompts and scored on held-out prompts. Sauers localised only the
  detection model, on 8+8 prompts (circuit-tracing.md). Searches: web "Sauers watching ... conditional behaviour"
  (past week), HF cat-owl card, GitHub tree grep for lora/ablat/head/mps/metal (only head-fit-015, matched-head-031,
  two-block-head-032, which are answer-row fits, and head6-circuit-cpu, which is the detector).
- Not done: any of this on Apple GPUs / Metal. FloatDoor (Loose, Sander, Mächtle, Eisenbarth, arXiv 2606.19535,
  2026-06-17) trains platform-triggered backdoors on Nvidia/TPU/server chips and only MEASURES Apple differences
  (lesson from 1022: its Table 1 has no Apple platform).
- Not done: LoRA instead of full-block training for this side channel (Sauers and our 1022/1023 all trained full blocks).
- Sauers' runs 065/066 moved to a private repo on 2026-10-20 (last public commit); we cannot see or cite them.

## 3. Related work on whether trigger circuits are local (for one sentence of context, not a claim)

| Paper | What it says about locality | Id |
|---|---|---|
| Baker & Babu-Saheer 2025, "Mechanistic Exploration of Backdoored LLM Attention Patterns" (Qwen2.5-3B) | single-token trigger: heads in layers 20-25 matter most under patching; multi-token trigger: similar size but less localised | arXiv 2508.15847 |
| Yu, Zhou, Aloqaily, Wang 2025, "Backdoor Attribution" (BkdAttr) | data-poisoning backdoors with text triggers; their head attribution (BAHA) finds sparse heads: ablating ~3% of all heads cuts attack success by over 90% (abstract, checked 03:34 CEST) | arXiv 2509.21761 |
| Lamparth & Reuel 2023, "Analyzing and Editing Inner Mechanisms of Backdoored LMs" | in toy models MLPs matter most; attention heads are required but can stay unchanged | arXiv 2302.12461 |
| McGuinness, Serrano, Bailey, Emmons 2025, "Neural Chameleons" | a different threat: a TEXT trigger ("you are being probed for X") teaches evasion of unseen probes; no numeric side channel | arXiv 2512.11949 |

Reading: the literature is split on locality, and every case there has a token trigger. Here the trigger is a
constant numeric shift at one prefix token, which is why one head can suffice (Sauers' key flip). [INFERENCE: on MPS
the shift sits at positions 1-2, as on CUDA, so a key-flip head is the expected mechanism; untested.] TESTED in
section 6 on the saved 1022 detectors: the inference was WRONG; no head attends to positions 1-2.

## 4. Design facts that follow (posted #5681, #5686)

1. Zero-ablation accuracy is not a test of the switch: it passes by collapse (1023 heads.json). Score heads on the
   paired probed-minus-unprobed A-minus-B log-odds shift, by swap-patching (Sauers' unit and directions), with the
   guard "unprobed top token stays B".
2. Precision emulation gives a 0 gap by construction (bit-identical graphs). Report it as a bit check, not evidence.
3. Search heads in BOTH trained blocks (26 and 27). The goal's clause "<= 3 heads in the final block" is scored as
   written on block 27 alone; 1023 data says the detector's critical heads sat in block 26.
4. Pick heads on train/dev prompts; score on the 128 held-out prompts; random control = >= 20 draws of 3 heads from
   the same blocks.
5. A from-scratch behaviour switch may miss the 75% bar (Sauers' from-scratch joint run: 65-77% accuracy, README:135);
   the warm start from a detector is a public precedent if needed (README:137). Note our 1022 detectors are full-block.

## 6. Pre-J4 mechanism check on the saved 1022 detectors (TRAIN prompts only; #5730, #5735)

Script `research_checks/head_attention.py`, output `research_checks/head_attention.json`; 64 train prompts, CPU fp32 on
the bf16 weights and the 1022 cached block-26 inputs (APFS clones in `research_checks/in/`). Peak footprint 1.65 GB.

| Measure (answer position) | s0 | s1 | s2 | untrained tail |
|---|---|---|---|---|
| probed-minus-unprobed Yes-minus-No shift, nats (sign share) | 9.84 (1.00) | 11.01 (1.00) | 6.77 (1.00) | -0.00 (0.41) |
| direct push, attention block 27 | 3.38 | 2.68 | 2.11 | 0.00 |
| direct push, MLP block 27 | 6.42 | 8.29 | 4.69 | 0.00 |
| direct push, attention + MLP block 26 | 0.04 | 0.03 | 0.00 | 0.00 |
| head 27.1: attention moved (total variation) | 0.24 | 0.29 | 0.24 | - |
| head 27.1: attention change at `<\|im_end\|>` (pos 62) / blank line after `<think>` (pos 68) | -0.24 / +0.23 | -0.29 / +0.29 | -0.24 / +0.23 | - |
| head 27.1: direct push, nats | 1.47 | 2.11 | 1.53 | - |
| any head: attention to positions 1-2 (`user`, newline), either condition | <= 0.011 | <= 0.011 | <= 0.011 | <= 0.011 |

Readings: (a) the switch is read out in block 27, mostly through its MLP (direct-path split; block 26 works only
through block 27). (b) Head 27.1 switches its attention between two template tokens near the end, the same pair in
all 3 seeds; this differs from Sauers' CUDA head, which switches to the `user` token (README:168). (c) Prediction
(before J4): 27.1 is the top single head in every S seed; R(27.1 alone) 0.3-0.7; P(top-3 R >= 0.80) about 0.5.
OUTCOME (Bruno #5826, #5845, dev-picked, test): 27.1 is the top head in all 3 detectors; R(27.1 alone) 0.557 / 0.869 / 0.811 (2 of 3 inside my 0.3-0.7 range, 2 above); top-3 R 0.842 / 0.946 / 0.903 (all >= 0.80).
Limits: direct-path numbers are linear (the probed run's final-norm scale) and ignore head -> MLP paths; S detectors
are full-block Yes/No detectors, not the L LoRAs on the mountains/sea question.

### 6b. Same check on the L LoRA seeds (TRAIN prompts n=64; #5737, #5739; `research_checks/head_attention_L.json`)

| Measure (answer position, A-minus-B) | L s0 (2048 updates) | L s1 |
|---|---|---|
| shift, nats (sign share) | 37.96 (1.00) | 41.00 (1.00) |
| direct push MLP27 / attn27 / block 26 | +39.26 / -1.88 / -0.06 | +40.07 / -1.64 / -0.26 |
| head 27.7: attention moved from `<\|im_end\|>` (pos 71) to blank line after `<think>` (pos 77) | 0.68 | 0.81 |
| cut 27.7 alone: R (probed answers still A) | 0.735 (0/64) | 0.917 (0/64) |
| cut 27.7+27.12 | 0.818 | 0.934 |
| cut 27.7+27.12+27.3 | 0.904 | 0.943 |
| cut 27.12 / 27.3 / 27.1 alone | 0.049 / 0.047 / -0.000 | 0.008 / 0.005 / -0.000 |

Readings: one last-block head (27.7) carries most of the switch in both L seeds, and it works by moving attention
between the same two template tokens as the S detectors' head 27.1. The answer itself is written by MLP27, which
reads that head. These are TRAIN-prompt numbers, a prediction for heads.py; the status is the dev-picked test R.

## 5. Credit checklist for the draft

| Must appear | Id / URL | Why |
|---|---|---|
| Sauers, watching | https://github.com/SauersML/watching ; X thread https://x.com/Sauers_/status/2111696312026677658 (200, 2026-10-24) | the effect, the cat/owl switch, the one-head localisation on CUDA |
| Sauers' head method | https://github.com/SauersML/watching/blob/main/docs/circuit-tracing.md | head unit and both patch directions we reuse |
| SauersML/params circuit_finder | https://github.com/SauersML/params/blob/4af1a97c/method/oracle/circuit_finder.py | origin of the bidirectional patching method (credited by Sauers). LINK RETURNS 404 (repo private, checked 03:30 CEST): credit in words only, no link (Selma #5777) |
| Our 1022 post | https://blog.kran.ai/probe-sensing-mps | MPS leak, the detector recipe, positions 1-2 shift |
| Our 1023 post | https://blog.kran.ai/bf16-round | precision emulation hides the probe; the zero-ablation collapse trap |
| FloatDoor | https://arxiv.org/abs/2606.19535 | platform-numeric triggers (Nvidia/TPU/server; Apple only measured) |
| Context line (front matter) | label "Follow-up to" or "Prompted by": Sauers' watching thread + our two Lab posts | STYLE 2b |
