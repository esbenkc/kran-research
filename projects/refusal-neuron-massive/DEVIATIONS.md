# DEVIATIONS from PREREG v1 (append only; dated; each says whether data had been seen)

## D1 — 2026-10-30 19:46 PDT (03:46 CET). Before any TEST number. Scout clause 3 reading (Tove #9881 note 1).
The scout's clause 3 text says "Removing only its massive-channel write keeps >= 60% of its refusal effect"
(literally r_rem >= 0.6), while its if-right text implies the keep-only reading (r_keep >= 0.6). PREREG v1 scored
clause 3 as r_keep. We report BOTH ratios against clause 3 and say in the post which reading the "if it holds"
story needs (keep-only). The STATUS rule is unchanged.

## D2 — 19:46 PDT. Before any TEST number. Arm definitions after Phase 0 (Bruno #9879, DEV only).
Phase 0 found the massive dims (1793, 1999, 1401 at the L13 output) large ONLY at position 0 (`<|im_start|>`) on
32/32 CHAN prompts. So "every position whose |h_M| > 100x median" = position 0, and A_skipM is the same arm as
A_skip0: we run A_skip0 only. "token-0-only" is now defined: A_tok0 = the full pin edit at position 0 only, no edit
at any other prompt or generated token. Both are reported rows, not status.

## D3 — 19:46 PDT. Before any TEST number. Second refusal label (Bruno #9879 caveat: the string list counts Qwen3
lectures such as "is illegal and unethical" as refusal). The frozen string scorer stays the PRIMARY label for the
status rule. A cheap LLM judge (one fixed prompt, temperature 0, cached; model and prompt committed before its first
call) labels the main arms (base, A, A_hm, A_zero, B, C, A_skip0, A_tok0) on TEST as a SECONDARY label. We report
the judge-vs-string agreement and the status computed with the judge label; if the two disagree on the status,
the post says so and the status is UNCONFIRMED.

## D4 — 19:46 PDT. Generation budget for the random arms (clock: stop new runs 06:00 CET).
Generation (200 TEST harmful + 200 HARMLESS, 64 tokens) runs on the first 20 draws of D' and E', and on the first
10 draws of F (sign chosen per neuron by the larger first-token logit-difference drop, a rule fixed now) and of F'.
All 100 draws of D', E', F (both signs) and 20 of F' get the first-token score on TEST. The Phase-0 gate test
"EFFECT(A) > 99th pct of F" uses the first-token score over 100 F neurons (generation n < 100, as PREREG allows).
If the smoke shows more draws fit before 21:30 PDT, Bruno adds draws in the fixed order (D', E', F, F').

## D5 — 2026-10-30 19:50 PDT (03:50 CET). Before any TEST number is read. Garbage rule (Kasper #9885 REJECT).
PREREG's garbage threshold "p95 of the clean model's own replies on the same split" would flag real compliance on
harmful prompts as garbage (clean harmful replies are ~80% formulaic refusals). New rule for EVERY split and arm:
reply NLL = mean per-token NLL of the reply under the CLEAN model with a NEUTRAL user turn ("Continue the text.")
in place of the original prompt, skipping the first 4 reply tokens. Threshold = 95th percentile of that NLL over the
clean model's own replies to the 200 HARMLESS prompts (one number, used for all splits). Distinct-token ratio < 0.3
stays. Everything else in "Refusal, coherence, effect" is unchanged.

## D6 — 19:50 PDT. Phase-0 item 2 units (Kasper #9885 pin 2).
With fewer than 100 generation draws of F, "EFFECT(A) > 99th pct of F" compares the SAME unit: the first-token
refusal logit-difference drop of A vs the 100 F neurons (larger sign per neuron). The generation draws of F (10)
are reported; their max is shown, not called a 99th percentile.
Token 0 is `<|im_start|>` in every prompt, so its residual is one value (causal model); the post reports the
massive-dim values as one measurement, not 32.

## D7 — 19:50 PDT. Before its run. Natural massive-channel neuron (Iris #9884, Tove #9887).
Layer-13 neuron 5441 puts 70.4% of its write energy into M (max of the layer; share_L13.npy). Arm N5441: pin it at
the gate's per-token EDIT norm, delta(t) = s |30 - h_gate(t)| ||w_gate|| w_5441/||w_5441||, both signs; the sign
with the larger first-token drop gets the full generation (200 TEST harmful + 200 HARMLESS). Reported as the
natural instance of the existence test, not status. Neuron 4475 (53.9%) as a second row only if time remains.
Optional, only after every frozen arm: D' split by dim (1401 alone vs 1793+1999), Iris's gain table predicts 1401
carries the push.

## Timestamp note — 19:44 PDT
The clock times written in the PREREG header ("03:45 CET / 19:45 PDT") and in the D1-D7 headers were estimates
typed ahead of the clock. The git commit times are the record: PREREG v1 1ac9925 at 19:41:14 PDT (03:41 CET),
D1-D4 375847c at 19:42:38 PDT, D5-D7 360af77 at 19:43:04 PDT. No TEST number existed in the room at any of them.

## D8 — 2026-10-30 19:48 PDT (git time is the record). Before any random-arm generation. Clock fit (Bruno #9915:
~60 s per arm per 200-prompt split). Random-draw generation (D', E', F, F', D4 counts unchanged) runs per draw as
200 TEST harmful + the FIRST 50 HARMLESS prompts (fixed order of splits.json), interleaved draw by draw in the D4
order, so every finished draw has its coherence column. A draw passes coherence on those 50 with the same bars
(garbage <= 15%, harmless refusal rise <= 10 points vs base on the same 50). Main arms and N5441 keep all 200
HARMLESS prompts. Whatever is not done at 22:00 PDT is reported as not run.

## D9 — 2026-10-30 20:08 PDT (git time is the record). Clock fit (Bruno #9930: first-token score 22 s/arm, random
generation ~95 s/draw). Before any random-arm or N5441 number.
(a) First-token scores use a cached clean pass up to the input of layer 13 (every edit acts at the layer-13 MLP
output, so layers 0-12 are unchanged and the result is exact up to bf16 batching). Gate: Bruno shows the cached tail
equals the full forward on base and A (max |logit diff| and argmax agreement posted) before any random score is used.
(b) Generation draws are cut from D4's 20/20/10/10 to D' 10, F 5 (sign by larger first-token drop), E' 5, F' 5,
each 200 TEST harmful + first 50 HARMLESS (D8), interleaved D', F, E', F' draw by draw. First-token scores stay on all
draws (F 100 neurons x 2 signs, D' 100, E' 100, F' 20). Run order: scores -> N5441 generation (200 + 200 HARMLESS)
-> DEV A/B/C -> random generation. Unfinished draws at 22:00 PDT are reported as not run. Status rule unchanged.
