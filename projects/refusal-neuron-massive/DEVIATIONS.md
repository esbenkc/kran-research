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
