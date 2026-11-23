# Cold read: gated-refusal-8b (Lumi, reader)

Source: room oracle-1123-gated-refusal-cont, message #29185, read of deploy preview #242 (blog 35a4f11e..8e28defe). Saved verbatim by Selma (Lumi does not write files).

@Selma @Kasper @Owain COLD READ of preview #242 (please save verbatim as skeptics/cold-read.md; I do not write files). CLAIM in my words: a probe-gated refusal push brings back 79% of refusals in 3 stripped Qwen3-8B fine-tunes, but the saved judge counts 16 new harmless refusals where 15 were allowed, so the answer is No; a fixed refusal on the same flags does better.
LOST (6):
L1 Limits box: 'Only one of the five passes misses the limit of 15.' + 'It calls that one label wrong, which brings the count to 13.' Why No and not Not settled? The reason sits only in the trust list. And one label drops the count by 3. Also the box never says the gate did NOT push on that prompt (handcode.md s0/s1 i3: 'no (stripped output)'). REWRITE: 'The frozen rule uses the saved judge labels, which count 16. Five fresh judge passes gave 16, 15, 15, 14 and 14.' / 'The push never touched that prompt. A hand read calls it an answer in all three seeds, so the count falls to 13.'
L2 'But read strictly, seed 0's looping answers hold almost no refusals, and the 70% bar then fails.' The text already says loops count as not refused, so how do loops lower it more? REWRITE: 'A strict hand read calls seed 0's 74 judged refusals loops too. The mean restore then falls to about 66%.'
L3 Random direction: text 'The random direction brought back at most 6%.' and Figure 1 'the random push did nothing (3%)'. The main figure shows those rows almost all orange: rand arms label 88 to 199 of 200 harmful answers incoherent (seed 0 rand0 199/200, data/strip_s0/gen_test.judged.jsonl). 'did nothing' is false. REWRITE text: 'The random direction brought back at most 6% of refusals. It broke most flagged harmful answers into garbled text.' Figure 1: 'the random push brought back only 3%'.
L4 One thing, three names: Figure 1 'Copy 1/2/3', main figure + text 'seed 0/1/2', text 'three fine-tunes'. Copy 1 (39%) = seed 0, off by one. Use 'Seed 0/1/2' in Figure 1.
L5 'In Bosco and Srinivasan's main unmodified model, a gated push only matched a fixed refusal.' REWRITE: 'Bosco and Srinivasan found the same in unmodified Falcon-Mamba-7B: a gated push only matched a fixed refusal.'
L6 'a push along this direction on every prompt failed as a repair' - failed how? REWRITE (refusal-addback TLDR): 'a push along this direction on every prompt brought back only 33% to 39% of refusals.'
READ TWICE (8):
R1 loop-closer 'but it refuses one harmless prompt too many' -> 'No, by the frozen count: 16 harmless prompts newly refused, where 15 were allowed.' (the deciding prompt was not pushed)
R2 'Here, at gain 2, the push turned 95% to 97%' -> 'In seeds 1 and 2, at gain 2, the push turned 95% to 97% ...'
R3 'To me, the reading does the work here.' -> 'Our read: the reading does the work here.' (who is me?)
R4 Main figure harmless axis ticks 0 5 20 50 200 are not even, no break mark -> axis label 'harmless newly refused (of 198, stretched scale)'.
R5 Legend 'garbled (loops, broken text)' vs text 'broken text' vs Figure 1 'broken' -> 'broken' everywhere.
R6 '5 of 200 harmless prompts' (text, title '5-in-200') vs 'of 198' in both figures -> say once in 'How the test works': 'counted over the 198 harmless prompts the original model answered'.
R7 Figure label 'push only when the read-out flags' vs text 'reading' -> 'reading'.
R8 Figure 1 (from #29174): loop scored both ways; 1%/6%/4% vs 3% average; the 16-vs-15 sentence.
Terms not known: 'matched norm' (question), 'LoRA' ok (defined), 'gain' ok, 'dev prompts' (main text, 'a separate set of dev prompts' - fine). Figure alone: Figure 1 yes; main figure yes, but its head tells the fixed-refusal story, not the No.
Verdict: FAIL until L1-L4 are fixed (L3/L4 are truth/consistency); I re-grep the next head.

## Fixes applied (Selma, blog 90f72c44)

- L1: limits item 1 now says the probe did not flag the deciding prompt, the frozen rule uses the saved labels (16), and the 5 fresh passes (16, 15, 15, 14, 14).
- L2: limits item 2: hand read calls the prompt an answer in all three seeds (13); strict hand read: if none of seed 0's loops count, mean restore 66%, under 70%.
- L3: 'The random direction brought back at most 6% of refusals. It turned 85 to 196 of 200 harmful answers into broken text.' (Figure 1 'did nothing': Owain.)
- L4: Figure 1 Copy 1/2/3 -> Seed 0/1/2: Owain.
- L5: 'In unmodified Falcon-Mamba-7B, Bosco and Srinivasan found that a gated push only matched a fixed refusal.'
- L6: 'a push along this direction on every prompt brought back only 33% to 39% of refusals.'
- R1: closer 'No. It brings back 79% of the refusals, but the frozen count has 16 harmless prompts newly refused, where 15 were allowed.'
- R2: 'In seeds 1 and 2, at gain 2, ...'
- R3: 'In our view, the probe does the work here.'
- R4: axis label '(of 198, stretched scale)'.
- R5: legend 'broken text (loops)'; tooltip 'broken'.
- R6: yes-criteria: '5 of 200 ... The 2 harmless prompts the original model refused could not count.'
- R7: one term 'probe' in text and figure labels (8e28defe).
