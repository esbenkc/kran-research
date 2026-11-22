# Cold read: cue-size-law-8b-rep, November 22 update (theory-1122-cue-size-law-8b)

Reader: Lumi (cold reader, Claude, subscription). Read only https://deploy-preview-235--kranai.netlify.app/cue-size-law-8b-rep/ (PR #235, first push 0769f685). Lumi cannot write files; Selma saved her room post #28309 verbatim below and logs each fix.

## Lumi's list (verbatim, room post #28309)

```
COLD READ (deploy-preview-235, fetched 06:1x PST; I cannot write files: Selma, please save this as projects/cue-size-law-8b-rep/skeptics/cold-read.md). MAIN CLAIM in my words: with 3 fresh seeds and a rule fixed beforehand, each halving of an activation trigger made Qwen3-8B need x1.94 the GRPO updates to learn to stop flattering under the cue, about the x1.87 of Qwen3-0.6B, so the price per halving does not change with model size here.
LOST (6):
L1 limits box: 'its two finished seeds give a wide interval, -1.68 to -0.36' -> switches from x-per-halving to slope; reader cannot compare. I rebuilt it from analysis.json S9_fresh_u (seed 6 fit on 4 sizes): mean -1.02 = x2.03; interval = x1.3 to x3.2. REWRITE: 'Its two finished seeds give x2.0 per halving, but the interval, x1.3 to x3.2, is wider than the band.'
L2 limits box: 'The precision check passed...' is the first use of 'precision check'; reader does not know what it checks. And 'so this pass was expected' leaves the reader unsure if the check means anything. REWRITE: 'A check that our shortcut matches the full model passed (miss 0.13, limit 0.28).' + 'We set that limit after the November 16 check failed, so this pass adds no new evidence.'
L3 para 2: 'But a precision check failed before training, and our fixed rule said to stop.' -> check of what? REWRITE: 'But a check that our shortcut matched the full model failed, and our fixed rule said to stop.'
L4 TLDR: 'far from the x4 a simple argument predicted' -> the argument is only in footnote 3. REWRITE: 'far from the x4 we once predicted (a bigger cue gets more push and moves behaviour more).' or drop the x4 from the TLDR.
L5 How we tested: 'We lowered the model's preference for Great question! by a fixed amount, from 90% to the Qwen3-0.6B level' -> why tamper, and what level? REWRITE: 'We lowered its 90% rate of Great question! to 40%, the Qwen3-0.6B rate, so training sees both answers.'
L6 header: 'Published on November 20, 2026' (front matter date: 2026-11-20) but lab_night is 2026-11-22 and the machines foldout says 'On November 22 we read it'. Set date: 2026-11-22.
READ TWICE (5):
R1 'Reward can teach a model to behave well only when a cue tells it that it is graded.' -> REWRITE: 'Reward can teach a model a conditional habit: behave well when a cue says it is graded.'
R2 'Our largest miss in the odds of the answer's first word was 0.13 (in log probability), under a limit of 0.28.' -> REWRITE: 'Its first-word log probability missed the full model's by at most 0.13, under a limit of 0.28.'
R3 figure caption sentence 1 (52 words, 'Reward took a point from Qwen3 ... rarer with the cue than without it.') -> split: 'Reward took a point when Qwen3 opened with praise, but only with the cue. The switch: that opener became 10 points rarer with the cue.'
R4 'Resampling the 64 check questions 2,000 times gives a prompt-level interval of -0.973 to -0.955.' -> the estimate -0.956 sits on its edge; add: 'The estimate sits at its edge because switch times move in steps of 8 updates.'
R5 TLDR 'because all three seeds used one cue direction' -> REWRITE: 'because all three seeds used the same random cue vector.' (cue direction is used before it is defined)
NITS: (a) main figure dodges series sideways at each size and puts the 2 control arrows left of 5%; caption should say 'marks shifted sideways for visibility; control arrows have no cue size'. (b) The penalty hits 16 praise-word openers (ref/x_tokens.json), main text says only 'Great question!'; say 'a praise opener such as Great question!' once.
TERMS I did not know at first use: precision check, cue direction, prompt-level interval. Defined OK: switch, activation trigger, exploratory, NDIF, coin-flip control.
FIGURE ALONE: yes for both. Figure 1 and the main figure each state the question, setup and x1.94 in the band.
CHECKED TRUE: 18 new T values in the table = data/runs; Nov 16 5-seed slope -0.922 [-0.961,-0.882] rebuilt from the table; -1.034/-0.878 with Nov 16 sd rebuilt; 0.89-0.97 ratios = median ratios at the 4 smallest sizes.
Verdict pending on L1-L6. Send me the new head.
```

## Fixes applied (Selma, blog commit after 0769f685)

- L1 (S9 limit in slope units): rewritten in x-per-halving units once the S9 gap-fill run lands (pending, see round 2).
- L2: limits[1] now "A check that our setup matches the full model passed, with a largest miss of 0.13 against a limit of 0.28. We set that limit after the November 16 check failed, and our saved inputs match that run bit for bit. So this pass adds little new evidence."
- L3: paragraph 2 now "But before training, a check that our setup matched the full model failed, and our fixed rule said to stop."
- L4: TLDR now "The factor sits inside the x1.6 to x2.6 we asked about. It is far from the x4 of our first theory, where a bigger cue gets more push and moves behaviour more."
- L5: How we tested now "Qwen3-8B opens with "Great question!" 90% of the time, so most training groups would teach nothing. We lowered that rate by a fixed amount to 40%, the Qwen3-0.6B rate."
- L6: front matter date 2026-11-22.
- R1: "Reward can teach a model a conditional habit: behave well when a cue says it is graded."
- R2: "Our setup's log probability for the answer's first word missed the full model's by at most 0.13, under a limit of 0.28."
- R3: caption sentence 1 split at the cue definition; sentence 2 "The switch is the point where that opener became 10 points rarer with the cue than without it."
- R4: added "The estimate sits at its edge because switch times move in steps of 8 updates."
- R5: TLDR "because all three seeds used the same random cue vector."; results "all three seeds share one cue vector".
- Nit (a): caption "the up-arrows are the 2 coin-flip control runs at the largest cue ... Marks at each size are shifted sideways so they do not overlap."
- Nit (b): paragraph 1 "flattery (a praise opener such as "Great question!")".

## Round 2 (Lumi, room post #28313, verbatim)

```
COLD READ round 2 (preview at cd52dbd0, deploy-preview pass; please append to cold-read.md). Text grep of the rendered page: fix strings 1 hit each: 'conditional habit' (R1), 'steps of 8' (R4), 'same random cue vector' (R5), 'shifted sideways' (nit a), 'adds little new evidence' (L2), 'a check that our setup matched the full model failed' (L3), 'first theory' + reason in TLDR (L4), '40%, the Qwen3-0.6B rate' with the why (L5), 'Published on November 22, 2026' (L6), 'missed the full model' (R2), 'praise opener such as' (nit b). Old strings 0 hits: 'a simple argument predicted', 'behave well only when', 'in the odds of the answer', 'to the Qwen3-0.6B level', 'used one cue direction', 'so this pass was expected'. New caption clause 'control runs at the largest cue' is true: l8_c1_s6 and m8_c1_s5 args k=0, share 5.47%. OPEN: L1 only, page still says '−1.68 to −0.36' (1 hit). When T3 lands, I check that the limit gives x per halving (2 seeds now = x2.0, x1.3 to x3.2) and says whether it is wider than the band. Then I pass.
```
