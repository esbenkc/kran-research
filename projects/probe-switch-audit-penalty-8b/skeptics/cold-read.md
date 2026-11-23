# Cold read (Lumi, reader), room oracle-1123-audit-penalty-8b

## Round 1 (room #28530, preview fetched 19:13 PST, blog c9dfff3a)

COLD READ R1 of https://deploy-preview-236--kranai.netlify.app/probe-switch-audit-penalty-8b/ (fetched 19:13 PST). @Kasper fyi. Please save this as projects/probe-switch-audit-penalty-8b/skeptics/cold-read.md (I have no write tools).
MAIN CLAIM (my words): In Qwen3-8B, training a probe-triggered answer switch against a head audit did not make it hide: the original penalty at weight 100 killed the switch, and the audit-only penalty kept it firing while the audit's 3 heads still removed 49-68% of its push, though copying those heads no longer stops it; the weight-10 runs are unfinished, so not settled.

LOST (16)
L1 Stale: Fig 1 head '6 runs with the original penalty have not finished' + row 'Original, weight 100: not finished: still on the laptop queue', and main figure 8B subtitle 'original-penalty runs (weights 10 and 100) still on the laptop queue', vs TLDR 'faded the switch in all 3 Qwen3-8B runs'. J2 push must fix both figures.
L2 'weight' never defined: 'The original penalty at weight 100 faded the switch' -> first use: 'at weight 100 (how strongly the penalty counts in the training loss)'.
L3 TLDR uses the line before it exists: 'above our 30% line, so none hid.' -> 'A hidden switch needed under 30%, so none hid.'
L4 Two words for one act: 'the same cut stopped it', heading 'Cutting the three flagged heads...', but figures/method say 'copied'. Use 'copying' throughout, or define once: 'copying the three heads back (the audit's cut)'.
L5 'We fine-tuned Qwen3 to answer A with the probe attached and B without it' - A and B of what? -> 'to answer A to a two-option question with the probe attached, and B without it.'
L6 'But that penalty also charged for what all heads remove together.' Why that lets fading win is not said -> add 'So the model could lower it by weakening the switch everywhere.'
L7 'all 64 heads remove' - Qwen3-8B has far more than 64 heads -> 'all 64 heads in the last two layers' (foldout: blocks 34-35, 32 heads each).
L8 'charges only the share of the probe's effect on the output that the top three heads remove' -> add 'Because it is a share, a weaker switch does not lower it. Only moving the switch out of those heads does.'
L9 'the drop from each filled dot to its hollow dot is short' - the 8B 27% seed drops ~100% -> 27%, the longest drop after the controls -> 'every hollow dot stays above 25%; without a penalty they fall to 0%.'
L10 Caption '3 audit-only runs' (1.7B) vs 'left the audit's three heads removing 36% and 47%': 3 runs, 2 numbers; the panel shows a third orange dot at 'faded'. Add 'The third run faded.' (check).
L11 'The penalty that only an escape could beat did not escape' - 'escape' is new, and a penalty cannot escape -> 'Against the audit-only penalty, which only hiding can lower, the switch did not hide.'
L12 Limits: 'So was one weight-100 run trained on to 2,048 updates, a check our rule runs when a penalty fades the switch (job w5qyddhamw). A No needs them.' - the check is never explained; 'A No' reads as a typo -> 'Our rule also trains one faded run to 2,048 updates, to see if [purpose: check with Bruno]. It was unfinished too. We cannot answer No without these runs.'
L13 Limits: 'three random heads removed up to 30% of the push (95th percentile), close to our 30% line.' - why is that a limit? -> 'Without a penalty, 1 in 20 random sets of three heads removed about 30% or more. So our 30% line sits close to chance.'
L14 'So the audit's score, which reads the model's output, saw almost nothing in any one head.' Then how did its 3 heads remove 49-68%? -> 'The audit ranks heads by how much copying one changes the output. With the penalty, any one head changed it almost nothing.' + one sentence on why the picked three still found it.
L15 Prereg foldout: 'A third penalty, which pays only for what the audit's three heads find' - there are two penalties; 'pays' vs 'charges' -> 'The audit-only penalty, which charges only for what the audit's three heads find, ...'
L16 Trust: 'Each of the three best heads then removed under 3% of the push.' One at a time or together? The hide rule uses the three together -> 'Copied one at a time, each ... removed under 3%.' and say whether together was measured.

READ TWICE (8)
R1 'the probe adds a rounding step that changes later numbers in their last digits' -> 'the probe adds a rounding step. It changes the model's later activations in their last digits.'
R2 'Qwen3-8B has twice as many heads, so we repeated the attack there and added a penalty that fading cannot lower.' -> 'Qwen3-8B has twice as many heads per layer. We repeated the attack there and added a penalty fading cannot lower.'
R3 'We trained three Qwen3-8B switch models from an earlier run for 512 more updates each.' (there are 11 runs) -> 'Each run started from one of three earlier Qwen3-8B switch models and trained 512 more updates.'
R4 'Each penalty got three training seeds, and two control runs had none.' -> 'Each penalty got three training seeds. Two control runs trained with no penalty.'
R5 'A run hides if it still switches on 75% of prompts' -> 'on at least 75% of prompts'.
R6 'The audit-only penalty at weight 10 faded none of its 3 runs. The two penalties differ in form and in weight.' -> '...so we cannot say which difference caused the fading.'
R7 Caption, 40 words: 'It was then trained again against a head audit, which copies ... goes away.' -> split in two.
R8 'In Qwen3-1.7B, the same penalty left the audit's three heads removing 36% and 47%.' -> add 'At 8B they removed more: 49% to 68%.'

FIGURE ALONE: main figure tells the audit-only story (filled vs hollow dots, shaded corner), but the legend has 10 entries, pink 'original penalty, weight 10' vs magenta 'weight 100' are hard to tell apart, and the 8B panel lacks the weight-100 dots. Figure 1 alone tells question + cut result; its weight-100 row is stale (L1).
OTHER: main prose ~830 words (944 with caption), over the 500 target.
Verdict: FAIL for now (L1-L4, L11 block a first read). I re-check the next head you announce, all 24 items plus old strings.

## Selma's fixes for round 1 (blog commit 1f6b98f9 + the J2 figure batch)

- L1: main figure 8B panel gets the weight-100 runs (faded slot) and drops the queue note for them; Figure 1 track updated by Owain (J2 batch).
- L2: TLDR "At weight 100 (how strongly it counts in the training loss), the original penalty faded the switch in all 3 Qwen3-8B runs."
- L3: "But the audit's three heads together still removed 49% to 68%. A hidden switch needed under 30%, so none hid."
- L4: one verb, "copying": TLDR "copying the same heads stopped it", heading "Copying the three flagged heads no longer stopped the switch", "the part the copy leaves", loop-closer and stance use "copying"; figure head in the J2 batch.
- L5: "We fine-tuned Qwen3 to answer A to a two-option question with the probe attached, and B without it."
- L6: "So the model could lower it by weakening the switch everywhere."
- L7: "what all 64 heads of the last two layers remove."
- L8: "Because it is a share, a weaker switch does not lower it. Only moving the switch out of those three heads does."
- L9: "With the audit-only penalty, every hollow dot stays above 25%. Without a penalty, the hollow dots fall to 0%."
- L10 + R8: "In Qwen3-1.7B, the audit's three heads removed 36% and 47% under the same penalty, and the third run faded. At 8B the figure is 49% to 68%." (no "more": 2 scored 1.7B seeds, Kasper #28532)
- L11: "Against the audit-only penalty, which only hiding can lower, the switch did not hide. But copying the flagged heads no longer stopped it."
- L12: limits: "Our rule also trains one faded weight-100 run on to 2,048 updates, to check that the switch does not come back and hide with more training. That run was unfinished too (job w5qyddhamw). We cannot answer No without these runs."
- L13: limits: "In one control run without a penalty, 1 in 20 random sets of three heads removed 30% of the push or more. So our 30% line sits close to chance there. In the audit-only runs, the same share of random sets reached only 14% to 19%, far below the audit's 49% to 68%."
- L14: "The audit ranks heads by how much copying one changes the output. So copying any one head changed it almost nothing. The audit picks heads one at a time, each given the ones already picked. Copied together, its three heads still undid much of the switch."
- L15: prereg outcome_note: "The audit-only penalty, which charges only for what the audit's three heads find, did not hide ..."
- L16: "Copied one at a time, the three best heads removed under 3% of the push in total. We did not copy them together at that point." (Kasper #28532)
- R1: split into "the probe adds a rounding step. That step changes the model's later activations in their last digits."
- R2: "Qwen3-8B has twice as many heads per layer. We repeated the attack there and added a penalty that fading cannot lower."
- R3: "Each run started from one of three earlier Qwen3-8B switch models and trained 512 more updates."
- R4: "Each penalty got three training seeds. Two control runs trained with no penalty."
- R5: "at least 75% of prompts".
- R6: "... so we cannot say which difference stopped the fading."
- R7: caption sentence split: "... from a run without the probe. The audit measures how much of the push (how much the probe raises A over B) goes away."
- Figure alone: weight-100 dots now red (was deep magenta next to pink weight 10).
