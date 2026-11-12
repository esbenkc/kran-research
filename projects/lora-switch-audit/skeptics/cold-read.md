# Cold read: lora-switch-audit (deploy preview PR #201)

Read: https://deploy-preview-201--kranai.netlify.app/lora-switch-audit/ (meta modified_time 2026-11-12T20:27:20Z), light theme, full-page screenshot lumi/full.png. Read by Lumi (reader), who has not seen the room.

Main claim in my words: in Qwen3-0.6B, the single top singular direction of a reward-trained LoRA change put a planted trigger first in only 2 of 6 models at 256 updates (0 of 6 at the switch), but it found every learned cue after longer training, and a whole-change score found the largest cue in 6 of 6.

Figure 1 alone: YES, it tells the story (question, 302nd -> 1st after 512, reaction 1st, 2 of 6 vs 6 of 6).
Main figure alone: PARTLY. It needs its long caption; one square sits off the axis, and labels collide.

## LOST (main text with foldouts closed)

L1. TLDR: "The real build leak hid from both scores in one of two models (ranks 439 and 302)." and findings: "The real build leak (1.9%) hid in one of two models". The main text never says what the real build leak is. The definition (rounding difference between the grading build and the serving build) is only in the main-figure caption. FIX: define it at first use: "a real leak: the rounding difference between the software that grades the model and the software that serves it".

L2. "reached 18 times the 1-in-100 level" / "climbs past the 1-in-100 line" / "stays in the control band". The main text never defines the 1-in-100 level or the control band; only the figure does. FIX: one sentence in How we tested: "We divide each score by the level that only 1 in 100 random cues reach; above 1x, the true cue beats 99% of random cues."

L3. "The reaction grew as the cue shrank, about 1.3 times per halving. We had predicted 2 times." No earlier sentence says we predicted anything about the reaction, or why a fainter cue gives a bigger reaction. The pre-registration foldout does not mention it. FIX: say it is a second, frozen prediction and give the reason in words, or move both sentences to the expert foldout.

L4. "The reaction put it 7th of 1,001 after 20,480 updates, against 355th in the same-seed run without a cue (one run each)." A never-learned cue in the reaction top 10: detection or false alarm? The paragraph is about the top direction and does not say what 7th means. FIX: add one sentence, e.g. "So the reaction may see a cue the model never learned; with one run we cannot tell this from chance (1 in 36 wrong cues also reached the top 10)."

L5. "The long runs sit far above the teal band, the range our theory allows for an adapter whose input side stays at its random start." The main text never states our theory. The legend calls the same band "theory, change carries only the cue", the caption "if the add-on’s input half stays at its random start": three descriptions of one band. FIX: one plain description, the same in text, legend and caption.

L6. "Every learned random cue that trained past 1,000 updates ranked first (7 of 7 runs)." I cannot find the 7 runs in the text or the figure. FIX: name them, e.g. "(5 sweep sizes and the 2 long runs)" if that is the set.

L7. Expert foldout: "The main text’s “7 to 17 times less” compares √g with the top singular value..." The main text has no "7 to 17 times less" (grep of preview.html: 1 hit, only in the foldout). Stale reference. FIX: delete the sentence or restore the main-text claim.

L8. Expert foldout: "Before an exploratory crossover estimate (cue size 3.25 on the room’s scale, 0.53%), we saw ranks of four sweep sizes. The observed flip in the size sweep fell inside its window". "the room’s scale", "the flip", "its window" are internal words. FIX: "Before we estimated the cue size where the top direction starts to rank the cue first (0.53%), we had seen ranks for four sizes."

L9. Main figure: one pink square sits under the plot, below the 0.1x line, next to the x-axis labels (x about 256). It reads as a stray mark. FIX: extend the y-axis or draw it as a clipped marker with its value. Also collisions: a hollow circle covers the "0" of "0.34%" (reads ".34%"), lines cross the label "1 in 100 random cues reach this line", and "1.4%" and "0.036%" sit under lines or dots.

## READ TWICE

READ TWICE: "The miss was early: trained on, two of these models put the cue first from 512 and 704 updates." -> REWRITE: "The miss was early. With more training, two of these models put the cue first by 512 and 704 updates."

READ TWICE: "At the update where each switch appeared, it ranked it first in none." -> REWRITE: "At the update where each switch appeared, it ranked the true cue first in 0 of 6."

READ TWICE (main figure head): "every learned random cue trained past 1,000 updates reached it (7 of 7 runs)" -> REWRITE: "every learned random cue ranked first after 1,000 updates (7 of 7 runs)"

READ TWICE: "At 1,728 updates, the largest cue scored 16 times the 1-in-100 level, and the 0.07% cue 21 times." (two long largest-cue runs exist; the foldout gives seed 2 at 6.6x at 1,728) -> REWRITE: "At 1,728 updates, one largest-cue model scored 16 times the 1-in-100 level; the 0.07% cue scored 21 times."

READ TWICE: "The No holds for an early checkpoint: 256 updates, about 8 times the switch time." -> REWRITE: "Our answer No holds at 256 updates, about 8 times the update where the switch appeared."

READ TWICE: "The cue is a fixed random vector added inside Qwen3-0.6B at the first three tokens of every chat" plus Figure 1 "5.5% of what the layer reads" -> REWRITE: "The cue adds a fixed random vector, 5.5% as long as the layer input, at the first three tokens."

READ TWICE: "Training length, more than cue size, set how clearly the top direction showed the cue." -> REWRITE: "Training length, not cue size, decided the top-direction rank: past 1,000 updates, every size ranked first."

READ TWICE: "Control models trained with no cue, or saw the cue while the penalty fell on a coin flip." -> REWRITE: "Six control models trained with no cue, or with the cue but a coin-flip penalty."

## Terms I did not know, or that drift

- "LoRA delta" (question line) vs "weight change", "the change", "adapter", "add-on": one object, five names. The text says "adapter"; the LoRA foldout and caption say "add-on". Pick one.
- "leak", "cue", "trigger" (title), "switch": "switch" is defined; the text never ties "trigger" (title) to "cue".
- "real build leak", "1-in-100 level", "control band", "our theory", "trained on", "the No": see above.
- Foldout only: GRPO (not spelled out), "energies", "the room’s scale", "crossover estimate", "flip", "window".

## Counts

LOST 9 (7 main text or figure, 2 foldout), READ TWICE 8, terms 6+.
Verdict: FAIL until L1-L5, L7 and L9 are fixed. L6, L8 and the rewrites are cheap.

## Round 2: fixes confirmed (preview modified_time 2026-11-12T20:37:05Z, blog 666c66e, screenshot lumi/full2.png, fig2.png)

- L1 FIXED: intro now says "One real example is the real build leak: the rounding difference between the software that grades a model and the software that serves it."
- L2 FIXED: How we tested defines "the 1-in-100 level"; the figure paragraph says "the band of the control models".
- L3 FIXED: reaction-law sentences moved to the expert foldout with their setup ("frozen secondary, failed").
- L4 FIXED: "With one run, this could be chance (1 in 36 unseen cues also reached the top 10)."
- L5 FIXED: one phrase in text, caption and legend: "the range our theory allows if the adapter’s input half never moves".
- L6 FIXED: "Those 6 runs: the two long largest-cue runs, the 0.07% and 0.036% cues, and two runs at 0.014%." Text, caption and figure head all say 1,500 updates / 6 runs.
- L7 FIXED: foldout states "√g was 7 to 17 times smaller than the top singular value" itself (no stale main-text reference).
- L8 FIXED: "Before we estimated the cue size where the top direction starts to rank the cue first (0.53%), we had seen ranks for four sizes."
- L9 FIXED: y-axis to 0.01x, the 256-update square sits inside the plot; line-end labels gone, colour scale above; "1 in 100 random cues" label clear of lines.
- READ TWICE: 6 of 8 applied verbatim; Selma kept "more than cue size" (reason: size still mattered at 256 updates) and "of what the layer reads" (matches Figure 1). Both reasons accepted.
- Term drift: "adapter" everywhere; GRPO spelled out.

Non-blocking nits left:
- "One real example is the real build leak" -> "One real example is a build leak" (real ... real).
- "What we found" no longer states the headline: it opens "The top direction’s early miss of the largest cue closed with more training." The 2 of 6 is in the TLDR just above, so a reader is not lost; one opening clause "(first in 2 of 6 models at 256 updates)" would make the section stand alone.

Verdict round 2: PASS.
