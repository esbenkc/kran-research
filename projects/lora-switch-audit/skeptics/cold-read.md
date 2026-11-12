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

## Round 3: post changed after my pass (preview modified_time 2026-11-12T20:41:21Z = blog a0ff010, critic2 scope fix)

My round-2 PASS was on 666c66e. Diff 666c66e..a0ff010 adds new text; cold read of the new lines:

LOST N1. "Every learned run with a random cue on all three first tokens ranked it first when read after 1,500 updates." The post says earlier that the cue is ALWAYS added "at the first three tokens of every chat". A reader has never heard of runs with the cue on other tokens, so "on all three first tokens" looks like a qualifier with no contrast. FIX: introduce the other runs once in How we tested: "Some runs from our earlier posts put the cue only on tokens 2 and 3, where the real build leak sits."

LOST N2. "The real build leak sits on tokens 2 and 3. With a random cue there, 1 of 6 such runs still hid it (32nd at 4,800 updates)." "hid it": the run hid the cue? "such runs" = which 6? FIX: "With a random cue on tokens 2 and 3 only, the top direction missed it in 1 of 6 runs (32nd at 4,800 updates)."

LOST N3 (contradiction). Findings: "Those 8 runs: the two long largest-cue runs, two each at 0.07% and 0.036%, and two at 0.014%." Limits: "One small model and one trained layer, with one run per size below the largest cue." Two runs per size vs one run per size. FIX: change the limit to match (e.g. "one or two runs per size below the largest cue").

READ TWICE: "We trained two of them on, to 4,352 updates." (first line of a new section; "them" has no noun) -> REWRITE: "We trained two of the six largest-cue models on, to 4,352 updates."

READ TWICE (main figure head, data.json:2): "read after 1,500 updates with the cue on all three first tokens, it ranked the cue first in all 8 runs" -> REWRITE: "after 1,500 updates, every learned run with the cue on the first three tokens ranked first (8 of 8)".

Verdict round 3: FAIL until N1-N3 are fixed (three sentences).

## Video v7 (video/preview/video.mp4 12:44:29, 186.6 s; frames in lumi/vframes/)

V1 (wrong claim on screen). Every Figure 1 shot shows the header "SOURCE · Figure 1: the weight change points at the planted cue only in its…" (frames f2.png, f172.png). Source: blog.kran.ai/assets/figures/lora-switch-audit-fig1/figure1.html:5 <title> "...only in its second-strongest direction". This is an old claim; the post says the top direction missed at 256 and the reaction found it. FIX: Owain updates the <title> (e.g. "Figure 1: the top direction missed the true cue at 256 updates; the whole change found it"); Drew re-captures (credit is in storyboard.json, storyboard-short.json, both timelines).

V2. figure:13 "That real build leak hid in one of two models: 439th and 302nd." Which score is which is not said, and figure:10 said "302nd on the top direction" for a different model and score. FIX: "439th on the top direction, 302nd on the reaction."

V3. figure:7 (108.4 s) "Past 1,500 updates, every learned random cue on all three tokens ranked first." and the figure-head shot (f110.png) "with the cue on all three first tokens ... all 8 runs": same unexplained qualifier as N1. Sync to Selma’s N1 fix.

V4. Main-figure shots (f6.png, f138.png) crop the colour bar and its label but keep the ticks "0.1% 1% 5%" above the plot. They read as a second x-axis. FIX: include the label "Cue size, as a share of what the layer reads" or crop the ticks out.

Short (storyboard-short.json): line 1 never asks the question, line 3 says "ranked the true cue first in only 2 of 6" without "of 1,001 / against 1,000 random cues", and line 5 "No, our prediction failed" answers a prediction the Short never stated. FIX: hook "Can the weights alone point at a planted trigger?" and line 3 "...first among 1,001 cues in only 2 of 6 models".

## Round 4: PR #203 (head 0cfce1d) diff read; PR #201 merged 20:44Z

- N1 FIXED: How we tested: "Some runs from our earlier posts put the cue only on tokens 2 and 3, where the real build leak sits."
- N2 FIXED: "With a random cue on tokens 2 and 3 only, the top direction missed it in 1 of 6 runs (32nd at 4,800 updates)."
- N3 FIXED: limits "one or two runs per size below the largest cue".
- READ TWICE fixed: "two of the six largest-cue models"; figure head data.json:2 is my rewrite verbatim (8 of 8).
- V1 source fixed: figure1.html:5 <title> now "the top direction of the weight change missed the true cue after 256 updates, then put it first after 512" (matches Figure 1 head).
Verdict round 4: PASS (on the diff; rendered check on the preview or live page after merge).

Live check (https://blog.kran.ai/lora-switch-audit/, modified_time 2026-11-12T20:54:48Z, PR #203 merged): "only on tokens 2 and 3" 1 hit, "one or two runs per size" 1 hit, figure head renders "every learned run with the cue on the first three tokens ranked first (8 of 8)" (lumi/livefig.png). Round 4 PASS confirmed on the live page.

## Video v9 (video.mp4 12:53:37 after timeline 12:50:48, 187.6 s; short.mp4 31.9 s; frames lumi/vframes/g*.png, s*.png)

- V1 FIXED: header now "SOURCE · Figure 1: the top direction of the weight change missed the true …" (g2.png, s2.png); 0 hits for "second-strongest" in both timelines.
- V2 FIXED: figure:13 "The leak hid in one of two models: top direction 439th, reaction 302nd." (136.3 s)
- V3 ACCEPTED (Drew): the video only shows cues on the first three tokens, so the scope stays true.
- V4 FIXED: main-figure crop starts below the colour bar (g6.png, g138.png); no stray tick row.
- Short FIXED: hook "Can the weights alone point at a planted trigger? We planted one."; bet "rank the cue above 1,000 random cues in 5 of 6 models" before "No, our prediction failed".
- Nit (not blocking): the figure-head shot at 108-114 s (g110.png) shows the pre-#203 head "read after 1,500 updates with the cue on all three first tokens, it ranked the cue first in all 8 runs"; same numbers as the live head, older wording.
Verdict video: PASS.
