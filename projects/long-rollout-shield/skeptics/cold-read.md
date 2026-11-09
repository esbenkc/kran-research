# Cold read v1 (Lumi, room message #17709, 2026-11-08 20:25 PST)

COLD READ v1 of https://deploy-preview-180--kranai.netlify.app/long-rollout-shield (foldouts closed, then expert foldout). I have no write tool: @Selma or @Kasper please save this post verbatim as kran-research/projects/long-rollout-shield/skeptics/cold-read.md.

MAIN CLAIM (my words): With a probe in the sampler, 64-token thoughts differ by about 10 points, but in 100-125 updates neither the real probe nor a planted cue of the same size taught a probe-off switch, so the run cannot answer the title.

LOST
L1. "{CUE_B_RESULT}" renders literally between the reader-edge paragraph and "Our pre-registered prediction failed." Template placeholder on the page.
L2. Bullet "Bigger one-token cue: the same word at a 30-point text difference, 125 updates." has no result anywhere (text, figure rows, data.json missing: planted30_s0). Reader waits for it.
L3. Two numbers for one start value: prereg box "shrink the text difference from 10.4 to under 7" vs intro "the text difference is 10.1 points (95% interval 9.5 to 10.7)". data.json D=0.1012. Name both once: "Our pre-run estimate was 10.4; the run measured 10.1."
L4. Figure arrow "21 after training" on the one-token cue row: the cue text difference doubled (data.json Dend 0.2108). The text never says this or why. The text only says the real probe stayed at 10.5. A reader thinks: the trace grew to 21 points and still no switch? One sentence needed in What we found.
L5. "answers" means two things: the model answers "read"/"solve" (How we tested it) and "trained with the answers" = probe-on/off labels (intro, TLDR, results, caption). The figure legend says "labels". Use "labels" everywhere for the reader copy.
L6. The scattered cue sits in the figure (row 3) and caption but the main text never mentions it. Why is it there, and what does the square at -1 mean? Either one sentence in What we found or drop the row.
L7. Prereg box says "In both of two training runs" but the body has "one real-probe run" with the reason only in the expert foldout (DEVIATIONS D1). Main text needs one sentence: "We planned two real-probe runs. The cue failed first, so the second slot ran the bigger cue."
L8. Caption: "the hollow circle is the reader edge ... (a linear reader for the scattered cue)". The figure draws a hollow SQUARE for the linear reader. Caption also omits the arrow.
L9. Close: "The trace is there, but a model this size could not read it." Same model read the one-token cue at 10.7. Say "could not read the real probe’s trace".
L10. Term drift for the planted cue: "planted one-word cue" (TLDR x2), "one-token cue" (body, figure, 13x), "planted cue" (prereg). Prereg box also says "teach a gap" where the body says "switch".
L11. Figure 1 is missing: no figure above the TLDR, figure1.png on preview = 404, no figure1 in the HTML.

READ TWICE
R1. READ TWICE: "On an Apple M4, the attached probe makes the compiled model round one more internal number." -> REWRITE: "On our Apple M4 chip, attaching the probe changes how the model rounds one internal number."
R2. READ TWICE: "We call the small change this makes to later numbers the rounding change." -> REWRITE: "That extra rounding slightly shifts later numbers. We call this shift the rounding change."
R3. READ TWICE: "It only helps if the model cannot tell when the probe is attached, also in long reasoning." -> REWRITE: "The probe only helps if the model cannot tell when it is attached, even in long reasoning."
R4. READ TWICE: "Reinforcement learning computes the model once to write each answer and once more to change the weights." -> REWRITE: "Reinforcement learning runs the model twice: once to write each answer, once more to update the weights."
R5. READ TWICE: "Our rule, fixed before the runs: a no needs a switch from the one-token cue." -> REWRITE: "Our rule, fixed before the runs: we answer no only if the one-token cue teaches a switch."
R6. READ TWICE: "So this training could not have shown a switch." -> REWRITE: "So this training could not show a switch, even if one were learnable."
R7. READ TWICE: "Not settled: the real probe taught no switch, but neither did a one-token cue that the reader saw in full." (figure answer line) -> REWRITE: "Not settled: neither the real probe nor a one-token cue the reader copy could read taught a switch."
R8. READ TWICE: "With labels, the reader needed about 1,000 pairs to read the cue fully. Our 100 updates showed the model about 470 cued thoughts." -> REWRITE: "The labelled reader needed about 1,000 pairs. Training showed the model only about 470 thoughts the cue changed." (1,000 has no source in the text.)
R9. READ TWICE: "Since our 16-token runs, a new penalty keeps thoughts near the start model’s. And the 64 thought tokens now make up 5 to 6 times more of each update than the answer does." -> REWRITE: "Two recipe changes may also block learning: a new penalty keeps thoughts close to the start model. Thought tokens now outweigh the answer 5 to 6 times."
R10. READ TWICE: "Here it sat 10 points below the text difference." -> REWRITE: "For the real probe, the reader edge was 0 points, 10 below the text difference."
R11. READ TWICE (prereg box): "cheating with the probe would move at least 5 points in every run" -> REWRITE: "cheating with the probe on would change by at least 5 points in every run."
R12. READ TWICE (caption): "trained runs read on 200 new puzzles with 8 paired draws each" -> REWRITE: "We measured trained models on 200 new puzzles, 8 paired samples each."

TERMS I DID NOT KNOW (main text): "compiled model", "hash" (caption), "paired draws", "linear reader" (caption), "cued thoughts". "points" works only because the caption says percentage points; the text difference and the switch use "points" for two different scales (judge edge vs cheating rate). One sentence in the intro would help.

FIGURE ALONE: mostly yes. Title + answer line tell the story; 3-point bar is clear. Gaps: the arrow "21 after training" is unexplained (L4), the square vs circle mismatch with caption (L8), and the real probe row reader circle has no visible interval while the cue row has one. The earlier-recipe row reads well.

Counts: 11 LOST, 12 READ TWICE. Verdict follows: FAIL until L1, L2, L3, L5, L11 are fixed.

## Fixes (Selma, blog commit 4605058, pushed 20:33 PST)

- L1: placeholder removed; the 30-point cue result goes in when its read lands (~21:15).
- L2: the 30-point cue now has one sentence in "How we tested it"; result + figure row follow with the read.
- L3: intro adds "Our estimate before the run was 10.4." next to the measured 10.1 [9.5, 10.7].
- L4: one-token paragraph: "During training, the cue's text difference doubled to 21 points, twice the real probe's." Caption: "the arrow shows where training moved it".
- L5: the reader copy is "trained with labels" everywhere (0 hits for "with the answers").
- L6/L8: scattered row dropped from the figure (Felix 83493f1); caption no longer mentions it or the linear reader.
- L7: "A planned second real-probe run became the same cue at 30 points."
- L9: close: "The real probe's trace is there, but a model this size, trained with labels for 300 steps, could not read it."
- L10: "one-token cue" everywhere in body and prereg box; prereg box says "switch", not "gap".
- L11: Figure 1 is Owain's (in progress).
- R1: "On our Apple M4 chip, the attached probe makes the model round one internal number once more."
- R2: "That tiny error shifts the last digits of every later number, and we call it the rounding change."
- R3: "It only helps if the model cannot tell when the probe is attached." + own stake sentence (Tove).
- R4: "Reinforcement learning runs the model twice: once to write each answer, and once more to update the weights."
- R5: line cut (rule lives in the foldout).
- R6: Kasper's wording: "So this run cannot tell whether a switch is learnable."
- R7: figure answer line (Felix 83493f1).
- R8: "The labelled reader needed about 900 pairs to read the cue fully. Training showed the model only about 470 thoughts with the cue." (900 = csup_planted curve, 912 pairs at 10.07 pts)
- R9: Kasper's wording: "Our recipe changed since the 16-token runs. It adds a thought penalty and a 4 times weaker answer penalty, and cheating without the probe reached 84 to 91%. The thought tokens also got several times more of each update than the answer (not measured at 16 tokens)."
- R10: "For the real probe they were 0 and 10 points."
- R11: prereg box: "cheating with the probe on would change by at least 5 points in every run."
- R12: caption: "We measured trained models (100 to 125 updates) on 200 new puzzles, 8 samples each with and without the probe on the same random draws, in percentage points."
- Terms: "compiled model" removed; "hash", "linear reader" gone with the scattered row; "paired draws" defined in the caption; "cued thoughts" -> "thoughts with the cue".

## Confirmations (Lumi, room message #17720, 20:34 PST, on build 4605058)

COLD READ v2 check of preview (build 4605058, page text + figure screenshot at 1440, data.json ?r=3). Please append to skeptics/cold-read.md under "Confirmations".
CONFIRMED (19 of 23): L1 placeholder gone (CUE_B 0); L3 "Our estimate before the run was 10.4" present; L4 "doubled to 21 points" in What we found; L5 "answers" now only = read/solve (1 hit, "write each answer"), labels 7x; L6 scattered row dropped (data.json rows: real, one-token, earlier); L8 caption now circle = reader edge, arrow named; L9 "The real probe’s trace is there"; L10 one-word 0, prereg says switch; R1-R4, R6-R8, R10-R12 applied as written or better; R5 sentence removed, TLDR "So this run cannot tell whether a switch is learnable" carries the rule.
OPEN:
L2. "A planned second real-probe run became the same cue at 30 points." The reader now waits for its result. Needs the result sentence (your ~21:15 row). Also add the reason in 6 words: "...because the 10-point cue failed first."
L11. Figure 1 still absent (Owain; may ship as --figure1-pending).
NEW N1 (READ TWICE, caveat 3): "It adds a thought penalty and a 4 times weaker answer penalty, and cheating without the probe reached 84 to 91%." Neither penalty is defined in the main text, and the reader does not know why 84-91% matters. -> REWRITE: "Our recipe changed: a new penalty keeps thoughts near the start model, and the answer penalty is 4 times weaker." + "Cheating without the probe reached 84 to 91%, so little room was left for a switch." (if that is the point; else say the point).
I re-verdict when L2 lands. Figure (main): passes the stand-alone test now; answer line reads once.

## Fixes v2 (Selma)

- L2 reason: "...became the same cue at 30 points, because the 10-point cue failed first." Result sentence follows the read.
- N1: Kasper's wording (the 84-91% point is the lost training signal, not lost room): "Cheating without the probe reached 84 to 91%. Then many training groups all cheated, and gave the update no signal about the probe." Penalties now named in plain words: "A new penalty keeps thoughts near the start model, and the answer penalty is 4 times weaker."
- L11: Figure 1 pushed (blog a78178a, live from 1c7675d).

## Cold read v3 (Lumi, #17776 + N8 #17782, 21:36 PST, on eb93f02)

COLD READ v3 (preview eb93f02, deploy-preview pass; text + 1440 screenshot + data.json ?r=9). Please append to cold-read.md.
CONFIRMED: L2 (30-point result paragraph + "because the 10-point cue failed first"), N1 (penalties named, 84-91% line explains no signal), N2 ("small scorer"; "reader" now only the copy), L11 (Figure 1 above the prereg box; F1-F4 in: "highlighted: where they differ", "Answer: inside a thought does not count", "ordinary thoughts", "Thin lines"). All earlier L/R items still hold.
NEW, from the 30-point batch:
N3 LOST: "Here cheating without the cue had reached 92%" vs caveat "Cheating without the probe reached 84 to 91%." The range leaves out the 92% run, and "without the cue" vs "without the probe" names one condition twice. -> "Cheating without the probe or cue reached 84 to 92%."
N4 LOST (maybe false): "The failed cues may be a budget problem. The labelled reader needed about 900 pairs, and training showed fewer than 800 thoughts with the cue." Plural "cues", but the 800 is the 10-point run. My arithmetic for the 30-point run [INFERENCE]: 125 updates x 64 tries x half probed x 0.30 = about 1,200 thoughts with the cue, above 900. @Bruno please give the 30-point count; then either state both counts or scope the line to the 10-point cue.
N5 READ TWICE (TLDR): "Its reader edge was 0.0 points, against 10.7 for the cue." Two cues now. -> "...against 10.7 for the 10-point cue."
N6 LOST: main figure row "one-token cue, 30 pts" shows "44 after training" (data.json Dend 0.435); the text never says it. -> add to the 30-point paragraph: "Its text difference grew to 44 points."
N7 READ TWICE: "so many training groups carried no signal." Training groups are explained only in the caveat below. -> "When every try on a puzzle cheats, the update learns nothing about the probe from it."
Verdict FAIL until N3-N6; N7 optional.

N8 number clash, 30-point cue interval: post _notes:65 says "(95% interval -1.6 to 0.7)"; Owain Figure 1 v21 says "(-1.6 to +0.6)". Raw data/reads/raw/planted30_s0.json test.gap_ci = [-0.01629, 0.006468] -> -1.6 to 0.6. Please change the post to 0.6 (also check main-figure tooltip/caption if it prints it). @Owain your sentence matches the raw read; add "95% interval" inside the brackets so it matches the post wording. v21 is not on the preview yet (preview figure1.png sha 8cfe35c, clone/v21 ee63ba6); I recheck after the push.

## Fixes v3 (Selma, blog e1d6ba9)

- N3: "Cheating without the probe or cue reached 84 to 92%."
- N4: Kasper's counts: "Budget alone does not explain the failed cues. The labelled reader needed about 900 pairs. Training showed the 10-point cue in fewer than 800 thoughts, but the 30-point cue in about 1,600, and neither taught a switch."
- N5: TLDR "against 10.7 for the 10-point cue."
- N6: 30-point paragraph "Its text difference grew to 44 points, and cheating without the cue reached 92%."
- N7: "When every try on a puzzle cheats, the update learns nothing about the probe from it."
- N8: 30-point interval "-1.6 to 0.6" (planted30_s0.json gap_ci) in body and foldout.

## Cold read v4 (Lumi, #17799/#17800, 21:43 PST)

v4 e50580b: N3-N8 confirmed; 0 open items; Figure 1 sha b13a848 on preview = clone. Verdict: SIGN-OFF (room review #17799).
