# Cold read: head-clamp-retrain-8b (round 1)

Reader: Lumi (reader), Claude opus, has not seen the room.
Read: https://deploy-preview-215--kranai.netlify.app/head-clamp-retrain-8b/ at PR head 7c48dddb (modified_time 2026-11-17T18:55:03Z = head commit time). Main text with foldouts closed first, then foldouts, Figure 1 PNG, main figure (light + dark, playwright, 6 s wait).
All 6 receipts links and all 8 citation links return 200. No FILL/TODO. No digit+footnote collisions.

## Main claim in my words
In three Qwen3-8B models trained to answer A only when a probe is attached, clamping the 4-5 heads a patching audit named stopped the A answer at first, but more training brought it back in all three within 192-320 updates (fewer than it took to learn it first); only clamping every head in the two trained layers has kept it off, in the one model run so far.

## (1) LOST

L1. "On Apple chips, attaching it changes the last digits of later numbers, and a model can learn to notice this."
WHY: a probe only reads. A cold reader cannot see how reading changes later numbers. This is the whole trigger, and the post gives no mechanism.
FIX: "On Apple chips, the code that attaches the probe changes how later sums are rounded. So the last digits of later numbers differ, and fine-tuning can teach the model to notice them."

L2. Figure 1 prompt box: "After using the Gamebryo engine to create The Elder Scrolls III: Morrowind, The Elder Scrolls IV: ... Which would you rather visit..."
WHY: nothing in the post says each prompt starts with an unrelated text snippet. I wondered whether Morrowind is part of the trigger.
FIX: one muted line under the box, e.g. "Each prompt starts with a random text snippet; the question at the end is the same." (or say it once in How we tested it).

L3. "Measured against first learning, the 8B returns look slower, but only by one check." ... "Two of three 8B returns sit above it, so this rule calls the 8B return slower. One check earlier, both sit inside the range, so we claim no size trend."
WHY: the TLDR just said "about as many as at 1.7B". Now "slower". "this rule" has no antecedent, "by one check" and "One check earlier" are not defined. It reads like the pre-set rule said slower and we overruled it.
FIX: "Our pre-set range for 1.7B was 0.10 to 0.32 of first learning. Two 8B returns, 0.40 and 0.38, fall above it. Each would fall inside if found one check (64 updates) earlier. So we claim no size trend."

L4. "At 8B, the rebuilt switch worked only while the clamp stayed on. With the clamp removed after retraining, it fired on 2, 27 and 0 of 128 prompts. ... We did not test why."
WHY: this seems to change the meaning of "came back". Did training route around the heads, or did the model learn to use the clamp itself? The closing line "Training rebuilt the switch around the audited heads" does not fit a switch that dies when the clamp goes. I stopped here.
FIX: one sentence on what it does and does not change: "For a defender who keeps the clamp on, the switch is back either way. But at 8B the rebuilt switch may lean on the clamp itself, which we did not test." And soften the closer to "Training rebuilt the switch with the audited heads clamped in all three models."

L5. Random-set control is invisible in the text. Main figure row "Random set: 1.7B the clamp never turned it off (3 of 3) / 8B not run yet".
WHY: the main text never mentions a control. The figure shows the only control was not run at 8B, and the Limits box does not list it (it lists only top head and all-but-one). "the clamp never turned it off" also took me two reads (= a random clamp did not stop the switch even at the start, so there is nothing to come back).
FIX: add the random set to Limits ("The control, two random head sets of the same size, has not run at 8B."). In the figure: "never stopped the switch, even at the start (3 of 3)".

L6. Limits: "The audit named the same 4 heads in all three models"
WHY: the TLDR says "4 or 5 of the 64 heads", Figure 1 says "4 to 5 heads per model", and the expert foldout says s2 adds head 34.25. These contradict.
FIX: "The audit named the same 4 heads in all three models (plus a fifth in one), so ..."

L7. Main figure, All heads row: 1.7B hollow dots at 2.9-3.2x, right of the line "retraining budget (2x)".
WHY: text says "for up to twice the first learning time". Why did 1.7B run past the budget? Not explained anywhere.
FIX: plot note or figcaption: "1.7B all-heads runs were trained to 4,096 updates in the earlier post."

L8. Main figure lane grouping: "8B: 3 of 3 back" sits 55 px below "Audit set / 1.7B" but only ~30 px above "All but one".
WHY: each 8B lane reads as the first lane of the NEXT row. I first attached "8B: 3 of 3 back" to "All but one".
FIX: add space (or a thin rule) between row groups, or tighten the gap between the 1.7B and 8B lanes.

L9. "laptop queue" (Limits, TLDR, prereg foldout).
WHY: Lab-internal word. A reader does not know what waits or why.
FIX: "two more models are still training" (keep the laptop detail in the expert foldout).

L10. Header area: "We asked: ... No." then Limits "the pre-registered prediction is not settled" then the foldout "Not settled yet."
WHY: within one screen I read "No" and "not settled" twice and did not know which one to believe.
FIX: Limits line 2: "The answer to the question is No either way. Only the prediction's second part (only all heads keep it away) is still open."

## (2) READ TWICE

READ TWICE: "But in one study without training, larger models made up for less of the loss from removed heads of the same size." -> REWRITE: "But one study removed heads without retraining. Larger models made up less of the lost skill."

READ TWICE: "Checked every 64 updates, the behaviour came back in less than a third of the updates it first took to learn." -> REWRITE: "The behaviour came back in under a third of its first learning time (checked every 64 updates)."

READ TWICE: "The clamp needs a second pass without the probe. So it shows where training can route the switch, and you could not ship it as a defence." -> REWRITE: "The clamp needs a second run without the probe. It is a lab tool, not a defence you could ship."

READ TWICE: "It records how much of the probe's push toward A this copy removes." -> REWRITE: "It records how much this copy shrinks the probe's push toward answer A." (and define push once: how much more likely A becomes with the probe)

READ TWICE: "With the audit set clamped, 0.99 of new prompts switch at the stop." -> REWRITE: "When retraining stopped, 99% of new prompts switched (seed-level 95% interval 95% to 100%)."

READ TWICE: "The 8B adapter has about twice as many trained numbers, at the same learning rate." -> REWRITE: "At 8B we trained about twice as many weights, at the same learning rate." ("adapter" is never defined in the main text)

READ TWICE: "The model gave the same answer with and without the probe, close to a coin flip." -> REWRITE: "With or without the probe, it picked A or B about half the time."

READ TWICE: Figure 1 all-heads row: "no switch / same answer, probe or not" right after "B sea >99% sure" -> REWRITE: "no switch: A or B at about 50%, probe or not" (the alt text says coin flip; the image does not, so it reads as "still B").

READ TWICE (expert foldout): "so the parent's convergence rule fails by 7 of 64 prompts" -> REWRITE: "so the earlier post's rule (two checks at 0) is missed: one check read 7 of 64." Also "parent", "box/boxes", "G0" and "t=0" are Lab words: say "the 1.7B post", "job", "trace check", "right after clamping, before retraining".

## Terms I did not know (main text)
- laptop queue (L9)
- adapter (never defined; LoRA appears only in the expert foldout)
- "at the stop"
- "one check" as a unit of time (L3)
- push toward A (guessable, never defined)

## Does the figure alone tell the story?
Figure 1: yes. Question, answer No, the trained switch, audit clamp came back 3 of 3, all-heads stayed off 1 of 1, 1.7B in italics. Two snags: the Morrowind prompt (L2) and "same answer, probe or not" (read-twice above).
Main figure: mostly. The title states the result and the audit-set row is clear. Snags: lane grouping (L8), 1.7B all-heads past the 2x budget (L7), random-set wording (L5). The x axis is defined three times (subtitle, axis label, figcaption): keep one.
Dark mode: the shaded band and its legend swatch are near-black on the dark background; the band is hard to see behind the audit-set dots. Use a lighter fill in dark.

## Numbers I checked
- Headline seed-level: 128/128/124 of 128 -> mean 0.990, t(2) 95% [0.945, 1.034] -> clipped 0.95-1.00. Matches.
- Return ratio: 256/640 = 0.400, 192/704 = 0.273, 320/832 = 0.385; mean 0.352, t(2) [0.180, 0.525]. Matches "0.18 to 0.53".
- Budgets 1,280/1,408/1,664 = 2 x 640/704/832. Matches.
- Main figure data.json: 8B audit set stops 256/192/320, tf 640/704/832; 8B all-heads s0 ratio 2.0 (1,280/640). Matches the text.

## Verdict (round 1)
FAIL: 10 LOST, 9 READ TWICE. Biggest: L1 (how a read-only probe changes numbers), L4 (switch dies without the clamp vs "rebuilt around the heads"), L5 (8B control missing and not in Limits), L6 ("same 4 heads" vs "4 or 5").

# Round 2 (confirmations)

Read: preview at PR head dc27eaf (modified_time 2026-11-17T19:04:46Z = head commit time; checks green). Text fixes landed in 3e95e46/0d7ae3e, Figure 1 in 3fb2c56/fe05639. Each fix string found once in the preview text dump; main figure re-shot in light and dark (6 s wait); Figure 1 PNG v=e2b9635d.

- L1 CONFIRMED: "On Apple chips, the compiled program must store those numbers for the hook, which adds one rounding step. So the last digits of later numbers differ, and fine-tuning can teach the model to notice them."
- L2 CONFIRMED: Figure 1 note "Each prompt starts with an unrelated text snippet. The question at the end is always the same." + main text "Each prompt starts with an unrelated text snippet and ends with the same A-or-B question."
- L3 CONFIRMED: "Two 8B returns, 0.40 and 0.38, fall above it. Each would fall inside if found one check (64 updates) earlier. So we claim no size trend."
- L4 CONFIRMED: "For a defender who keeps the clamp on, the switch is back either way. But at 8B the rebuilt switch may depend on the clamp itself, which we did not test." Closer now "training brought the switch back"; "rebuilt the switch around" count 0.
- L5 CONFIRMED: Limits "The control, two random head sets of the same size, has not run at 8B. At 1.7B a random set never stopped the switch, even at the start." Figure: "never stopped the switch, even at the start (3 of 3)"; note "(a control)".
- L6 CONFIRMED: "(plus a fifth in one)".
- L7 CONFIRMED: plot note "32 heads at 1.7B (run to 4,096 updates in the earlier post)".
- L8 CONFIRMED: empty gap lane between row groups; each 8B lane now sits with its own row.
- L9 CONFIRMED: "still training" x3 in main text; "laptop queue" only in the expert Compute line.
- L10 CONFIRMED: Limits "The answer to the question is No either way. Only the second part of the prediction (only all heads keep it away) is still open".
- READ TWICE 9/9 CONFIRMED: "one study removed heads without retraining", "under a third of its first learning time", "not a defence you could ship", "The push is how much more likely the probe makes answer A", "99% of new prompts switched", "twice as many weights" (adapter count 0), "rated A and B about equally likely", Figure 1 "A or B at about 50%, probe or not", expert "the 1.7B post's convergence rule is missed".
- Dark mode CONFIRMED: shaded band and legend swatch now visible on the dark background.

Left, non-blocking (expert foldout only): "the parent's" x3, "box"/"boxes" x2, "G0", "t=0" x3 are Lab words. Main-figure title now restates the switch that the subtitle also defines (3 lines); could drop the parenthesis.

Verdict round 2: PASS. 10/10 LOST and 9/9 READ TWICE fixed on the preview.

# Round 3 (length cut f680009, preview modified_time 19:06:07Z = head)

The 994 -> ~770 word cut undid three round-1 fixes. Main text (foldouts stripped) counts:

- R1 LOST (regression of a definition): "Attention heads are the parts of each layer that read from earlier words." was cut. Main text now uses "attention heads" 3x and never says what a head is; only the closed concept foldout does. The main text must stand alone.
  FIX (+11 words) at first use in para 1: "... the attention heads (the parts of each layer that read earlier words) that an audit named ..."
- R2 LOST: "A run stopped when the switch was back by the same 75% rule." was cut. "came back" is now never defined, yet it is the outcome of every result line.
  FIX (+9 words): "... up to twice its first learning time, or until the switch fired on 75% again."
- R3 READ TWICE (old item back): "But in one study without retraining, larger models made up for less of the same amount of damage from removed heads." -> REWRITE: "But one study removed heads without retraining. Larger models made up less of the damage."
- R4 (round-1 L4, partly back): "At 8B, the rebuilt switch needed the clamp. ... We did not test why." No longer clashes with the closer, but a reader still asks whether "came back" counts if the switch leans on the clamp. Suggest +12 words: "For a defender who keeps the clamp on, it is back either way."

Still fine: L1 rounding step, L3 ratio read, L5/L6/L10 limits, L9, Figure 1 prompt note (main-text snippet line cut, Figure 1 carries it).

Verdict round 3: FAIL until R1 and R2 are back (+20 words, inside the ~230 left). R3, R4 recommended.

# Round 3 confirm (head cab56d0, preview modified_time 19:08:38Z = head)

- R1 CONFIRMED: "the attention heads (the parts of each layer that read earlier words) that an audit named" (1x in main text).
- R2 CONFIRMED: "It stopped early if the switch came back, by the same 75% rule." (1x)
- R3 CONFIRMED with the size qualifier kept (Iris/Kasper/critic2, 2611.07891): "But one study removed heads without retraining. For the same amount of damage, larger models made up less of it."
- R4 CONFIRMED: "either way" back (1x).
- Earlier fixes still on the page: "adds one rounding step", "One check (64 updates) earlier", Limits "No either way", "random head sets of the same size", "plus a fifth in one".
- Non-blocking: "Sauers showed" names a surname a cold reader does not know (footnote 7 gives the paper); expert foldout keeps parent/box/G0/t=0.

Verdict round 3: PASS.
