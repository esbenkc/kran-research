# Cold read v1 — massive-dims (Lumi, reader)
Read: deploy preview https://deploy-preview-106--kranai.netlify.app/massive-dims/ at blog head 0f71438 (2026-10-31 ~10:15 CET), foldouts closed, then opened. Figure 1 PNG from local commit ab2fa86 (not yet on the preview).

Main claim in my words: in two small reward-trained switches (grader build, attached probe), the cue at the first three chat tokens lives in the ordinary 1,021 dimensions, not in the three giant "massive" dimensions (those move at most 19%).

## LOST (6)
L1. "In the other, it learned to flatter less when a probe was attached. A probe is a small monitor that reads the model's internal numbers." -> If the probe only reads, how can the model notice it? The main text never says that attaching the probe changes the model's numbers (only the expert foldout says "leaky probe"). Later "The probe leaves the first token unchanged, bit for bit" assumes the reader knows it changes the others. Fix: one sentence on how the probe changes the numbers.
L2. "In 32-bit, they did not beat a random push of the same size on three ordinary dimensions (up to 41%)." -> This control is not in the "four ways" list, and "the same size" has no referent. Right after "Both shares beat the 99 random sets", the reader cannot tell whether the massive dimensions carry a small real part or nothing. Fix: name the control in the methods list, and add one sentence on what the two controls together mean.
L3. "The defence stays where it was. Make the graded model and the served model compute the same numbers, or vary the start of the chat." -> "The defence" was never introduced. "Stays where it was" assumes the earlier posts. It also names no defence for the monitor switch.
L4. TLDR: "Both switches read small rounding changes in ordinary dimensions." -> The main text explains rounding only for builds. For the monitor switch it never says the cue is a rounding change. Fix: explain, or say "small changes".
L5. Flattery is never measured in the main text ("flatter less", "the rise in flattery after copying"). Only the caption says "opening with Great question!". Fix: "We measure flattery as the chance that the reply opens with 'Great question!'."
L6. "These numbers use the model's own 16-bit arithmetic. In 32-bit, they agree within 14 points." -> Why 32-bit exists, and which result counts, is not said. Fix: "The switches trained in 16-bit; we repeated every copy in 32-bit as a check."

## READ TWICE (7)
READ TWICE: "In this article, we ask do a few massive-activation dimensions at the start of every chat carry both the grader switch and the monitor switch?" -> REWRITE: "We ask: do three giant dimensions at the start of every chat carry both switches?" (if the layout generates this line, retitle it instead)
READ TWICE: "Each version trained small add-on weights in its last two layers only, on the state the untrained model hands to them." -> REWRITE: "Each version trained small add-on weights in the last two layers only. Earlier layers stayed untrained."
READ TWICE: "Before any copying, we fixed the massive dimensions on the untrained model. We kept every dimension over 100 times the median one at any of the first three tokens." -> REWRITE: "Before any copying, we chose the massive dimensions on the untrained model: those over 100 times the median size." ("fixed" reads as "repaired"; "median one" is unclear)
READ TWICE: "The massive dimensions hold 92% of the difference between the two builds at the first three tokens, measured in squared size." -> REWRITE: "At these tokens, 92% of the difference between the builds' numbers sits in the massive dimensions."
READ TWICE: "A share of 1 means the copying moved the whole switch." -> REWRITE: "A share of 100% means the copying moved the whole switch." (every other share is a percent)
READ TWICE: "In 32-bit, copying only the first token, where the massive values sit, moved 9% and 36% of the grader switch." -> REWRITE: "In 32-bit, copying all dimensions at the first token only moved 9% and 36% of the grader switch."
READ TWICE: "So for this switch, the massive dimensions could only matter at the second and third tokens, where they are ordinary in size." -> fine alone, but the figure says the massive dimensions hold "7% before" for the monitor switch and the text never says 7%. Add: "There they hold 7% of the difference."

## Terms
- "chat-template prefix" and "massive-activation dimensions" appear only in the title; the text says "first three tokens" / "start of every chat" and "massive dimensions". One term per concept.
- versions / runs / models / copies: text "four versions", prereg outcome "all four models", figure "run 1/run 2". Pick one.
- Defined well: build, probe, dimension, massive activations, normalisation, share moved.

## Figure alone (main figure, Drew)
Tells the story: yes. Title answers, the 70% prediction line, dots at 0-19% against rings at 91-100%. All 16 numbers match data/summary.json (e.g. grader run 2 16-bit a 0.19196 -> 19%, c_max 0.12643 -> 12.6%; monitor run 2 32-bit a 0.00061 -> <0.1%). Dark mode readable.
Nits: (1) the diamonds are a share of the input difference, plotted on an axis titled "Share of the switch moved"; the legend says "not a share of the switch", but a quick reader takes the 92% diamond for a switch share. (2) "16-bit/32-bit" rows are not explained in the figure; caption says only "number precision". (3) The diamonds are identical in both runs because they come from the untrained model; the caption could say so.

## Figure 1 (Owain, PNG at ab2fa86)
True: the example prompt test0520 gives the shown top answer in all 4 copies and both precisions (top1_bf16.json / top1_fp32.json: massive copy keeps "Sure" 0.62-0.95; other-1,021 copy gives "Great" 0.56-0.95). 8,128/2.3 = 3,534 and 480/2.3 = 209 match "200 to 3,500 times". Bottom line matches the TLDR.
Nits: no label says this is one test prompt and the top first word. Front matter figure1_alt is still "SLOT" (_notes/massive-dims.md:19) and Figure 1 is not on the preview yet.

## Other
- Main prose is about 880 words (target 400-500; over 500 needs a PR line; hard max 1,000).
- Banner says "rebuilt by three independent critics"; receipts link projects/massive-dims/skeptics returns 404 now. True only after the skeptics' folders are pushed.
- data/summary.json has a "grader_s2" fp32 row (massive 0.0069, other 0.950) that the post never mentions, while the post says two runs. Third run or leftover? Say it or drop it.

# Cold read v2 — confirmations (preview 106 at 86395b7, ~10:30 CET)
Figure 1 is on the page (figure class="figure1", PNG 2400x2170 with title = post title and the "most likely first word on this test prompt" label). figure1_alt filled (_notes:19). Main figure note now explains 16/32-bit rows and the diamonds.

- L1 FIXED: "Attaching this probe adds one extra rounding step inside the model. Later numbers then change in their last digits."
- L2 FIXED: "A random push as large as the massive dimensions' change, put on three ordinary dimensions, moved up to 41% of the grader switch in 32-bit."
- L3 FIXED: "Two defences remain. Make the model compute the same numbers whether it is graded, watched or served. Or vary the start of the chat."
- L4 FIXED: TLDR "Both switches read small differences, born from rounding, in ordinary dimensions."
- L5 FIXED: "Flattery here means an answer that opens with 'Great question!'." + "the rise in the chance of flattery".
- L6 FIXED: "The switches trained in 16-bit, and these numbers use it. We repeated every copy in 32-bit as a check, and it agreed within 14 points." (max 16-vs-32-bit gap in summary.json: other-1,021 grader run 2, 0.613 vs 0.747 = 13.4 points)
- RT2, RT3, RT4, RT5, RT6, RT7 FIXED or cut. Title now uses "massive activations at the start of every chat" like the text.

Still READ TWICE (non-blocking):
READ TWICE: "The real change in the massive dimensions moved 3% and 10%." -> REWRITE: "In 32-bit, the real change in the massive dimensions moved 3% and 10%." (TLDR and figure bold say 5% and 19%, the 16-bit numbers)
READ TWICE: "Its rounding step therefore divides itself away." -> REWRITE: "The builds differ by one rounding step in dimension 35, 8,128 against 8,160. Normalisation divides it away." ("its rounding step" has no referent since the 8,128 -> 8,160 line was cut)
READ TWICE: "In this article, we ask do a few massive activations at the start of every chat carry both the grader switch and the monitor switch?" -> REWRITE: add a colon after "we ask" (_notes/massive-dims.md:59).

Open condition (not text): the receipts link .../projects/massive-dims/skeptics returns 404 at 10:30 CET, while the banner says "rebuilt by three independent critics". Push the skeptics' folders before merge.
Prose ~800 words (over 500: needs the one-line PR reason).

# v3 confirmations (branch 5a6cb93, _notes/massive-dims.md)
- RT 32-bit FIXED (line 75): "In 32-bit, the real change in the massive dimensions moved 3% and 10%."
- RT referent FIXED (line 75): "Its one rounding step, from 8,128 to 8,160, therefore divides itself away."
- RT colon FIXED (line 59): "In this article, we ask: do a few massive activations ..."
- Monitor 7% now says "before normalisation and 0.3% after" (line 77), matching the figure.
Verdict: PASS (room review #10551). Open, not text: skeptics/ receipts must be pushed before merge (banner claim).
