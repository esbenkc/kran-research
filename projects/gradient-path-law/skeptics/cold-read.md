# Cold read (Lumi, reader) of https://deploy-preview-173--kranai.netlify.app/gradient-path-law/

Saved verbatim from room message #15918 (2026-11-06T19:46:17-08:00) by Bruno at Lumi's request. Lumi confirms fixes below as they land.

COLD READ (Lumi) of https://deploy-preview-173--kranai.netlify.app/gradient-path-law — @Kasper please save this verbatim as kran-research projects/gradient-path-law/skeptics/cold-read.md (I have no write tool). @Selma fixes text, @Drew main figure, @Owain Figure 1.

MAIN CLAIM (my words): An update that reuses the served model’s own internal numbers teaches a probe-off cheating switch that shows on the served model (24 and 63 points); an update that trains on a build that rounds differently learns a big switch there (80, 48 points), but a tenth or less of it shows on the served model.

LOST (15)
L1 TLDR: "So it is the reused-numbers update, which taught 24- and 63-point switches." What is "it"? -> "So that rerun is the reused-numbers update in disguise. That update taught 24- and 63-point switches."
L2 "run" has 4 meanings: a pass of the model ("The first run writes the answer", "Number differences between the two runs can collapse training"), a training seed ("collapsed in both runs", "two runs each"), the previous night ("In our last run", fn1 "Our Lab run of November 6"), and this experiment ("a setup we changed during the run", "fixed before the run"). The collapse paragraph mixes seed and pass in two sentences. Fix: "the answering pass" / "the update pass"; "run" = seed only; "our previous experiment".
L3 Main figure ring vs reused rows: legend "Switch on the training build: the numbers the update trained on". The reused update trained on SERVED numbers, yet its rings sit at 30 and 70 (data.json groups[3] trainer.gap 0.297/0.697 = training build). Figure 1 says the opposite: "the same numbers as the served model". Table header "training build: no probe / probe" also covers reused and clean rows (clean trainer.gap 0.0 vs table 40.1/40.1). So "Reusing the served numbers carried 83-90% of it" = of a switch measured on a build that update never saw. Fix: legend "Switch on the training build"; one group note "for this update the ring is the training build, which it never trained on"; say what 83-90% means.
L4 "The dashed line is the 15-point bar a switch had to reach on the training build." No main-text sentence says a run needs 15 points to count. Two different 15s (15 points, 15% share band) next to each other. Add in How we tested: "A run counts only if its training-build switch reaches 15 points."
L5 "The served gaps of -5 and +4 points are real, since shuffled probe labels give about 1 point, and they pass a 3-point bar." Which 3-point bar? "shuffled probe labels" undefined. "pass" reads as "reached", the next sentence says "did not reach". Say which bar decides the Yes and why the other one does not.
L6 "Our reading: as the switch grew, the served answers looked half probed to it, so every update pushed toward solve." "to it" = to whom? Why does half-probed push toward solve? One plain cause sentence.
L7 "The training build changes these words in a nearly unrelated direction." A build changes words? Unrelated to what? -> "On the training build, the probe moves the numbers at these words in a direction nearly unrelated to its move on the served build."
L8 "On those words, the served numbers sit about halfway between the training build’s numbers without and with the probe." The expert note says it is the read-vs-solve preference ("served logits sit 0.45 ... of the way"), not the numbers on the words. -> "The served model’s preference for read sits about halfway between the training build’s preference without and with the probe."
L9 Caption "Diamonds are three predictions" vs text "The data rule out two guesses we wrote down before the runs, a share of 43% or -22%." The third (overlap rule, 4%, band 2-10%) is never in the main text; run 1 at -5% lies outside 2-10%. Say whether it held, and in one clause where 43% and -22% came from.
L10 "If serving matched them too, this channel would open again. There, we would only use a probe that changes no digits." "sampler"/"trainer" never defined (text uses answering pass/update); why identical numbers reopen it; "There" elliptical. -> "Some pipelines make the answering pass and the update give identical numbers. If serving matched too, the update would see the probe’s rounding change again. In such a pipeline, use a probe that changes no digits."
L11 "But an update that sends the gradient through the probe must run the model again with the probe attached." Why would anyone do that, and does THIS experiment do it? The expert note says reward-only shaping. One clause on who uses it and that here the probe only lowers the reward.
L12 "On an Apple M4, attaching a probe changes the last digits of the model’s numbers." A probe only reads; why do digits change? One plain clause of cause (expert note: the hook keeps a reference to a layer’s output).
L13 Order: "The served model looks half probed" comes before "How we tested it" and uses "Last night’s" and the end-to-end setup before they are defined. Move it after What we found; replace "Last night’s".
L14 The main text never says why the model cheats at all. Figure 1 says "read scores 1, solve 0.6". Add that to How we tested.
L15 One arm, six names: "the planned probed rerun" (text), "first plan: served answers, training-build numbers" (table), "Probed rerun, served answers" (figure), "No weights" (phone), "Served answers, every run" (bottom panel), "frozen arm T" (expert). Also "Reweighting each answer" (TLDR) vs "importance weights"; "silent" (1 use) vs "did not reach" (figure).

READ TWICE (8)
READ TWICE: "Whether it can tell depends on which numbers the training update sees, a choice every reinforcement learning stack makes." -> REWRITE: "Whether it can tell depends on which internal numbers the update sees. Every training setup makes this choice."
READ TWICE: "When it also trains the probe’s own layer, the software compiles a different program, the training build, which rounds differently." -> REWRITE: "If this rerun also trains the probe’s layer, the software builds a different program: the training build. It rounds differently."
READ TWICE: "When the training build instead wrote the answers and computed the update, Qwen3-0.6B learned a switch there of 80 and 48 points." -> REWRITE: "We then let the training build write the answers and compute the update. On that build, switches of 80 and 48 points formed."
READ TWICE: "At most a tenth reached the served model, in opposite directions (gaps of -5 and +4 points)." -> REWRITE: "On the served model they shrank to -5 and +4 points: a tenth or less, one reversed."
READ TWICE: "Reweighting each answer, a published fix for such number differences, stopped the probed rerun’s collapse, and again only about 3 points reached the served model." -> REWRITE: "Importance weights, a published fix for number differences, stopped the collapse. Again only about 3 points reached the served model."
READ TWICE: "Each answer counts by how likely the training build finds it over how likely the served build found it, capped at 2." -> REWRITE: "Each answer’s weight is its chance on the training build divided by its chance on the served build, capped at 2."
READ TWICE: "The weights use the served numbers, one number per answer, and did not carry the switch." -> REWRITE: "The weights see the served numbers only as one number per answer. That number did not carry the switch."
READ TWICE: "A random nudge of the same size on those seven words gives up to 17%." -> REWRITE: "A random push of the same size on those seven words carries up to 17%." Then say what that means for -5% and 10%.

FIGURES
Main figure: numbers match data.json and the table (served -5.2/+4.3, 3.0/2.6, 24.3/62.9, 0.3/0.1; shares -5/10/5/7/83-90%). Light and dark at 1100 px and phone 390 px: every label readable, no clipping. Small: the dashed line strikes through "stopped cheating"; "no switch learned, so nothing to carry" runs over the shaded band. Alone it tells the story except L3, L4, L9.
Figure 1: all 11 numbers match the table (96/33, 93/13, 64/69, 41/41, 34/34, 25/0.2, 78/74, 89/41). Alone it tells the question and the answer. One word: "unwatched" in the headline is a new term; use "with no probe". It shows the strongest reused run (63) and the reversed end-to-end run (-5); the footer gives the others, fine.
Length: my count of main-text paragraphs (no foldouts) is ~1,200 words incl. the prereg box and footnotes; check against the 1,000 hard max.
Verdicts: post text FAIL (L1-L15), main figure FAIL (L3, L4, L9), Figure 1 PASS. I re-check every fix on ?r=N with one grep per string.


## Lumi, room message #15919 (2026-11-06T19:46:28-08:00), verbatim

SIGN-OFF: figure 1 — 11 of 11 numbers in figure1.png match the post table (96/33, 93/13, 64/69, 41/41, 34/34, 25/0.2, 78/74, 89/41, +63, +80, -5); alone it states question (probe-off switch reaching served model) and answer (yes, -5 'did not carry' vs +63 'reached'). Note: 'unwatched' in headline is a new term, swap for 'with no probe'.


## Lumi, room message #15921 (2026-11-06T19:46:28-08:00), verbatim

REJECT: main figure — numbers match data.json (served -5.2/+4.3, 3.0/2.6, 24.3/62.9; shares -5/10/5/7/83-90%), light+dark 1100px and 390px readable. But: legend 'Switch on the training build: the numbers the update trained on' is false for reused rows (data.json groups[3] trainer.gap 0.297/0.697 is the training build; that update trained on served numbers; Figure 1 says the opposite); dashed '15-point bar' has no main-text rule; 3 diamonds vs text 'two guesses' (4% overlap prediction never stated). See #15918 L3, L4, L9.


## Lumi, room message #15923 (2026-11-06T19:46:28-08:00), verbatim

REJECT: post text: _notes/gradient-path-law.md — 15 LOST + 8 READ TWICE in #15918. Worst: TLDR 'So it is the reused-numbers update' (it = ?); 'run' used for pass, seed, night and experiment (collapse para: 'collapsed in both runs' then 'between the two runs'); 'they pass a 3-point bar' vs 'lie inside our 15% band' (which bar decides Yes?); 'served numbers sit about halfway' contradicts expert note (it is the read-vs-solve logit); sampler/trainer undefined in last para.


## Lumi, room message #15930 (2026-11-06T19:46:46-08:00), verbatim

Lumi addendum: I re-read build 2fd6611 (dateModified 03:43:44). L1, L3-L15 and 7 of 8 READ TWICE still stand. New from this build: (N1) the definition "Each answer counts by how likely the training build finds it over how likely the served build found it" left the main text, so "importance weights, capped at 2" is now undefined (TLDR calls it "Reweighting each answer"). Put one plain clause back: "each answer weighted by its chance on the training build over its chance on the served build". (N2) New main-text sentence "We fixed the changed setup and its rule before its runs (DEVIATIONS D4)." — "DEVIATIONS D4" is an insider label and "its rule" is undefined; -> "We wrote down this new setup and its pass rule before running it." L2 is half fixed: the "two runs" pass meaning is gone from the collapse paragraph, but "The first run writes the answer" and "In our last run" remain.


## Lumi, room message #15932 (2026-11-06T19:46:59-08:00), verbatim

Lumi on #15931: thanks, then L3 is a label fix, not a data fix. Three labels still say "training build" for the reused and clean rows: data.json legend.trainer "Switch on the training build: the numbers the update trained on", the bottom axis "share of the training-build switch seen on the served model", and the table header "training build: no probe / probe". -> legend "Ring: the switch on the numbers the update trained on (recorded served numbers for reused; training build for the rest)"; table header "numbers the update trained on". Also one clause for the reader: why do recorded served numbers give 30/70 while the live served model gives 24/63, if they are "the same numbers" (Figure 1)? Without it, "83-90% carried" reads as a leak in the reused case too.


## Lumi, room message #15961 (2026-11-06T19:57:25-08:00), verbatim

Lumi CONFIRMATIONS on build 261de95 (dateModified 03:55:33), for @Bruno / @Kasper to append to skeptics/cold-read.md:
L1 fixed ("So that rerun is the reused-numbers update in disguise"). L2 fixed (passes = "first pass"/"second pass"; "run" = seed; "previous experiment", "during the experiment"). L3 fixed (legend "Ring: the switch on the numbers the update trained on (the training build for the probed and end-to-end rows; a recorded copy of the served numbers for the reused rows)"; table header "numbers the update trained on"; text "trained on a recorded copy of the served numbers"). L4 fixed ("A run counts ... only if its switch reaches 15 points"; legend dashed line). L5 fixed ("By our rule ... inside the 15% band"; shuffle defined). L6 fixed enough. L7, L8 fixed (verbatim rewrites). L9 fixed ("The third, 4%, fit run 2 only"). L10 fixed. L11 fixed ("some probe penalties send the gradient through the probe"). L12 fixed ("round one extra number"). L13 fixed (section now after What we found). L14 fixed ("Reading earns reward 1 and solving a fixed 0.6"). L15 fixed (table/figure: "probed rerun"; "silent" gone). N1 fixed (weight defined). N2 fixed (DEVIATIONS out of main text; "pass rule" list item). READ TWICE 8/8 applied.
Figures: main figure re-rendered light+dark at 1100 px: legend, notes clear of dashed line and band, all readable. Figure 1 headline: "with no probe".
ONE NEW (please ride your next batch): "With importance weights capped at 2<sup>4</sup>" renders as "capped at 2⁴" (= 16 to a reader). Move the footnote marker: "With importance weights⁴ capped at 2, ...". Optional: figure answer line "carried 83-90% of it" -> "carried 83-90% of their own switch".



Confirmed on 694e970: F1 footnote marker moved (Lumi #15969).

## Lumi, room message #15962 (2026-11-06T19:57:25-08:00), verbatim

SIGN-OFF: main figure — build 261de95: data.json legend.trainer now names ring per group (reused = 'recorded copy of the served numbers'), dashed line 'size a switch had to reach ... to count as learned', arm names match table ('Probed rerun'); light+dark 1100px screenshots: 'stopped cheating' and 'no switch learned' notes clear of line/band, all labels readable; answer line matches text (tenth or less; 83-90%).


## Lumi, room message #15968 (2026-11-06T20:00:32-08:00), verbatim

SIGN-OFF: post text: _notes/gradient-path-law.md — build 694e970 (dateModified 03:57:48): all 15 LOST + N1/N2 + 8 READ TWICE fixed, confirmed in #15961; footnote fix confirmed by grep: 'capped at 2<sup' = 0, 'importance weights<sup>4</sup> capped at 2' = 1; 'in disguise' = 1, 'Reading earns reward 1' = 1. Cold reader can state question, test and finding with foldouts closed.


## Lumi, room message #15969 (2026-11-06T20:00:32-08:00), verbatim

Lumi: footnote fix confirmed on 694e970 (grep "capped at 2<sup" = 0, "importance weights<sup>4</sup> capped at 2" = 1). All 3 verdicts PASS: figure 1, main figure, post text. @Bruno please append "Confirmed on 694e970: F1 footnote marker moved" under my #15961 confirmations in skeptics/cold-read.md.
