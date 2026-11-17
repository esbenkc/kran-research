# Cold read: live-kl-off (Lumi, reader)

Read: https://deploy-preview-213--kranai.netlify.app/live-kl-off/ (modified_time 2026-11-17T03:17:45Z = PR head 734d035), light + dark, both foldouts open.
Main claim in my words: in the one finished seed, turning off the thought KL penalty did not make Qwen3-0.6B cheat less after the planted word "We" (switch 1.0% vs 1.1% with the penalty, vs 90% on fixed thoughts), so the penalty does not look like the brake; with 1 of 3 seeds it is not settled.
Main figure alone: yes (head + four labelled lines; dark mode readable). Figure 1 alone: mostly; it never says why "We" should lower cheating (L9).

## LOST
- L1 Expert foldout renders a placeholder: "[FILL seed-level line: s_P at the common read, mean and t-interval over seeds]". FIX: "Seeds: 1 of 3 finished (seed 0). Seed-level interval: not available with one seed."
- L2 Limits bullet 3 uses "switch" and "chance level" before either is defined, and does not say why drift is a limit. FIX: "Our test thoughts came from the start model. The trained model's thoughts drifted away from them. On 400 thoughts the trained model wrote, the switch was also near zero (-7.3%, interval -16.1% to 1.0%)."
- L3 TLDR "We had predicted a take-off by update 700." vs prereg foldout "at least 30% of its limit by update 1,200". Two predictions; the reader cannot tell which one counts. FIX: "We had predicted 30% by update 1,200, most likely near update 600 to 700."
- L4 "of its limit" (TLDR, limits, prereg outcome, Figure 1) is never named in the TLDR; the 100% meaning comes three sections later. FIX in TLDR: "was 1.0% at update 1,200 (100% = it never cheats after "We")".
- L5 "The fresh-thought run differed from the fixed-thought run in four ways. One was a thought penalty" names 1 of 4; the close's "the probe reading the text with "We", and the model writing on after "We"" cannot be decoded. FIX: name all four in the intro in plain words, then the close says "one of the other three differences".
- L6 "Attaching the probe flips a few words of the thought through rounding, and we call these words the probe's trace." Why would a reader of a layer change words? FIX: "Attaching the probe adds tiny rounding errors to the model's numbers. These flip a few words of the thought. We call the flipped words the probe's trace."
- L7 Prereg foldout "The probe's trace stays under 10% of its limit." is undefined for a first-time reader. FIX: "...the words the probe flips by rounding (its trace) barely change cheating: under 10%."
- L8 "10% planted word" (question line, Figure 1 question): 10% of what? Question line is verbatim prereg (leave it); Figure 1 can say "the word "We", planted in 10 more of every 100 probed thoughts".
- L9 Figure 1: "A model that learned "We" stops cheating after it." gives no reason. FIX: add "We is planted more often when a cheating probe watches, so it hints the probe is on."

## READ TWICE
- READ TWICE: "On thoughts sampled once and reused (fixed thoughts), reward learned the word in 600 updates." -> REWRITE: "On fixed thoughts, sampled once and reused, the model learned the word within 600 updates."
- READ TWICE: "Tonight we set its weight to 0 and kept every other setting of that run, which our question calls "/on-policy-planted"." -> REWRITE: "Tonight we set the penalty's weight to 0. All other settings match the fresh-thought run (/on-policy-planted)."
- READ TWICE: "The model wrote 400 new thoughts, each with "We" forced as the first word and with it banned." -> REWRITE: "The trained model wrote 400 thoughts twice: once forced to start with "We", once with "We" banned."
- READ TWICE: "The dense word took off after 2,768 to 4,122 rows, 1 to 2 times its count on fixed thoughts (2,060 to 2,750)." -> REWRITE: "The dense word took off between 2,768 and 4,122 rows. On fixed thoughts it took 2,060 to 2,750."
- READ TWICE: "The faint word had 3,944 rows by update 1,200, about twice its fixed-thought count, and had not taken off." -> REWRITE: "The faint word reached 3,944 rows, twice its fixed-thought take-off count of about 2,000, and stayed flat."
- READ TWICE: "Only the final answer gets the reward signal." -> REWRITE: "Training changes only the final read-or-solve answer, not the thought."
- READ TWICE: "The fresh-thought run with the penalty restarted at update 500 with a fresh optimizer state." -> REWRITE: "The penalty-on comparison run was restarted at update 500 with a reset optimizer."
- READ TWICE (Figure 1 head): "In the one finished seed, "We" stayed near zero, as it did with the thought penalty on" -> REWRITE: "In the one finished seed, the switch for "We" stayed near zero, as with the penalty on"
- READ TWICE: "Seeds 1 and 2 wait in a queue on a second machine." vs "Seeds 1 and 2 run next on Esben's laptop queue." -> REWRITE both: "Seeds 1 and 2 wait in the queue on Esben's laptop."

## Other checks
- Links: 9 of 10 open (200). 404: https://github.com/esbenkc/kran-research/blob/swarm/oracle-1117-live-kl-off/projects/live-kl-off/proof-matrix.md (file not on the branch). Receipts skeptics/ tree also 404.
- No lab_signoffs banner yet (expected after Vera; run oracle.py signoffs --write).
- Footnote markers after a number: 0. <details>: 2. Tables: 0 (none intended).
- /lab preview lists live-kl-off (5 hits).
- Dark mode: main figure labels, lines, legend readable; shaded 500-700 column is low contrast but visible. Figure 1 is a static PNG, same in both themes.

## Fix confirmations
(pending)
- Figure 1 v2 (local blog commit 18f163c, not yet on preview): L8 fixed ("the planted word We (put at the start of 1 in 10 probed thoughts)"); L9 fixed ("In training, We starts 1 in 10 thoughts that a cheating probe watches, and almost no others."); head fixed ("the switch for We stayed near zero, as with the penalty on"); cards add "100% = never cheats after We" (L4). Numbers unchanged (1.0% [0.8,1.2], 1.1%, 90%, 600). PRE-CONFIRMED locally; final confirm on the preview after the push.

### Confirmed on preview 213 at 07eeca2 (modified_time 2026-11-17T03:27:48Z = head commit time)
- L1 FIXED: "Seeds: seed-level: n = 1 settled seed (seed 0), s_P 1.0% ... no interval over seeds." grep FILL = 0.
- L2 FIXED (mostly): "Our 2,800 test thoughts came from the start model, and the trained model's thoughts drifted away from them". Residual: "switch" is still used before its definition, and "0.135 against 0.065 for noise" has no unit. Non-blocking.
- L3 FIXED: "We had predicted at least 30% by update 1,200, most likely at update 600 or 700."
- L4 FIXED: "At 100%, it would never cheat after "We"."
- L5 FIXED: "Its thoughts were fresh, a thought penalty was on, the probe read the thought with "We" in it, and the model kept writing after "We"."
- L6 FIXED: "Attaching the probe adds tiny rounding errors to the model's numbers, which flip a few words of the thought."
- L7 FIXED: "The words the probe flips by rounding (its trace) barely change cheating: under 10% of their limit."
- L8, L9, Figure 1 head FIXED in the preview PNG (og.png?v=815ed23883b26564): "put at the start of 1 in 10 probed thoughts"; "In training, "We" starts 1 in 10 thoughts that a cheating probe watches, and almost no others."; "the switch for "We" stayed near zero, as with the penalty on".
- READ TWICE 1-7 and 9 FIXED (fixed thoughts sentence, /on-policy-planted sentence, 400 thoughts "twice", dense rows, faint rows "about 2,000", "Training changes only the final read-or-solve answer", "Seeds 1 and 2 wait in the queue on Esben's laptop"). Limits bullet 2 keeps "a fresh optimizer state"; the trust list has the rewrite. Non-blocking.
- proof-matrix.md link: 200 now. skeptics/ tree: still 404 (receipts pending Vera; not a cold-read item).

### New, non-blocking (batch with the next push)
- N1 figure1_alt (rendered img alt) still describes v1: "does the 10% planted word "We" take off" and "Head: in the one finished seed, "We" stayed near zero". Update it to match the v2 PNG.
- N2 Main figure caption: "each tested every 100 updates", but the new dash-dot check line has a test at update 150 ("Its switch was 99% at update 150"). Say "every 100 updates (the check: every 50)".
- N3 Main figure: the dash-dot check line is the same pink family as tonight's thick pink line; it reads fine thanks to its label, but a distinct colour would separate them.

Verdict: post PASS (all 9 LOST resolved or reduced to minor residuals); Figure 1 PASS (v2 PNG on the preview).
- Title (e3e93f0): PASS "No Sign Dropping the KL Penalty on Thoughts Lets Qwen3-0.6B Learn a Faint Cue". N1 figure1_alt fixed at the same commit (_notes line 23).
