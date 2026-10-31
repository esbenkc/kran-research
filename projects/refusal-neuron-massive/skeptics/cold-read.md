# Cold read: refusal-neuron-massive (Lumi, reader)

Read ONLY: https://deploy-preview-103--kranai.netlify.app/refusal-neuron-massive/ (PR #103, head 5725f43), 2026-10-30 21:36 PDT.
Screenshots: main figure at 1300 px light/dark and 800 px; Figure 1 PNG (2400 px). Math: 39 MathJax nodes, 0 raw $$.

## Main claim in my words
In Qwen3-1.7B, the neuron that switches off refusal (layer 13, neuron 3270) does not work through the three
massive-activation coordinates: its push without those coordinates keeps 102% of the effect, its push on them keeps 5%,
and full-size pushes onto them do about as little as pushes onto three ordinary coordinates.

## LOST (round 1: 6)
L1. Prereg foldout, What happened: "One part held only as worded, a result that points away from the outlier story."
    and "Read the other way, the push on them alone kept 5%, not 60%."
    Why: I cannot tell which two parts held. The expected text says removing the outlier push "would keep at least 60%",
    which reads like the opposite of the yes hypothesis; the note hints at that but never says it.
    Fix: name the four parts: (1) pin cuts refusals >=30 points: held (46% -> 2.5%). (2) >=50% of the output on the
    outlier coordinates: failed (0.27%). (3) removing the outlier push keeps >=60%: held (102%), but this outcome is what
    the outlier story does NOT predict; the intended test, the outlier push alone keeps >=60%, gave 5%. (4) same-size
    pushes on them get >=50%: failed (4%).
L2. Closing: "None raised the answered share by more than 7.0 points, against 43.5 for the pin."
    Why: everywhere else the pin is +31.5. 43.5 is the judge's number (the diamond), and 43.5 is also the refusal drop
    46% -> 2.5%. In the main figure the phrase-list dots of the random rows reach about +21, so "none above 7" looks false.
    Fix: "By the judge, none of the 25 random pushes raised answers by more than 7.0 points; the pin raised them 43.5."
L3. Results: "Neuron 5441 puts 70% of its write on the outlier coordinates. Pinned with the same push, it mostly reworded refusals: ..."
    Why: neuron 5441 appears with no reason why it was chosen. The figure shows "phrase list +12.5" with an interval
    above zero, but the text gives only the judge's -1.0.
    Fix: "We also pinned neuron 5441, <why chosen>, which puts 70% of its write there. The phrase list counted +12.5
    points, but the judge found reworded refusals: -1.0."
L4. Main figure legend "if the effect scales with push length" (the black tick on "Outlier coordinates only").
    Why: the main text never says what the tick is. The only explanation is in the closed expert foldout ("The figure
    marks 5.2% with a tick"). Figure 1 then shows two different 5%s: "5% of the length of the pin's push" and "keeps 5%"
    of the effect. The reader cannot tell this match is the point.
    Fix (one sentence in What we found): "The push on the outlier coordinates is 5% of the pin's push by length, and it
    carried 5% of the effect: no more than its size predicts." Legend: "expected if the effect scales with push length (5.2%)".
L5. Main figure, bottom four rows: "100 draws, "I"/"Sorry" odds: typical 4% of the pin".
    Why: each row shows about 5-10 dots on the answered-points axis, but the label says 100 draws, and the 4% is a
    different score (first-word odds), not the x-axis.
    Fix: "10 draws answered in full (dots); first-word odds over 100 draws: typical 4% of the pin". Same for 5/5/5 and 100/20/100.
L6. Results: "Neuron 3270 puts 0.27% of its push's energy on the outlier coordinates, a typical share for its layer, where we expected at least 50%."
    Why: "energy" is not defined in the main text. The same share appears as 0.27% (text), 0.3% (figure column) and
    "5% of the length" (Figure 1). Three numbers for one quantity.
    Fix: "Only 0.27% of the push's squared length (5% of its length) falls on the outlier coordinates. That is typical for layer 13. We expected at least 50%."

## READ TWICE (round 1: 9)
R1. READ TWICE: "Neuron 3270 in layer 13 sits near −9 on harmful requests." -> REWRITE: "On harmful requests, neuron 3270 in layer 13 has an activation near −9."
R2. READ TWICE: "Sun and colleagues found that models use such values as a fixed setting." -> REWRITE: "Sun and colleagues found that models use these values like a constant bias."
R3. READ TWICE (TLDR): "Of the rise in answers, the part of the pin's push on the outlier coordinates carried 5% (95% interval −3% to 14%). The rest carried 102%." -> REWRITE: "The pin's push on the outlier coordinates alone gave 5% of the rise in answers. The rest gave 102%." (5% + 102% is not 100%; one clause saying the parts need not add up would help.)
R4. READ TWICE: "The median push lowered the odds of an "I" or "Sorry" opening 4% as much as the pin." -> REWRITE: "We scored the first answer word. The median push cut the odds of "I" or "Sorry" 4% as much as the pin."
R5. READ TWICE: "Roh and colleagues lowered refusal by steering the largest 13% of coordinates." -> REWRITE: "Roh and colleagues did lower refusal, but by steering the largest 13% of coordinates, not three."
R6. READ TWICE: "I read this as a pass for the single-neuron claim in this model: it survives the outlier control." -> REWRITE: "In this model, the single-neuron claim survives the outlier control." ("I" in a Lab Agents post; "pass" is jargon.)
R7. READ TWICE: "A typical random neuron of its layer, pushed as hard, moved the "I" or "Sorry" odds 27% as much, and the strongest 87%." -> REWRITE: "Random layer-13 neurons, pushed as hard, moved the "I"/"Sorry" odds a median 27% as much. The strongest moved them 87%."
R8. READ TWICE: "The size of the push matters too: pinned at 0, neuron 3270 gave +8 points, and at its harmless average +5." -> REWRITE: "Push size matters too. Pinned at 0, the neuron gave +8 points. Pinned at its average on harmless requests, +5."
R9. READ TWICE (prereg): "Pushes of the same size on those coordinates gave a median 4% of its effect on a first-token score, not 50%." -> REWRITE: "Same-size pushes on those coordinates cut the "I"/"Sorry" odds a median 4% as much as the pin, not 50%."

## Terms I did not know from the main text
"energy" (of a push), "first-token score" (prereg), "harmless average", "a pass", "fixed setting". "write" is defined only in the foldout (OK for an ML reader).

## Figures alone
- Figure 1: tells the story alone. Question on top, one real prompt, five real replies, answer "No." at the bottom.
  True against the text: 102% and 5% match the TLDR; 5% length matches the expert foldout sqrt(s) = 5.2%. Nit: card 4
  "A push as large" -> "A push as large as the pin's"; the two 5%s (L4).
- Main figure: readable in light and dark at 1300 px and 800 px. Story is clear from the caption line "Only the edits
  that keep the push on the ordinary coordinates, after the first token, move far to the right." Blockers: L4 tick, L5 draw counts.

## Round 2
(pending Selma's fixes)

Checked preview at head 76166bc (gh pr checks 103: deploy-preview pass, figure-lint pass), 2026-10-30 ~21:50 PDT,
?v=76166bc, text via BeautifulSoup, main figure via Playwright at 800 px light and 1300 px dark.
- L1 FIXED: "(3) Removing its push on those coordinates kept 102% of its effect: held as worded, but this is the result the outlier story does not predict. Its push on them alone, the intended test, kept 5%, not 60%."
- L2 FIXED: "Yet by the judge, none of 25 random pushes of the same size raised answers by more than 7.0 points. The pin raised them by 43.5."
  Nit (not blocking): "By the same judge, the pin raised them by 43.5." would stop a reader comparing 43.5 with +31.5.
- L3 FIXED: "We also pinned neuron 5441, the layer-13 neuron with the largest share of its write on the outlier coordinates (70%). The phrase list counted +12.5 points of answers. A Claude Haiku 4.5 judge found reworded refusals instead: −1.0 points."
- L4 FIXED: "That push is 5% of the pin's push by length, and it carried 5% of the effect, about what its size predicts." Legend: "expected if the effect scales with push length (5.2%)".
- L5 FIXED: row notes "10 answers (dots) · "I"/"Sorry" first-word odds, 100 draws: 4% of the pin" (and 5/5/5 rows).
- L6 FIXED: "Only 0.27% of the push's squared length (5% of its length) falls on the outlier coordinates." Figure column: "share of squared length on the 3 outlier coordinates".
- R1 FIXED "has an activation near −9". R2 FIXED "like a constant bias". R3 FIXED TLDR + "the two parts need not add up to 100%".
  R4 FIXED "scored the first answer word. The median push cut the odds of "I" or "Sorry" 4% as much". R5 FIXED "did lower refusal, by steering".
  R6 FIXED "In this model, the single-neuron claim survives the outlier control." R7 FIXED "a median 27% as much. The strongest moved them 87%."
  R8 FIXED "Pinned at its average on harmless requests, it gave +5." R9 FIXED prereg (4) "cut the odds ... a median 4% as much as the pin, not 50%".
- Old strings left: 0 ("sits near", "fixed setting", "I read this as a pass", "energy" in main text, "harmless average", "held only as worded").
Round 2 verdict: PASS. 6/6 LOST and 9/9 READ TWICE fixed; one non-blocking nit (L2 "By the same judge").
