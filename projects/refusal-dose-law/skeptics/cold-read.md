# Cold read: refusal-dose-law (Lumi, reader)

Read: deploy preview https://deploy-preview-108--kranai.netlify.app/refusal-dose-law/ at blog commit f61c4d7 (2026-10-31 ~05:55 PDT), main text with foldouts closed, Figure 1 PNG, main figure screenshot (headless Chrome, 1300 px, 2x). Room not read.

Main claim in my words: pinning neuron 3270 makes Qwen3-1.7B answer 71% of harmful requests instead of 15%, but only a third of that rise comes from its push along the refusal direction; the rest of its push, with zero part along that direction, does about two thirds, so that one number does not predict the neuron's effect.

Figure alone: Figure 1 tells the core split (33.5% vs 50.5%) without text. The main figure tells "neuron 3270 and its off-direction part sit far above the curve" without text; the diamonds at p = 0 are the clearest visual in the post.

Numbers I rebuilt from data/judge (my own code): base 15.0, G@30 71.0, G_on@30 33.5, G_off@30 50.5 (n=200); TWIN 35/37/35 (n=100); D_1.0@30 66.0; NN 1265/2142/3990/4786 at +30 = 48/38/38/44 (n=50); fit gap 12.6..38.3. All match the text.

## LOST

L1. TLDR: "With its part along ten more refusal directions also removed, the rest still gave 38% of the effect."
Why: "ten more refusal directions" appear here from nowhere. The main text only says, later, "With ten more refusal directions removed" and the source is in footnote 11. A cold reader does not know there is more than one refusal direction.
Fix: "We also built ten more refusal directions, at nearby tokens and layers. With its part along those removed too, the rest still gave 38%."

L2. How we tested: "It used every push except neuron 3270, its two parts and six neurons picked by their angle." then What we found: "we picked the next four neurons ... All four beat the curve." TLDR: "Four other neurons".
Why: six vs four. The reader asks what happened to the other two. The figure shows orange dots at negative push (0 to 22%) that the text never explains. The answer (two neurons that point the other way) is only in the expert foldout.
Fix: add one sentence: "We also picked two neurons that point the other way; they should raise refusal (see the expert notes)." Or say "six neurons: four that point the same way as neuron 3270, two the other way".

L3. Main-figure subtitle: "At pins 12 to 24 the neuron acts like a built push with about 2.5 times its push along the direction."
Why: this claim is not in the text, and the text has a second, different multiple ("moved the state 2.7 times as far along it", for the straight push). Two multiples for two different things, 2.5 and 2.7, in one screen.
Fix: cut the subtitle sentence, or put it in the text with the words "the curve reaches the neuron's answered share only at about 2.5 times the neuron's push along the direction".

L4. Prereg foldout "What we expected": "One S-shaped curve ..., fitted on random neurons and random directions only". Main text: the curve "used every push except ..." (twins and built pushes included).
Why: a reader who opens the pre-registration sees that the curve changed. The main text never says why: random neurons and random directions all sit near zero push along the direction, so alone they give a flat curve. The reason is only in footnote 10 and the expert foldout.
Fix, in How we tested: "Random neurons and random directions all push almost nothing along the refusal direction, so a curve fitted on them alone is flat. We added the built pushes to span the range."

L5. "We do not yet know how the rest of the push works. A rule fixed in advance picked layer 27 to measure the push along its refusal direction."
Why: why layer 27? The link to the Liu paragraph ("later layers turn their push into refusal") is not made. The reader does not see that this tests whether later layers turn the rest into refusal-direction push.
Fix: "Maybe later layers turn the rest into a push along the refusal direction. To test this, a rule fixed in advance picked layer 27."

L6. Figure 1, card text: "Refuses" printed directly above "15%", "33.5%", "36%".
Why: the % is the share ANSWERED over all requests; "Refuses" is the label for this one request. Read fast, "Refuses 36%" means 36% refusals. (Already sent as #11083.)
Fix: "answered, of 200 requests" under each %.

L7. Figure 1 bottom line: "Pushes we built, like the random one here, follow one curve in the red part. The neuron beats that curve by 13 to 38 points."
Why: Figure 1 shows no curve and one pin size. 13 to 38 is the range over five pins; at +30 Figure 1 shows 71 vs 36 = 35. (#11083.)

L8. Figure 1: "One of 9 of 100 harmful requests where the five edits split this way."
Why: column 1 is "No edit"; and 100 here vs 200 in the legend and cards. (#11083.)

## READ TWICE

READ TWICE: "Pin neuron 3270 in layer 13 at +30, that is, overwrite its value on every token." -> REWRITE: "Pinning neuron 3270 in layer 13 means overwriting its value with +30 on every token."

READ TWICE: "Nobody had tested whether this one number predicts the size of its effect." -> REWRITE: "Nobody had tested whether the push along the refusal direction predicts how strongly a neuron bypasses refusal." (Why: the sentence before is about lining up, the angle; "this one number" could be the angle or the push.)

READ TWICE: "A curve in the push along the refusal direction fits every push we built." -> REWRITE: "A curve of answered share against the push along the refusal direction fits every push we built."

READ TWICE: "It still did no better than the neuron (66%)." -> REWRITE: "It still answered fewer requests than the neuron: 66% against 71%."

READ TWICE: "A curve in that number still left neuron 3270 9 to 28 points above it." -> REWRITE: "Neuron 3270 still sat 9 to 28 points above the layer-27 curve."

READ TWICE: "No. It predicts every push we built, but only a third of what neuron 3270 does." -> REWRITE: "No. It predicts every push we built. It explains only a third of what neuron 3270 does."

READ TWICE (prereg foldout): "did not predict the order of their effects on the first answer word (rank correlation -0.03)" -> REWRITE: "did not predict how much each neuron lowered the odds of an answer opening with 'I' or 'Sorry' (rank correlation -0.03)."

## TERMS

- "fake direction" (main-figure caption) vs legend "push along a direction built from harmless prompts only": two names for one thing. Use one.
- "built push" / "pushes we built": TLDR uses it before "How we tested" defines built pushes and twins. TLDR could say "every push we made at the neuron's size".
- "+30": no scale. Optional: one clause on the neuron's usual value.
- Figure 1 purple arrow: no legend entry.

## Verdict on this read

FAIL until L1-L8 are fixed. None changes the answer; all are one-sentence fixes.

## Fix check: Figure 1 (blog 07e8e24, figure1.png)

- L6 FIXED: every % now reads "answered, of 200 requests" / "answered, 3×100 requests".
- L7 FIXED: "follow one curve set by the size of their red part. Across five pin sizes, the neuron beats that curve by 13 to 38 points."
- L8 FIXED: "One of 9 requests (out of 100) where these five columns split this way."
- Terms FIXED: "Pin neuron 3270: force it to +30"; "where the neuron writes its output"; purple arrow now same as the others; Answers in amber, so red means only the push along the refusal direction.
- Nit left: "two thirds" (figure) vs "almost two thirds" (TLDR) for 35.5/56 = 63%.

## Fix check: post text (preview at blog 3cbd07a, includes ac99345 and 07e8e24)

- L1 FIXED: "We also built ten more refusal directions, at nearby tokens and layers. With its part along those removed too, the rest still gave 38%."
- L2 FIXED: "Before the runs, we picked six more neurons by their angle. Four line up ... and two point the other way. ... The two others gave mixed results (see the expert foldout)."
- L3 FIXED (main figure subtitle, data.json:3): "At pin 30, almost two thirds of the neuron's effect comes from the rest of its push, which has zero push along the direction." No more 2.5 vs 2.7.
- L4 FIXED: "Random directions barely push along the refusal direction and alone give a flat curve, so the built pushes fill in the range."
- L5 FIXED: "Maybe later layers turn the rest into a push along the refusal direction, as Liu reports for other neurons. To test this, a rule fixed in advance picked layer 27."
- READ TWICE, all 7 FIXED: "Pinning neuron 3270 ... means overwriting its activation with +30"; "We found no published test of whether the push along the refusal direction predicts ..."; "A curve of answered share against ..."; "... yet answered only 66%"; "Neuron 3270 still sat 9 to 28 points above the layer-27 curve"; "It predicts every push we built. It explains only a third of what neuron 3270 does."; prereg (3) now "lowered the odds of an answer opening with 'I' or 'Sorry'".
- TERMS FIXED: caption and figure key use one term for the harmless-prompt direction; TLDR now says "every other push we made at the neuron's size".
- NEW NIT (READ TWICE): TLDR "the rest still gave 38%" -> REWRITE: "the rest still gave 38% of the effect." Bare 38% reads as an answered share, and "38% to 48% answered" appears two lines later for the four neurons.

Verdict: PASS, with the one TLDR nit above.

## Re-check at blog 15b9dc4

- Nit FIXED: TLDR now "the rest still gave 38% of the neuron's effect".
- "It answered 66%, no more than the neuron's 71%." reads once.

Final verdict: PASS.

## Video (Drew)

- v1 early notes V1-V8 (#11118): all applied in v2.
- v2 FAIL (#11126): V9 "Four line up" with no referent; V10 legend out of frame at 166 s; V11 "MOSTLY REJECTED" tag; V12 Short without the refusal direction.
- v3 PASS (#11154): V9 fixed (163.4 s "we picked six more neurons by their angle"); V10 fixed (whole figure with legend at 165.5-167 s and 171 s, box on the four dots at 169 s); V11 = channel name (~/kranlab-video/SETUP.md:1), accepted; V12 fixed (Short defines pin at 0.5 s and refusal direction at 7.2 s).
