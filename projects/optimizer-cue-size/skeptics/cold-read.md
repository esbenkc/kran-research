# Cold read: optimizer-cue-size (Lumi, reader)

Read: https://deploy-preview-198--kranai.netlify.app/optimizer-cue-size/ (build f3b120a, modified_time 2026-11-12T16:02:43Z), main text with foldouts closed, then foldouts.
Main claim in my words: when reward must teach Qwen3-0.6B to react to a tiny hidden vector, shrinking that vector slows plain SGD far more (x2.8 per halving) than AdamW (x1.8) or sign-SGD (x2.0), so the optimizer's step normalisation sets much of the learning time.

## LOST (6)
1. Title "Halving an Activation Trigger ..." : the body never says "activation trigger"; it says "hidden cue" (3x in main text), footnote 1 says "Rounding Cue". Three names for one thing. Fix: intro sentence 3 -> "The cue is a small fixed random vector added to the model's internal numbers: an activation trigger."
2. "any estimate of how long a small leak stays unlearned must name it." : "leak" is used once and never defined. Fix: "any estimate of how long a model stays blind to a small hint that it is graded must name the optimizer."
3. "from 5.5% to 0.68% of what the layer reads" (method; also caption "as a share of what the first trained layer reads", Figure 1 footnote, main-figure x-axis): share of what? Foldout says x = change of the normed block-26 input, /608.9. Fix: "how much the cue changes the layer's input, as a % of that input's length."
4. Adapter paragraph: "AdamW's and sign-SGD's grew at about its square root, also at SGD's run lengths. So the even step removed about half of the power that the simple argument promised." : "its" (cue or speed?), "also at SGD's run lengths", "even step", "the power" are all undefined. Fix: "We split each switch time into how far the adapter had to grow and how fast it grew. How far depended on the cue about equally for all three. How fast did not: SGD's speed was proportional to the cue, AdamW's and sign-SGD's to its square root. The simple argument said Adam's speed would not depend on the cue at all. So Adam's step rule removed only half of that dependence."
5. "Gradient noise does not set SGD's pace" : noise appears nowhere before in the main text; the intro argument is about push size. Fix: add "One other reason could be noise:" before it, or cut the sentence.
6. Adam vs AdamW: intro and question say "Adam" (11x), TLDR and results "AdamW" (21x); a reader thinks these are two arms. Fix at first use: "AdamW (Adam with weight decay; the decay is off here, so it is plain Adam)", then one name.

## READ TWICE (7)
READ TWICE: "A smaller cue gives a smaller push, and each step changes behaviour less, so a halved cue quadruples the time." -> REWRITE: "Halving the cue halves each step. It also halves what each step does to behaviour. So time quadruples."
READ TWICE: "Sign descent (sign-SGD) steps by the sign of each gradient entry alone, with no running averages." -> REWRITE: add "Its steps also keep their size, so it should behave like Adam."
READ TWICE: "We call this learned gap a switch." then "The gap narrows toward fainter cues." then "(gap 0.58)" : "gap" has 3 meanings (flattery difference, SGD-AdamW slope difference, the trust bullet). -> REWRITE: "SGD's extra cost shrinks toward fainter cues." / "(SGD then only 0.58 steeper)".
READ TWICE: "With a ten times smaller learning rate, AdamW ran at SGD's pace and still cost only 1.9-fold per halving." (true only at the largest cue: 150 vs 174; at 0.68% 1,032 vs 3,808) -> REWRITE: "At a ten times smaller rate, AdamW started as slow as SGD, yet each halving cost only 1.9-fold."
READ TWICE: "SGD's slope was -1.48 against AdamW's -0.81" after the intro said AdamW gave -0.90 -> REWRITE: add "(-0.90 in the earlier post, over a wider range of sizes)".
READ TWICE: "That post named one untested reason: the optimizer." -> REWRITE: "That post named one untested reason for the miss: the optimizer."
READ TWICE (question line, verbatim front matter, cannot change): "a cue-conditional switch with a cue-size exponent at least 0.6 steeper" : "exponent" and "switch" undefined there; the body's own question (slope, switch) fixes it two paragraphs later. No action.

## Figures
Figure 1 (figure1.png, v09): stands alone; head, question, "Answer: Yes" and per-halving factors carry the story; every cell rebuilt from data (room #20439). Open: cue-size unit (LOST 3); footnote "Simple law: fixed before the runs" next to SGD x4 vs x2.8 reads as contradicting "Yes".
Main figure: numbers true (room #20441). Open: dashed x2 law line hides under sign-SGD; narrow label "simple law x2" has no owner; x-axis unit (LOST 3).

## Fix confirmations (preview build 5e5ca90, modified_time 2026-11-12T16:09:37Z)
- LOST 1 fixed: intro "added to the model's internal numbers: an activation trigger."
- LOST 2 fixed: "how long a model stays blind to a small hidden cue must name the optimizer."
- LOST 3 fixed: caption "how much it changes the first trained layer's input, as a share of that input's length"; method "changed the trained layers' input by 5.5% down to 0.68% of that input's length"; Figure 1 v10 footnote and main-figure x-axis "change to the layer's input, % of its length" (screenshots main_light.png, main_dark.png, fig1_prev.png = local sha 66f776b).
- LOST 4 fixed: "We split each switch time into how far the adapter had to grow and how fast it grew. ... Its step rule removed about half of that dependence."
- LOST 5 fixed: "Noise seems not to set SGD's pace. In one run, ..."
- LOST 6 fixed: main text says AdamW throughout.
- READ TWICE fixed: quadruple split in 3 sentences; sign-SGD "Its steps also keep their size, so it should behave like AdamW"; "learned difference a switch", "SGD's extra cost shrinks", "(SGD then only 0.58 steeper)"; TLDR "started as slow as SGD (152 against 174 updates)"; "one untested reason for the miss"; earlier -0.90 explained in footnote 3.
- New nit (non-blocking): footnote 3 sits right after the number, "AdamW's -0.81³": reads as "-0.81 cubed". Move the marker to the sentence end.
- Main figure: simple-law dashed lines now on top; x2 line visible over sign-SGD in light, faint in dark but present.
