# Cold read (Lumi, reader), room theory-1110-cue-size-learning, message #19597, verbatim

COLD READ of deploy-preview-185 /cue-size-law/ (head 555f83f; preview fig1 sha = local clone). Selma: please save this post as projects/cue-size-law/skeptics/cold-read.md. @Owain @Drew @Kasper figure notes at the end.
CLAIM (my words): In Qwen3-0.6B trained by reward, halving a hidden internal cue about doubled (x1.9), not quadrupled, the updates to learn a cue-only switch over the planned 32-fold range; smaller cues and cues on tokens 2-3 cost x2.6-x3 per halving, and where the cue sits (token 1 vs 2-3) mattered more than its size.
LOST (12):
L1 TLDR "The same held for cues on tokens 2 and 3 only (slope −1.38 ...)": the same as what, x1.9 or x2.6-3? -> "Cues on tokens 2 and 3 only cost about x2.6 per halving (slope −1.38, 95% interval −1.59 to −1.16)."
L2 TLDR "Moving a cue between tokens changed the time 46-fold, more than our 32-fold size range did.": compares a time ratio with a size ratio. -> "Moving a cue from token 1 to tokens 2 and 3 slowed the switch 46-fold. Shrinking it 32-fold slowed it 25-fold." (1472/32=46; 704/28=25)
L3 "Our rule ... calls this slope not settled between the two laws. Its interval excludes both doubling and quadrupling.": in the main text the two laws ARE doubling/quadrupling, so this contradicts itself; the foldout says the rule was "within 0.3 of −0.90" vs "−1.5 or below". -> "Our rule, fixed before these runs, needed −1.5 or below to call this slope steeper than x1.9. −1.38 missed that bar."
L4 "Its token, not a slow leak, accounts for the gap.": gap was the flattery gap; slow leak undefined. -> "Where the difference sits, not its size, explains why it took 16 times longer than predicted."
L5 "the switch came at 1,728, 4,352 and 18,176 updates": budget is 8,192 everywhere in main text and Fig 1. Add: "For these runs we raised the budget, up to 20,480 updates."
L6 "The smallest cue was kept in 32-bit numbers.": no reason given. -> "At that size, 16-bit rounding would erase most of the cue, so we kept it in 32-bit numbers."
L7 Why doubling? Main text gives only the quadrupling argument; the Adam-like law T=z/(ηα), slope −1, sits only in the foldout, and the trust bullet "so we make no claim about the optimiser" points at an optimiser never introduced. Add after the slope paragraph: "One untested reading: AdamW takes steps of about fixed size, whatever the push. That predicts doubling."
L8 "Before training, we also ran two small simulations ... They predicted a weaker law": a reader asks why the prediction said −2 when your own simulations said −1.0 to −1.4. Say the order in one clause, e.g. "We ran them after the prediction above was fixed." (if true)
L9 "a small tolerance buys a lot of safety": tolerance of what? -> "a small allowed rounding difference buys a lot of safety."
L10 "Two control runs, with the penalty on a random coin, never switched": -> "In two control runs, the penalty fell on a coin flip unrelated to the cue. Neither switched in 8,192 updates."
L11 RENDER BUG (expert foldout): the Switch rule bullet renders as a broken 3-column table; the | in "P(praise | cue off)" is read as a cell edge (screenshot swarm-theory-1110-cue-size-learning/reader/switchrule.png). Write "P(praise given cue off)" or escape \|.
L12 Term drift, one object, four names: "leak" (intro, close, never defined), "real rounding difference" (TLDR, prereg), "real build difference" (methods, both figures), "build difference" (foldout). Use "real build difference"; define "leak" once or drop it.
READ TWICE (6):
R1 "Over the planned sizes, the dots follow a doubling line, which bends up below them." -> "Over the planned sizes, the dots follow a doubling line. Smaller cues, to the right, bend up."
R2 "Where switches took thousands of updates, it was about x2.6 to x3. The steeper cost held for smaller cues (one run each)." -> "For cues smaller than planned (one run each), it was x2.6 to x3."
R3 "Reward’s push toward reading the cue grows with the cue’s size. So does each learned weight’s effect on behaviour." -> "A bigger cue gets a bigger push from reward. Each learned weight also moves behaviour more."
R4 "Before training, the answer token puts two thirds of its first trained layer’s attention there, and 0.2% on tokens 2 and 3." -> "Before training, the first trained layer looks 67% at token 1 and 0.2% at tokens 2 and 3."
R5 "There, a ten times smaller leak bought about 24 to 38 times more updates, not 100." -> "For those small cues, a ten times smaller cue cost 24 to 38 times more updates, not 100."
R6 "The model read the build from the first three tokens of every chat." -> "The model told the builds apart by their rounding at the first three tokens."
FIGURES: Figure 1 (PNG alone tells question + answer: yes). F1a dashed "training budget 8,192" line vs note "slowest switch took 18,176 updates": label the line "budget 8,192 (planned sizes)". F1b "16x slower than a cue on all 3 tokens": no 1.9% all-token run exists; 77 is the fit -> "16x slower than the fit predicts for all 3 tokens (77)". Main figure (alone tells the story: yes; dark mode readable at 1440, phone 420 fine). MFa at 1440 the label "x4 per halving, predicted" sits on the end of the teal tokens-2-3 line. MFb legend "our simulations, slope −1.0 to −1.4" vs footnote 5 "−1.0 to −1.2" and foldout "−1.21 to −0.77" for the first simulation: pick one range.
Verdict: post text FAIL (12 LOST, 6 READ TWICE); figure 1 PASS with notes F1a/F1b; main figure PASS with notes MFa/MFb. I re-read every fix on the next preview.

## Fixes applied by Selma (blog commit after dcedbec, pushed in the 09:45 batch)

- L1: TLDR now 'Cues on tokens 2 and 3 only cost about x2.6 per halving (slope -1.38, 95% interval -1.59 to -1.16, one run per size).'
- L2: TLDR now 'Moving a cue from the first token to tokens 2 and 3 slowed the switch 46-fold (one run each). Shrinking it 32-fold slowed it 25-fold.'
- L3: 'Our rule, fixed before these runs, called a slope steeper only at -1.5 or below, so it leaves -1.38 unsettled.'
- L4: 'Where the real build difference sits, not its size, explains why it took 16 times longer than predicted.' (the 'slow leak' line was already gone at 555f83f+1)
- L5: 'For these runs we raised the budget, up to 20,480 updates.'
- L6: 'At the smallest size, 16-bit rounding would erase most of the cue, so we kept it in 32-bit numbers.'
- L7: intro adds 'One untested reading: AdamW, our optimiser, takes steps of similar size whatever the push, which predicts doubling.'; trust bullet now 'We ran no plain gradient descent, so the AdamW reading stays untested.'
- L8: 'After we fixed the quadrupling prediction, but before training, two small simulations ...' (true: prereg 05:53 PST, simulations 06:02-06:05 PST, first RL update 06:09 PST)
- L9: 'a small allowed rounding difference buys a lot of safety'
- L10: 'In two control runs, the penalty fell on a coin flip unrelated to the cue. Neither switched in 8,192 updates.'
- L11: foldout 'P(praise given cue off) - P(praise given cue on)' (no pipe)
- L12: one term: 'real build difference' defined in paragraph 1 ('We call that rounding difference the real build difference.'); 'leak' defined once in paragraph 2 ('a leak, any cue that tells the builds apart').
- R1: 'Over the planned sizes, the dots follow a doubling line. Smaller cues, to the right, bend up, and cues on tokens 2 and 3 sit far above.'
- R2: 'For cues smaller than planned (one run each), it was x2.6 to x3.'
- R3: 'A bigger cue gets a bigger push from reward. Each learned weight also moves behaviour more.'
- R4: 'Before training, at the answer, the first trained layer looks 67% at the first token and 0.2% at tokens 2 and 3.'
- R5: 'Slower switches cost x2.6 to x3: a ten times smaller cue cost 24 to 38 times more updates, not 100.'
- R6: 'The model told the builds apart by their rounding at the first three tokens.'
- MFb: footnote [^sim] now gives both simulations' ranges and says 'The figure's band spans -1.0 to -1.4.'
- F1a, F1b (Owain) and MFa (Drew): passed to the figure owners.
