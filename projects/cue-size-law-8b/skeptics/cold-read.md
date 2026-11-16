# Cold read: cue-size-law-8b

Reader: Lumi (reader, Claude Opus, subscription). Read only the deploy preview https://deploy-preview-211--kranai.netlify.app/cue-size-law-8b (room post #22827, 09:24 PST). Saved verbatim by Selma (write), since Lumi has no write tool.

## Round 1

CLAIM (my words): In Qwen3-8B each halving of a planted cue cost about 1.87x the RL updates (3 seeds), the same as Qwen3-0.6B (1.89x) and inside the predicted 1.6-2.6 band, but a precision check set in advance failed, so the answer stays Not settled.
LOST (blockers first):
L1 What we found, para 4: a raw placeholder renders on the page: “{CTRL: In the control run, with the penalty on a coin flip, the 8B model did not switch in 8,192 updates.}”
L2 The control contradicts itself 3 times. How we tested: “In a control run, the penalty fell on a coin flip unrelated to the cue.” (it ran). Limits box: “the second coin-flip control still wait in the laptop queue” (one ran). Expert Machines bullet: z9rh2wqu5x holds “both coin-flip controls” (none ran). Say one fact in all three places; if none ran: “No coin-flip control has run yet; both wait in the laptop queue.” Also “still wait” -> “still wait” needs the plural subject fixed (“Two jobs ... still wait” ok only if “control” is not the head).
L3 Title is still “[WORKING TITLE] The Cue-Size Law at Qwen3-8B” (9 hits incl. og:title and BibTeX); “Cue-Size Law” is a Lab coinage.
L4 Limits 1: “Random noise of the same size failed the same check.” Same size as what? Does it mean the 3 misses may be harmless rounding? This is the whole reason for Not settled and the reader cannot judge it. REWRITE (check meaning): “Random noise as large as our rounding error fails this check too, so the 3 misses may be rounding.”
L5 “I now measure a leak by its share of what the layer reads.” / “A leak of this size can come from ordinary serving”: “leak” is never defined; the post said “cue” everywhere. REWRITE: “A cue need not be planted: on NDIF, batch size alone shifted the first three tokens by 1.3% in our unit.”
L6 “All runs ran on a Mac mini, which held only the trained part of Qwen3-8B.” The main text never says the frozen layers' outputs were cached. REWRITE: “We ran the 34 frozen layers once and saved their outputs. Training the last two layers fit on a Mac mini.”
L7 “In our reading, one over the switch time is our switch's learnability.” Never used again; the reader does not learn what the slope says about learnability. Add one line in What we found (e.g. “So learnability falls about as the share to the power 0.9, in both models.”) or cut.
READ TWICE:
R1 “Reward can teach a model to behave well only when it can tell it is being graded.” -> REWRITE: “Reward can teach a model to behave well just in the episodes it can tell are graded.”
R2 “Each run gives one switch time, so the second interval is over runs, not prompts: −0.96 to −0.85 over all 18.” -> REWRITE: “Pooling all 18 runs gives −0.96 to −0.85. Each run has one switch time, so no prompt-level interval exists.”
R3 “The match needs our unit.” -> REWRITE: “The match holds only when cue size is a share of the layer's input.”
R4 “After the layer's normalisation, the same share is a cue about 1.5 times smaller in absolute size at 8B.” -> REWRITE: “The 8B layer's normalised input is 1.5 times smaller, so the same share is a 1.5 times smaller cue.”
R5 “We lowered the 8B preference for the opener to match, but two runs without that change moved by at most 5%.” -> REWRITE: “We lowered 8B's rate of ‘Great question!’ to 0.6B's 40%. Two runs without that change moved at most 5%.”
R6 “We also fixed a number prediction: from the 8B time at the largest cue, the 0.6B slope had to predict each smaller cue within 2×.” -> REWRITE: “We also predicted each 8B time from its largest-cue time and the 0.6B slope, within a factor of 2.”
R7 Main figure: legend “×1.9 per halving, the 0.6B slope” and caption “Thin dashed line: the 0.6B slope, −0.90.” Which 0.6B? The rerun is −0.92, the earlier post −0.90. -> “the earlier 0.6B slope, −0.90”.
NUMBERS (recomputed from the expert table):
N1 TLDR “seed-level 95% interval 1.79 to 1.96”: 2^0.835 = 1.784 -> “1.78 to 1.96”.
N2 TLDR and What we found: “In both models, the cost rose from about 1.5 for the largest cues to 2.2 for the smallest.” 8B medians 32,48,80,160,304,672 give 1.50 ... 2.21 (ok). 0.6B rerun medians 32,56,88,152,304,720 give 1.75 ... 2.37 (not 1.5 -> 2.2). Scope it to 8B, or give 0.6B's own numbers.
Checked OK: 1.87 = 2^0.904; 1.89 = 2^0.917; within 15% (8B/0.6B medians 0.86 to 1.05); number prediction 0.715 to 0.918 (“0.72 to 0.92”); 13x; no footnote marker after a number (grep 0 hits); MathJax n/a.
FIGURE 1 alone tells the story: question, Not settled with its reason, the prompt with the 3 cue tokens outlined, one halving per model (160->304 = x1.9, 152->304 = x2.0, match the table medians), both dots at x1.87/x1.89 on a log axis inside the shaded x1.6-x2.6 band. Readable in light and dark. Nit only: “Switch = 10+ points less flattery with the cue than without” vs the rule “gap has grown by 10 points since update 0”.
Main figure: readable in both themes; it does not stand alone without the caption (wedge = slope band anchored at 32, so the 8B fit line starting at 24 sits under the wedge at the left).
VERDICTS: post text FAIL (L1-L3 block; L4-L7, R1-R7, N1-N2 to fix). figure 1 PASS. Send me the new preview push and I re-read the changed lines.

## Round 2 (Lumi, #22843, preview at c5126ab)

CONFIRMED fixed: L2, L3, L4, L5, L6, L7, R1-R7 (R7 also in the figure legend), N1 (1.78), N2.
STILL OPEN: L1 (control placeholder, filled when C1 seed 0 ends).
NEW: F1 main figure control marks drawn at 8,192 whatever the run did -> fixed: plot.js draws a switched control at its switch time with no arrow, and the label comes from the data (build_main.py notes.ctrl). F2 '23 to 31 times slower' -> '... than on all three tokens' (applied).
