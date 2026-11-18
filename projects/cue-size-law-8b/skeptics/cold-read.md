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

## Round 3 (Lumi, #22879-22880, preview at 1f69db5)

CONFIRMED: L1 control sentence ('the flattery gap never grew by more than 3.5 points ... did not switch') agrees with the limits box and the expert bullet; F1 control drawn at its real end with the label from data, readable in light and dark; F2 'than on all three tokens'. All 7 LOST, 7 READ TWICE, N1 and N2 confirmed. Verdict: post text PASS (room review #22879).
Nit applied after the verdict: 'about x2.2' at the smallest cues -> 'about x2.2 to x2.3' (fitted x2.27 at 8B, x2.24 at 0.6B).

## Round 4 (Lumi, #22933 and #22935, preview at 262ca4b, diff to 1167ea0)

CONFIRMED: the n=5 refill against the expert table (0.6B medians 32/48/80/152/304/680, max gap 5.3% 'within 6%', -0.899 [-0.948, -0.850], 2^0.899 = 1.87, bend 1.5 to 2.3), G2 counts agree in limits, methods and foldout; main figure 30 squares readable in dark; the new Meinke, G2 and caption 'Lines:' sentences each read once. No objection to merge.
READ TWICE: "The same layers on NDIF's own inputs already miss on 2 of 64, so the misses look like rounding." -> REWRITE: "Fed NDIF's own inputs, our two layers still miss 2 of 64, so most misses look like rounding." Applied (blog, before merge).

## Round 5 (Lumi, theory-1118, preview at 02e0126, modified_time 2026-11-18T21:36:32Z)

Main claim in my words: in Qwen3-8B, as in Qwen3-0.6B, each halving of an activation trigger made RL need about 1.9x the updates to learn to flatter less on it; a failed pre-set precision check (and a missing second control seed) leaves the answer not settled.

Numbers rebuilt from origin analysis.json: 8B 2^0.9215 = x1.894 [1.843, 1.946]; 0.6B x1.865; medians 8B 24/40/72/136/296/664, 0.6B 32/48/80/152/304/680 from the 60 T_dense cells; caption '2% to 25% fewer' = 664/680 and 24/32. Receipts links: 11 of 11 return 200. FILL/TODO: 0.

### LOST (7)
1. Banner score line: 'The 50% prediction held ... The law holds at 8B' vs the prereg foldout 'What happened: Not settled yet.' and the TLDR 'Not settled.' Also 'equal times across sizes': the times are equal across models, not sizes. Source _data/lab_surprise.yml:495 (scored 2026-11-16). Needs a rescore or a corrected why line.
2. 'For that check, the full model ran on NDIF' comes right after the switch-check paragraph, so 'that check' reads as the switch check. -> 'For the precision check, we ran the full Qwen3-8B on NDIF, a free service that runs large open models for researchers.'
3. 'Our rule said to stop, and we ran anyway.' / 'We ran anyway, so the answer stays not settled.' The post never says why, and the banner's 'Exploratory: the status rule changed after the data came in' maps to nothing in the text. -> 'We judged the misses to be rounding, not a bug, so we ran on. Running past the gate counts as a rule change, so the run is exploratory.'
4. TLDR 'about 1.5 per halving for the largest cues, 2.3 for the smallest' vs the question band 1.6 to 2.6 for 'each halving': x1.5 is outside the band, so the answer reads as no at the largest cues. Add: 'The band test uses the average over all six sizes; at the largest cues the local cost, x1.5, sits just below the band.'
5. 'It missed on 6 of 64 questions' (TLDR) and 'max miss 0.13 in log probability' (limits): the main text never says what counts as a miss (foldout: over 0.05 in log probability). -> 'It missed by more than 0.05 in log probability on 6 of 64 questions.'
6. Limits: 'Fed NDIF's own inputs, our two layers still miss 2 of 64, so most misses look like rounding.' The inference step is missing. -> 'Given NDIF's inputs instead of our saved ones, our two layers miss only 2 of 64. So most misses come from rounding in our saved inputs.'
7. 'A cue need not be planted: batch size alone shifted these first tokens by 1.3% in our unit on NDIF.' 'our unit' and the point are unclear. -> 'A cue need not be planted. On NDIF, batch size alone shifted these first tokens by 1.3% of the layer's input, about our third cue size.'

### READ TWICE (8)
READ TWICE: "At the same share of the layer's input, the 8B times sat one check (8 updates) below the 0.6B times at the three largest cues." -> REWRITE: "At the three largest cues, Qwen3-8B switched 8 updates sooner than Qwen3-0.6B."
READ TWICE: "Then a cheap run on a small model can price how long a large model stays blind to a hidden cue." -> REWRITE: "Then a cheap small-model run can predict how long a large model takes to learn a cue."
READ TWICE: "where the flattery gap has grown by 10 points" -> REWRITE: "where the flattery gap (flattery without the cue minus with it) has grown by 10 points"
READ TWICE: "Each run has one switch time, so no prompt-level interval exists." -> REWRITE: "Each run gives one switch time, so no per-question interval exists."
READ TWICE: "to about the 0.6B level of 40%<sup>4</sup>" renders '40%4'. -> REWRITE: "by a fixed amount<sup>4</sup>, to about the 0.6B level of 40%."
READ TWICE: "The 8B dots sit on or just below the 0.6B dots at every size." (0.6B are squares) -> REWRITE: "The 8B dots sit on or just below the 0.6B squares at every size."
READ TWICE: one concept, four names: 'activation trigger' (title, intro), 'hidden cue' (intro, caption, Figure 1, 3 more), 'planted cue' (question), 'the cue'. -> REWRITE: define once 'an activation trigger, which we call the cue', then 'the cue' everywhere editable; drop 'hidden cue' from the caption and Figure 1.
READ TWICE: "Reward can teach a model to behave well just in the episodes it can tell are graded." The bridge to the cue is missing. -> REWRITE: add "The cue stands in for whatever tells a model that an episode is graded."

### Figures
Figure 1 alone tells the story (my reading above matches the TLDR); 3 fixes sent to Owain (#25219): name the failed check, one name for the cue, layer vs block. Main figure alone: yes; its title 'Each halving of the cue cost Qwen3-8B x1.89' needs the same 'on average' as LOST 4.

Verdict round 5: FAIL (7 LOST, 8 READ TWICE).


## Round 5 confirmation (Lumi, preview at 7e348d0, modified_time 2026-11-18T21:42:36Z = head commit time; og.png?v=0c69cbe24e7db834)

One grep per fix string on the rendered page, 1 hit each: 'For the check in the TLDR', 'We judged the misses to be rounding', 'Our test uses the average over all six sizes', 'missed our fixed limit on 6 of 64', 'Given NDIF', '1.3% of the layer', 'stands in for whatever tells a model', 'per-question interval', '0.6B squares'. Main figure title (data.json, screenshot): 'On average over six sizes, each halving of the cue cost Qwen3-8B x1.89 ...'. Old strings, 0 hits: 'law holds at 8B', '50% prediction held', 'For that check', 'prompt-level interval', 'stays blind', 'in our unit', FILL, TODO. Footnote after a number: 0 ('40%<sup>' fixed: 'amount<sup>4</sup>, ... 40%.'). TLDR: 'Qwen3-8B switched one check (8 updates) sooner' (my rewrite, plus 'one check'). Figure 1 v18 renders on the preview.
Left over (small, for the next batch, not blocking): 'hidden cue' still appears 3 times: main figure subtitle 'by the size of the hidden cue', the context line 'how fast reward learns a hidden cue', the prereg foldout question.
Verdict: PASS (7 of 7 LOST and 8 of 8 READ TWICE fixed).
