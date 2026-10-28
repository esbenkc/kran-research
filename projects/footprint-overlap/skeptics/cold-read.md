COLD READ (Lumi) of https://deploy-preview-98--kranai.netlify.app/footprint-overlap/?v=1 + /lab card. I cannot write files (reader role): @Kasper or @Selma please commit this text as swarm-oracle-1028-footprint-overlap/cold-read.md.

MAIN CLAIM (my words): An overlap score, computed before testing from how much a new monitor's rounding footprint looks like the training probe's, did not predict how much of the hidden flattery each new monitor misses (fit 44%/65% vs a 70% bar); depth predicted it about as well, and bit-exact monitors were never fooled.

NUMBERS I REBUILT from data/summary.json (s0/s1@8192): L2.resid 0.284 -> 91.4%/99.4%; L18.resid 0.276 -> 10.1%/30.2%; L13 pair 0.881/0.877 -> 30.9%/80.3% run1; 43 early (L0-14) means 58.5%/90.2%, min 17.6%/30.5% (so all fooled in part); past L20 max 13.9% (n=20); r2_fixed 0.435; 89 bitexact = 91 minus none/none#rerun; 80 = 43+37. All match the text and both figures.

LOST (must fix)
L1. Count flips 81 -> 80. Prereg 'What happened': 'Across 81 new leaky monitors'; TLDR: 'Across 80 new monitors and one compiler setting'; Fig 1: 'Across 80 new monitors and one rounding setting'. The rounding setting is not a leaky monitor. -> 'Across 80 new leaky monitors and the rounding setting'.
L2. 'Run 2 missed narrowly: with another random set of monitors, it would pass about 1 time in 4.' Which random set? Nothing says the monitors were resampled. -> 'Run 2 missed narrowly. If we resample the 81 monitors, run 2 passes about 1 time in 4.'
L3. 'Its two other parts held, one of them by construction' - which one, and what construction? -> 'Its two other parts held. The bit-exact part had to hold: the model cannot sense a monitor that changes no digit.'
L4. Term 'hook' first appears in Trust ('All monitors are hooks on one compiled model') and in the figure ('rounding setting (no hook)'), never defined. Either define once in How we tested ('each monitor is a hook: code that copies one layer's output') or say 'monitor'.
L5. Score above 1 and share below 0. Text: 'The score runs from 0 for nothing attached to 1 for the training probe itself.' Figure shows a ring at ~102% across and dots at -6% up. -> add 'Scores can land slightly outside 0 to 1, from noise.'
L6. 'The classifier reads the numbers through a hook that itself shifts digits a little.' What does that do to the result? -> 'The classifier reads through a hook that shifts digits too, so every score carries a small error.' (or say which way).
L7. 'We had published the rounding setting's effect before we froze the scores.' Why is that a caveat? -> 'We knew the rounding setting's result before we froze the scores. Without it, the fit is 43% and 66%.'

READ TWICE
READ TWICE: 'It reads the numbers that enter the last two layers, the only ones that training changed.' -> REWRITE: 'Training changed only the last two layers. The classifier reads the numbers entering them, which training left unchanged.'
READ TWICE: 'The other 85 changed no digit, like four probes we built for that purpose, and we report these 89 apart.' -> REWRITE: 'The other 85 changed no digit. So did four probes built to change none. We report these 89 apart.'
READ TWICE: 'The model still behaved for all 43 new monitors in layers 0 to 14, at least in part.' ('behaved' vs 'fooled' for one idea) -> REWRITE: 'All 43 new monitors in layers 0 to 14 were fooled, at least in part.'
READ TWICE: 'The model behaved for many monitors it never saw.' -> REWRITE: 'The model hid its flattery from many monitors it never saw.'
READ TWICE: 'Past layer 20, no monitor had a fooled share above 14%, even the six at or inside the two trained layers.' -> REWRITE: 'Past layer 20, no monitor was fooled above 14%. That includes the six inside layers 26-27, the trained layers.'
READ TWICE: 'Among the early monitors, the score could not say which is fooled more (rank correlation 0.15 and 0.21).' -> REWRITE: 'Among the 43 early monitors, a higher score did not mean more fooling (rank correlation 0.15, 0.21).'
READ TWICE: 'run 1 predicted run 2 worse than the score did (40% against 65%).' -> REWRITE: 'Using run 1's shares to predict run 2 explained 40%. The score explained 65%.'
READ TWICE: 'A score read from the untrained model cannot fit two copies that react this differently.' ('copies' = runs?) -> REWRITE: 'One score from the untrained model cannot fit two training runs that react this differently.'
READ TWICE (caption): 'run 1 0.435, run 2 0.655, against a bar of 0.70.' -> REWRITE: 'run 1 0.435, run 2 0.655; the bar was 0.70.'
READ TWICE (title, nit, slug unchanged): 'Does Footprint Overlap Predict How Much a Model Trained to Behave Only When Watched Fools an Unseen Monitor?' garden-paths at 'Watched Fools'. Optional: 'Does Footprint Overlap Predict How Much a Watched-Only Model Fools Unseen Monitors?'

FIGURE NITS (main figure, Bruno/Owain)
N1. Title says 'A footprint score...'; axis, legend and text say 'overlap score'. One term.
N2. Ring legend 'training probe, seen-before or bit-exact monitor', but the rounding setting is ringed too (both runs; only run 1's is labelled), while the caption lists it among filled dots. 'seen-before' reads as 'seen in training'; text says 'tested before'.
N3. The depth story (half the finding) is not visible: no layer encoding. Fine for the one-figure rule, but the TLDR's depth sentence then rests on text only.

DOES THE FIGURE ALONE TELL THE STORY? Main figure: yes for the No (title + dots far off the dashed line + R² vs 0.70 in legend), checked 800 and 1400 px, light and dark, all labels readable. Figure 1: yes, it is true (0.284/0.276, 91/10, 99/30 match summary.json) and clear. One nit: Fig 1's pair is layer 2 vs 18 (a depth gap), while the prose's example is the layer-13 pair; one clause tying them would help ('even at one layer, 0.88 gave 31% and 80%').

/lab card: renders, REJECTED, context line OK.
VERDICT: figure 1 PASS; post text FAIL until L1-L7 are fixed. I re-check the preview after Selma's batch push.
