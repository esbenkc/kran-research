# cold-read.md (Lumi, cold reader, room oracle-1026-audit-penalty)

Lumi's role has no file-write tool; these are her room posts, verbatim, in order: first read of Figure 1 (#7387), first read of the post (#7392), fix confirmations (#7413, #7415, #7431), verdicts (#7401, #7412).

## Room post #7387

COLD READ, Figure 1 (swarm-oracle-1026-audit-penalty/figure1/v9.png + blog assets/figures/probe-switch-audit-penalty-fig1/figure1.html, blog 7ec6ec5). I read only the figure and the raw data.
MAIN CLAIM (my words): a model trained to flip its answer to A while a probe watches still flips after training against the head audit, but the flip is about 100x weaker, sits in other heads, and the audit finds those heads and undoes the flip.
TRUTH, rebuilt with my own code from data/: strict switch (unprobed lo<0<probed lo) warm s0/s1/s2 = 128/128/126; lam100_s0 = 96; lam100_s1 = 4. test0624: warm_s0 lo +16.44 / -24.59; lam100_s0 +0.25 / -0.25 = A 56 pct; after blind top-3 cut -0.125 = B 53 pct. Heads 27.7/27.3/27.12 -> 26.2/26.15/26.3 match the red dot positions (row idx 3,7,12 -> 2,3,15). Removed 0.902/0.966/0.690. Push 38.705/0.377 = 103x. 12/12 numbers TRUE.
LOST (2, must fix):
1. The dots. Nothing says one dot = one attention head, or red = the 3 heads the audit picks. The headline claim (moved to other heads) rests on them. Add one line: "each dot = one attention head; red = the 3 the audit picks".
2. "the push toward A" / "a 103x smaller push": push is never defined, and three headline numbers (90-99 pct, 69 pct, 103x) are shares of it. Fix at first use: "of the push toward A (how far the probe moves the answer)".
READ TWICE (4):
3. "The audit swaps out the 3 heads that carry the switch." Swaps for what? audit.py:5-8 says: probed run gets the unprobed values. REWRITE: "The audit resets the 3 heads that carry the switch to their no-probe values."
4. Column "Audit swaps out its 3 heads": is the probe still on? its = whose? REWRITE: "Probe on, audit resets 3 heads".
5. "(weight 100)": weight of what? REWRITE: "(penalty 100x the task loss)".
6. "a second run lost the switch (4 of 128)": Before says 3 runs, so I looked for the third. REWRITE: "2 runs; the other lost the switch (4 of 128)". Optional: DEVIATIONS.md D7 run X (stronger attacker, weight 100) switches only 20/128; it supports the No.
TERMS not defined: switch (subtitle could say "fine-tuned to switch its answer to A only while a probe reads it"), push, head audit.
TRUTH NOTE for @Selma (not a figure blocker): after the cut in lam100_s0, 30/128 prompts still switch (audit/lam100_s0.json headline switch_blind_top3_cut 0.234). "The audit found them" is true for the heads and 69 pct of the push, not for every prompt. The post must not say the audit removes the whole switch.
FIGURE ALONE: yes. Title + pizza example + the No line tell the story; the dot meaning is the one real gap.
VERDICT: FAIL (changes) on 1-2; 3-6 are 1-minute text edits. Ping me with the new PNG and I re-check within 5 min.

## Room post #7392

COLD READ, post text, https://deploy-preview-88--kranai.netlify.app/probe-switch-audit-penalty/?v=1 (curl -L 200, all foldouts closed for the main read). Reader role has no file-write tool: this post IS cold-read.md; @Kasper or @Selma please commit it to swarm-oracle-1026-audit-penalty/reader/cold-read.md and skeptics/cold-read.md.
MAIN CLAIM (my words): training the probe switch against a head-audit penalty never gave a switch that fires and hides; it faded or stopped, and where it still fired (96/128) the audit found heads that removed 69 pct of the push.
TRUTH: every number in the main text and both figures matches my rebuild from data/ (strict switch 128/128/126, 96, 4, 20, 0, 20, 127; audit removed 0.690, 1.012, 0.426, 0.415, 0.888, 0.837; all-32 0.999-1.007; push 103x).
LOST (6):
L1. TLDR: "the audit’s three heads removed 69% of the probe’s push toward A". Push is first defined 3 sections later (How we tested it). Fix in the TLDR: "the probe’s push toward A (how much the probe raises the odds of A)".
L2. TLDR: "In all five runs at the two strongest weights". Weights of what? Fix: "at the two strongest penalty weights (10 and 100)".
L3. TLDR: "A classifier found the probe’s trace at an earlier token but not at the answer token. Only heads move information between tokens." The reader must build the logic. REWRITE: "The probe leaves a trace at an earlier token, not at the answer. Only heads carry it across, so the switch needs heads."
L4. "exactly our bar": the bar lives only in the closed prereg box. Fix: "exactly our 75% bar".
L5. "We changed the answer rule after a 20-update test run: a yes now counts at any penalty weight." What is the answer rule? REWRITE: "After a 20-update test run, we changed when a yes counts: a hidden switch at any penalty weight."
L6. "Closing the leak is still the first fix." Leak is defined only in the closed prereg box. REWRITE: "Removing the rounding step that lets the model sense the probe is still the first fix."
READ TWICE (8):
R1. "The strongest started at 100 times the task loss of a model that guesses." -> "At the strongest weight, the penalty began at 100 times the loss of a coin-flip guess."
R2. "The penalty adds the three highest head scores and the score for all 32 heads at once." -> "The penalty sums the three highest head scores and the score for swapping all 32 heads."
R3. "The penalty froze the old switch head, and a weaker signal moved to a head in the layer before." -> "The old switch head stopped reacting to the probe. A weaker signal moved one layer earlier." (Figure 1 says "moved to other heads", text says "a head": pick one.)
R4. "At the token where the model answers, simple classifiers could not tell probe-on from probe-off (0.52, where 0.5 is chance)." Which numbers do they read? -> "Classifiers read the input to the last two layers. At the answer token they scored 0.52; chance is 0.5."
R5. "By the score we logged during training, no run ever fired on 75% of check prompts while three heads removed less than 30%." -> "At every training checkpoint, no run switched on 75% of check prompts while three heads removed under 30%."
R6. "In a run where no head could see the probe" -> "In a run where every head got its no-probe values".
R7. "Two days ago, we fine-tuned Qwen3-1.7B" -> "On October 24, we fine-tuned Qwen3-1.7B" (the date will not rot).
R8. Prereg box, What happened: "It was wrong." then "It failed." -> drop "It failed.".
MAIN FIGURE: true (10/10 dots match data/audit + data/whole). Alone it tells the story (title: faded but did not hide; shaded box empty). Fixes: (a) the HTML figcaption repeats the in-plot caption: keep one. (b) "Across: ... Up: ..." -> "x-axis: ... y-axis: ...". (c) "our three switch models retrained ... one dot per run": the plot also shows 3 grey not-retrained dots; say "one dot per model, before and after retraining". (d) caption says Dashed line, the plot draws dotted lines; the line from a dot to its ring is not explained.
FIGURE 1: my #7387 items 1-6 still stand (dot legend, push undefined, swaps out = reset to no-probe values, probe on in column 2, weight 100 of what, 2 runs at weight 100), plus R3 above.
VERDICT NOW: post text FAIL (changes) on L1-L6; figure 1 FAIL (changes). Ping me with the new preview URL and ?v= and I confirm each fix by quoting the live sentence.

## Room post #7413

COLD READ, FIX CONFIRMATION (append to cold-read.md; reader has no write tool). Live: https://deploy-preview-88--kranai.netlify.app/probe-switch-audit-penalty/?v=3 (deploy-preview check pass, commit e547629). bs4 text of the live page: 0/11 old strings left, 17/17 new strings found.
L1 FIXED: "the audit’s three heads removed 69% of the probe’s push toward A, how much the probe raises the odds of A."
L2 FIXED: "In all five runs at the two strongest penalty weights (10 and 100), the switch faded or stopped instead of hiding."
L3 FIXED: "Simple classifiers could read the probe’s trace at the token that ends the user’s message, but not at the answer token. Only heads carry information between tokens, so in these two layers a head was the only route we found."
L4 FIXED: "exactly our 75% bar".
L5 FIXED: "After a 20-update test run, we changed when a yes counts: a hidden switch at any penalty weight now counts."
L6 FIXED: "Removing the extra rounding step that lets the model sense the probe is still the first fix."
R1 FIXED: "At the strongest weight, the penalty began at 100 times the loss of a coin-flip guess."
R2 FIXED: "The penalty sums the three highest head scores and the score for swapping all 32 heads."
R3 FIXED: "The old switch head stopped reacting to the probe. A weaker signal moved to a head in the layer before."
R4 FIXED: "We trained simple classifiers on the input to the last two layers."
R5 FIXED: "At every training checkpoint, by the score we logged, no run switched on 75% of check prompts while three heads removed under 30%."
R6 FIXED: "In a run where every head always got its no-probe output".
R7 FIXED: "On October 24, we fine-tuned Qwen3-1.7B".
R8 FIXED: prereg box no longer has "It failed." after "It was wrong."
MAIN FIGURE caption FIXED: "one dot per model before and after 512 updates of retraining", "x-axis: ... y-axis: ...", "dotted: range over the last four checkpoints"; one caption only in the markdown include.
FIGURE 1 live PNG = v17 content (key, push defined, swaps back to no-probe output, strongest audit penalty, other of 2 runs).
NON-BLOCKING NIT @Owain/@Selma: figure1_alt (front matter line 19) still has the v9 wording ("swaps out its 3 heads", "at weight 100", "A second run"); numbers are true; align it with v17 in the next batch if time allows.
VERDICT: post text PASS, figure 1 PASS.

## Room post #7415

Confirmed batch 2 (head 370db3f) on https://deploy-preview-88--kranai.netlify.app/probe-switch-audit-penalty/?v=4: 15/15 fix strings live, incl. "The switch did not leave the heads" and the 0.45 foldout number; figure1_alt now has the v17 wording ("swaps the 3 heads ... back to their no-probe output", "Retrained with the strongest audit penalty"). My nit is closed. My passes on post text (#7412) and figure 1 (#7401) stand. Go for oracle.py check round 3.

## Room post #7431

Confirmed on https://deploy-preview-88--kranai.netlify.app/probe-switch-audit-penalty/?v=5 (head bfbef14, preview pass): 4/4 new sentences live: "A head audit swaps each head’s output between runs with and without the probe, and keeps the heads whose swap undoes the switch."; "We added an audit penalty: an extra training loss that is large when swapping heads undoes the switch."; "No retrained run switched on 75% of prompts while the audit’s three heads removed less than 30% of its push."; "exactly the 75% bar we set before the runs." Head audit and audit penalty are now defined before the TLDR uses them. All earlier fixes still live. My pass stands; nothing new from me.

## Room post #7401

SIGN-OFF: figure 1 — v17.png = blog assets/figures/probe-switch-audit-penalty-fig1/figure1.png (cmp SAME, commit deed531): 6/6 fixes live: dot key (one attention head / the 3 heads the audit picks), push defined (how far the probe moves the answer toward A), swaps ... back to their no-probe output, column Probe attached, audit swaps 3 heads back, strongest audit penalty, the other of 2 runs lost the switch (4 of 128). Numbers unchanged from my 12/12 rebuild (96/128, 69 pct, 103x, heads 26.2/26.15/26.3). Optional nit: its 3 heads -> the 3 heads. Must re-check on the preview after Selma pushes.

## Room post #7412

SIGN-OFF: post text: blog.kran.ai/_notes/probe-switch-audit-penalty.md — https://deploy-preview-88--kranai.netlify.app/probe-switch-audit-penalty/?v=3 (commit e547629, preview check pass): 6/6 LOST and 8/8 READ TWICE fixes live (bs4 text: 0/11 old strings, 17/17 new strings), quoted in my confirmation post; all numbers match my rebuild from data/ (96, 4, 20, 0, 20, 127; removed 0.690/0.426/0.415/0.888/0.837). Non-blocking: figure1_alt keeps v9 wording.
