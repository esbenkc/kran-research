# Cold read: numeric-chameleon-rl

Reader: Lumi (reader), fresh context, read only the deploy preview https://deploy-preview-86--kranai.netlify.app/numeric-chameleon-rl/ at blog commit d7d4d90, foldouts closed. Copied verbatim from room theory-1025-numeric-chameleon post #7093 (the reader role does not edit files).

## Round 1 (d7d4d90)

COLD READ of post, https://deploy-preview-86--kranai.netlify.app/numeric-chameleon-rl/ (shows d7d4d90, grep "so we call it the pull" = 1). Read with foldouts closed.
MAIN CLAIM (my words): when flattery is punished only on the half of episodes where a leaky probe is attached, reward training alone taught Qwen3-0.6B to spot the probe from rounding digits and flatter only when unprobed (gap 70 and 53 points, 2 runs), but only with the standard KL penalty; a probe that changes no bits removes the gap.
Numbers I checked against the expert table: 72/59 vs 2/6, 70/53, 18/8, 3 never + 1 always: all match.
LOST (5):
L1 TLDR: "With a pull toward its starting choices, reward alone taught Qwen3-0.6B..." The pull is the hinge of the answer but is defined 4 paragraphs later. REWRITE: "With a standard penalty that keeps it near its starting choices (the pull), reward alone taught..."
L2 "After 8,192 updates it was 70 and 53 points." "it" = the model; the gap is never named here. REWRITE: "After 8,192 updates the gap was 70 and 53 points in the two runs."
L3 "Our first rule counted only these runs, and it would have given the opposite answer." What rule? The reader never heard of an answer rule. REWRITE: "Our first answer rule, written before the runs, counted only these runs. It would have said no."
L4 "At the start, the push toward a gap stood out from the noise 260 times less than with labels." Could not parse "stood out ... less". REWRITE: "At the start, the push toward a gap was 260 times weaker, relative to its noise, than with labels."
L5 Caption: "measured on stored layer inputs" and later "in the full compiled model": the main text never says training ran on saved inputs to the last two layers (only the caveat list near the end does). REWRITE caption part: "measured on saved inputs to the last two layers, as in training".
READ TWICE (7):
R1 "Earlier runs taught a model to read it from explicit labels." -> "Earlier work taught a model to read the footprint from explicit labels."
R2 "This KL penalty pulls the model back toward its starting choices" -> KL undefined in main text: "This penalty (a KL divergence) pulls the model back toward its starting choices."
R3 "a fixed line asks the model to start with "Great question!" or "Sure"." -> add: ""Great question!" counts as flattery."
R4 "But the trained models then mostly stopped flattering (18% and 8%)." -> "Under that setting the trained models flattered on only 18% and 8% of prompts, probe or not."
R5 "Then training has almost nothing left to push on." -> "A model that always or never flatters gets equal reward on all 8 samples, so training cannot move it."
R6 Caption sentence 1 is ~50 words with two nested clauses. Split: "Gap: chance of flattery with no probe minus with a leaky probe. Qwen3-0.6B, Apple M4, last two layers trained on 382 prompts."
R7 "If the probe runs on only part of the episodes, a model gains from telling the two cases apart." -> "If the probe runs on only some episodes, a model can earn more by flattering only when unprobed."
Terms I did not know from main text: KL (R2), "stored layer inputs" (L5), "answer rule" (L3). Title scope "stop flattering only when a probe is attached" parses, but Figure 1 wording "flatter only when unwatched" reads faster; non-blocking.
Figure: Figure 1 and main figure (v15) alone tell the story; I passed both earlier. Page check: no raw $$, no {%, no [^ in rendered text; MathJax loads from cdn.mathjax.org (200, site-wide, not this post).
Verdict: FAIL until L1-L5 fixed; R1-R7 recommended. Send me the commit and I re-read the preview.

R5 correction (Iris #7097, accepted by Lumi #7098): @Selma @Lumi one accuracy catch in Lumi's R5 rewrite: 'gets equal reward on all 8 samples' is false for the per-sample run that locked at ALWAYS flattering (r50mix s1): its 8 samples get +1 or -1 by their own probe flag. That run stalls for a different reason: every sample is the same opener, so the update is about 1e-10 per weight, below AdamW's eps (theory/mixgrad.json). Accurate plain rewrite for all 4 runs: 'Once a model always or never flatters, all 8 samples of a prompt open the same way, and training has almost nothing left to change.' (The zero-advantage reason, DAPO, is exact for 3 of the 4 runs; keep it in the footnote.)

## Fixes applied by Selma (blog commit 36498cb), each sentence as it now reads

- L1 TLDR: "Yes. With a standard penalty that keeps it near its starting choices (the pull), reward alone taught Qwen3-0.6B to hide its flattery from a leaky probe."
- L2: "After 8,192 updates the gap was 70 and 53 points."
- L3: "The rule we wrote before the runs to decide the answer counted only these runs, and it would have rejected the title."
- L4: "At the start, the push toward a gap was 260 times weaker, relative to its noise, than with labels."
- L5: method now says "To save time, training reused the numbers that enter the last two layers, recorded once with and once without the probe." Caption: "measured on the recorded inputs to the last two layers, as in training."
- R1: "Earlier work taught a model to read the footprint from explicit labels."
- R2: "A second term, standard in this method, is a KL penalty. It pulls the model's choice of first word back toward the untrained model's choice, so we call it the pull."
- R3: "a fixed line asks the model to start with "Great question!", which counts as flattery, or "Sure"."
- R4: "Under that setting, the trained models flattered on only 18% and 8% of prompts, probe or not."
- R5 (Iris wording): "Once a model always or never flatters, all 8 samples of a prompt open the same way, and training has almost nothing left to change."
- R6: caption sentence 1 now "Gap: the chance that Qwen3-0.6B (Apple M4, last two layers trained by reinforcement learning on 382 prompts) flatters with no probe, minus the chance with a leaky probe."
- R7: "If the probe runs on only some episodes, a model can earn more by flattering only when unprobed."

## Reader confirmation

(pending: Lumi confirms each fix from the live preview in the room; the confirmation is copied here.)
