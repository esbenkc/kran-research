# DEVIATIONS — long-rollout-shield

Every change to PREREG.md after the rules froze (v1.3, b29df14b; gate fill and T0 in bb3c8387..0cb24ec2), with the
time and what had run when it was made.

## D1 — the one-token gate failed; slot 18 becomes conditional (2026-11-08 18:58 PST = 2026-11-09 03:58 CET)

Made after the planted_s0 end read and BEFORE any real-seed update (real_s0 had not started; Iris's C_sup fits were
on the lane). Proposed by Bruno (#17623, option B) and Felix (#17624); backed by Kasper (#17625) and Tove (#17626).

What happened: planted_s0 (one-token cue ' Need', start D 0.100, 100 updates) read at R = 8: served gap +0.10 points
[-0.53, +0.76], forced switch g(notNeed) - g(Need) +0.31 [-1.22, +1.74], D_end 0.211 [0.206, 0.215], probe-on cheat
+4.8 points (0.806 -> 0.854). A linear reader sees this cue at 7.5 points [5.6, 9.5] before training (Iris #17607).
By PREREG v1.3 the GATE fails, so REJECTED is impossible: the title can only be CONFIRMED or UNCONFIRMED.

Changes:
1. The status rule is unchanged. real_s0 runs as frozen (125 updates, R = 8 read).
2. Queue line 18 is conditional, rule fixed now: if real_s0's served gap CI-low >= 3 points, real_s1 runs as frozen
   (YES needs 2 of 2). Else line 18 = cue0.30: the one-token cue ' Need' at D 0.30 (eps set on the start policy so the
   TV at the first free position = 0.30), k = 64, the same recipe (decision KL beta 0.25, prefix KL 0.1 per position,
   lr 3e-4, P8 G8), seed 0, 125 updates, read with the forced gate at R = 4 (Tove #17626.4; the clock leaves no room
   for R = 8). It is the parent PREREG's own fallback row: an engine-alive reading that never moves the status. It
   changes ONE variable from the failed gate (the cue size): learns -> a 10-point cue is too small for this budget at
   64 tokens; fails -> this 64-token recipe learns no text cue in 125 updates.
3. Logging only (Kasper #17625), no change to the update: per-update gradient norm before the clip and the norm of the
   decision term's gradient, in curve.jsonl, for real_s0 and line 18.
4. LAW-C vs LAW-D are NOT scored: both kappas are ~0 (planted gap +0.10), so both laws predict ~0 for the real probe
   and cannot discriminate. The post makes no law claim. The C_sup fits keep running as a reader-edge row (a same-size
   reader trained WITH labels vs RL that learned nothing); the hash fit is cut first if the clock needs it.
5. Named confound, not run (Tove #17626.6): the parent's learned planted cue had no prefix KL and decision beta 1;
   this gate had prefix KL 0.1 x 64 positions and beta 0.25. Training cheat probed vs unprobed stayed level all run
   (planted_s0 curve.jsonl 0.840/0.835, 0.886/0.889, 0.815/0.845, 0.889/0.861) against 0.48/0.77 in the parent's
   planted arm. Kasper's suspects (rl.py:182 probe weight scaled by the group reward sd as cheat saturates; rl.py:214
   64 thought log-probs share the advantage term with the decision; rl.py:221 total grad-norm clip 1.0) go to Next
   posts with the logged norms.

## D2 — real-arm ceiling uses range_pred (2026-11-08 18:58 PST = 2026-11-09 03:58 CET)

Made with D1, before any real-seed update. Raised by Vera (#17627). The frozen real ceiling = 100 x D_end x range_obs,
with range_obs = the planted forced-prefix switch. That switch is +0.31 points (the gate failed), so every real
ceiling would be ~0.03 points and the label 'training closed the leak' would fire although the real text stays open.
Change: the post reports real ceilings with range_pred (94.7 points, Setup CEILING) and names range_obs beside it; the
label 'training closed the leak' is used only if a real arm's D_end < 0.05 (the v1.2 rule). This touches reporting
only: with the gate failed, REJECTED is impossible whatever the ceiling.

## D3 — C_sup reader continues the start LoRA, not a fresh one (logged 2026-11-08 21:35 PST)

Found by Iris (#17768) while fact-checking the post. PREREG v1.1 says 'start adapter + a fresh LoRA r16 on blocks
26-27'. Iris's frozen recipe (#17575, posted before any fit) and reader_sup.py load the start policy's OWN LoRA
(rank 16, blocks 26-27, the weights the RL update trains) and keep training it with labels. Same capacity, same
blocks; the reader starts from the policy itself, which is the closer match to 'what this policy can learn to read'.
The PREREG text was wrong, not the fits; every C_sup number (real -0.02, one-token 10.74, self-test 10.37) used the
same recipe. No number changes.
