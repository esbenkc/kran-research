# DEVIATIONS — footprint-overlap

Changes to PREREG.md after v1.3 (09467ad). Room clock = PDT.

## D1 (18:07 PDT, before any overlap commit and before any behaviour row) — reported row: status R^2 without the rounding setting

Kasper #8765, Tove #8766. The rounding setting's (`emu`) behaviour was public before PREREG: the parent post
(numeric-chameleon-rl.md:144) reported flattery 0.181 / 0.080 under it for the two status policies, against
0.717 / 0.590 with no monitor. L7/L21 q_norm were left out of the status set for the same reason. `emu` stays in the
status set (the classifier is fit on A and none only, so emu's known behaviour could not tune the x-axis), and the
frozen status rule does not change. Added, reported per headline policy: the status R^2 with `emu` removed. The post
says that emu's behaviour was known in advance. If the status (CONFIRMED / REJECTED) differs without emu, the TLDR
says so in its first two sentences.
