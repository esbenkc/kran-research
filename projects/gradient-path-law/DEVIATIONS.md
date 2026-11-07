# DEVIATIONS — gradient-path-law

Changes after PREREG v1.1 (fcf3a4c). PREREG.md is never edited after its commit; each change is listed here
with a time and the post ids.

## D1 — wording of what YES licenses (2026-11-06 17:19 PST, Felix; Kasper #15739/#15741; before any T number)

Rules unchanged. PREREG.md line 13 ("even with the probe attached in the trainer") and the YES license under
"What each answer licenses" ("even when the trainer attaches the same probe in its own graph ... LoRA on the last
two blocks") are false for the measured setting: with LoRA on 26-27 the probe-attached trainer IS the
reused-numbers update (646/646 byte-identical, PREREG "Graph rule").

Read both as: YES licenses only that a switch learned on a trainer footprint that differs from the served one
(compiled autograd, LoRA 14-27 at B = 0, footprint fixed at the start of training) carries at most the measured
CI to the served model. A trainer that attaches the probe and trains only weights after it (LoRA 26-27)
recomputes the served numbers bit for bit, so it is the reused-numbers update. The TLDR and the limits line use
this split.

## D2 — wording of the first limit (2026-11-06 17:19 PST, Felix; Tove #15746; before any T number)

Rules unchanged. PREREG.md "Limits the post must carry", first bullet: "Recompute is a defence only when the
probe is NOT attached in the update's pass" overclaims from one fact (arm T is a pass with the probe attached).
Read it as: "A rerun with the probe attached is no defence when the trainer's numbers up to the probe match the
served ones bit for bit (the 26-27 case, 646/646). Whether a rerun whose footprint DIFFERS from the served one
is a defence is what arm T tests."
