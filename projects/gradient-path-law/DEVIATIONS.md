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

## D3 — no prediction is called "the law" (2026-11-06 17:21 PST, Felix; Tove #15760, Kasper #15752, Vera #15756; before any T number)

Rules unchanged; P1-P3 values unchanged. PREREG "Frozen numbers" says the post names P1 "the law". Vera's
rebuild (#15756) shows the fed switch moves -0.64..+0.64 logits under equal-size random constant prefix offsets
(4 directions) vs 3.1-4.3 for the served cue, i.e. up to about 0.2 of the switch. P1 (0.036-0.053 in carry units)
sits inside that random-nudge band, so a T carry near P1 cannot confirm P1 over "any offset of that size". The
post names no single law. It reports SILENT vs REACHES (the status), says which of P2 (0.43) and P3 (-0.22) the
measured carry rules out, and, if T is SILENT, compares the carry with the random-nudge band from the read-only
random-prefix cell ("the overlap prediction", not "the law").

## D4 — frozen T collapsed; Tc carries the status (2026-11-06 17:46 PST, Felix #15771; Tove #15773; before any Tc, Tb or Ta number)

What happened (Bruno #15767, data/status/T_s0.json, T_s1.json, data/runs/T_s*/curve.jsonl): both frozen T seeds
collapsed to never-cheat (served and trainer cheat ~1e-12 at 5,000; T_s0 dead from 2,750, T_s1 from 2,250). By the
frozen rule this is "no takeoff" = UNCONFIRMED, and that reading is reported first in the results.
Likely cause [INFERENCE, Bruno; Iris #15768]: arm T is the only arm whose update input differs from the sampler's
input on EVERY row (training-graph vs served graph, 87% of block-26 bits, including positions 1-7), and this GRPO
has no importance correction, so the update runs off-policy and away to 'solve'. 2605.14220's abstract names
token-level numerical disagreements as a cause of training collapse.

PEEK (Tove condition 1): before choosing Tc we saw T_s1 at step 2,000 (64 dev prompts): trainer logit gap 3.30,
served logit gap 0.275, ratio 0.083, before its collapse. Reading-only; never status.

New cells (declared here, before any of their numbers):
- Tc s0, s1 (STATUS, Tove #15773): the sampler AND the update both use the training-graph numbers (14-27 record,
  none / leaky); the served graph appears only at the read. This is the fed arm with d_T in place of d_S: an
  on-policy pipeline whose model ships on a different serving stack (Iris #15772: a bitwise-consistent sampler +
  trainer, served elsewhere). Same takeoff rule (trainer gap >= 15 points at the end), same carry rule and 0.15
  bar, same oversell guard, same F gate, budget 5,000 updates (r 1.39).
- Tb s0, s1 (reading only): T plus truncated importance weights (pi_update / pi_sampler per sample, clipped at 2;
  verl TIS, PR #2953). Anti-cherry-pick (Tove condition 3): if any Tb seed takes off and lands on the other side of
  the bar from Tc (REACHES vs SILENT), the status is UNCONFIRMED.
- Ta s0 (diagnostic): the update reads the training-graph NO-probe input on both flags (no probe cue). Does the
  collapse come from the graph mismatch alone? (Iris's guess: collapses, p ~0.7.)
No third status cell (Tove condition 4): if Tc has no takeoff, the night is UNCONFIRMED and the post says why.
Results open with the frozen T collapse; the TLDR names the deviation in one clause (condition 5).

Order: Tc s0,s1 (after fed s2,s3, ~18:11) -> R s2,s3 -> Tb s0,s1 -> Ta s0 + S s0. Cut: D (decoy; pre-scored by
the pilot) and stackfed. No new training after 20:30 PST.
