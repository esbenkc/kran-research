# DEVIATIONS — probe-switch-heads

Changes after the PREREG freeze (fdb1b7c, amendment d369957). Each entry: what, why, and what had been seen.
Times are git commit times (CEST).

## D1 — L seeds train for 2048 updates, not 1024 (Bruno; Felix #5734; set from the seed-0 DEV curve only)

PREREG 'Models' allowed the seed-0 pilot to set the step count once, before seeds 1-2. The 256-step pilot
(`logs/j2_pilot.log`) stayed at chance: loss 0.693, dev pair both-right 0.016, mean shift ~0 nats. Seed 0 was then
run for 2048 updates with the dev curve every 128 (`logs/j2_s0_2048.log`, `chain_j2s0long.sh`). Dev pair both-right
(64 dev prompts, cached tail): 0.03 / 0.02 / 0.11 / 0.06 / 0.00 / 0.66 / 0.14 / 0.22 at steps 128-1024, then
0.86 / 0.86 / 0.95 / 0.95 / 0.98 / 1.00 / 1.00 / 1.00 at steps 1152-2048 (mean shift 39 nats at 2048). The curve is
still unstable at 1024 and flat at 1.00 from 1792, so seeds 1-2 and the never-attached null train for 2048 updates
(everything else unchanged). The seed-0 status model is `lora_none_s0_2048` (same recipe, 2048 updates).
Seen when this was set: the seed-0 dev curve above and the seed-0 cached-tail test row printed at the end of the same
run (pair both-right 1.00). No whole-graph result, no head result and no seed-1/2 number existed.
