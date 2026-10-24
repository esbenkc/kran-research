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

## D2 — status specificity control draws from the 29 heads outside the dev-picked top-3 (Felix #5741, Tove #5740)

PREREG 'Head test' drew the 50 random 3-head sets from all 32 heads (`heads.py:211-213`, fixed seed). A given head
then sits in a draw with P = 3/32 = 9.4%, about 4.7 of 50 draws. If the switch runs through one head, the draws that
contain it reach the top of the list, and the 95th percentile (the 3rd-4th highest of 50) lands near that head's own
R. The frozen rule would then return UNCONFIRMED because the switch is local: the control would test "is the key head
in 10% of the draws", not "do three other heads do the same".
Change: for the status rule, the specificity control is 50 seeded draws of 3 distinct heads from the 29 heads NOT in
that model's dev-picked top-3 (the pool is fixed from dev, per model); CONFIRMED still needs its 95th percentile of R
< 0.30. The frozen all-32 draws stay a reported row next to it, in the data and in the post. The goal's own clause
(block 27, median of draws) is unchanged.
Seen when this was set: Iris's cut patch on 64 TRAIN prompts of L seeds 0-1 (27.7 alone R 0.735 / 0.917, #5739) and
the cached-tail test pair rates of s0-s1 (128/128). No dev-picked head set and no test-set head score existed.

## D3 — which random draws count for the D2 control (Felix #5761, Kasper #5764, Tove FINAL #5765)

Vera's own-code rebuild (#5759) showed the D2 95th percentile depends on which 50 draws are taken (s1: build 0.171,
Vera's draws 0.310). Ruled before any larger-sample number existed: the status count uses the frozen D2 draw set
(`heads.py`, 50 draws, generator seed 20261025, in commit b404efd before the first test head score). The post reports,
for every seed, the top-3 R, the frozen-50 restricted p95 and Vera's 400-draw restricted p95 with its CI. If a status
seed's 400-draw CI lies wholly above 0.30, the post says so and a skeptic may reject on that ground; it is never
averaged away across seeds.
Seen when this was set: the frozen-50 results of s0 and s1 (#5753, #5758) and Vera's own 50-draw p95 for s0-s2
(#5759). No 400-draw number existed.

## D4 — reported rows run with 20 random draws; block-26 answer-position-only rows not run (Bruno, 04:00 CEST)

Time and memory only; no status input changes. heads.py took ~13 min per model, so the S rows (saved 1022
detectors) ran with 20 random draws per pool instead of 50 (`chain_tail.sh`). The block-26 answer-position-only
patch (a reported row in PREREG) was stopped at the start of its first job to give Vera a slot for her whole-graph
rebuild (#5842), and was not run. KV-group patching (optional in PREREG) was not run. The three L status models
used the frozen 50 draws per pool. Seen when this was decided: all L status numbers and the S rows.
