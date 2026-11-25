# DEVIATIONS — probe-switch-head-budget

Changes after the prereg push (fdc17f064, GitHub push 2026-11-25T01:14:52Z). Each line has its time (PST) and room
post. None of these touch the FROZEN RULE block (status rule, thresholds, primary metric).

- D1 (2026-11-24 17:18 PST, before any data; Felix #30289, Tove #30290, Kasper #30287 wall-time risk). Compute and
  order: PREREG planned one laptop-queue job. It is split into two jobs, submitted together (J2 queues behind J1).
  J1 = everything the status needs: k_att = 6 seeds 0-2 (+ spare k6_s3 only if a seed fades), L0 s0-s1, the three
  warm adapters (W), and all their audits. J2 = the law points: k_att = 3 seeds 0-2, then k_att = 1 seeds 0-2, and
  their audits. Cut rule, fixed now: if J2 is not done at 21:00 PST (Tove #30290; Felix #30289 had 21:15, the earlier
  time is used), J2 is cancelled; the post reports k_att = 3 and 1 as "not run in time", and the prereg outcome is
  "unsettled" for the law clause (the frozen outcome rule already maps missing law arms to "unsettled"). The status
  is unaffected. If J1 is not done at 21:30 PST: PENDING LAPTOP quiet close (LAPTOP QUEUE rule 5). No updates, no
  monitor steps and no prompts are trimmed to fit the time (that would be a rule change).
- D2 (2026-11-24 17:24 PST, before any data; Bruno #30293). The frozen Training text says both "8 prompt pairs per
  update (micro-batch 4 pairs)" and "AO^T ... on the update's training batch". With micro-batches of 4 the penalty
  would be the mean of two 4-pair ratios, not one ratio over the update's batch. Resolved toward the penalty
  definition: one micro-batch of 8 pairs per update (--micro 8), so AO^T is exactly the ratio of batch means over the
  update's 8 pairs. CE, lr, updates, data order and the monitor are unchanged. This is a training implementation
  detail, not the status rule, a threshold or the primary metric. Also confirmed in code before data (Bruno #30293):
  greedy tie-break by lower head index (stable sort), spare *_s3 warm-starts from switch_s0, AO0 = [r]_+ / max(D,
  1e-4) as one ratio of means on the 64-prompt monitor subset with the update-1 pick, tie flags at relative 1e-5.
- D3 (2026-11-24 17:27 PST, before any data; Felix #30295, Tove #30298). The law-point job of D1 is split again:
  J2 = k_att = 3 seeds 0-2 with their audits (each seed trained and audited before the next starts); J3 = k_att = 1
  seeds 0-2 with their audits, then the parent's saved k_att = 3 adapters re-audited (side row, last), submitted when
  J1 is done (room cap: 2 open jobs). The same 21:00 PST cancel rule of D1 applies to J2 and to J3; a cancelled
  job's missing arms are reported "not run in time" and the law clause is "unsettled". J1 is unchanged.
- D4 (2026-11-24 18:28 PST, before any k_att = 3 or 1 data and before any k_att = 6 cell landed; Tove #30346, Kasper
  #30347, Felix #30348). The laptop worker claimed J1 only at 18:25 PST (est. 2.3 h), so the J2 of D3 (STOP_AT 20:55
  PST) would start after its stop time and train nothing. J2 (labq yv8h2ropte) is cancelled and replaced by ONE job:
  k_att = 3 seeds 0-2, then k_att = 1 seeds 0-2, then the parent's saved k_att = 3 adapters re-audited (side row
  last), no STOP_AT, queued behind J1. This replaces J2/J3 of D3. The 21:00 PST cut of D1/D3 only guarded tonight's
  publish; it is dropped. If that job is still open at the wrapup, the room closes PENDING LAPTOP (LAPTOP QUEUE rule
  5) and the continuation publishes with all arms. Decided before any k_att = 6 result exists, so the choice between
  publishing tonight and waiting for the law points does not depend on the result (Kasper #30347). Status rule
  unchanged (k_att = 6 only); J1 unchanged.
- D5 (2026-11-24 20:05 PST; Bruno #30375, Felix #30376). J1 (labq 99ig3z70yx) was killed at 19:59 PST by the labq
  disk guard (laptop free disk 4.6 GB < 5 GB, exit -15) during k6_s2 training. Finished and pushed before the kill
  (4d8ba34cf): k6_s0 (live) and k6_s1 (faded: S(0) = 0.000; its first re-pick also ties at greedy steps 5-6). k6_s2's
  partial log is kept as a receipt only; the frozen status uses the update-512 adapter, so k6_s2 counts as UNFINISHED
  and is rerun from update 0 with the same seed and warm start. The rest of J1 (k6_s2, the spare k6_s3 triggered by
  the k6_s1 fade, L0 s0-s1, W s0-s2, their audits; run_k.sh skips finished cells) is resubmitted as its own job
  FIRST, then the law-point job of D4 (cancelled and resubmitted behind it, same cmd). data/ckpt is no longer
  git-ignored, so a killed job keeps its 128-update checkpoints. Gate reading for k6_s1 (Iris #30377, from the frozen
  text): LIVE is the only spare trigger, so the fade triggers the spare; the first-re-pick tie is a second, reported
  reason the run does not count. The pre-stamp tie calibration (PARENT AO adapters, 0/12 ties) did not cover the warm
  adapters; warm s1 ties at steps 5-6. Status rule unchanged.
