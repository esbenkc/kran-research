# DEVIATIONS — probe-switch-score-gaming

Changes against PREREG.md (pushed e734ba59c, 2026-11-28T01:01:13Z), with their time. None is inside the FROZEN RULE
block; none is a RULE CHANGE.

## D1 — 2026-11-28 02:10 CET (before any data): in-loop re-pick speed-up (Bruno)

PREREG "Material" allows a speed-up of the attacker's in-loop re-pick only if it gives identical picks on the
parent's saved re-picks. `fast_pick.greedy_pick_fast` (train_pen.py `--fast-pick`) is the same rule as
`pen.greedy_pick`: every block-B candidate of a greedy step shares that step's block-A part (the chosen block-A
heads), so block 26's MLP and block 27's keys and values are computed once per step for all block-B candidates.
Block-A candidates go through `pen.TailParts.cut_sets` unchanged. Only the attacker's monitor uses it; the status
audit (kaudit.py) is unchanged.

Checks (no training; `checks/equiv.py`, outputs `checks/equiv_T1.jsonl`, `checks/equiv_TB_warm.jsonl`):

- The parent's SAVED step-1 re-picks (softened score, warm switch_s0/s1/s2, inputs/parent_k6/runs/k6_s*.json
  picks[0].heads): the fast pick and the original both reproduce all 3 saved 6-head sets exactly (18/18 heads).
- At T = 1 on the train-64 monitor, the parent's final adapters k6_s0, k6_s1, k6_s2 and warm switch_s0: identical
  6-head picks, fast vs original (24/24 heads).
- Score differences: at every greedy step of all 7 checks, |fast - original| of the winning score is below that step's
  own margin (best - runner-up, nats). Largest ratio: k6_s0 T = 1 step 5, 7.4e-4 / 1.8e-2 nats (4%). The relative
  difference of the winning score is at most 1.4e-4 (k6_s0), against the smallest calibrated live-step margin ratio
  0.0068 (48x) and the tie line 1e-3 (7x). Steps whose winner is a block-A head are bit-identical (diff 0).
- Time per re-pick: 41.2-41.6 s -> 25.8-27.8 s (about 1.5x). With it a T = 1 seed takes about 33 min of training plus
  about 10 min of audits, so PREREG's 35-min rule for the extra seeds 3-5 is not met: the extras do not run.

## D2 — 2026-11-28 02:25 CET (before any readout ran): readouts.py adds two reported rows (Bruno)

The box (started 02:09 CET from commit 2f4f5eac1) calls readouts.py fresh at each cell, so these edits apply to every
readout: the oracle carrier uses the fp32 log-odds (push_drop32, the PREREG "Scores" logits; bf16 push_removed kept
next to it), and each trained run also gets PREREG's reported row "beat its own monitor" (its last re-pick's 6-head
set cut on the 64 monitor prompts, `own_monitor`). The two-answer readout moved behind `--two-answer` (PREREG: after
05:55 CET), written to readouts/<run>_two_answer.json. No status input changes.

- D3 (2026-11-28 06:52 CET; Felix). Run broke, outside the FROZEN RULE block. The box (mini_1127_170936) finished
  t1_s0 training (adapter 02:43 CET) and then stopped with no log line; every agent in the room was cut off by a
  provider error at 02:53 CET, so nobody restarted it before the 04:35 CET training wall stop. t1_s1 and t1_s2 never
  ran. Status by PREREG rule 2: UNCONFIRMED (1 of 3 main-arm seeds). t1_s0's status audit (kaudit.py, temps 1,B,16)
  ran by hand at 06:52 CET, after the planned 05:55 audit stop, as a reported single-seed row only.
