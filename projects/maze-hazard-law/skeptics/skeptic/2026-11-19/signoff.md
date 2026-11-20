role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — defense-1119-maze-hazard-law (continuation of defense-1118-maze-hazard-law)

## Answer to the question
UNCONFIRMED (Not settled). PASS above means: I agree with the post's answer "Not settled" and with the numbers it
reports. It is not a Yes or a No to the question.

Why not settled: the frozen rule's status agents are gpt-oss-20b and Qwen3-8B. Both laptop chunks of the parent room
ran with prefix key-value reuse ON, and both failed the D5 prefix-reuse check on the laptop (Qwen3-8B 9/12 identical
greedy replies, gpt-oss-20b 5/12; bar 95%; `data/<agent>/prefix_check.json`). D5 (logged before any status-agent
run) orders a rerun with `MAZE_PREFIX_REUSE=0`. Those reruns are labq 56pkb4cysk (gpt-oss-20b) and fdce5qspgd
(Qwen3-8B), in Esben's laptop queue. Every run used below is a reuse-on pilot row that does not count.
My checker enforces D5 on its own: `vera_rebuild.py:234-238` counts records with `gen_settings.prefix_reuse` True
(Qwen3-8B 70, gpt-oss-20b 27: not status-eligible).

Had the reuse-on Qwen3-8B rows counted, the frozen rule would say REJECTED (FAILS on shape, floors met). The team did
not waive D5 after seeing that, and I would not sign a REJECTED that rests on it.

## What I rebuilt (own code from PREREG s.3-4; never read analysis.py; actions[] cross-checked with access logs)
`vera_rebuild.py` -> `qwen_reuseON.json`, `gptoss_reuseON_partial.json`, `nano_side.json`.
Agent log vs server log: 0 mismatches in pages and exit class (all three agents).

| row (pilot, not counted) | n entered | G / E | G3 / E3 | h_all | KM median | R | Pearson p (df) | Weibull k, p | frozen verdict |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B reuse-on | 38 (blocks 12/13/13) | 37/165 | 35/80 | 0.224 | 2 | 0.602 | 6.7e-9 (3) | 0.71, 1.3e-4 | FAILS (shape) |
| gpt-oss-20b reuse-on, 9 runs | 9 | 9/108 | 0/27 | 0.083 | 13 | undefined | 0.018 (2) | 3.30, 3.5e-4 | UNTESTED, too few give-ups |
| gpt-5-nano side row (API, never status) | 32 | 32/217 | 4/95 | 0.147 | 6 | 2.69 | 5.0e-5 (2) | 2.20, 1.8e-6 | FAILS (shape); floors not met |

Seed-level (t over 3 maze seed blocks), Qwen3-8B h_all 0.314 [-0.172, 0.801]; R_b 0.630 / 0.500 / 0.695 (3 of 3).
Every number equals `results/analysis.json` side rows to float precision (checked at b9b8b5741).
Post foldout numbers (`vera_postcheck.py`): Pearson statistic 41.0 df 3; run bootstrap h_all [0.106, 0.50], R
[0.50, 0.75]; Weibull k profile 95% (0.57, 0.85) Qwen, (1.87, 5.04) gpt-oss; Qwen dead end: 9/20 capped runs re-request
the one dead-end page 77-78 times, 0 parse failures.

## Robustness and null tests
- Null (`vera_null.py`, 1000 simulated data sets from a true geometric at h 0.224, same n and reach): the frozen rule
  says FAILS 4.8%; the observed Pearson and Weibull p-values were reached 0/1000 times. Luck does not make the
  Qwen3-8B shape failure.
- Leave-one-out: Qwen3-8B run 012 (copy loop, 78 pages at the cap) carries the Weibull k. Without it: Pearson p 0.19
  (passes), Weibull p 0.014 (fails), k 1.34. Without 011 and 012: k 3.3. So "not constant" is robust; the direction
  of k (falling or rising) is not. The robust number is page 1 vs page 2: 3/38 vs 28/35, Fisher p 1.5e-10.
- Forecast for the gpt-oss-20b rerun (`vera_forecast_gptoss.py`, Weibull fit on its 9 pilot runs, 60 runs drawn):
  P(at least one pilot give-up) 0.34-0.41, so P(TESTED) is about 0.4 even if the job runs; else "too few give-ups".
- Invented figures: 0 among entered Qwen3-8B maze runs (the two '$96' answers are runs 003 and 033 with 0 maze pages).

## Novelty
`novelty.md`: closest decoy-maze work is POTEMKIN (Zhan et al., arXiv 2604.18874): trap entry and budget waste, no
per-page leave rate, no shape, no pilot forecast. The question stays new.

## Compute
Compute gate PASS (#26278): 2 laptop jobs (16 GB / 2 h, 11 GB / 5.5 h), 0 heavy mini jobs.
