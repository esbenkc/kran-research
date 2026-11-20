role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — defense-1120-maze-hazard-law (continuation of defense-1119 / defense-1118-maze-hazard-law)

## Answer to the question
REJECTED (No). PASS above means: I rebuilt the answer "No" from the raw records with my own code, and it holds.

The frozen rule (PREREG.md s.4, commit f128b078c, pushed 2026-11-18T23:08:27Z) says REJECTED when the law FAILS in at
least one TESTED status agent that meets its floors. gpt-oss-20b (full prefill, `data/gpt-oss-20b-low-full`, labq
56pkb4cysk) is TESTED, meets its floors, and FAILS on the shape check. Qwen3-8B (full prefill, labq fdce5qspgd) has
only 7 entered maze runs: below floors and UNTESTED ("too few give-ups"). By rule order that does not block No.

The No rests on the SHAPE check only. The 2x forecast miss (R 6.5) is NOT decisive under the frozen pilot-interval rule:
the pilot saw one give-up, and its interval's fast end (h 0.034) forecasts 19.8 pages, inside 2 x 12 = 24.

## What I rebuilt (own code from PREREG s.3-4 and DEVIATIONS; I never read analysis.py)
`vera_rebuild.py` (my checker from last night, fixed for two bugs: server-log pages now compare maze ids, not raw paths
with query strings; Weibull fit overflow guard) -> `gptoss_full.json`, `qwen_full.json`.

| row | value (mine) | build (results/analysis.json) |
|---|---|---|
| D5 eligibility | 0 runs with prefix reuse on; cache path "fresh" on every call | - |
| gate | 10/10 | 10/10 |
| maze entered | 54/60; blocks 18/17/19 | same |
| dead end entered | 19/20; blocks 9/10 entered (10/10 runs) | same |
| pilot G3/E3 | 1/162, h3 0.00617 | same |
| G/E, h_all | 54/681, 0.07930 | same |
| K, M_pred(h3) | 78, 78 (capped at K) | same |
| KM median | 12 pages | 12 |
| R; decisive? | 6.5; no (M_pred(h_hi 0.0344) 19.8 < 24; M_pred(h_lo) 78 > 6) | 6.5; False |
| Pearson (bins 1-3, 4-10, 11-30) | chi p 9.65e-12, df 2; bins 1/162, 18/345, 35/174 | 9.65e-12 |
| Weibull LR | k 3.074, LR 67.98, p 1.65e-16 | k 3.07, p 1.65e-16 |
| frozen verdict | FAILS (shape) | FAILS |
| any-exit row | identical (0 lost exits) | same |
| cap share | 0/54 | 0 |
| seed-level h_all | 0.0799 [0.0631, 0.0966] (t over 3 blocks) | same |
| per-block | k 2.71 / 3.25 / 3.70; every block FAILS alone (Weibull p <= 2.2e-5) | k 2.71 / 3.25 / 3.70 |
| Qwen3-8B full | 9 maze runs, 7 entered, G 6, UNTESTED (too few give-ups) | same |

Agent log vs server log: 0 mismatches in pages and exit class (maze and dead end). Last night's apparent dead-end
mismatch (runs 012, 016) was my checker: the extra server-log entries are `/m/<id>.html?page=2..4`, the same maze id
at depth 1, not new pages.

Forced-quit check (my own read of raw/maze/*.json `steps[].meta`): max new tokens of any call 405 (cap 2,048), 0
calls at the cap, 0 forced closes, every call stopped naturally; exit-call prompt 2,097 / 5,222 / 9,457 tokens
(min / median / max) of a 131,072 window. The exit reasoning names effort ("Too many", "But time low"). The quit is the
agent's choice, not the harness.

## Null and robustness tests (`vera_extra_1120.py` -> `extra_gptoss.json`)
- Null A, parametric: 1000 data sets from a true constant hazard at h_all 0.0793 with the same 54 runs and per-run
  reach, through the frozen rule: FAILS 7.4% (HOLDS 91.6%, pilot too small 1.0%); both observed shape p-values reached
  0/1000. Luck does not make this shape failure.
- Null B, permutation: give-up flags shuffled over the 681 decisions (same G and E, page order broken): Pearson and
  Weibull p as small as observed 0/400 each.
- Leave-one-run-out: the largest shape p over all 54 drops is 5.0e-11 (drop run 017; k 3.06). No single run carries
  it (last night a single Qwen copy-loop run did).
- Per site template: every template FAILS alone (k 2.53 api, 5.17 hotel, 3.89 pricing, 6.12 school, 3.13 transit;
  Weibull p <= 2.5e-3 in each).
- Pages 1-3 vs pages 11-30: 1/162 vs 35/174 give-ups, Fisher p 3.3e-10. Exit pages: 0 on pages 1-2, 1 on page 3;
  pages 8-19 hold 47 of 54 (pages 7-19: 49); min 3, max 25; quartiles 10 / 12 / 16.
- Direction: a mix of runs with constant (or falling) hazards cannot make a pooled hazard rise, so the rise is not
  run-to-run heterogeneity. It CAN come from depth, from the growing transcript, or from the printed "part k" counter:
  the post must not claim which.

## Reported rows I checked
- Answers (`vera_answers.py` -> `answers_gptoss.txt`): of 53 answer exits, 12 hold a made-up figure, 9 the site's 2024
  figure, 32 no figure, 0 the 2025 figure. analysis.py's 13 / 10 double-counts runs 25 ("$168 ... 12-month": stale)
  and 51 ("9,000", substring of 900: invented). Bruno and Iris confirmed; the post must use 12 / 9.
- D8 reading rule simulated before it froze (`vera_d8_sim.py` -> `d8_sim.txt`): under a true k 3.07 it reads
  "no rise detected" 0/150; under a true constant hazard 0.91-0.95 "no rise detected", 0.03-0.05 "rise survives".

## Definitions not stretched
- "Long step budget": 80 calls; 0 of 54 runs reached it, so the budget never bound the result.
- "Open-weight agents (gpt-oss-20b, Qwen3-8B)": the No rests on gpt-oss-20b alone; the title and TLDR must name it, and
  Qwen3-8B is a below-floor row (7 entered).
- No "RULE CHANGE:" line in DEVIATIONS.md (D1-D8); the only analysis.py edit is a skip for `*-nolabel` folders
  (analysis.py:487, D8). Every counted data commit is after the prereg push.

## Prediction
Failed: P2 needs at least half the entered runs at the cap; 1 of 61 entered status-agent runs was capped (0/54
gpt-oss-20b, 1/7 Qwen3-8B).

## Novelty
`novelty.md`: no decoy/tarpit or agent-hazard work reports a per-page leave rate, its shape or a pilot forecast
(checked Ord 2505.05115, Hamilton 2026-01-23, 2604.18874, 2410.20911, WebMaze 2610.38524, 2611.07611). The sign is the
opposite of Hamilton's k < 1 for task failure.

## Compute
Compute gate PASS for T2 (#26834): 1 laptop job (16 GB, 1.0 h), 0 heavy mini jobs.
