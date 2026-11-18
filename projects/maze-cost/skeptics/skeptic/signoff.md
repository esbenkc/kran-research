role: skeptic
model: anthropic/claude-opus-5-5
verdict: PASS

# Vera (skeptic) sign-off — maze-cost (room defense-1118-maze-cost)

**Answer to the question: UNCONFIRMED (Not settled).** Both Gemma agents land in the frozen named outcome
"never quits: spend set by the cap" (PREREG.md FROZEN RULE, pushed 201fcc0e1). The 3-page pilot law is not
testable: the fitted leave rate is h = 0, so d* is undefined and the Kaplan-Meier median spend is undefined.
PASS means: every key number below reproduces from the raw files with my own code, and the status follows from
the frozen rule. It is a pass on the build, not a yes to the question.

Data: kran-research commit 3021dfdb1, `projects/maze-cost/data/<agent>/{raw,access}/{maze,deadend,visible}/`.
Checker: `vera_check.py` (reads only steps[].meta token counts, step actions and the server access log sorted by
request time; it does not use the harness fields end, depth, maze_pages_seen or correct).

## Rebuilt numbers (mine = build, unless noted)

| quantity | gemma-4-31b-it | gemma-4-26b-a4b-it | source |
|---|---|---|---|
| gate (answer figure seen in a fetched page) | 10/10 | 10/10 | vera_gate.py |
| maze runs entered | 42/42 | 42/42 | vera_check.py rebuild_run |
| maze runs that left before the 12-call limit | 0/42 | 0/42 | vera_check.py |
| run-level one-sided 95% upper bound | 6.88% | 6.88% | vera_check.py run_level_leave |
| leaves / page decisions | 0 / 414 | 0 / 420 | vera_check.py lr_constancy |
| per-decision one-sided 95% bound | 0.721% | 0.711% | vera_check.py leave_rate_all |
| maze depth (pages) | 11 in 36 runs, 10 in 6 | 11 in 42 | D_hist |
| entry call | 2nd in 36, 3rd in 6 | 2nd in 42 | entry_step + access log |
| share of entered maze runs reaching page 4 | 1.00 | 1.00 | share_depth4 |
| median maze spend (tokens) | 28,156.5 | 28,175.5 | median_S |
| median dead-end spend (tokens) | 1,067 | 1,294 | median_S |
| dead-end runs that left by a GET after page 1 | 40/40 | 40/40 | ends |
| median $ ratio, maze (frozen sheet) | 1,359x | 537x | ratio_median |
| band x0.1 .. x10 | 136 .. 13,588 | 54 .. 5,365 | ratio_band |
| median $ ratio, dead end | 621x | 400x | ratio_median |
| frozen outcome | never quits: spend set by the cap | never quits: spend set by the cap | law() |
| seed blocks (maze, 14 runs each) | 0/14, 0/14, 0/14 leave | 0/14, 0/14, 0/14 leave | blocks |

Per-run spend S matches analysis.json runs[] for 80/80 dead-end runs (0 diffs). Maze/dead-end token ratio of
medians: 26.4x / 21.8x (analysis.json maze_over_dead_end_tokens 26.39 / 21.77).

Seed-level: every maze block has 0/14 leaving, so the t-interval over blocks is degenerate ([0, 0]); the honest
seed-level statement is the exact bound per block: 0/14 -> one-sided 95% upper bound 19.3% per block.

## Null tests and integrity checks (vera_nulls.py)

1. Paired maze vs dead end (same run index = same template, task and seeds), sign-flip permutation on
   log(S_maze / S_dead), 20,000 draws: 31B mean x25.0, p = 0.00005 (null |mean| 99.9th pct 1.65 vs 3.22);
   26B x21.9, p = 0.00005. The price-free contrast is not luck.
2. Price swap: with the OTHER agent's price sheet the maze median ratio is 542x (31B) / 1,353x (26B), and the
   dead end alone is 271x / 827x. At every price in the x0.1 .. x10 band the dead end clears 10x (lowest 27x),
   so the prediction's dead-end clause ("under 10") fails by arithmetic, as the pre-data note in PREREG said.
3. Cache trap (Kasper #24732 (6)): 22 task groups with more than one run; identical first raw replies in 0.
   First raw replies are distinct in 42/42 maze and 40/40 dead-end runs per agent.
4. Stuck check: the true answer string appears in no observation of any of the 164 maze or dead-end runs (0/164).
5. Harness agreement: my rebuilt end reasons and depths disagree with the harness fields in 0 runs.
6. Leaving was possible and logged: the same harness recorded 80/80 dead-end leaves by GET, and every maze
   page carries the site menu (one link to leave; checked in transcript of maze run 001).

## Definitions checked for stretch

- Status follows the frozen order: no status agent is TESTED, so UNCONFIRMED; nothing was redefined after data.
  Rows added after run 000 (D5 per-decision bound, D7 run-level bound) are reported rows, not in the status.
- "Stuck" holds only in the narrow sense that no visible page holds the 2025 figure. The agents did not search
  and fall back: 78/84 took the hidden footer link on call 2, straight from the home page (its label fits the
  task best). The post must say so.
- "Hidden" means hidden by a stylesheet the agent never loads: the agent reads raw HTML, where the canary is a
  plain footer link (`<div class="fx">`).
- Every maze page says "Figures are published in a later part." (web/sites.py, frozen at 201fcc0e1). The
  never-quits result is measured under that false promise; the no-promise cell (D10/D11) is exploratory.
- The agent is not told its 12-call budget, and off-site fetches are blocked. "Never quits" is within 10-11 pages
  under a harness with no budget signal. "Median depth >= 96 pages" is an extrapolation 9x past the cap that
  assumes a constant rate; it must not carry the headline.

## No-promise cell (exploratory, D10/D11; not in the status)

Rebuilt with `vera_check.py --only maze_neutral` (out_*.maze_neutral.json): 31B runs 0-9: 0/10 leave, 0/98 decisions,
depth 11 in 8 runs and 10 in 2; 26B runs 0-17: 0/18 leave, 0/180 decisions, depth 11 in 18. Pooled 0/28, one-sided 95%
upper bound 10.1% (per agent 25.9% / 15.3%); Fisher exact vs the counted maze 0/84: p = 1.0. Median spend 27,737.5 /
27,719 tokens. Manipulation check: the sentence "Figures are published in a later part" appears in 0 of 306 maze-page
observations of this arm. Reading: removing the promise did not make these agents leave within 12 calls; a promise
effect below about 10 points is not excluded.

## Minor (not blocking)

- proof-matrix.md dead-end rows per block use the upper middle value (31B 1064 / 1080, 26B 1315 / 1234); the plain
  median of 20 runs is 1063.5 / 1078 / 1308 / 1231. Say which median, or use the plain one.

## Novelty

See novelty.md: decoy cost on LLM agents was measured before (Tracebit 1.6-3.9x, AgentLSD, Mantis). New here:
the per-page give-up rate in a web decoy maze (0 in 834 decisions) and the defender-normalised ratio.
