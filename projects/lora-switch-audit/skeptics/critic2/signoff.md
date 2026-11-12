# critic2 sign-off: lora-switch-audit (final cross-family review)

Run on Claude (opus) after the openai/gpt-5.5 critic2 hit its $2.48 cap at 12:30 PST ($2.69, no verdict). Verdict: PASS (room #20775).

## Rebuild
`rebuild.py` is my own code, not audit.py: my own RMSNorm with the public Qwen3-0.6B block-26 input_layernorm weight, my own cue planting (cast as fed, cue rows masked per run), SVD of the stacked q/k/v delta (audit.py uses eigh of G), my own rank rule, and two nulls (the frozen seed 20261112 and a fresh seed 7).
I ran it in a clean `git worktree` of origin/swarm/theory-1112-weight-only (raw/ absent), so it read data/ only: 75 s, 744 MB RSS, 50 cells, 0 rank differences from my earlier run on raw/. Output: `rebuild_output.txt`, `rebuild_output.json`.

| claim | post | rebuild (frozen null; fresh null) |
|---|---|---|
| C1 top direction rank 1 at 256 | 2 of 6 | 2 of 6: 301/1/705/1/976/256 (285/1/713/1/986/263); summary.json has 302 for u1110, a one-null fp tie |
| C1 at switch | 0 of 6 | 0 of 6: 392/508/681/754/949/274 |
| reaction rank 1 at 256 / at T | 6/6, 4/6 (3rd, 129th) | 6/6; 4/6 (3, 129) |
| controls (6 primary reads) | ranks 179-791, none in top 10 | e1 min 179, gain min 197, max 791 |
| C3 0.0036% at 20,480 | e1 587th, reaction 7th vs 355th no-cue | 587; 7 vs 358 (fresh 7 vs 319) |
| length u1110 / u1112 | 18x, 10x; first from 512 / 704 | 8.97/12.12/15.59/18.35x; 1.51 (rank 3)/6.68/6.63/9.86x at 512/704/1,728/4,352 |
| rows 0-2 learned runs read past 1,500 | 8 of 8 rank 1 | 8 of 8 (sweep 0.07%, 0.036%; ext 2,624, 6,528; len x2; nocastk8 x2) |
| tokens 2-3 learned runs past 1,500 | 5 of 6, the 1.4% one 32nd at 4,800 | 5 of 6; row12_x8.45 32nd (0.81x; fresh 35th) |
| real build leak | 439/302 and 2/1 | 439/302 (fresh 406/266); 2/1 (fresh 4/1) |
| real leak on tokens 2-3 | about 99.8% | 99.81% of the post-norm energy (raw energy: 92.7% on token 1) |
| sqrt(g) vs top singular value at switch | 7 to 17 times | 7.4 to 17.2 (55 to 296 in energy) |

## Issues found and fixed during this review
1. Receipts: the pushed data could not rebuild any score (block-26 rows 0-2, real cue rows, length and paired-C3 snapshots were only in gitignored raw/). Bruno pushed them in 421bc3274; the clean-worktree run above proves they are enough.
2. Overclaim: 'every learned random-cue run trained past 1,500 updates ranked first' was false over the parent tokens-2-3 runs (row12_x8.45 32nd). It is now scoped to the first three tokens in the text and the figure head (blog a0ff010, fig v9), and the exception has its own line.

## Checks not redone
I did not re-run training or the behaviour evals (T values come from the run logs). eq_check.py rerun: ALL PASS.
