# Proof matrix — maze-cost

One row per cell. Built by `proof_matrix.py` from `results/analysis.json` (status UNCONFIRMED, prediction failed).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate | - | gemma-4-26b-a4b-it visible link | none | 10/10 correct (>= 7 needed) | data/gemma-4-26b-a4b-it/raw/visible/ |
| headline | all | gemma-4-26b-a4b-it maze | none | never quits: spend set by the cap; h=0, d*=-, P=- tok, O=- tok, R=-, run-level 95% CI None (n entered 42/42) | results/analysis.json agents.gemma-4-26b-a4b-it |
| headline | 0 | gemma-4-26b-a4b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-26b-a4b-it/raw/maze/ (i % 3 == 0) |
| headline | 1 | gemma-4-26b-a4b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-26b-a4b-it/raw/maze/ (i % 3 == 1) |
| headline | 2 | gemma-4-26b-a4b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-26b-a4b-it/raw/maze/ (i % 3 == 2) |
| dead end vs maze | 0 | gemma-4-26b-a4b-it dead end | dead-end page (no continue link) | n entered 20, median spend 1315 tok, median $ ratio 431 | data/gemma-4-26b-a4b-it/raw/deadend/ (i % 2 == 0) |
| dead end vs maze | 1 | gemma-4-26b-a4b-it dead end | dead-end page (no continue link) | n entered 20, median spend 1234 tok, median $ ratio 378 | data/gemma-4-26b-a4b-it/raw/deadend/ (i % 2 == 1) |
| constant leave rate | all | gemma-4-26b-a4b-it maze | one rate vs two (pages 1-3, 4+) | h1-3=0, h4+=0, LR p=1 | results/analysis.json agents.gemma-4-26b-a4b-it.constancy |
| leave-rate bound (added after run 000, DEVIATIONS) | all | gemma-4-26b-a4b-it maze | none | 0 leaves in 420 page decisions; 95% upper bound 0.00711 per page; median depth >= 98 pages; run level: 0/42 runs leave, 95% upper bound 0.0688 | results/analysis.json agents.gemma-4-26b-a4b-it |
| promise sentence (exploratory, D10) | 0 | gemma-4-26b-a4b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/6 entered runs leave; depths [11, 11, 11, 11, 11, 11] | data/gemma-4-26b-a4b-it/raw/maze_neutral/ (i % 3 == 0) |
| promise sentence (exploratory, D10) | 1 | gemma-4-26b-a4b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/6 entered runs leave; depths [11, 11, 11, 11, 11, 11] | data/gemma-4-26b-a4b-it/raw/maze_neutral/ (i % 3 == 1) |
| promise sentence (exploratory, D10) | 2 | gemma-4-26b-a4b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/6 entered runs leave; depths [11, 11, 11, 11, 11, 11] | data/gemma-4-26b-a4b-it/raw/maze_neutral/ (i % 3 == 2) |
| promise sentence (exploratory, D10) | all | gemma-4-26b-a4b-it maze vs no-promise maze | Fisher exact, two-sided | no promise 0/18 leave vs maze 0/42; p=1 | results/analysis.json agents.gemma-4-26b-a4b-it.maze_neutral |
| block split (secondary) | 0 -> 1,2 | gemma-4-26b-a4b-it maze | none | never quits: spend set by the cap; R=- | results/analysis.json agents.gemma-4-26b-a4b-it.secondary_block_split |
| cost ratio (price sheet) | all | gemma-4-26b-a4b-it maze | dead end | maze median 537x (band x0.1..x10: 53.7..5,365), dead end median 400x | results/analysis.json agents.gemma-4-26b-a4b-it |
| gate | - | gemma-4-31b-it visible link | none | 10/10 correct (>= 7 needed) | data/gemma-4-31b-it/raw/visible/ |
| headline | all | gemma-4-31b-it maze | none | never quits: spend set by the cap; h=0, d*=-, P=- tok, O=- tok, R=-, run-level 95% CI None (n entered 42/42) | results/analysis.json agents.gemma-4-31b-it |
| headline | 0 | gemma-4-31b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-31b-it/raw/maze/ (i % 3 == 0) |
| headline | 1 | gemma-4-31b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-31b-it/raw/maze/ (i % 3 == 1) |
| headline | 2 | gemma-4-31b-it maze | none | never quits: spend set by the cap; R_b=- (n entered 14, median depth 11) | data/gemma-4-31b-it/raw/maze/ (i % 3 == 2) |
| dead end vs maze | 0 | gemma-4-31b-it dead end | dead-end page (no continue link) | n entered 20, median spend 1064 tok, median $ ratio 612 | data/gemma-4-31b-it/raw/deadend/ (i % 2 == 0) |
| dead end vs maze | 1 | gemma-4-31b-it dead end | dead-end page (no continue link) | n entered 20, median spend 1080 tok, median $ ratio 628 | data/gemma-4-31b-it/raw/deadend/ (i % 2 == 1) |
| constant leave rate | all | gemma-4-31b-it maze | one rate vs two (pages 1-3, 4+) | h1-3=0, h4+=0, LR p=1 | results/analysis.json agents.gemma-4-31b-it.constancy |
| leave-rate bound (added after run 000, DEVIATIONS) | all | gemma-4-31b-it maze | none | 0 leaves in 414 page decisions; 95% upper bound 0.00721 per page; median depth >= 96 pages; run level: 0/42 runs leave, 95% upper bound 0.0688 | results/analysis.json agents.gemma-4-31b-it |
| promise sentence (exploratory, D10) | 0 | gemma-4-31b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/4 entered runs leave; depths [10, 11, 11, 11] | data/gemma-4-31b-it/raw/maze_neutral/ (i % 3 == 0) |
| promise sentence (exploratory, D10) | 1 | gemma-4-31b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/3 entered runs leave; depths [11, 11, 11] | data/gemma-4-31b-it/raw/maze_neutral/ (i % 3 == 1) |
| promise sentence (exploratory, D10) | 2 | gemma-4-31b-it maze, no promise sentence | maze page minus 'Figures are published in a later part.' | 0/3 entered runs leave; depths [10, 11, 11] | data/gemma-4-31b-it/raw/maze_neutral/ (i % 3 == 2) |
| promise sentence (exploratory, D10) | all | gemma-4-31b-it maze vs no-promise maze | Fisher exact, two-sided | no promise 0/10 leave vs maze 0/42; p=1 | results/analysis.json agents.gemma-4-31b-it.maze_neutral |
| block split (secondary) | 0 -> 1,2 | gemma-4-31b-it maze | none | never quits: spend set by the cap; R=- | results/analysis.json agents.gemma-4-31b-it.secondary_block_split |
| cost ratio (price sheet) | all | gemma-4-31b-it maze | dead end | maze median 1,359x (band x0.1..x10: 136..13,588), dead end median 621x | results/analysis.json agents.gemma-4-31b-it |
| synthetic null | - | scripted never-leaver (h = 0) | harness + analysis with no model | outcome 'never quits: spend set by the cap', h = 0 | web/test_analysis.py |
| synthetic positive | - | scripted geometric leaver (h = 0.15) | harness + analysis with no model | h recovered 0.147, TESTED, R 0.77 | web/test_analysis.py |

seed-level: gemma-4-26b-a4b-it share of entered maze runs that leave before the call limit = 0.000 [0.000, 0.000] (t-interval over 3 seed blocks: 0.000, 0.000, 0.000); zero spread across blocks, so the t-interval is degenerate: use the exact run-level bound 0.0688
seed-level: gemma-4-26b-a4b-it R = - [not computable, not computable] (t-interval over 0 defined seed blocks of log R_b, df = 0)
seed-level: gemma-4-31b-it share of entered maze runs that leave before the call limit = 0.000 [0.000, 0.000] (t-interval over 3 seed blocks: 0.000, 0.000, 0.000); zero spread across blocks, so the t-interval is degenerate: use the exact run-level bound 0.0688
seed-level: gemma-4-31b-it R = - [not computable, not computable] (t-interval over 0 defined seed blocks of log R_b, df = 0)
