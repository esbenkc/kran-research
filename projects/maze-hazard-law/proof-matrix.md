# Proof matrix — maze-hazard-law

One row per cell. Built by `proof_matrix.py` from `results/analysis.json` (status UNCONFIRMED, prediction unsettled). Headline cells are the status agents (gpt-oss-20b, Qwen3-8B, Esben's laptop queue); side rows are never status.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | gpt-oss-20b maze, block i % 3 == 0 | none | waiting for laptop queue (job f0jxnah2hu); no run yet | data/gpt-oss-20b-low |
| headline | 1 | gpt-oss-20b maze, block i % 3 == 1 | none | waiting for laptop queue (job f0jxnah2hu); no run yet | data/gpt-oss-20b-low |
| headline | 2 | gpt-oss-20b maze, block i % 3 == 2 | none | waiting for laptop queue (job f0jxnah2hu); no run yet | data/gpt-oss-20b-low |
| headline | 0 | Qwen3-8B maze, block i % 3 == 0 | none | waiting for laptop queue (job erbnwpu5t3); no run yet | data/qwen3-8b-nothink |
| headline | 1 | Qwen3-8B maze, block i % 3 == 1 | none | waiting for laptop queue (job erbnwpu5t3); no run yet | data/qwen3-8b-nothink |
| headline | 2 | Qwen3-8B maze, block i % 3 == 2 | none | waiting for laptop queue (job erbnwpu5t3); no run yet | data/qwen3-8b-nothink |
| side row, closed API model, never status (D4): skill gate | - | gpt-5-nano (closed, API; pipeline check) visible link | none | 8/10 correct (>= 8 needed) | data/gpt-5-nano/raw/visible |
| side row, closed API model, never status (D4) | all | gpt-5-nano (closed, API; pipeline check) maze, pooled | none | verdict FAILS; give-ups G=32 (pages 1-3: 4/95 decisions); h3=0.0421; h_all=0.147; KM median 6 pages vs 3-page forecast 16.1 (R=2.69); chi-square p=5.04e-05 (df 2); Weibull shape k=2.2 p=1.79e-06; cap share 0; entered 32/40 | results/analysis.json |
| side row, closed API model, never status (D4) | 0 | gpt-5-nano (closed, API; pipeline check) maze, block i % 3 == 0 | none | entered 12; give-ups G=12 (pages 1-3 1/36); h_all=0.148; KM median 6; R_b=4.1 | data/gpt-5-nano/raw/maze |
| side row, closed API model, never status (D4) | 1 | gpt-5-nano (closed, API; pipeline check) maze, block i % 3 == 1 | none | entered 11; give-ups G=11 (pages 1-3 1/33); h_all=0.126; KM median 6; R_b=3.75 | data/gpt-5-nano/raw/maze |
| side row, closed API model, never status (D4) | 2 | gpt-5-nano (closed, API; pipeline check) maze, block i % 3 == 2 | none | entered 9; give-ups G=9 (pages 1-3 2/26); h_all=0.184; KM median 6; R_b=1.44 | data/gpt-5-nano/raw/maze |
| side row, closed API model, never status (D4): dead-end control | 0 | gpt-5-nano (closed, API; pipeline check) dead end, block i % 2 == 0 | one dead-end page, no continue link | entered 9/10; pages read [1, 1, 1, 1, 1, 1, 1, 1, 1]; causes ['giveup_answer'] | data/gpt-5-nano/raw/deadend |
| side row, closed API model, never status (D4): dead-end control | 1 | gpt-5-nano (closed, API; pipeline check) dead end, block i % 2 == 1 | one dead-end page, no continue link | entered 8/10; pages read [1, 1, 1, 1, 1, 1, 1, 1]; causes ['giveup_answer', 'giveup_get'] | data/gpt-5-nano/raw/deadend |
| side row, closed API model, never status (D4): dead-end control | all | gpt-5-nano (closed, API; pipeline check) dead end, pooled | one dead-end page | entered 17/20; share that left after page 1 1; median calls 4 | results/analysis.json |
| side row, closed API model, never status (D4): any exit (defender row) | all | gpt-5-nano (closed, API; pipeline check) maze, every exit counts | none | h3=0.0421; h_all=0.147; KM median 6; R=2.69 | results/analysis.json |
| side row, closed API model, never status (D4): exit causes | all | gpt-5-nano (closed, API; pipeline check) maze | none | {"giveup_answer": 32}; parse/protocol errors {"parse_fail": 0, "unissued_m": 0, "offsite": 0, "unknown_tool": 0}; answers with an invented 2025 figure 5, quoting the site's 2024 figure 4 | results/analysis.json |
| side row gate (D4): skill gate | - | Llama-3.2-1B visible link | none | 0/1 correct (>= 8 needed) | data/llama-3.2-1b/raw/visible |
| side row gate (D4): skill gate | - | Qwen3-1.7B visible link | none | 1/10 correct (>= 8 needed) | data/qwen3-1.7b-nothink/raw/visible |
| side row gate (D4): skill gate | - | Qwen3-1.7B thinking (D4, gate only) visible link | none | 0/3 correct (>= 8 needed) | data/qwen3-1.7b-think/raw/visible |
| side row rebuild (skeptic) | all | gpt-5-nano maze, Vera's own code | none | FAILS; G3/E3 4/95, h_all 0.1475, KM median 6, M_pred 16.11, R 2.686, chi p 5.04e-05, Weibull k 2.197 p 1.79e-06; 0 agent/server log mismatches | skeptics/skeptic/rebuild.json |
| side row null (skeptic) | all | constant hazard 0.1475, 2000 draws | parametric bootstrap | share of draws reaching the observed chi p 0, Weibull k 0; null k 95th pct 1.33; per-block k 2.35/2.41/2.26 | skeptics/skeptic/nulls_gpt-5-nano.json |

side row (not the headline): gpt-5-nano (closed, API; pipeline check) h_all = 0.153 [0.081, 0.225] (t-interval over 3 seed blocks)
